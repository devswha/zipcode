use std::io::{self, IsTerminal, Stdout, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::Value;
use termimad::crossterm::cursor::{Hide, MoveTo, Show};
use termimad::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers,
    MouseEvent, MouseEventKind,
};
use termimad::crossterm::execute;
use termimad::crossterm::style::{
    Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor,
};
use termimad::crossterm::terminal::{
    self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen,
};
use zipcode_inference::Role;
use zipcode_runtime::{ConversationLoop, Session, StreamCallback};
use zipcode_tools::PermissionMode;

use crate::render::{terminal_width, truncate_to_width};
use crate::repl::{
    clear_session, compact_session, help_text, load_session_into_loop, parse_slash_command,
    prepare_loop, run_interactive, session_status_lines, CompactFeedback, ParsedSlashCommand,
    SlashCommand,
};
use crate::tui_composer::Composer;
use crate::UiMode;

/// Whether the TUI panic hook is currently installed.
static TUI_PANIC_HOOK_INSTALLED: AtomicBool = AtomicBool::new(false);

const HEADER_LINES: u16 = 2;
const STATUS_LINES: u16 = 2;
const HINT_LINES: u16 = 1;
const MAX_COMPOSER_LINES: usize = 5;
const STREAM_REDRAW_INTERVAL: Duration = Duration::from_millis(33);
const STREAM_REDRAW_MIN_BYTES: usize = 24;
const MOUSE_WHEEL_SCROLL_LINES: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TranscriptViewport {
    width: usize,
    height: usize,
}

pub fn run_interactive_with_ui(
    model_path: Option<&Path>,
    permission_mode: Option<&str>,
    backend_override: Option<&str>,
    session_id: Option<&str>,
    ui_mode: UiMode,
) -> Result<()> {
    let force_plain = std::env::var("ZIPCODE_NO_TUI").is_ok()
        || std::env::var("TERM").as_deref() == Ok("dumb")
        || !io::stdin().is_terminal()
        || !io::stdout().is_terminal();

    let effective_ui = if matches!(ui_mode, UiMode::Fullscreen) && force_plain {
        UiMode::Plain
    } else {
        ui_mode
    };

    match effective_ui {
        UiMode::Plain => run_interactive(model_path, permission_mode, backend_override, session_id),
        UiMode::Fullscreen => {
            run_interactive_fullscreen(model_path, permission_mode, backend_override, session_id)
        }
    }
}

fn run_interactive_fullscreen(
    model_path: Option<&Path>,
    permission_mode: Option<&str>,
    backend_override: Option<&str>,
    session_id: Option<&str>,
) -> Result<()> {
    let launch = prepare_loop(model_path, permission_mode, backend_override, session_id)?;
    let backend = match launch.effective_backend {
        zipcode_inference::Backend::LlamaCpp => "llama-cpp",
        zipcode_inference::Backend::LlamaServer => "llama-server",
        zipcode_inference::Backend::Candle => "candle",
    };
    let mut conv = launch.conv;
    let mut ui = FullscreenUi::new(&conv, backend.to_string(), &launch.startup_notices)?;
    ui.draw()?;

    if !run_automation_script_from_env(&mut ui, &mut conv)? {
        return ui.restore();
    }

    loop {
        match event::read().context("failed to read terminal input")? {
            Event::Key(key) => {
                if !ui.handle_key_event(key, &mut conv)? {
                    break;
                }
            }
            Event::Mouse(mouse) => ui.handle_mouse_event(mouse)?,
            Event::Resize(_, _) => ui.draw()?,
            _ => {}
        }
    }

    ui.restore()
}

fn run_automation_script_from_env(
    ui: &mut FullscreenUi,
    conv: &mut ConversationLoop,
) -> Result<bool> {
    let Ok(script) = std::env::var("ZIPCODE_TUI_AUTOMATION_SCRIPT") else {
        return Ok(true);
    };

    for line in script.lines() {
        if !ui.handle_submitted_input(line.to_string(), conv)? {
            return Ok(false);
        }
    }

    Ok(true)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntryKind {
    User,
    Assistant,
    /// Gemma 4 private reasoning channel. Rendered dimmed so users can
    /// follow the model's thinking without mistaking it for the final
    /// answer. Never persisted into session history — see `ConversationLoop`.
    Thinking,
    ToolStart,
    ToolResult,
    Info,
    Error,
    Permission,
}

struct TranscriptEntry {
    kind: EntryKind,
    content: String,
}

struct Overlay {
    title: String,
    body: Vec<String>,
}

struct FullscreenUi {
    stdout: Stdout,
    transcript: Vec<TranscriptEntry>,
    transcript_cache: TranscriptCache,
    composer: Composer,
    transcript_scroll: usize,
    status: String,
    backend: String,
    session_id: String,
    tool_count: usize,
    cwd: String,
    raw_enabled: bool,
    overlay: Option<Overlay>,
    draw_drops: usize,
    esc_armed: bool,
    /// Bytes of Gemma 4 reasoning streamed during the current turn.
    /// Surfaced in the status bar as "Thinking… (N chars)" so the user
    /// sees live progress without the reasoning itself polluting the
    /// transcript. Reset to 0 at the start of each `run_turn`.
    thinking_bytes: usize,
}

impl FullscreenUi {
    fn new(conv: &ConversationLoop, backend: String, startup_notices: &[String]) -> Result<Self> {
        // Install a panic hook that restores the terminal before printing the
        // panic message.  Without this, a panic leaves the terminal in raw mode
        // + alternate screen and the user sees a garbled shell.
        if !TUI_PANIC_HOOK_INSTALLED.swap(true, Ordering::SeqCst) {
            let prev_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                // Best-effort terminal restoration — ignore errors.
                let _ = terminal::disable_raw_mode();
                let _ = execute!(
                    io::stdout(),
                    Show,
                    DisableMouseCapture,
                    LeaveAlternateScreen
                );
                prev_hook(info);
            }));
        }

        let mut stdout = io::stdout();
        if let Err(e) = execute!(stdout, EnterAlternateScreen, EnableMouseCapture, Hide) {
            clear_panic_hook_flag();
            return Err(e.into());
        }
        if let Err(e) = terminal::enable_raw_mode() {
            let _ = execute!(stdout, Show, DisableMouseCapture, LeaveAlternateScreen);
            clear_panic_hook_flag();
            return Err(e.into());
        }

        let mut ui = Self {
            stdout,
            transcript: initial_transcript_entries(&conv.session),
            transcript_cache: TranscriptCache::default(),
            composer: Composer::new(),
            transcript_scroll: 0,
            status: "Ready".to_string(),
            backend,
            session_id: conv.session.id.clone(),
            tool_count: conv.tools.names().len(),
            cwd: conv.cwd.display().to_string(),
            raw_enabled: true,
            overlay: None,
            draw_drops: 0,
            esc_armed: false,
            thinking_bytes: 0,
        };
        for notice in startup_notices {
            ui.push_entry(EntryKind::Info, format!("[notice] {notice}"));
        }
        if !startup_notices.is_empty() {
            ui.status = "Check startup notices".to_string();
        }
        ui.draw()?;
        Ok(ui)
    }

    fn restore(&mut self) -> Result<()> {
        if self.raw_enabled {
            terminal::disable_raw_mode()?;
            self.raw_enabled = false;
        }
        execute!(self.stdout, Show, DisableMouseCapture, LeaveAlternateScreen)?;
        self.stdout.flush()?;
        Ok(())
    }

    fn handle_key_event(&mut self, key: KeyEvent, conv: &mut ConversationLoop) -> Result<bool> {
        if self.overlay.is_some() {
            match key.code {
                KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') => {
                    self.overlay = None;
                    self.status = "Ready".to_string();
                    self.draw()?;
                    return Ok(true);
                }
                _ => return Ok(true),
            }
        }

        if !matches!(key.code, KeyCode::Esc) {
            self.esc_armed = false;
        }

        match key {
            KeyEvent {
                code: KeyCode::Char('d'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => return Ok(false),
            KeyEvent {
                code: KeyCode::Char('c'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                self.composer.clear();
                self.status = "Input cancelled".to_string();
            }
            KeyEvent {
                code: KeyCode::Char('l'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                self.status = "Screen refreshed".to_string();
            }
            KeyEvent {
                code: KeyCode::Esc, ..
            } => {
                if self.composer.is_empty() {
                    match handle_empty_composer_escape(&mut self.composer, &mut self.esc_armed) {
                        EscapeOutcome::Armed => {
                            self.status = "Press Esc again to edit previous message".to_string();
                        }
                        EscapeOutcome::LoadedPrevious => {
                            self.status = "Editing previous message".to_string();
                        }
                        EscapeOutcome::NoPreviousMessage => {
                            self.status = "No previous message to edit".to_string();
                        }
                    }
                } else {
                    self.composer.clear();
                    self.esc_armed = false;
                    self.status = "Input cleared".to_string();
                }
            }
            KeyEvent {
                code: KeyCode::F(1),
                ..
            } => self.open_help_overlay(),
            KeyEvent {
                code: KeyCode::BackTab,
                ..
            } => cycle_permission_mode(self, conv),
            KeyEvent {
                code: KeyCode::PageUp,
                ..
            } => {
                let moved = self.scroll_transcript_by_page(ScrollDirection::Older)?;
                self.status = if moved == 0 {
                    "Already at the oldest visible history".to_string()
                } else {
                    format!("Scrolled up {} line(s)", moved)
                };
            }
            KeyEvent {
                code: KeyCode::PageDown,
                ..
            } => {
                let moved = self.scroll_transcript_by_page(ScrollDirection::Newer)?;
                self.status = if self.transcript_scroll == 0 {
                    "Back to latest output".to_string()
                } else {
                    format!("Scrolled down {} line(s)", moved)
                };
            }
            KeyEvent {
                code: KeyCode::Home,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                let moved = self.jump_to_oldest_transcript()?;
                self.status = if moved == 0 {
                    "Already at the oldest visible history".to_string()
                } else {
                    "Jumped to oldest visible history".to_string()
                };
            }
            KeyEvent {
                code: KeyCode::End,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                let moved = self.jump_to_latest_transcript();
                self.status = if moved == 0 {
                    "Already at latest output".to_string()
                } else {
                    "Jumped to latest output".to_string()
                };
            }
            KeyEvent {
                code: KeyCode::Left,
                ..
            } => self.composer.move_left(),
            KeyEvent {
                code: KeyCode::Right,
                ..
            } => self.composer.move_right(),
            KeyEvent {
                code: KeyCode::Home,
                ..
            } => self.composer.move_home(),
            KeyEvent {
                code: KeyCode::End, ..
            } => self.composer.move_end(),
            KeyEvent {
                code: KeyCode::Up, ..
            } => {
                if !self.composer.move_up() {
                    let _ = self.composer.history_previous();
                }
            }
            KeyEvent {
                code: KeyCode::Down,
                ..
            } => {
                if !self.composer.move_down() {
                    let _ = self.composer.history_next();
                }
            }
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => self.composer.backspace(),
            KeyEvent {
                code: KeyCode::Delete,
                ..
            } => self.composer.delete_forward(),
            KeyEvent {
                code: KeyCode::Enter,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL)
                || modifiers.contains(KeyModifiers::SHIFT)
                || modifiers.contains(KeyModifiers::ALT) =>
            {
                self.composer.insert_newline();
            }
            KeyEvent {
                code: KeyCode::Char('j'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => self.composer.insert_newline(),
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => {
                if self.composer.try_escape_newline() {
                    self.status = "Inserted newline".to_string();
                    self.draw()?;
                    return Ok(true);
                }
                let submitted = self.composer.submit();
                return self.handle_submitted_input(submitted, conv);
            }
            KeyEvent {
                code: KeyCode::Char(ch),
                modifiers,
                ..
            } if modifiers.is_empty() || modifiers == KeyModifiers::SHIFT => {
                self.composer.insert_char(ch);
            }
            _ => {}
        }

        self.draw()?;
        Ok(true)
    }

    fn handle_mouse_event(&mut self, mouse: MouseEvent) -> Result<()> {
        if self.overlay.is_some() {
            return Ok(());
        }

        let Some(direction) = mouse_scroll_direction(mouse.kind) else {
            return Ok(());
        };

        let moved = self.scroll_transcript_by_lines(direction, MOUSE_WHEEL_SCROLL_LINES)?;
        self.status = match direction {
            ScrollDirection::Older if moved == 0 => {
                "Already at the oldest visible history".to_string()
            }
            ScrollDirection::Older => format!("Scrolled up {} line(s)", moved),
            ScrollDirection::Newer if self.transcript_scroll == 0 => {
                "Back to latest output".to_string()
            }
            ScrollDirection::Newer => format!("Scrolled down {} line(s)", moved),
        };
        self.draw()?;
        Ok(())
    }

    fn handle_submitted_input(
        &mut self,
        input: String,
        conv: &mut ConversationLoop,
    ) -> Result<bool> {
        self.transcript_scroll = 0;
        if input.trim().is_empty() {
            self.status = "Ready".to_string();
        } else {
            match parse_slash_command(input.trim()) {
                ParsedSlashCommand::Command(command, argument) => {
                    if !self.handle_slash_command(command, argument.as_deref(), conv)? {
                        return Ok(false);
                    }
                }
                ParsedSlashCommand::Error(message) => {
                    self.push_entry(EntryKind::Error, message);
                    self.status = "Command error".to_string();
                }
                ParsedSlashCommand::NotCommand => self.run_turn(input, conv)?,
            }
        }
        self.draw()?;
        Ok(true)
    }

    fn handle_slash_command(
        &mut self,
        command: SlashCommand,
        argument: Option<&str>,
        conv: &mut ConversationLoop,
    ) -> Result<bool> {
        match command {
            SlashCommand::Help => self.open_help_overlay(),
            SlashCommand::Status => {
                self.overlay = Some(Overlay {
                    title: "Session Status".to_string(),
                    body: vec![
                        format!("Session ID: {}", conv.session.id),
                        format!("Messages: {}", conv.session.messages.len()),
                        format!("Tools: {}", conv.tools.names().len()),
                        format!("Backend: {}", self.backend),
                        format!(
                            "Permission mode: {}",
                            permission_mode_label(conv.permission.mode())
                        ),
                        format!("Working dir: {}", conv.cwd.display()),
                    ],
                });
                self.status = "Status opened".to_string();
            }
            SlashCommand::SessionShow => {
                for line in session_status_lines(conv) {
                    self.push_entry(EntryKind::Info, line);
                }
                self.status = "Session details shown".to_string();
            }
            SlashCommand::SessionLoad => {
                match load_session_into_loop(conv, argument.expect("session load requires an id")) {
                    Ok(message) => {
                        self.set_transcript_from_session(&conv.session);
                        self.push_entry(EntryKind::Info, message);
                        self.session_id = conv.session.id.clone();
                        self.status = "Session loaded".to_string();
                    }
                    Err(error) => {
                        self.push_entry(EntryKind::Error, error.to_string());
                        self.status = "Session load failed".to_string();
                    }
                }
            }
            SlashCommand::Compact => match compact_session(conv) {
                Ok(feedback) => {
                    self.set_transcript_from_session(&conv.session);
                    self.push_entry(EntryKind::Info, feedback.message().to_string());
                    self.status = match feedback {
                        CompactFeedback::Compacted(_) => "Compaction complete".to_string(),
                        CompactFeedback::Skipped(_) => "Compaction skipped".to_string(),
                    };
                }
                Err(error) => {
                    self.push_entry(EntryKind::Error, error.to_string());
                    self.status = "Compaction failed".to_string();
                }
            },
            SlashCommand::Clear => match clear_session(conv) {
                Ok(message) => {
                    self.transcript.clear();
                    self.transcript_cache = TranscriptCache::default();
                    self.push_entry(EntryKind::Info, message);
                    self.session_id = conv.session.id.clone();
                    self.status = "Conversation cleared".to_string();
                }
                Err(error) => {
                    self.push_entry(EntryKind::Error, error.to_string());
                    self.status = "Conversation clear failed".to_string();
                }
            },
            SlashCommand::Quit => return Ok(false),
        }

        Ok(true)
    }

    fn open_help_overlay(&mut self) {
        let mut body: Vec<String> = help_text().lines().map(str::to_string).collect();
        body.extend([
            "".to_string(),
            "Fullscreen keys:".to_string(),
            "  Enter      submit".to_string(),
            "  \\ + Enter  newline (Claude Code quick escape)".to_string(),
            "  Ctrl+J      newline".to_string(),
            "  Shift+Enter newline".to_string(),
            "  Option+Enter newline".to_string(),
            "  Arrow keys  move cursor / recall history".to_string(),
            "  PgUp/PgDn   scroll transcript by page".to_string(),
            "  Mouse wheel scroll transcript by line".to_string(),
            "  Ctrl+Home   jump to oldest visible history".to_string(),
            "  Ctrl+End    jump to latest output".to_string(),
            "  Shift+Tab   cycle permission mode".to_string(),
            "  F1          help".to_string(),
            "  Esc Esc     edit previous message".to_string(),
            "  Esc         clear input / close overlay".to_string(),
            "  Ctrl+L      refresh screen".to_string(),
            "  Ctrl+C      cancel input".to_string(),
            "  Ctrl+D      exit (same as /quit or /exit)".to_string(),
        ]);
        self.overlay = Some(Overlay {
            title: "Help".to_string(),
            body,
        });
        self.status = "Help opened".to_string();
    }

    fn run_turn(&mut self, input: String, conv: &mut ConversationLoop) -> Result<()> {
        self.push_entry(EntryKind::User, input.trim_end().to_string());
        self.status = "Thinking…".to_string();
        self.thinking_bytes = 0;
        self.draw()?;

        let mut cb = TuiCallback {
            ui: self,
            last_stream_draw: None,
            pending_stream_bytes: 0,
        };
        let result = conv.run_turn(&input, &mut cb);
        cb.flush_pending_stream_draw();
        cb.ui.session_id = conv.session.id.clone();
        cb.ui.status = if result.is_ok() {
            "Ready".to_string()
        } else {
            "Last turn failed".to_string()
        };
        cb.ui.draw()?;
        result
    }

    fn push_entry(&mut self, kind: EntryKind, content: String) {
        self.transcript.push(TranscriptEntry { kind, content });
        self.transcript_scroll = 0;
        let Some(entry) = self.transcript.last() else {
            return;
        };
        self.transcript_cache.append_entry(entry);
    }

    fn append_assistant_token(&mut self, token: &str) {
        match self.transcript.last_mut() {
            Some(TranscriptEntry {
                kind: EntryKind::Assistant,
                content,
            }) => content.push_str(token),
            _ => self.push_entry(EntryKind::Assistant, token.to_string()),
        }
        self.refresh_last_transcript_cache_entry();
    }

    /// Append a Gemma 4 reasoning chunk to the transcript. Reasoning entries
    /// are kept separate from assistant entries so styling can dim them and
    /// so the next regular `append_assistant_token` opens a fresh Assistant
    /// entry — visually marking the transition from "thinking" to "answer".
    fn append_thinking_token(&mut self, token: &str) {
        match self.transcript.last_mut() {
            Some(TranscriptEntry {
                kind: EntryKind::Thinking,
                content,
            }) => content.push_str(token),
            _ => self.push_entry(EntryKind::Thinking, token.to_string()),
        }
        self.refresh_last_transcript_cache_entry();
    }

    fn transcript_viewport(&mut self) -> Result<TranscriptViewport> {
        let (width, height) = terminal::size()?;
        let usable_width = width.saturating_sub(2) as usize;
        let composer_width = width.saturating_sub(4) as usize;
        let composer_lines = self.composer.wrapped_lines(composer_width.max(1));
        let composer_visible = composer_lines.len().clamp(1, MAX_COMPOSER_LINES);
        let composer_height = composer_visible as u16;
        let status_y = height.saturating_sub(HINT_LINES + STATUS_LINES + composer_height);
        let transcript_height = status_y.saturating_sub(HEADER_LINES) as usize;
        let viewport = TranscriptViewport {
            width: usable_width.max(8),
            height: transcript_height.max(1),
        };
        self.ensure_transcript_cache(viewport.width);
        Ok(viewport)
    }

    fn max_transcript_scroll(&mut self) -> Result<usize> {
        let viewport = self.transcript_viewport()?;
        Ok(self
            .transcript_cache
            .lines
            .len()
            .saturating_sub(viewport.height))
    }

    fn scroll_transcript_by_lines(
        &mut self,
        direction: ScrollDirection,
        lines: usize,
    ) -> Result<usize> {
        let previous = self.transcript_scroll;
        let max_scroll = self.max_transcript_scroll()?;
        self.transcript_scroll = scroll_offset_after_delta(previous, max_scroll, lines, direction);
        Ok(previous.abs_diff(self.transcript_scroll))
    }

    fn scroll_transcript_by_page(&mut self, direction: ScrollDirection) -> Result<usize> {
        let viewport = self.transcript_viewport()?;
        let page = viewport.height.saturating_sub(2).max(1);
        self.scroll_transcript_by_lines(direction, page)
    }

    fn jump_to_oldest_transcript(&mut self) -> Result<usize> {
        let previous = self.transcript_scroll;
        self.transcript_scroll = self.max_transcript_scroll()?;
        Ok(previous.abs_diff(self.transcript_scroll))
    }

    fn jump_to_latest_transcript(&mut self) -> usize {
        let previous = self.transcript_scroll;
        self.transcript_scroll = 0;
        previous
    }

    fn set_transcript_from_session(&mut self, session: &Session) {
        self.transcript = initial_transcript_entries(session);
        self.transcript_cache = TranscriptCache::default();
        self.transcript_scroll = 0;
    }

    fn prompt_for_permission(&mut self, message: &str) -> Result<bool> {
        self.push_entry(EntryKind::Permission, format!("{message} [Y/n]"));
        self.status = "Permission required".to_string();
        self.draw()?;

        loop {
            if let Event::Key(key) = event::read().context("failed to read permission input")? {
                match key.code {
                    KeyCode::Enter => return Ok(true),
                    KeyCode::Char('y') | KeyCode::Char('Y') => return Ok(true),
                    KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => return Ok(false),
                    _ => {}
                }
            }
        }
    }

    fn draw(&mut self) -> Result<()> {
        let (width, height) = terminal::size()?;
        let transcript_top = HEADER_LINES;
        let composer_width = width.saturating_sub(4) as usize;
        let composer_lines = self.composer.wrapped_lines(composer_width.max(1));
        let composer_visible = composer_lines.len().clamp(1, MAX_COMPOSER_LINES);
        let composer_height = composer_visible as u16;
        let status_y = height.saturating_sub(HINT_LINES + STATUS_LINES + composer_height);
        let composer_label_y = status_y + 1;
        let composer_top = composer_label_y + 1;
        let hint_y = height.saturating_sub(1);
        let transcript_height = status_y.saturating_sub(transcript_top) as usize;

        self.begin_sync_output();

        execute!(self.stdout, MoveTo(0, 0), Clear(ClearType::All))?;
        self.draw_header(width)?;
        self.draw_transcript(width, transcript_top, transcript_height)?;
        self.draw_footer(width, status_y, composer_top, composer_height, hint_y)?;
        if self.overlay.is_some() {
            self.draw_overlay(width, height)?;
        } else {
            self.position_cursor(width, composer_top, composer_lines.len(), composer_visible)?;
        }

        self.end_sync_output()?;
        Ok(())
    }

    fn draw_stream_frame(&mut self) -> Result<()> {
        if self.overlay.is_some() {
            return self.draw();
        }

        let (width, height) = terminal::size()?;
        let transcript_top = HEADER_LINES;
        let composer_width = width.saturating_sub(4) as usize;
        let composer_lines = self.composer.wrapped_lines(composer_width.max(1));
        let composer_visible = composer_lines.len().clamp(1, MAX_COMPOSER_LINES);
        let composer_height = composer_visible as u16;
        let status_y = height.saturating_sub(HINT_LINES + STATUS_LINES + composer_height);
        let composer_label_y = status_y + 1;
        let composer_top = composer_label_y + 1;
        let hint_y = height.saturating_sub(1);
        let transcript_height = status_y.saturating_sub(transcript_top) as usize;

        self.begin_sync_output();
        self.clear_region(transcript_top, transcript_height as u16)?;
        self.clear_region(status_y, height.saturating_sub(status_y))?;
        self.draw_transcript(width, transcript_top, transcript_height)?;
        self.draw_footer(width, status_y, composer_top, composer_height, hint_y)?;
        self.position_cursor(width, composer_top, composer_lines.len(), composer_visible)?;
        self.end_sync_output()?;
        Ok(())
    }

    fn clear_region(&mut self, top: u16, height: u16) -> Result<()> {
        for row in 0..height {
            execute!(
                self.stdout,
                MoveTo(0, top + row),
                Clear(ClearType::CurrentLine)
            )?;
        }
        Ok(())
    }

    fn begin_sync_output(&mut self) {
        let _ = self.stdout.write_all(b"\x1b[?2026h");
    }

    fn end_sync_output(&mut self) -> Result<()> {
        let _ = self.stdout.write_all(b"\x1b[?2026l");
        self.stdout.flush()?;
        Ok(())
    }

    fn draw_header(&mut self, width: u16) -> Result<()> {
        execute!(
            self.stdout,
            MoveTo(0, 0),
            SetForegroundColor(Color::Cyan),
            SetAttribute(Attribute::Bold),
            Print(truncate_to_width(
                &format!(
                    " zipcode  [{}]  fullscreen  session:{}  tools:{} ",
                    self.backend,
                    truncate_to_width(&self.session_id, 12),
                    self.tool_count
                ),
                width as usize
            )),
            ResetColor,
            SetAttribute(Attribute::Reset),
            MoveTo(0, 1),
            SetForegroundColor(Color::DarkGrey),
            Print(truncate_to_width(
                &format!(" cwd: {}", self.cwd),
                width as usize
            )),
            ResetColor,
        )?;
        Ok(())
    }

    fn draw_transcript(&mut self, width: u16, top: u16, height: usize) -> Result<()> {
        let usable_width = width.saturating_sub(2) as usize;
        self.ensure_transcript_cache(usable_width.max(8));

        let visible = visible_tail(&self.transcript_cache.lines, height, self.transcript_scroll);
        for (idx, line) in visible.iter().enumerate() {
            execute!(
                self.stdout,
                MoveTo(0, top + idx as u16),
                SetForegroundColor(line.color),
                Print(truncate_to_width(&line.text, width as usize)),
                ResetColor,
            )?;
        }
        Ok(())
    }

    fn ensure_transcript_cache(&mut self, width: usize) {
        if self.transcript_cache.width == Some(width)
            && self.transcript_cache.entry_line_counts.len() == self.transcript.len()
        {
            return;
        }
        self.transcript_cache.rebuild(&self.transcript, width);
    }

    fn refresh_last_transcript_cache_entry(&mut self) {
        let Some(entry) = self.transcript.last() else {
            self.transcript_cache = TranscriptCache::default();
            return;
        };
        self.transcript_cache.replace_last_entry(entry);
    }

    fn draw_footer(
        &mut self,
        width: u16,
        status_y: u16,
        composer_top: u16,
        composer_height: u16,
        hint_y: u16,
    ) -> Result<()> {
        execute!(
            self.stdout,
            MoveTo(0, status_y),
            SetForegroundColor(Color::DarkGrey),
            Print("─".repeat(width as usize)),
            ResetColor,
            MoveTo(0, status_y + 1),
            SetForegroundColor(Color::Yellow),
            Print(truncate_to_width(
                &format!(
                    " {}  •  {}  •  composer:{} line(s)",
                    self.status,
                    scroll_status_label(self.transcript_scroll),
                    self.composer
                        .wrapped_lines(width.saturating_sub(4) as usize)
                        .len()
                ),
                width as usize
            )),
            ResetColor,
            MoveTo(0, composer_top - 1),
            SetForegroundColor(Color::DarkGrey),
            Print(truncate_to_width(
                " Compose (Enter submit • Ctrl+J newline • F1 help) ",
                width as usize,
            )),
            ResetColor,
        )?;

        let composer_lines = self
            .composer
            .wrapped_lines(width.saturating_sub(4) as usize);
        let visible_start = composer_lines
            .len()
            .saturating_sub(composer_height as usize);
        for row in 0..composer_height {
            let line = composer_lines
                .get(visible_start + row as usize)
                .cloned()
                .unwrap_or_default();
            execute!(
                self.stdout,
                MoveTo(0, composer_top + row),
                SetForegroundColor(Color::Green),
                Print(if row == 0 { "> " } else { "· " }),
                ResetColor,
                Print(truncate_to_width(&line, width.saturating_sub(2) as usize)),
            )?;
        }

        execute!(
            self.stdout,
            MoveTo(0, hint_y),
            SetForegroundColor(Color::DarkGrey),
            Print(truncate_to_width(
                " /help • /status • /session • /compact • /clear • /quit (/exit) • Ctrl+D exit • PgUp/PgDn page • Ctrl+Home/End top/latest ",
                width as usize,
            )),
            ResetColor,
        )?;
        Ok(())
    }

    fn draw_overlay(&mut self, width: u16, height: u16) -> Result<()> {
        let Some(overlay) = &self.overlay else {
            return Ok(());
        };
        let box_width = width.saturating_sub(8).max(20);
        let box_left = (width.saturating_sub(box_width)) / 2;
        let body_width = box_width.saturating_sub(4) as usize;
        let mut body_lines = Vec::new();
        for line in &overlay.body {
            body_lines.extend(wrap_plain(line, body_width.max(1)));
        }
        let box_height = (body_lines.len() as u16 + 4)
            .min(height.saturating_sub(2))
            .max(6);
        let box_top = (height.saturating_sub(box_height)) / 2;

        for row in 0..box_height {
            execute!(
                self.stdout,
                MoveTo(box_left, box_top + row),
                SetForegroundColor(Color::DarkGrey),
                Print(" ".repeat(box_width as usize)),
                ResetColor,
            )?;
        }

        execute!(
            self.stdout,
            MoveTo(box_left, box_top),
            SetForegroundColor(Color::Cyan),
            SetAttribute(Attribute::Bold),
            Print(truncate_to_width(
                &format!(" {} ", overlay.title),
                box_width as usize
            )),
            ResetColor,
            SetAttribute(Attribute::Reset),
        )?;

        let available_body = box_height.saturating_sub(3) as usize;
        for (idx, line) in body_lines.into_iter().take(available_body).enumerate() {
            execute!(
                self.stdout,
                MoveTo(box_left + 2, box_top + 1 + idx as u16),
                Print(truncate_to_width(&line, body_width)),
            )?;
        }

        execute!(
            self.stdout,
            MoveTo(box_left + 2, box_top + box_height - 1),
            SetForegroundColor(Color::DarkGrey),
            Print("Esc / Enter / q to close"),
            ResetColor,
        )?;
        Ok(())
    }

    fn position_cursor(
        &mut self,
        width: u16,
        composer_top: u16,
        total_lines: usize,
        visible_lines: usize,
    ) -> Result<()> {
        let (cursor_row, cursor_col) = self
            .composer
            .cursor_visual_position(width.saturating_sub(4) as usize);
        let visible_start = total_lines.saturating_sub(visible_lines);
        let display_row = cursor_row
            .saturating_sub(visible_start)
            .min(visible_lines.saturating_sub(1));
        let prefix = 2u16;
        execute!(
            self.stdout,
            MoveTo(
                prefix + cursor_col as u16,
                composer_top + display_row as u16
            )
        )?;
        Ok(())
    }
}

impl Drop for FullscreenUi {
    fn drop(&mut self) {
        let _ = self.restore();
        clear_panic_hook_flag();
    }
}

/// Mark the TUI panic hook as no longer needed.  The hook itself remains
/// installed (it is harmless when the TUI is not active) but the flag is
/// cleared so a future TUI session can re-install if needed.
fn clear_panic_hook_flag() {
    TUI_PANIC_HOOK_INSTALLED.store(false, Ordering::SeqCst);
}

struct TuiCallback<'a> {
    ui: &'a mut FullscreenUi,
    last_stream_draw: Option<Instant>,
    pending_stream_bytes: usize,
}

impl StreamCallback for TuiCallback<'_> {
    fn on_token(&mut self, text: &str) {
        self.ui.append_assistant_token(text);
        self.ui.status = "Responding…".to_string();
        self.pending_stream_bytes = self.pending_stream_bytes.saturating_add(text.len());
        if self.should_draw_stream_update(text) {
            self.try_draw();
            self.last_stream_draw = Some(Instant::now());
            self.pending_stream_bytes = 0;
        }
    }

    fn on_thinking(&mut self, text: &str) {
        // Silent default: keep the "Thinking…" header in the status bar
        // and update a live character counter so the user sees progress
        // without the reasoning itself flooding the transcript. Verbose
        // mode (`--verbose` / `ZIPCODE_VERBOSE=1`) additionally streams
        // the raw reasoning into a dedicated dimmed transcript entry.
        self.ui.thinking_bytes = self.ui.thinking_bytes.saturating_add(text.len());
        self.ui.status = format!("Thinking… ({} chars)", self.ui.thinking_bytes);

        if crate::render::is_verbose() {
            self.ui.append_thinking_token(text);
        }

        self.pending_stream_bytes = self.pending_stream_bytes.saturating_add(text.len());
        if self.should_draw_stream_update(text) {
            self.try_draw();
            self.last_stream_draw = Some(Instant::now());
            self.pending_stream_bytes = 0;
        }
    }

    fn on_tool_start(&mut self, name: &str, args: &Value) {
        self.ui.push_entry(
            EntryKind::ToolStart,
            format!(
                "{name}({})",
                truncate_to_width(&args.to_string(), terminal_width().min(120))
            ),
        );
        self.ui.status = format!("Running {name}…");
        self.pending_stream_bytes = 0;
        self.last_stream_draw = Some(Instant::now());
        self.try_draw();
    }

    fn on_tool_result(&mut self, name: &str, result: &str) {
        self.ui.push_entry(
            EntryKind::ToolResult,
            format!(
                "{name}: {}",
                truncate_to_width(result.trim(), terminal_width().min(120))
            ),
        );
        self.ui.status = "Thinking…".to_string();
        self.pending_stream_bytes = 0;
        self.last_stream_draw = Some(Instant::now());
        self.try_draw();
    }

    fn on_permission_prompt(&mut self, message: &str) -> bool {
        self.ui.prompt_for_permission(message).unwrap_or(false)
    }

    fn on_error(&mut self, error: &str) {
        self.ui.push_entry(EntryKind::Error, error.to_string());
        self.ui.status = "Error".to_string();
        self.pending_stream_bytes = 0;
        self.last_stream_draw = Some(Instant::now());
        self.try_draw();
    }
}

impl TuiCallback<'_> {
    fn should_draw_stream_update(&self, text: &str) -> bool {
        should_draw_stream_update(self.last_stream_draw, self.pending_stream_bytes, text)
    }

    fn flush_pending_stream_draw(&mut self) {
        if self.pending_stream_bytes > 0 {
            self.try_draw();
            self.last_stream_draw = Some(Instant::now());
            self.pending_stream_bytes = 0;
        }
    }

    /// Attempt to redraw; on failure, record the drop in the status bar so the
    /// user can see that frames are being lost.
    fn try_draw(&mut self) {
        if let Err(e) = self.ui.draw_stream_frame() {
            self.ui.draw_drops += 1;
            self.ui.status = format!("draw error (drop={}): {e}", self.ui.draw_drops);
        }
    }
}

fn should_draw_stream_update(
    last_stream_draw: Option<Instant>,
    pending_stream_bytes: usize,
    text: &str,
) -> bool {
    if text.contains('\n') || pending_stream_bytes >= STREAM_REDRAW_MIN_BYTES {
        return true;
    }

    match last_stream_draw {
        None => true,
        Some(last_draw) => last_draw.elapsed() >= STREAM_REDRAW_INTERVAL,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EscapeOutcome {
    Armed,
    LoadedPrevious,
    NoPreviousMessage,
}

fn handle_empty_composer_escape(composer: &mut Composer, esc_armed: &mut bool) -> EscapeOutcome {
    if *esc_armed {
        *esc_armed = false;
        if composer.history_previous() {
            EscapeOutcome::LoadedPrevious
        } else {
            EscapeOutcome::NoPreviousMessage
        }
    } else {
        *esc_armed = true;
        EscapeOutcome::Armed
    }
}

fn next_permission_mode(mode: PermissionMode) -> PermissionMode {
    match mode {
        PermissionMode::ReadOnly => PermissionMode::WorkspaceWrite,
        PermissionMode::WorkspaceWrite => PermissionMode::FullAccess,
        PermissionMode::FullAccess => PermissionMode::ReadOnly,
    }
}

fn permission_mode_label(mode: PermissionMode) -> &'static str {
    match mode {
        PermissionMode::ReadOnly => "read-only",
        PermissionMode::WorkspaceWrite => "workspace-write",
        PermissionMode::FullAccess => "full-access",
    }
}

fn cycle_permission_mode(ui: &mut FullscreenUi, conv: &mut ConversationLoop) {
    let next = next_permission_mode(conv.permission.mode());
    conv.permission.set_mode(next);
    let message = format!("Permission mode changed to {}", permission_mode_label(next));
    ui.push_entry(EntryKind::Info, message.clone());
    ui.status = message;
}

#[derive(Default)]
struct TranscriptCache {
    width: Option<usize>,
    entry_line_counts: Vec<usize>,
    lines: Vec<StyledLine>,
}

impl TranscriptCache {
    fn rebuild(&mut self, entries: &[TranscriptEntry], width: usize) {
        let mut lines = Vec::new();
        let mut entry_line_counts = Vec::with_capacity(entries.len());
        for entry in entries {
            let rendered = format_entry(entry, width);
            entry_line_counts.push(rendered.len());
            lines.extend(rendered);
        }
        self.width = Some(width);
        self.entry_line_counts = entry_line_counts;
        self.lines = lines;
    }

    fn append_entry(&mut self, entry: &TranscriptEntry) {
        let Some(width) = self.width else {
            return;
        };
        let rendered = format_entry(entry, width);
        self.entry_line_counts.push(rendered.len());
        self.lines.extend(rendered);
    }

    fn replace_last_entry(&mut self, entry: &TranscriptEntry) {
        let Some(width) = self.width else {
            return;
        };
        let rendered = format_entry(entry, width);
        if let Some(previous_count) = self.entry_line_counts.last_mut() {
            let keep = self.lines.len().saturating_sub(*previous_count);
            self.lines.truncate(keep);
            *previous_count = rendered.len();
            self.lines.extend(rendered);
        } else {
            self.entry_line_counts.push(rendered.len());
            self.lines.extend(rendered);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zipcode_runtime::Session;

    #[test]
    fn stream_redraw_policy_batches_small_tokens() {
        assert!(!should_draw_stream_update(
            Some(Instant::now()),
            STREAM_REDRAW_MIN_BYTES - 1,
            "a"
        ));
    }

    #[test]
    fn stream_redraw_policy_flushes_newlines() {
        assert!(should_draw_stream_update(Some(Instant::now()), 1, "\n"));
    }

    #[test]
    fn transcript_cache_append_matches_full_rebuild() {
        let mut entries = vec![TranscriptEntry {
            kind: EntryKind::User,
            content: "hello".to_string(),
        }];
        let mut cache = TranscriptCache::default();
        cache.rebuild(&entries, 8);

        entries.push(TranscriptEntry {
            kind: EntryKind::Assistant,
            content: "world".to_string(),
        });
        cache.append_entry(entries.last().unwrap());

        let mut rebuilt = TranscriptCache::default();
        rebuilt.rebuild(&entries, 8);

        assert_eq!(cache.entry_line_counts, rebuilt.entry_line_counts);
        assert_eq!(cache.lines, rebuilt.lines);
    }

    #[test]
    fn transcript_cache_updates_last_entry_without_rebuilding_history() {
        let mut entries = vec![
            TranscriptEntry {
                kind: EntryKind::User,
                content: "hello".to_string(),
            },
            TranscriptEntry {
                kind: EntryKind::Assistant,
                content: "안녕".to_string(),
            },
        ];
        let mut cache = TranscriptCache::default();
        cache.rebuild(&entries, 4);

        entries[1].content.push_str("하세요");
        cache.replace_last_entry(&entries[1]);

        let mut rebuilt = TranscriptCache::default();
        rebuilt.rebuild(&entries, 4);

        assert_eq!(cache.entry_line_counts, rebuilt.entry_line_counts);
        assert_eq!(cache.lines, rebuilt.lines);
    }

    #[test]
    fn empty_composer_escape_requires_double_press_to_recall_history() {
        let mut composer = Composer::new();
        for ch in "previous".chars() {
            composer.insert_char(ch);
        }
        let _ = composer.submit();

        let mut esc_armed = false;
        assert_eq!(
            handle_empty_composer_escape(&mut composer, &mut esc_armed),
            EscapeOutcome::Armed
        );
        assert!(esc_armed);
        assert_eq!(
            handle_empty_composer_escape(&mut composer, &mut esc_armed),
            EscapeOutcome::LoadedPrevious
        );
        assert_eq!(composer.text(), "previous");
    }

    #[test]
    fn shift_tab_cycles_permission_modes() {
        assert_eq!(
            next_permission_mode(PermissionMode::ReadOnly),
            PermissionMode::WorkspaceWrite
        );
        assert_eq!(
            next_permission_mode(PermissionMode::WorkspaceWrite),
            PermissionMode::FullAccess
        );
        assert_eq!(
            next_permission_mode(PermissionMode::FullAccess),
            PermissionMode::ReadOnly
        );
    }

    #[test]
    fn transcript_entries_from_session_rehydrates_saved_history() {
        let mut session = Session::new();
        session.messages = vec![
            zipcode_inference::ChatMessage::system("system prompt"),
            zipcode_inference::ChatMessage::user("hello"),
            zipcode_inference::ChatMessage::assistant("world"),
            zipcode_inference::ChatMessage::tool_result("call_1", "tool output"),
        ];

        let entries = transcript_entries_from_session(&session);
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].kind, EntryKind::Info);
        assert!(entries[0].content.contains("saved system prompt"));
        assert_eq!(entries[1].kind, EntryKind::User);
        assert_eq!(entries[1].content, "hello");
        assert_eq!(entries[2].kind, EntryKind::Assistant);
        assert_eq!(entries[2].content, "world");
        assert_eq!(entries[3].kind, EntryKind::ToolResult);
        assert_eq!(entries[3].content, "tool output");
    }

    /// The Gemma 4 reasoning channel is rendered as a dedicated, dimmed
    /// entry so users can follow the model's thinking without mistaking it
    /// for the final answer.
    #[test]
    fn format_entry_renders_thinking_with_dimmed_style() {
        let entry = TranscriptEntry {
            kind: EntryKind::Thinking,
            content: "I should read the README".to_string(),
        };

        let lines = format_entry(&entry, 80);

        assert!(!lines.is_empty(), "thinking entry should produce output");
        assert!(
            lines
                .iter()
                .all(|line| matches!(line.color, Color::DarkGrey)),
            "every thinking line must be DarkGrey: {lines:?}"
        );
        assert!(
            lines.iter().any(|line| line.text.contains("I should read")),
            "thinking content must appear in formatted lines"
        );
    }

    #[test]
    fn scroll_status_label_is_human_friendly() {
        assert_eq!(scroll_status_label(0), "latest");
        assert_eq!(scroll_status_label(1), "1 line above latest");
        assert_eq!(scroll_status_label(12), "12 lines above latest");
    }

    #[test]
    fn mouse_wheel_maps_to_transcript_scroll_direction() {
        assert_eq!(
            mouse_scroll_direction(MouseEventKind::ScrollUp),
            Some(ScrollDirection::Older)
        );
        assert_eq!(
            mouse_scroll_direction(MouseEventKind::ScrollDown),
            Some(ScrollDirection::Newer)
        );
        assert_eq!(mouse_scroll_direction(MouseEventKind::Moved), None);
    }

    #[test]
    fn scroll_amount_clamps_within_transcript_bounds() {
        assert_eq!(
            scroll_offset_after_delta(0, 10, 4, ScrollDirection::Older),
            4
        );
        assert_eq!(
            scroll_offset_after_delta(8, 10, 4, ScrollDirection::Older),
            10
        );
        assert_eq!(
            scroll_offset_after_delta(10, 10, 3, ScrollDirection::Newer),
            7
        );
        assert_eq!(
            scroll_offset_after_delta(2, 10, 5, ScrollDirection::Newer),
            0
        );
    }

    #[test]
    fn visible_tail_clamps_scroll_without_empty_overscroll() {
        let items = vec![1, 2, 3, 4, 5];
        assert_eq!(visible_tail(&items, 3, 0), &[3, 4, 5]);
        assert_eq!(visible_tail(&items, 3, 1), &[2, 3, 4]);
        assert_eq!(visible_tail(&items, 3, 999), &[1, 2, 3]);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StyledLine {
    color: Color,
    text: String,
}

fn format_entry(entry: &TranscriptEntry, width: usize) -> Vec<StyledLine> {
    let (prefix, color) = match entry.kind {
        EntryKind::User => ("You", Color::Green),
        EntryKind::Assistant => ("Zip", Color::White),
        // DarkGrey gives the reasoning channel a visually dimmed appearance
        // distinct from normal assistant output while remaining readable.
        EntryKind::Thinking => ("…", Color::DarkGrey),
        EntryKind::ToolStart => ("Tool", Color::Yellow),
        EntryKind::ToolResult => ("Out", Color::Blue),
        EntryKind::Info => ("Info", Color::Cyan),
        EntryKind::Error => ("Err", Color::Red),
        EntryKind::Permission => ("Perm", Color::Magenta),
    };
    let indent = " ".repeat(prefix.len() + 2);
    let mut out = Vec::new();
    let mut first = true;
    for raw_line in entry.content.lines() {
        let wrapped = wrap_plain(raw_line, width.saturating_sub(indent.len()).max(1));
        if wrapped.is_empty() {
            out.push(StyledLine {
                color,
                text: format!("{prefix}: "),
            });
            first = false;
            continue;
        }
        for line in wrapped {
            let rendered = if first {
                format!("{prefix}: {line}")
            } else {
                format!("{indent}{line}")
            };
            out.push(StyledLine {
                color,
                text: rendered,
            });
            first = false;
        }
    }
    if out.is_empty() {
        out.push(StyledLine {
            color,
            text: format!("{prefix}:"),
        });
    }
    out
}

fn wrap_plain(text: &str, width: usize) -> Vec<String> {
    crate::width::wrap_display(text, width)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScrollDirection {
    Older,
    Newer,
}

fn mouse_scroll_direction(kind: MouseEventKind) -> Option<ScrollDirection> {
    match kind {
        MouseEventKind::ScrollUp => Some(ScrollDirection::Older),
        MouseEventKind::ScrollDown => Some(ScrollDirection::Newer),
        _ => None,
    }
}

fn scroll_offset_after_delta(
    previous: usize,
    max_scroll: usize,
    amount: usize,
    direction: ScrollDirection,
) -> usize {
    match direction {
        ScrollDirection::Older => previous.saturating_add(amount).min(max_scroll),
        ScrollDirection::Newer => previous.saturating_sub(amount),
    }
}

fn scroll_status_label(offset: usize) -> String {
    match offset {
        0 => "latest".to_string(),
        1 => "1 line above latest".to_string(),
        value => format!("{value} lines above latest"),
    }
}

fn initial_transcript_entries(session: &Session) -> Vec<TranscriptEntry> {
    let mut entries = vec![TranscriptEntry {
        kind: EntryKind::Info,
        content: format!(
            "zipcode v{} — fullscreen TUI (Codex-style MVP+) ready",
            env!("CARGO_PKG_VERSION")
        ),
    }];
    entries.extend(transcript_entries_from_session(session));
    entries
}

fn transcript_entries_from_session(session: &Session) -> Vec<TranscriptEntry> {
    let mut entries = Vec::new();
    let mut system_prompt_reported = false;

    for message in &session.messages {
        match message.role {
            Role::System => {
                if !system_prompt_reported {
                    entries.push(TranscriptEntry {
                        kind: EntryKind::Info,
                        content: "Resumed with saved system prompt.".to_string(),
                    });
                    system_prompt_reported = true;
                }
            }
            Role::User => entries.push(TranscriptEntry {
                kind: EntryKind::User,
                content: message.content.clone(),
            }),
            Role::Model => entries.push(TranscriptEntry {
                kind: EntryKind::Assistant,
                content: message.content.clone(),
            }),
            Role::Tool => entries.push(TranscriptEntry {
                kind: EntryKind::ToolResult,
                content: message.content.clone(),
            }),
        }
    }

    entries
}

fn visible_tail<T>(items: &[T], max_len: usize, scroll_offset: usize) -> &[T] {
    if items.len() <= max_len {
        return items;
    }
    let max_scroll = items.len().saturating_sub(max_len);
    let clamped_scroll = scroll_offset.min(max_scroll);
    let end = items.len().saturating_sub(clamped_scroll);
    let start = end.saturating_sub(max_len);
    &items[start..end]
}
