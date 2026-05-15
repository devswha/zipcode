use std::io::{self, IsTerminal, Stdout, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::Value;
use termimad::crossterm::cursor::{Hide, MoveTo, Show};
use termimad::crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind,
};
use termimad::crossterm::execute;
use termimad::crossterm::style::{
    Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor,
};
use termimad::crossterm::terminal::{self, Clear, ClearType};
use zipcode_inference::Role;
use zipcode_runtime::{ConversationLoop, Session, StreamCallback};
use zipcode_tools::PermissionMode;

use crate::render::{summarize_tool_args, terminal_width, truncate_to_width};
use crate::repl::{
    clear_session, compact_session, help_text, load_session_into_loop, parse_slash_command,
    prepare_loop, run_interactive, session_status_lines, CompactFeedback, ParsedSlashCommand,
    SlashCommand,
};
use crate::tui_composer::Composer;
use crate::width::display_width;
use crate::UiMode;

/// Whether the TUI panic hook is currently installed.
static TUI_PANIC_HOOK_INSTALLED: AtomicBool = AtomicBool::new(false);

const HEADER_LINES: u16 = 1;
const STATUS_LINES: u16 = 3;
const HINT_LINES: u16 = 1;
const MAX_COMPOSER_LINES: usize = 5;
const STREAM_REDRAW_INTERVAL: Duration = Duration::from_millis(33);
const STREAM_REDRAW_MIN_BYTES: usize = 24;
const MOUSE_WHEEL_SCROLL_LINES: usize = 4;
const BRAND_WORDMARK_MIN_WIDTH: usize = 64;
const BRAND_WORDMARK: &[&str] = &[
    "███████╗██╗██████╗  ██████╗ ██████╗ ██████╗ ███████╗",
    "╚══███╔╝██║██╔══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝",
    "  ███╔╝ ██║██████╔╝██║     ██║   ██║██║  ██║█████╗  ",
    " ███╔╝  ██║██╔═══╝ ██║     ██║   ██║██║  ██║██╔══╝  ",
    "███████╗██║██║     ╚██████╗╚██████╔╝██████╔╝███████╗",
    "╚══════╝╚═╝╚═╝      ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝",
];

/// Saturating cast from `usize` to `u16`, clamping at `u16::MAX`.
/// Used throughout the TUI for terminal coordinates which are inherently
/// bounded by the terminal's u16 dimension.
#[allow(clippy::cast_possible_truncation)]
fn as_u16(n: usize) -> u16 {
    n.min(u16::MAX as usize) as u16
}

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
        // Each arm has a distinct semantic purpose even though two bodies are
        // identical `{}` — handled keys are consumed silently, unknown events
        // are also ignored — so suppress the pedantic identical-bodies lint.
        #[allow(clippy::match_same_arms)]
        match event::read().context("failed to read terminal input")? {
            Event::Key(key) if !ui.handle_key_event(key, &mut conv)? => break,
            Event::Key(_) => {}
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
        if !ui.handle_submitted_input(line, conv)? {
            return Ok(false);
        }
    }

    Ok(true)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntryKind {
    Brand,
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
    /// Thin visual separator inserted between major turn transitions
    /// (User→Tool, Tool→Assistant, etc.) so the transcript doesn't
    /// read as one undifferentiated wall of text.
    Separator,
}

struct TranscriptEntry {
    kind: EntryKind,
    content: String,
}

struct Overlay {
    title: String,
    body: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OverlayGeometry {
    box_width: u16,
    box_height: u16,
    box_left: u16,
    box_top: u16,
    body_width: usize,
    available_body: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TuiLayout {
    transcript_top: u16,
    status_y: u16,
    status_line_y: u16,
    composer_label_y: u16,
    composer_top: u16,
    hint_y: u16,
    transcript_height: usize,
    composer_height: u16,
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
    raw_enabled: bool,
    overlay: Option<Overlay>,
    overlay_scroll: usize,
    draw_drops: usize,
    esc_armed: bool,
    /// Bytes of Gemma 4 reasoning streamed during the current turn.
    /// Surfaced in the status bar as "Thinking… (N chars)" so the user
    /// sees live progress without the reasoning itself polluting the
    /// transcript. Reset to 0 at the start of each `run_turn`.
    thinking_bytes: usize,
    /// True when the scrollback mirror has an unterminated assistant line.
    /// The fullscreen UI renders on the main screen instead of alternate-screen
    /// so users can mouse-scroll and copy logs from terminal scrollback.
    scrollback_line_open: bool,
}

/// Normalize a streamed assistant token for scrollback mirror output.
///
/// Each `\n` is replaced with `\r\n` followed by 5 spaces so that
/// continuation lines in terminal scrollback align with the "Zip: " prefix
/// (4 chars + 1 space).
fn normalize_scrollback_token(token: &str) -> String {
    token.replace('\n', "\r\n     ")
}

impl FullscreenUi {
    fn new(conv: &ConversationLoop, backend: String, startup_notices: &[String]) -> Result<Self> {
        // Install a panic hook that restores the terminal before printing the
        // panic message. Without this, a panic leaves the terminal in raw mode
        // and the user sees a garbled shell.
        if !TUI_PANIC_HOOK_INSTALLED.swap(true, Ordering::SeqCst) {
            let prev_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                // Best-effort terminal restoration — ignore errors.
                let _ = terminal::disable_raw_mode();
                let _ = execute!(io::stdout(), Show);
                prev_hook(info);
            }));
        }

        let mut stdout = io::stdout();
        if let Err(e) = execute!(stdout, Hide) {
            clear_panic_hook_flag();
            return Err(e.into());
        }
        if let Err(e) = terminal::enable_raw_mode() {
            let _ = execute!(stdout, Show);
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
            raw_enabled: true,
            overlay: None,
            overlay_scroll: 0,
            draw_drops: 0,
            esc_armed: false,
            thinking_bytes: 0,
            scrollback_line_open: false,
        };
        if let Some(first_notice) = startup_notices.first() {
            ui.status = if startup_notices.len() == 1 {
                format!("Startup: {first_notice}")
            } else {
                format!(
                    "Startup: {first_notice} (+{} more)",
                    startup_notices.len() - 1
                )
            };
        }
        ui.draw()?;
        Ok(ui)
    }

    fn restore(&mut self) -> Result<()> {
        if self.raw_enabled {
            terminal::disable_raw_mode()?;
            self.raw_enabled = false;
        }
        if self.scrollback_line_open {
            let _ = self.stdout.write_all(b"\r\n");
            self.scrollback_line_open = false;
        }
        execute!(self.stdout, Show)?;
        self.stdout.flush()?;
        Ok(())
    }

    /// Handle a key event while an overlay (help, etc.) is active.
    /// Returns `Some(should_continue)` if the key was consumed, `None` to
    /// fall through to normal key handling.
    fn handle_overlay_key(&mut self, key: KeyEvent) -> Option<Result<bool>> {
        self.overlay.as_ref()?;
        match key.code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') => {
                self.overlay = None;
                self.overlay_scroll = 0;
                self.status = "Ready".to_string();
                Some(self.draw().and(Ok(true)))
            }
            KeyCode::Up => Some(
                self.scroll_overlay_by(1, ScrollDirection::Older)
                    .and(Ok(true)),
            ),
            KeyCode::Down => Some(
                self.scroll_overlay_by(1, ScrollDirection::Newer)
                    .and(Ok(true)),
            ),
            KeyCode::PageUp => Some(
                self.scroll_overlay_by_page(ScrollDirection::Older)
                    .and(Ok(true)),
            ),
            KeyCode::PageDown => Some(
                self.scroll_overlay_by_page(ScrollDirection::Newer)
                    .and(Ok(true)),
            ),
            KeyCode::Home if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.overlay_scroll = 0;
                Some(self.draw().and(Ok(true)))
            }
            KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => {
                Some(self.jump_to_overlay_end().and(Ok(true)))
            }
            _ => Some(Ok(true)),
        }
    }

    /// Handle control-key combinations (Ctrl+D, Ctrl+C, Ctrl+L) and the
    /// Escape key. Returns `Some(should_continue)` if the key was consumed,
    /// `None` otherwise.
    fn handle_control_keys(&mut self, key: KeyEvent) -> Option<Result<bool>> {
        match key {
            KeyEvent {
                code: KeyCode::Char('d'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => Some(Ok(false)),
            KeyEvent {
                code: KeyCode::Char('c'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                self.composer.clear();
                self.status = "Input cancelled".to_string();
                None
            }
            KeyEvent {
                code: KeyCode::Char('l'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                self.status = "Screen refreshed".to_string();
                None
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
                None
            }
            _ => None,
        }
    }

    /// Handle scroll keys (`PageUp`, `PageDown`, Ctrl+Home, Ctrl+End).
    fn handle_scroll_keys(&mut self, key: KeyEvent) -> bool {
        match key {
            KeyEvent {
                code: KeyCode::PageUp,
                ..
            } => {
                let moved = self
                    .scroll_transcript_by_page(ScrollDirection::Older)
                    .ok()
                    .unwrap_or(0);
                self.status = if moved == 0 {
                    "Already at the oldest visible history".to_string()
                } else {
                    format!("Scrolled up {moved} line(s)")
                };
                true
            }
            KeyEvent {
                code: KeyCode::PageDown,
                ..
            } => {
                let moved = self
                    .scroll_transcript_by_page(ScrollDirection::Newer)
                    .ok()
                    .unwrap_or(0);
                self.status = if self.transcript_scroll == 0 {
                    "Back to latest output".to_string()
                } else {
                    format!("Scrolled down {moved} line(s)")
                };
                true
            }
            KeyEvent {
                code: KeyCode::Home,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                let moved = self.jump_to_oldest_transcript().ok().unwrap_or(0);
                self.status = if moved == 0 {
                    "Already at the oldest visible history".to_string()
                } else {
                    "Jumped to oldest visible history".to_string()
                };
                true
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
                true
            }
            _ => false,
        }
    }

    /// Handle cursor movement and navigation keys (arrows, Home/End,
    /// Up/Down with history fallback, F1, `BackTab`).
    fn handle_navigation_keys(&mut self, key: KeyEvent, conv: &mut ConversationLoop) -> bool {
        match key {
            KeyEvent {
                code: KeyCode::F(1),
                ..
            } => {
                self.open_help_overlay();
                true
            }
            KeyEvent {
                code: KeyCode::BackTab,
                ..
            } => {
                cycle_permission_mode(self, conv);
                true
            }
            KeyEvent {
                code: KeyCode::Left,
                ..
            } => {
                self.composer.move_left();
                true
            }
            KeyEvent {
                code: KeyCode::Right,
                ..
            } => {
                self.composer.move_right();
                true
            }
            KeyEvent {
                code: KeyCode::Home,
                ..
            } => {
                self.composer.move_home();
                true
            }
            KeyEvent {
                code: KeyCode::End, ..
            } => {
                self.composer.move_end();
                true
            }
            KeyEvent {
                code: KeyCode::Up, ..
            } => {
                if !self.composer.move_up() {
                    let _ = self.composer.history_previous();
                }
                true
            }
            KeyEvent {
                code: KeyCode::Down,
                ..
            } => {
                if !self.composer.move_down() {
                    let _ = self.composer.history_next();
                }
                true
            }
            _ => false,
        }
    }

    /// Handle text editing keys (Backspace, Delete, Enter, Ctrl+J,
    /// character input). Returns `Some(result)` when the key produces an
    /// early return (submission), `None` otherwise.
    fn handle_editing_keys(
        &mut self,
        key: KeyEvent,
        conv: &mut ConversationLoop,
    ) -> Option<Result<bool>> {
        match key {
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                self.composer.backspace();
                None
            }
            KeyEvent {
                code: KeyCode::Delete,
                ..
            } => {
                self.composer.delete_forward();
                None
            }
            KeyEvent {
                code: KeyCode::Enter,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL)
                || modifiers.contains(KeyModifiers::SHIFT)
                || modifiers.contains(KeyModifiers::ALT) =>
            {
                self.composer.insert_newline();
                None
            }
            KeyEvent {
                code: KeyCode::Char('j'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                self.composer.insert_newline();
                None
            }
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => {
                if self.composer.try_escape_newline() {
                    self.status = "Inserted newline".to_string();
                    return Some(self.draw().and(Ok(true)));
                }
                let submitted = self.composer.submit();
                Some(self.handle_submitted_input(&submitted, conv))
            }
            KeyEvent {
                code: KeyCode::Char(ch),
                modifiers,
                ..
            } if modifiers.is_empty() || modifiers == KeyModifiers::SHIFT => {
                self.composer.insert_char(ch);
                None
            }
            _ => None,
        }
    }

    /// Main keyboard event dispatcher. Delegates to focused helpers for
    /// overlay, control, scroll, navigation, and editing keys.
    fn handle_key_event(&mut self, key: KeyEvent, conv: &mut ConversationLoop) -> Result<bool> {
        // Overlay has priority — dismiss keys close it, everything else is swallowed.
        if let Some(result) = self.handle_overlay_key(key) {
            return result;
        }

        // Disarm the double-Escape "edit previous" trigger on any non-Esc key.
        if !matches!(key.code, KeyCode::Esc) {
            self.esc_armed = false;
        }

        // Try each handler in priority order. Control keys can produce early
        // returns (Ctrl+D exits); scroll/navigation/editing return `true` when
        // they consume the key.
        if let Some(result) = self.handle_control_keys(key) {
            return result;
        }
        if self.handle_scroll_keys(key) {
            self.draw()?;
            return Ok(true);
        }
        if self.handle_navigation_keys(key, conv) {
            self.draw()?;
            return Ok(true);
        }
        if let Some(result) = self.handle_editing_keys(key, conv) {
            return result;
        }

        self.draw()?;
        Ok(true)
    }

    fn handle_mouse_event(&mut self, mouse: MouseEvent) -> Result<()> {
        if self.overlay.is_some() {
            if let Some(direction) = mouse_scroll_direction(mouse.kind) {
                self.scroll_overlay_by(MOUSE_WHEEL_SCROLL_LINES, direction)?;
            }
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
            ScrollDirection::Older => format!("Scrolled up {moved} line(s)"),
            ScrollDirection::Newer if self.transcript_scroll == 0 => {
                "Back to latest output".to_string()
            }
            ScrollDirection::Newer => format!("Scrolled down {moved} line(s)"),
        };
        self.draw()?;
        Ok(())
    }

    fn handle_submitted_input(&mut self, input: &str, conv: &mut ConversationLoop) -> Result<bool> {
        self.transcript_scroll = 0;
        if input.trim().is_empty() {
            self.status = "Ready".to_string();
        } else {
            match parse_slash_command(input.trim()) {
                ParsedSlashCommand::Command(command, argument) => {
                    if !self.handle_slash_command(command, argument.as_deref(), conv) {
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
    ) -> bool {
        match command {
            SlashCommand::Help => self.open_help_overlay(),
            SlashCommand::Status => {
                self.overlay_scroll = 0;
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
                if let Some(id) = argument {
                    match load_session_into_loop(conv, id) {
                        Ok(message) => {
                            self.set_transcript_from_session(&conv.session);
                            self.push_entry(EntryKind::Info, message);
                            self.session_id.clone_from(&conv.session.id);
                            self.status = "Session loaded".to_string();
                        }
                        Err(error) => {
                            self.push_entry(EntryKind::Error, format!("{error:#}"));
                            self.status = "Session load failed".to_string();
                        }
                    }
                } else {
                    self.push_entry(
                        EntryKind::Error,
                        "/session load requires a session id".to_string(),
                    );
                    self.status = "Missing session id".to_string();
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
                    self.session_id.clone_from(&conv.session.id);
                    self.status = "Conversation cleared".to_string();
                }
                Err(error) => {
                    self.push_entry(EntryKind::Error, error.to_string());
                    self.status = "Conversation clear failed".to_string();
                }
            },
            SlashCommand::Quit => return false,
        }

        true
    }

    fn open_help_overlay(&mut self) {
        let mut body: Vec<String> = help_text().lines().map(str::to_string).collect();
        body.extend([
            String::new(),
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
        self.overlay_scroll = 0;
        self.status = "Help opened".to_string();
    }

    fn run_turn(&mut self, input: &str, conv: &mut ConversationLoop) -> Result<()> {
        self.push_entry(EntryKind::User, input.trim_end().to_string());
        self.status = "Thinking…".to_string();
        self.thinking_bytes = 0;
        self.draw()?;

        let mut cb = TuiCallback {
            ui: self,
            last_stream_draw: None,
            pending_stream_bytes: 0,
        };
        let result = conv.run_turn(input, &mut cb);
        cb.flush_pending_stream_draw();
        cb.ui.session_id.clone_from(&conv.session.id);
        cb.ui.status = if result.is_ok() {
            "Ready".to_string()
        } else {
            "Last turn failed".to_string()
        };
        cb.ui.draw()?;
        result
    }

    fn push_entry(&mut self, kind: EntryKind, content: String) {
        // Auto-insert a separator on major role transitions so the
        // transcript visually groups related entries (e.g. User question,
        // then Tool+Out block, then Zip answer).
        if !matches!(
            kind,
            EntryKind::Separator | EntryKind::Info | EntryKind::Brand
        ) {
            if let Some(prev) = self.transcript.last() {
                let dominated = matches!(
                    (prev.kind, kind),
                    (
                        EntryKind::User | EntryKind::ToolResult,
                        EntryKind::Assistant | EntryKind::Thinking
                    ) | (EntryKind::Assistant, EntryKind::User)
                );
                if dominated {
                    let sep = TranscriptEntry {
                        kind: EntryKind::Separator,
                        content: String::new(),
                    };
                    self.mirror_entry_to_scrollback(&sep);
                    self.transcript_cache.append_entry(&sep);
                    self.transcript.push(sep);
                }
            }
        }
        self.transcript.push(TranscriptEntry { kind, content });
        self.transcript_scroll = 0;
        let Some(entry) = self.transcript.last() else {
            return;
        };
        let cached_entry = TranscriptEntry {
            kind: entry.kind,
            content: entry.content.clone(),
        };
        if !matches!(cached_entry.kind, EntryKind::Assistant) {
            self.mirror_entry_to_scrollback(&cached_entry);
        }
        self.transcript_cache.append_entry(&cached_entry);
    }

    fn append_assistant_token(&mut self, token: &str) {
        match self.transcript.last_mut() {
            Some(TranscriptEntry {
                kind: EntryKind::Assistant,
                content,
            }) => {
                content.push_str(token);
                self.mirror_assistant_token_to_scrollback(token);
            }
            _ => {
                self.push_entry(EntryKind::Assistant, token.to_string());
                self.mirror_assistant_token_to_scrollback(token);
            }
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
        let layout = layout_for_terminal(height, composer_visible);
        let viewport = TranscriptViewport {
            width: usable_width.max(8),
            height: layout.transcript_height.max(1),
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

    const fn jump_to_latest_transcript(&mut self) -> usize {
        let previous = self.transcript_scroll;
        self.transcript_scroll = 0;
        previous
    }

    fn set_transcript_from_session(&mut self, session: &Session) {
        self.transcript = initial_transcript_entries(session);
        self.transcript_cache = TranscriptCache::default();
        self.transcript_scroll = 0;
    }

    fn mirror_entry_to_scrollback(&mut self, entry: &TranscriptEntry) {
        if self.scrollback_line_open {
            let _ = self.stdout.write_all(b"\r\n");
            self.scrollback_line_open = false;
        }

        let width = terminal::size()
            .map(|(width, _)| width.saturating_sub(2) as usize)
            .unwrap_or(100)
            .max(80);
        for line in format_entry(entry, width) {
            let _ = self.stdout.write_all(line.text.as_bytes());
            let _ = self.stdout.write_all(b"\r\n");
        }
        let _ = self.stdout.flush();
    }

    fn mirror_assistant_token_to_scrollback(&mut self, token: &str) {
        if !self.scrollback_line_open {
            let _ = self.stdout.write_all(b"Zip: ");
            self.scrollback_line_open = true;
        }
        let normalized = normalize_scrollback_token(token);
        let _ = self.stdout.write_all(normalized.as_bytes());
        let _ = self.stdout.flush();
    }

    fn prompt_for_permission(&mut self, message: &str) -> Result<bool> {
        self.push_entry(EntryKind::Permission, format!("{message} [Y/n]"));
        self.status = "Permission required".to_string();
        self.draw()?;

        loop {
            if let Event::Key(key) = event::read().context("failed to read permission input")? {
                match key.code {
                    KeyCode::Enter | KeyCode::Char('y' | 'Y') => return Ok(true),
                    KeyCode::Char('n' | 'N') | KeyCode::Esc => return Ok(false),
                    _ => {}
                }
            }
        }
    }

    fn draw(&mut self) -> Result<()> {
        let (width, height) = terminal::size()?;
        let composer_width = width.saturating_sub(4) as usize;
        let composer_lines = self.composer.wrapped_lines(composer_width.max(1));
        let composer_visible = composer_lines.len().clamp(1, MAX_COMPOSER_LINES);
        let layout = layout_for_terminal(height, composer_visible);

        self.begin_sync_output();

        execute!(self.stdout, MoveTo(0, 0), Clear(ClearType::All))?;
        self.draw_header(width)?;
        self.draw_transcript(width, layout.transcript_top, layout.transcript_height)?;
        self.draw_footer(width, layout)?;
        if self.overlay.is_some() {
            self.draw_overlay(width, height)?;
        } else {
            self.position_cursor(
                width,
                layout.composer_top,
                composer_lines.len(),
                composer_visible,
            )?;
        }

        self.end_sync_output()?;
        Ok(())
    }

    fn draw_stream_frame(&mut self) -> Result<()> {
        if self.overlay.is_some() {
            return self.draw();
        }

        let (width, height) = terminal::size()?;
        let composer_width = width.saturating_sub(4) as usize;
        let composer_lines = self.composer.wrapped_lines(composer_width.max(1));
        let composer_visible = composer_lines.len().clamp(1, MAX_COMPOSER_LINES);
        let layout = layout_for_terminal(height, composer_visible);

        self.begin_sync_output();
        self.clear_region(layout.transcript_top, as_u16(layout.transcript_height))?;
        self.clear_region(layout.status_y, height.saturating_sub(layout.status_y))?;
        self.draw_transcript(width, layout.transcript_top, layout.transcript_height)?;
        self.draw_footer(width, layout)?;
        self.position_cursor(
            width,
            layout.composer_top,
            composer_lines.len(),
            composer_visible,
        )?;
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
                    " ◆ zipcode {}  {}  session:{}  tools:{} ",
                    crate::version::git_label(),
                    self.backend,
                    truncate_to_width(&self.session_id, 8),
                    self.tool_count
                ),
                width as usize
            )),
            ResetColor,
            SetAttribute(Attribute::Reset),
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
                MoveTo(0, top + as_u16(idx)),
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

    fn draw_footer(&mut self, width: u16, layout: TuiLayout) -> Result<()> {
        execute!(
            self.stdout,
            MoveTo(0, layout.status_y),
            SetForegroundColor(Color::DarkGrey),
            Print("─".repeat(width as usize)),
            ResetColor,
            MoveTo(0, layout.status_line_y),
            SetForegroundColor(Color::Yellow),
            Print(truncate_to_width(
                &format!(
                    " {}  ·  {}",
                    self.status,
                    scroll_status_label(self.transcript_scroll),
                ),
                width as usize
            )),
            ResetColor,
            MoveTo(0, layout.composer_label_y),
            SetForegroundColor(Color::DarkGrey),
            Print(truncate_to_width(
                " Message  Enter send · Ctrl+J newline · F1 help ",
                width as usize,
            )),
            ResetColor,
        )?;

        let composer_lines = self
            .composer
            .wrapped_lines(width.saturating_sub(4) as usize);
        let visible_start = composer_lines
            .len()
            .saturating_sub(layout.composer_height as usize);
        for row in 0..layout.composer_height {
            let line = composer_lines
                .get(visible_start + row as usize)
                .cloned()
                .unwrap_or_default();
            execute!(
                self.stdout,
                MoveTo(0, layout.composer_top + row),
                SetForegroundColor(Color::Green),
                Print(if row == 0 { "> " } else { "· " }),
                ResetColor,
                Print(truncate_to_width(&line, width.saturating_sub(2) as usize)),
            )?;
        }

        execute!(
            self.stdout,
            MoveTo(0, layout.hint_y),
            SetForegroundColor(Color::DarkGrey),
            Print(truncate_to_width(
                " /help /status /clear /quit · PgUp/PgDn scroll · mouse select copies logs ",
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
        let initial_geometry = overlay_geometry(width, height, overlay.body.len());
        let body_lines = wrapped_overlay_body(overlay, initial_geometry.body_width.max(1));
        let geometry = overlay_geometry(width, height, body_lines.len());
        let max_scroll = body_lines.len().saturating_sub(geometry.available_body);
        self.overlay_scroll = self.overlay_scroll.min(max_scroll);

        for row in 0..geometry.box_height {
            execute!(
                self.stdout,
                MoveTo(0, geometry.box_top + row),
                SetForegroundColor(Color::DarkGrey),
                Print(" ".repeat(width.max(geometry.box_width) as usize)),
                ResetColor,
            )?;
        }

        execute!(
            self.stdout,
            MoveTo(geometry.box_left, geometry.box_top),
            SetForegroundColor(Color::Cyan),
            SetAttribute(Attribute::Bold),
            Print(truncate_to_width(
                &format!(" {} ", overlay.title),
                geometry.box_width as usize
            )),
            ResetColor,
            SetAttribute(Attribute::Reset),
        )?;

        for (idx, line) in body_lines
            .into_iter()
            .skip(self.overlay_scroll)
            .take(geometry.available_body)
            .enumerate()
        {
            execute!(
                self.stdout,
                MoveTo(geometry.box_left + 2, geometry.box_top + 1 + as_u16(idx)),
                Print(truncate_to_width(&line, geometry.body_width)),
            )?;
        }

        let footer = if max_scroll == 0 {
            "Esc / Enter / q to close".to_string()
        } else {
            format!(
                "Up/Down scroll {}/{} · Esc / Enter / q",
                self.overlay_scroll + 1,
                max_scroll + 1
            )
        };
        execute!(
            self.stdout,
            MoveTo(
                geometry.box_left + 2,
                geometry.box_top + geometry.box_height - 1
            ),
            SetForegroundColor(Color::DarkGrey),
            Print(truncate_to_width(&footer, geometry.body_width)),
            ResetColor,
        )?;
        Ok(())
    }

    fn scroll_overlay_by(&mut self, lines: usize, direction: ScrollDirection) -> Result<()> {
        let max_scroll = self.max_overlay_scroll()?;
        self.overlay_scroll =
            overlay_scroll_offset_after_delta(self.overlay_scroll, max_scroll, lines, direction);
        self.draw()
    }

    fn scroll_overlay_by_page(&mut self, direction: ScrollDirection) -> Result<()> {
        let (width, height) = terminal::size()?;
        let Some(overlay) = &self.overlay else {
            return Ok(());
        };
        let initial_geometry = overlay_geometry(width, height, overlay.body.len());
        let body_lines = wrapped_overlay_body(overlay, initial_geometry.body_width.max(1));
        let geometry = overlay_geometry(width, height, body_lines.len());
        self.scroll_overlay_by(geometry.available_body.saturating_sub(1).max(1), direction)
    }

    fn jump_to_overlay_end(&mut self) -> Result<()> {
        self.overlay_scroll = self.max_overlay_scroll()?;
        self.draw()
    }

    fn max_overlay_scroll(&self) -> Result<usize> {
        let (width, height) = terminal::size()?;
        let Some(overlay) = &self.overlay else {
            return Ok(0);
        };
        let initial_geometry = overlay_geometry(width, height, overlay.body.len());
        let body_lines = wrapped_overlay_body(overlay, initial_geometry.body_width.max(1));
        let geometry = overlay_geometry(width, height, body_lines.len());
        Ok(body_lines.len().saturating_sub(geometry.available_body))
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
                prefix + as_u16(cursor_col),
                composer_top + as_u16(display_row)
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
        self.ui
            .push_entry(EntryKind::ToolStart, format_tool_start_content(name, args));
        self.ui.status = format!("Running {name}…");
        self.pending_stream_bytes = 0;
        self.last_stream_draw = Some(Instant::now());
        self.try_draw();
    }

    fn on_tool_result(&mut self, name: &str, result: &str) {
        self.ui.push_entry(
            EntryKind::ToolResult,
            format_tool_result_content(name, result),
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

    last_stream_draw.is_none_or(|last_draw| last_draw.elapsed() >= STREAM_REDRAW_INTERVAL)
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

const fn next_permission_mode(mode: PermissionMode) -> PermissionMode {
    match mode {
        PermissionMode::ReadOnly => PermissionMode::WorkspaceWrite,
        PermissionMode::WorkspaceWrite => PermissionMode::FullAccess,
        PermissionMode::FullAccess => PermissionMode::ReadOnly,
    }
}

const fn permission_mode_label(mode: PermissionMode) -> &'static str {
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
        } else {
            self.entry_line_counts.push(rendered.len());
        }
        self.lines.extend(rendered);
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

    // ── New tests: format_entry coverage for all EntryKind variants ──────

    #[test]
    fn format_entry_user_has_green_color_and_prefix() {
        let entry = TranscriptEntry {
            kind: EntryKind::User,
            content: "hello world".to_string(),
        };
        let lines = format_entry(&entry, 80);
        assert!(!lines.is_empty());
        assert!(lines[0].text.starts_with("You: "));
        assert!(lines.iter().all(|l| l.color == Color::Green));
    }

    #[test]
    fn format_entry_assistant_has_white_color() {
        let entry = TranscriptEntry {
            kind: EntryKind::Assistant,
            content: "response".to_string(),
        };
        let lines = format_entry(&entry, 80);
        assert!(!lines.is_empty());
        assert!(lines[0].text.starts_with("Zip: "));
        assert!(lines.iter().all(|l| l.color == Color::White));
    }

    #[test]
    fn format_entry_tool_start_has_yellow_color() {
        let entry = TranscriptEntry {
            kind: EntryKind::ToolStart,
            content: "bash\nls".to_string(),
        };
        let lines = format_entry(&entry, 80);
        assert!(!lines.is_empty());
        assert!(lines[0].text.starts_with("╭─ Running bash"));
        assert!(lines.iter().any(|l| l.text.contains("ls")));
        assert!(lines.iter().all(|l| l.color == Color::Yellow));
    }

    #[test]
    fn format_entry_tool_result_has_blue_color() {
        let entry = TranscriptEntry {
            kind: EntryKind::ToolResult,
            content: "file contents here".to_string(),
        };
        let lines = format_entry(&entry, 80);
        assert!(!lines.is_empty());
        assert!(lines[0].text.starts_with("╭─ Done"));
        assert!(lines.iter().any(|l| l.text.contains("file contents here")));
        assert!(lines.iter().all(|l| l.color == Color::Blue));
    }

    #[test]
    fn format_tool_start_content_summarizes_agent_spawn() {
        let args = serde_json::json!({
            "task": "read the README heading and summarize it",
            "tool_allowlist": ["read_file", "grep_search"],
            "max_tokens": 128,
        });

        let content = format_tool_start_content("agent", &args);

        assert!(content.contains("agent"));
        assert!(content.contains("read the README heading"));
        assert!(content.contains("tools: read_file, grep_search"));
        assert!(content.contains("budget: 128 tokens"));
        assert!(!content.contains("tool_allowlist"));
    }

    #[test]
    fn format_tool_result_content_summarizes_child_agent_result() {
        let content = format_tool_result_content(
            "agent",
            "Child agent complete.\nSummary: checked the README\nTool calls: 1",
        );
        let lines = format_entry(
            &TranscriptEntry {
                kind: EntryKind::ToolResult,
                content,
            },
            80,
        );

        assert!(lines[0].text.contains("Agent complete"));
        assert!(lines
            .iter()
            .any(|line| line.text.contains("checked the README")));
        assert!(lines.iter().any(|line| line.text.contains("1 tool call")));
    }

    #[test]
    fn format_entry_info_has_cyan_color() {
        let entry = TranscriptEntry {
            kind: EntryKind::Info,
            content: "session loaded".to_string(),
        };
        let lines = format_entry(&entry, 80);
        assert!(!lines.is_empty());
        assert!(lines[0].text.starts_with("Info: "));
        assert!(lines.iter().all(|l| l.color == Color::Cyan));
    }

    #[test]
    fn format_entry_error_has_red_color() {
        let entry = TranscriptEntry {
            kind: EntryKind::Error,
            content: "something went wrong".to_string(),
        };
        let lines = format_entry(&entry, 80);
        assert!(!lines.is_empty());
        assert!(lines[0].text.starts_with("Err: "));
        assert!(lines.iter().all(|l| l.color == Color::Red));
    }

    #[test]
    fn format_entry_permission_has_magenta_color() {
        let entry = TranscriptEntry {
            kind: EntryKind::Permission,
            content: "Allow bash? [Y/n]".to_string(),
        };
        let lines = format_entry(&entry, 80);
        assert!(!lines.is_empty());
        assert!(lines[0].text.starts_with("Perm: "));
        assert!(lines.iter().all(|l| l.color == Color::Magenta));
    }

    #[test]
    fn format_entry_separator_produces_dotted_line() {
        let entry = TranscriptEntry {
            kind: EntryKind::Separator,
            content: String::new(),
        };
        let lines = format_entry(&entry, 80);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].color, Color::DarkGrey);
        assert!(lines[0].text.contains("╌"));
    }

    #[test]
    fn format_entry_separator_width_capped_at_60() {
        let entry = TranscriptEntry {
            kind: EntryKind::Separator,
            content: String::new(),
        };
        let lines = format_entry(&entry, 200);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text.chars().count(), 60);
    }

    #[test]
    fn format_entry_empty_content_produces_prefix_only() {
        let entry = TranscriptEntry {
            kind: EntryKind::User,
            content: String::new(),
        };
        let lines = format_entry(&entry, 80);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "You:");
    }

    #[test]
    fn format_entry_multiline_content_produces_multiple_lines() {
        let entry = TranscriptEntry {
            kind: EntryKind::Assistant,
            content: "line one\nline two\nline three".to_string(),
        };
        let lines = format_entry(&entry, 80);
        assert_eq!(lines.len(), 3);
        assert!(lines[0].text.starts_with("Zip: line one"));
        // Continuation lines should be indented, not prefixed
        assert!(lines[1].text.starts_with("     "));
        assert!(lines[1].text.contains("line two"));
    }

    #[test]
    fn format_entry_assistant_code_block_uses_dark_yellow() {
        let entry = TranscriptEntry {
            kind: EntryKind::Assistant,
            content: "```\ncode here\n```".to_string(),
        };
        let lines = format_entry(&entry, 80);
        // The line inside the code block should be DarkYellow
        assert!(lines
            .iter()
            .any(|l| l.color == Color::DarkYellow && l.text.contains("code here")));
    }

    #[test]
    fn format_entry_non_assistant_code_blocks_stay_base_color() {
        let entry = TranscriptEntry {
            kind: EntryKind::User,
            content: "```\ncode\n```".to_string(),
        };
        let lines = format_entry(&entry, 80);
        // Non-assistant entries don't get code block coloring
        assert!(lines.iter().all(|l| l.color == Color::Green));
    }

    // ── as_u16 clamping tests ──────────────────────────────────────────

    #[test]
    fn as_u16_zero() {
        assert_eq!(as_u16(0), 0);
    }

    #[test]
    fn as_u16_small_value() {
        assert_eq!(as_u16(42), 42);
    }

    #[test]
    fn as_u16_max_does_not_clamp() {
        assert_eq!(as_u16(u16::MAX as usize), u16::MAX);
    }

    #[test]
    fn as_u16_clamps_values_above_max() {
        assert_eq!(as_u16(u16::MAX as usize + 1), u16::MAX);
        assert_eq!(as_u16(usize::MAX), u16::MAX);
    }

    // ── scroll_offset_after_delta additional edge cases ────────────────

    #[test]
    fn scroll_offset_zero_amount_does_not_move() {
        assert_eq!(
            scroll_offset_after_delta(5, 10, 0, ScrollDirection::Older),
            5
        );
        assert_eq!(
            scroll_offset_after_delta(5, 10, 0, ScrollDirection::Newer),
            5
        );
    }

    #[test]
    fn scroll_offset_saturating_sub_on_newer() {
        assert_eq!(
            scroll_offset_after_delta(2, 10, 5, ScrollDirection::Newer),
            0
        );
    }

    #[test]
    fn scroll_offset_saturating_add_on_older() {
        assert_eq!(
            scroll_offset_after_delta(0, 0, 100, ScrollDirection::Older),
            0
        );
    }

    #[test]
    fn overlay_scroll_uses_top_to_bottom_offsets() {
        assert_eq!(
            overlay_scroll_offset_after_delta(0, 10, 4, ScrollDirection::Older),
            0
        );
        assert_eq!(
            overlay_scroll_offset_after_delta(0, 10, 4, ScrollDirection::Newer),
            4
        );
        assert_eq!(
            overlay_scroll_offset_after_delta(9, 10, 4, ScrollDirection::Newer),
            10
        );
        assert_eq!(
            overlay_scroll_offset_after_delta(3, 10, 4, ScrollDirection::Older),
            0
        );
    }

    // ── should_draw_stream_update additional cases ─────────────────────

    #[test]
    fn stream_update_draws_immediately_on_first_token() {
        assert!(should_draw_stream_update(None, 0, "a"));
    }

    #[test]
    fn stream_update_draws_when_pending_exceeds_threshold() {
        assert!(should_draw_stream_update(
            Some(Instant::now()),
            STREAM_REDRAW_MIN_BYTES,
            "a"
        ));
    }

    #[test]
    fn stream_update_defers_small_token_within_interval() {
        assert!(!should_draw_stream_update(Some(Instant::now()), 0, "a"));
    }

    // ── visible_tail additional edge cases ─────────────────────────────

    #[test]
    fn visible_tail_empty_slice_returns_empty() {
        let items: Vec<i32> = vec![];
        let result: &[i32] = visible_tail(&items, 5, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn visible_tail_max_len_exceeds_items_returns_all() {
        let items = vec![1, 2, 3];
        assert_eq!(visible_tail(&items, 10, 0), &[1, 2, 3]);
    }

    #[test]
    fn visible_tail_max_len_zero_returns_empty() {
        let items = vec![1, 2, 3];
        let result: &[i32] = visible_tail(&items, 0, 0);
        assert!(result.is_empty());
    }

    // ── transcript_entries_from_session edge cases ─────────────────────

    #[test]
    fn transcript_entries_from_empty_session() {
        let session = Session::new();
        let entries = transcript_entries_from_session(&session);
        assert!(entries.is_empty(), "empty session should yield no entries");
    }

    #[test]
    fn initial_transcript_from_empty_session_has_no_chrome_noise() {
        let session = Session::new();
        let entries = initial_transcript_entries(&session);

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].kind, EntryKind::Brand);
    }

    #[test]
    fn brand_entry_renders_logo_and_version_without_role_prefix() {
        let entry = TranscriptEntry {
            kind: EntryKind::Brand,
            content: String::new(),
        };

        let lines = format_entry(&entry, 80);

        assert!(lines
            .iter()
            .any(|line| line.text.contains(crate::version::VERSION)));
        assert!(lines.iter().all(|line| !line.text.starts_with("Info:")));
        assert!(lines.iter().any(|line| line.text.contains("███████")));
        assert!(lines
            .iter()
            .any(|line| line.text.contains("local-only coding agent")));
    }

    #[test]
    fn brand_entry_uses_compact_wordmark_on_narrow_terminals() {
        let entry = TranscriptEntry {
            kind: EntryKind::Brand,
            content: String::new(),
        };

        let lines = format_entry(&entry, 40);

        assert!(lines.iter().any(|line| line.text.contains("ZIPCODE")));
        assert!(lines.iter().all(|line| !line.text.contains("███████")));
    }

    #[test]
    fn transcript_entries_deduplicate_multiple_system_messages() {
        let mut session = Session::new();
        session.messages = vec![
            zipcode_inference::ChatMessage::system("sys1"),
            zipcode_inference::ChatMessage::system("sys2"),
            zipcode_inference::ChatMessage::user("hello"),
        ];
        let entries = transcript_entries_from_session(&session);
        // Only one Info entry for system, even with multiple system messages
        let info_count = entries.iter().filter(|e| e.kind == EntryKind::Info).count();
        assert_eq!(info_count, 1);
        assert_eq!(entries[1].kind, EntryKind::User);
    }

    // ── TranscriptCache edge cases ─────────────────────────────────────

    #[test]
    fn transcript_cache_append_without_prior_width_is_noop() {
        let mut cache = TranscriptCache::default();
        let entry = TranscriptEntry {
            kind: EntryKind::User,
            content: "hello".to_string(),
        };
        cache.append_entry(&entry);
        assert!(cache.lines.is_empty());
        assert!(cache.entry_line_counts.is_empty());
    }

    #[test]
    fn transcript_cache_replace_last_on_empty_cache_appends() {
        let mut cache = TranscriptCache::default();
        let entry = TranscriptEntry {
            kind: EntryKind::Assistant,
            content: "world".to_string(),
        };
        cache.replace_last_entry(&entry);
        assert!(cache.lines.is_empty());
        // No width set, so replace_last_entry also no-ops
    }

    #[test]
    fn transcript_cache_rebuild_width_stored() {
        let entries = vec![TranscriptEntry {
            kind: EntryKind::User,
            content: "test".to_string(),
        }];
        let mut cache = TranscriptCache::default();
        cache.rebuild(&entries, 42);
        assert_eq!(cache.width, Some(42));
        assert_eq!(cache.entry_line_counts.len(), 1);
    }

    // ── handle_empty_composer_escape additional cases ──────────────────

    #[test]
    fn empty_composer_escape_no_history_returns_no_previous() {
        let mut composer = Composer::new();
        let mut esc_armed = true; // Pre-arm so second press tries to load
        assert_eq!(
            handle_empty_composer_escape(&mut composer, &mut esc_armed),
            EscapeOutcome::NoPreviousMessage
        );
        assert!(!esc_armed);
    }

    // ── permission_mode_label tests ────────────────────────────────────

    #[test]
    fn permission_mode_labels_match_string_representations() {
        assert_eq!(permission_mode_label(PermissionMode::ReadOnly), "read-only");
        assert_eq!(
            permission_mode_label(PermissionMode::WorkspaceWrite),
            "workspace-write"
        );
        assert_eq!(
            permission_mode_label(PermissionMode::FullAccess),
            "full-access"
        );
    }

    // ── scroll_status_label boundary cases ─────────────────────────────

    #[test]
    fn scroll_status_label_plural_vs_singular() {
        assert_eq!(scroll_status_label(1), "1 line above latest");
        assert_eq!(scroll_status_label(2), "2 lines above latest");
        assert_eq!(scroll_status_label(1000), "1000 lines above latest");
    }

    // ── mouse_scroll_direction non_scroll_events ───────────────────────

    #[test]
    fn mouse_non_scroll_events_return_none() {
        assert_eq!(mouse_scroll_direction(MouseEventKind::Moved), None);
    }

    #[test]
    fn layout_keeps_status_composer_and_hint_on_distinct_rows() {
        let layout = layout_for_terminal(24, 1);

        assert_eq!(layout.transcript_top, 1);
        assert_eq!(layout.status_y, 19);
        assert_eq!(layout.status_line_y, 20);
        assert_eq!(layout.composer_label_y, 21);
        assert_eq!(layout.composer_top, 22);
        assert_eq!(layout.hint_y, 23);
        assert_eq!(layout.transcript_height, 18);
        assert_eq!(layout.composer_height, 1);
    }

    #[test]
    fn layout_clamps_composer_height_to_visible_limit() {
        let layout = layout_for_terminal(30, MAX_COMPOSER_LINES + 3);

        assert_eq!(layout.status_y, 21);
        assert_eq!(layout.status_line_y, 22);
        assert_eq!(layout.composer_label_y, 23);
        assert_eq!(layout.composer_top, 24);
        assert_eq!(layout.hint_y, 29);
        assert_eq!(layout.transcript_height, 20);
        assert_eq!(layout.composer_height, as_u16(MAX_COMPOSER_LINES));
    }

    #[test]
    fn overlay_geometry_preserves_footer_space_on_short_terminals() {
        let geometry = overlay_geometry(80, 12, 30);

        assert_eq!(geometry.box_width, 72);
        assert_eq!(geometry.box_height, 10);
        assert_eq!(geometry.box_left, 4);
        assert_eq!(geometry.box_top, 1);
        assert_eq!(geometry.available_body, 7);
    }

    #[test]
    fn overlay_geometry_defaults_zero_sized_pty_to_renderable_area() {
        let geometry = overlay_geometry(0, 0, 6);

        assert_eq!(geometry.box_width, 72);
        assert_eq!(geometry.box_height, 10);
        assert_eq!(geometry.box_left, 4);
        assert_eq!(geometry.box_top, 7);
        assert_eq!(geometry.available_body, 7);
    }

    // ── normalize_scrollback_token tests ──────────────────────────────

    #[test]
    fn normalize_scrollback_token_no_newlines() {
        assert_eq!(normalize_scrollback_token("hello"), "hello");
    }

    #[test]
    fn normalize_scrollback_token_single_newline() {
        assert_eq!(
            normalize_scrollback_token("line1\nline2"),
            "line1\r\n     line2"
        );
    }

    #[test]
    fn normalize_scrollback_token_multiple_newlines() {
        assert_eq!(
            normalize_scrollback_token("a\nb\nc"),
            "a\r\n     b\r\n     c"
        );
    }

    #[test]
    fn normalize_scrollback_token_empty_string() {
        assert_eq!(normalize_scrollback_token(""), "");
    }

    #[test]
    fn normalize_scrollback_token_only_newline() {
        assert_eq!(normalize_scrollback_token("\n"), "\r\n     ");
    }

    #[test]
    fn normalize_scrollback_token_trailing_newline() {
        assert_eq!(normalize_scrollback_token("text\n"), "text\r\n     ");
    }

    #[test]
    fn normalize_scrollback_token_korean_multiline() {
        assert_eq!(
            normalize_scrollback_token("안녕하세요\n반갑습니다"),
            "안녕하세요\r\n     반갑습니다"
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StyledLine {
    color: Color,
    text: String,
}

fn format_entry(entry: &TranscriptEntry, width: usize) -> Vec<StyledLine> {
    if matches!(entry.kind, EntryKind::Brand) {
        return format_brand_entry(width);
    }

    if matches!(entry.kind, EntryKind::ToolStart | EntryKind::ToolResult) {
        return format_tool_entry(entry, width);
    }

    // Separator: thin dotted line spanning the width
    if matches!(entry.kind, EntryKind::Separator) {
        let line = "╌".repeat(width.min(60));
        return vec![StyledLine {
            color: Color::DarkGrey,
            text: line,
        }];
    }

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
        // Brand and Separator are handled by early returns above; these arms
        // exist solely to satisfy exhaustiveness.
        EntryKind::Brand | EntryKind::Separator => unreachable!(),
    };
    let indent = " ".repeat(prefix.len() + 2);
    let mut out = Vec::new();
    let mut first = true;
    // Track fenced code blocks (```) so lines inside them render in a
    // distinct color. Only applies to Assistant entries — other roles
    // don't produce markdown.
    let is_assistant = matches!(entry.kind, EntryKind::Assistant);
    let mut in_code_block = false;
    for raw_line in entry.content.lines() {
        if is_assistant && raw_line.trim_start().starts_with("```") {
            in_code_block = !in_code_block;
        }
        let line_color = if is_assistant && in_code_block {
            Color::DarkYellow
        } else {
            color
        };
        let wrapped = wrap_plain(raw_line, width.saturating_sub(indent.len()).max(1));
        if wrapped.is_empty() {
            out.push(StyledLine {
                color: line_color,
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
                color: line_color,
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

fn format_brand_entry(width: usize) -> Vec<StyledLine> {
    let card_width = width.clamp(36, 76);
    let inner_width = card_width.saturating_sub(2);
    let title = format!(" zipcode {} ", crate::version::build_label());
    let top = if title.len() + 1 >= inner_width {
        format!("╭{}╮", truncate_to_width(&title, inner_width))
    } else {
        let right = inner_width.saturating_sub(title.len());
        format!("╭{title}{}╮", "─".repeat(right))
    };
    let mut lines = vec![top];
    if width >= BRAND_WORDMARK_MIN_WIDTH {
        lines.extend(
            BRAND_WORDMARK
                .iter()
                .map(|line| centered_brand_line(line, inner_width)),
        );
    } else {
        lines.push(centered_brand_line("ZIPCODE", inner_width));
    }
    lines.push(centered_brand_line("local-only coding agent", inner_width));
    lines.push(centered_brand_line(
        "/help commands · Enter send",
        inner_width,
    ));
    let bottom = format!("╰{}╯", "─".repeat(inner_width));
    lines.push(bottom);

    lines
        .into_iter()
        .map(|text| StyledLine {
            color: Color::Cyan,
            text,
        })
        .collect()
}

fn centered_brand_line(text: &str, inner_width: usize) -> String {
    let text = truncate_to_width(text, inner_width.saturating_sub(2));
    let padding = inner_width.saturating_sub(display_width(&text));
    let left = padding / 2;
    let right = padding.saturating_sub(left);
    format!("│{}{}{}│", " ".repeat(left), text, " ".repeat(right))
}

fn format_tool_start_content(name: &str, args: &Value) -> String {
    if name == "agent" {
        return format_agent_start_content(args);
    }

    let summary_width = terminal_width().saturating_sub(12).clamp(24, 96);
    let summary = summarize_tool_args(name, args, summary_width);
    if summary.is_empty() {
        name.to_string()
    } else {
        format!("{name}\n{summary}")
    }
}

fn format_agent_start_content(args: &Value) -> String {
    let task = args["task"].as_str().unwrap_or("(missing task)").trim();
    let mut lines = vec!["agent".to_string(), task.to_string()];

    let mut meta = Vec::new();
    if let Some(tools) = args["tool_allowlist"].as_array() {
        let names = tools
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(", ");
        if !names.is_empty() {
            meta.push(format!("tools: {names}"));
        }
    }
    if let Some(max_tokens) = args["max_tokens"].as_u64() {
        meta.push(format!("budget: {max_tokens} tokens"));
    }
    if !meta.is_empty() {
        lines.push(meta.join(" · "));
    }

    lines.join("\n")
}

fn format_tool_result_content(name: &str, result: &str) -> String {
    if name == "agent" {
        return format_agent_result_content(result);
    }

    let preview_width = terminal_width().saturating_sub(12).clamp(24, 96);
    let preview = result
        .trim()
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("(no output)");
    format!(
        "{name}\n{}\nstats: {}",
        truncate_to_width(preview, preview_width),
        tool_result_stats(result)
    )
}

fn format_agent_result_content(result: &str) -> String {
    let summary = result.lines().find_map(|line| {
        line.trim()
            .strip_prefix("Summary:")
            .map(str::trim)
            .filter(|value| !value.is_empty())
    });
    let tool_calls = result.lines().find_map(|line| {
        line.trim()
            .strip_prefix("Tool calls:")
            .map(str::trim)
            .filter(|value| !value.is_empty())
    });
    let body = summary.unwrap_or_else(|| {
        result
            .trim()
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .unwrap_or("(no child response)")
    });
    let mut lines = vec!["agent".to_string(), body.to_string()];
    if let Some(count) = tool_calls {
        let label = if count == "1" {
            "1 tool call".to_string()
        } else {
            format!("{count} tool calls")
        };
        lines.push(format!("stats: {label}"));
    }
    lines.join("\n")
}

fn tool_result_stats(result: &str) -> String {
    let lines = result.lines().count().max(1);
    let bytes = result.len();
    format!("{lines} lines · {bytes} B")
}

fn format_tool_entry(entry: &TranscriptEntry, width: usize) -> Vec<StyledLine> {
    let color = match entry.kind {
        EntryKind::ToolStart => Color::Yellow,
        EntryKind::ToolResult => Color::Blue,
        _ => unreachable!(),
    };
    let mut lines = entry.content.lines();
    let first = lines.next().unwrap_or_default().trim();
    let mut details = lines
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();

    let has_structured_name = !details.is_empty();
    let name = has_structured_name
        .then_some(first)
        .filter(|value| !value.is_empty());
    let body = if has_structured_name {
        std::mem::take(&mut details)
    } else if first.is_empty() {
        Vec::new()
    } else {
        vec![first]
    };

    let stats = body
        .last()
        .and_then(|line| line.strip_prefix("stats: "))
        .map(str::to_string);
    let body = if stats.is_some() {
        body[..body.len().saturating_sub(1)].to_vec()
    } else {
        body
    };

    let title = match (entry.kind, name) {
        (EntryKind::ToolStart, Some("agent")) => "Spawning agent".to_string(),
        (EntryKind::ToolStart, Some(name)) => format!("Running {name}"),
        (EntryKind::ToolStart, None) => "Running tool".to_string(),
        (EntryKind::ToolResult, Some("agent")) => "Agent complete".to_string(),
        (EntryKind::ToolResult, Some(name)) => format!("Done {name}"),
        (EntryKind::ToolResult, None) => "Done".to_string(),
        _ => unreachable!(),
    };
    let footer = match entry.kind {
        EntryKind::ToolStart => Some("started".to_string()),
        EntryKind::ToolResult => stats,
        _ => unreachable!(),
    };

    format_tool_card(&title, &body, footer.as_deref(), width, color)
}

fn format_tool_card(
    title: &str,
    body: &[&str],
    footer: Option<&str>,
    width: usize,
    color: Color,
) -> Vec<StyledLine> {
    let text_width = width.saturating_sub(2).max(1);
    let mut out = vec![StyledLine {
        color,
        text: format!(
            "╭─ {}",
            truncate_to_width(title, text_width.saturating_sub(3))
        ),
    }];

    for raw_line in body {
        let wrapped = wrap_plain(raw_line, text_width);
        if wrapped.is_empty() {
            out.push(StyledLine {
                color,
                text: "│".to_string(),
            });
        } else {
            for line in wrapped {
                out.push(StyledLine {
                    color,
                    text: format!("│ {line}"),
                });
            }
        }
    }

    if let Some(footer) = footer.filter(|line| !line.is_empty()) {
        out.push(StyledLine {
            color,
            text: format!(
                "╰─ {}",
                truncate_to_width(footer, text_width.saturating_sub(3))
            ),
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

const fn mouse_scroll_direction(kind: MouseEventKind) -> Option<ScrollDirection> {
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

fn overlay_scroll_offset_after_delta(
    previous: usize,
    max_scroll: usize,
    amount: usize,
    direction: ScrollDirection,
) -> usize {
    match direction {
        ScrollDirection::Older => previous.saturating_sub(amount),
        ScrollDirection::Newer => previous.saturating_add(amount).min(max_scroll),
    }
}

fn scroll_status_label(offset: usize) -> String {
    match offset {
        0 => "latest".to_string(),
        1 => "1 line above latest".to_string(),
        value => format!("{value} lines above latest"),
    }
}

fn overlay_geometry(width: u16, height: u16, body_line_count: usize) -> OverlayGeometry {
    let terminal_width = if width == 0 { 80 } else { width };
    let terminal_height = if height == 0 { 24 } else { height };
    let box_width = terminal_width.saturating_sub(8).max(20).min(terminal_width);
    let max_box_height = terminal_height
        .saturating_sub(2)
        .max(6)
        .min(terminal_height);
    let box_height = as_u16(body_line_count)
        .saturating_add(4)
        .min(max_box_height)
        .max(1);
    OverlayGeometry {
        box_width,
        box_height,
        box_left: (terminal_width.saturating_sub(box_width)) / 2,
        box_top: (terminal_height.saturating_sub(box_height)) / 2,
        body_width: box_width.saturating_sub(4) as usize,
        available_body: box_height.saturating_sub(3) as usize,
    }
}

fn wrapped_overlay_body(overlay: &Overlay, body_width: usize) -> Vec<String> {
    let mut body_lines = Vec::new();
    for line in &overlay.body {
        body_lines.extend(wrap_plain(line, body_width.max(1)));
    }
    body_lines
}

fn layout_for_terminal(height: u16, composer_visible: usize) -> TuiLayout {
    let composer_height = as_u16(composer_visible.clamp(1, MAX_COMPOSER_LINES));
    let status_y = height.saturating_sub(HINT_LINES + STATUS_LINES + composer_height);
    let transcript_top = HEADER_LINES;
    TuiLayout {
        transcript_top,
        status_y,
        status_line_y: status_y + 1,
        composer_label_y: status_y + 2,
        composer_top: status_y + STATUS_LINES,
        hint_y: height.saturating_sub(1),
        transcript_height: status_y.saturating_sub(transcript_top) as usize,
        composer_height,
    }
}

fn initial_transcript_entries(session: &Session) -> Vec<TranscriptEntry> {
    let entries = transcript_entries_from_session(session);
    if entries.is_empty() {
        vec![TranscriptEntry {
            kind: EntryKind::Brand,
            content: String::new(),
        }]
    } else {
        entries
    }
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
