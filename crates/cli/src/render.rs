use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use serde_json::Value;
use termimad::MadSkin;

// ── Terminal utilities ──────────────────────────────────────────────

/// Returns the current terminal width, defaulting to 80 columns.
pub fn terminal_width() -> usize {
    termimad::crossterm::terminal::size()
        .map(|(w, _)| w as usize)
        .unwrap_or(80)
}

/// Truncate a string to fit within `max_width` visible characters.
/// Appends "…" if the string was truncated.
pub fn truncate_to_width(s: &str, max_width: usize) -> String {
    if max_width < 2 {
        return String::new();
    }
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_width {
        return s.to_string();
    }
    let truncated: String = chars[..max_width - 1].iter().collect();
    format!("{truncated}…")
}

// ── Synchronized output ─────────────────────────────────────────────

/// Write to stderr inside a synchronized output block (CSI 2026).
/// Prevents flicker by batching screen updates atomically.
fn sync_eprintln(text: &str) {
    let stderr = io::stderr();
    let mut lock = stderr.lock();
    let _ = lock.write_all(b"\x1b[?2026h");
    let _ = lock.write_all(text.as_bytes());
    let _ = lock.write_all(b"\n");
    let _ = lock.write_all(b"\x1b[?2026l");
    let _ = lock.flush();
}

// ── Spinner ──────────────────────────────────────────────────────────

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// A braille-dot spinner that runs on a background thread.
///
/// Writes to stderr so it doesn't interfere with streamed token output
/// on stdout. Clears its own line when stopped.
pub struct Spinner {
    active: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Spinner {
    /// Start a spinner with a label (e.g. "thinking").
    pub fn start(message: &str) -> Self {
        let active = Arc::new(AtomicBool::new(true));
        let active_clone = active.clone();
        let msg = message.to_string();

        let handle = std::thread::spawn(move || {
            let mut i = 0usize;
            let stderr = io::stderr();
            while active_clone.load(Ordering::Relaxed) {
                let frame = SPINNER_FRAMES[i % SPINNER_FRAMES.len()];
                {
                    let mut lock = stderr.lock();
                    let _ = write!(lock, "\r\x1b[2K\x1b[90m{frame} {msg}\x1b[0m");
                    let _ = lock.flush();
                }
                std::thread::sleep(std::time::Duration::from_millis(80));
                i = i.wrapping_add(1);
            }
            // Clear the spinner line before exiting
            let mut lock = stderr.lock();
            let _ = write!(lock, "\r\x1b[2K");
            let _ = lock.flush();
        });

        Spinner {
            active,
            handle: Some(handle),
        }
    }

    /// Stop the spinner, blocking until the thread finishes.
    pub fn stop(&mut self) {
        self.active.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.stop();
    }
}

// ── Markdown rendering ──────────────────────────────────────────────

/// Create a styled MadSkin for markdown rendering.
pub fn create_skin() -> MadSkin {
    let mut skin = MadSkin::default();
    skin.headers[0].set_fg(termimad::crossterm::style::Color::Cyan);
    skin.headers[1].set_fg(termimad::crossterm::style::Color::Blue);
    skin.headers[2].set_fg(termimad::crossterm::style::Color::Green);
    skin.bold.set_fg(termimad::crossterm::style::Color::Yellow);
    skin.italic
        .set_fg(termimad::crossterm::style::Color::Magenta);
    skin
}

/// Render a markdown block to the terminal.
#[allow(dead_code)]
pub fn render_markdown(text: &str) {
    let skin = create_skin();
    skin.print_text(text);
}

// ── Tool output formatting ──────────────────────────────────────────

/// Print a tool invocation header: `▶ tool_name(args…)`
pub fn print_tool_start(name: &str, args: &Value) {
    let width = terminal_width();
    // Reserve space for "▶ name(" + ")"  →  name.len() + 4 (▶ takes ~2 cols)
    let max_args = width.saturating_sub(name.len() + 5);
    let args_str = format_tool_args(args, max_args);
    sync_eprintln(&format!(
        "\x1b[33m▶\x1b[0m \x1b[1m{name}\x1b[0m\x1b[90m({args_str})\x1b[0m"
    ));
}

/// Print a tool result: `◀ tool_name: preview…`
pub fn print_tool_result(name: &str, result: &str) {
    let width = terminal_width();
    let preview = result.trim().replace('\n', " ");
    // Reserve space for "◀ name: " → name.len() + 4
    let max_preview = width.saturating_sub(name.len() + 5);
    let preview = truncate_to_width(&preview, max_preview);
    sync_eprintln(&format!(
        "\x1b[32m◀\x1b[0m \x1b[1m{name}\x1b[0m\x1b[90m: {preview}\x1b[0m"
    ));
}

/// Format tool arguments into "key=val, key=val", respecting max width.
fn format_tool_args(args: &Value, max_width: usize) -> String {
    let raw = match args {
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| {
                let val = match v {
                    Value::String(s) => {
                        let s = s.trim();
                        truncate_to_width(s, 60)
                    }
                    other => {
                        let s = other.to_string();
                        truncate_to_width(&s, 60)
                    }
                };
                format!("{k}={val}")
            })
            .collect::<Vec<_>>()
            .join(", "),
        other => other.to_string(),
    };
    truncate_to_width(&raw, max_width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_short_string_unchanged() {
        assert_eq!(truncate_to_width("hello", 10), "hello");
    }

    #[test]
    fn truncate_exact_width() {
        assert_eq!(truncate_to_width("hello", 5), "hello");
    }

    #[test]
    fn truncate_adds_ellipsis() {
        assert_eq!(truncate_to_width("hello world", 6), "hello…");
    }

    #[test]
    fn truncate_tiny_width_returns_empty() {
        assert_eq!(truncate_to_width("hello", 1), "");
        assert_eq!(truncate_to_width("hello", 0), "");
    }

    #[test]
    fn format_args_truncates_long_values() {
        let args = serde_json::json!({"command": "a]".repeat(100)});
        let formatted = format_tool_args(&args, 80);
        assert!(formatted.len() <= 80 + 3); // allow for ellipsis multi-byte
    }
}
