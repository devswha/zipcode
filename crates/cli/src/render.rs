use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use serde_json::Value;
use termimad::MadSkin;

// ── Verbose flag ─────────────────────────────────────────────────────
//
// Process-wide toggle for surfacing thinking reasoning inline. Set once
// at startup from `--verbose`/`-v` or `ZIPCODE_VERBOSE=1`, then read by
// the callbacks. Keeps the default experience Claude-Code-like (reasoning
// is happening silently behind a counter) while still letting developers
// flip the full stream on when debugging prompt behavior.

static VERBOSE: AtomicBool = AtomicBool::new(false);

/// Enable verbose rendering for the remainder of the process.
pub fn set_verbose(on: bool) {
    VERBOSE.store(on, Ordering::Relaxed);
}

/// Check whether verbose rendering is enabled.
pub fn is_verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed)
}

// ── Terminal utilities ──────────────────────────────────────────────

/// Returns the current terminal width, defaulting to 80 columns.
pub fn terminal_width() -> usize {
    termimad::crossterm::terminal::size()
        .map(|(w, _)| w as usize)
        .unwrap_or(80)
}

/// Truncate a string to fit within `max_width` display columns.
/// Appends "…" if the string was truncated.  Uses Unicode display-width so
/// CJK characters (2 columns each) are measured correctly.
pub fn truncate_to_width(s: &str, max_width: usize) -> String {
    crate::width::truncate_display(s, max_width)
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
/// on stdout. Clears its own line when stopped. Exposes a thinking-bytes
/// counter so the silent (non-verbose) thinking path can still surface
/// progress feedback to the user without dumping reasoning content.
pub struct Spinner {
    active: Arc<AtomicBool>,
    thinking_bytes: Arc<AtomicUsize>,
    handle: Option<JoinHandle<()>>,
}

impl Spinner {
    /// Start a spinner with a label (e.g. "thinking").
    pub fn start(message: &str) -> Self {
        let active = Arc::new(AtomicBool::new(true));
        let active_clone = active.clone();
        let thinking_bytes = Arc::new(AtomicUsize::new(0));
        let thinking_clone = thinking_bytes.clone();
        let msg = message.to_string();

        let handle = std::thread::spawn(move || {
            let mut i = 0usize;
            let stderr = io::stderr();
            while active_clone.load(Ordering::Relaxed) {
                let frame = SPINNER_FRAMES[i % SPINNER_FRAMES.len()];
                let count = thinking_clone.load(Ordering::Relaxed);
                let label = if count > 0 {
                    format!("{msg} ({count} chars)")
                } else {
                    msg.clone()
                };
                {
                    let mut lock = stderr.lock();
                    let _ = write!(lock, "\r\x1b[2K\x1b[90m{frame} {label}\x1b[0m");
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
            thinking_bytes,
            handle: Some(handle),
        }
    }

    /// Record additional thinking bytes streamed from the model so the
    /// spinner label can show live progress without printing the reasoning.
    pub fn add_thinking_bytes(&self, bytes: usize) {
        self.thinking_bytes.fetch_add(bytes, Ordering::Relaxed);
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
    let args_str = summarize_tool_args(name, args, max_args);
    sync_eprintln(&format!(
        "\x1b[33m▶\x1b[0m \x1b[1m{name}\x1b[0m\x1b[90m({args_str})\x1b[0m"
    ));
}

/// Per-tool argument summarization.
///
/// Claude-Code-style: show the most useful identifier (usually `path`)
/// and a size hint, and elide long free-form content so bulk edits do
/// not dump hundreds of lines of code into the terminal. Falls through
/// to the generic `format_tool_args` rendering for tools we do not
/// specialize.
pub(crate) fn summarize_tool_args(name: &str, args: &Value, max_width: usize) -> String {
    match name {
        "write_file" => {
            let path = args["path"].as_str().unwrap_or("?");
            let content = args["content"].as_str().unwrap_or("");
            let lines = content.lines().count().max(1);
            let bytes = content.len();
            truncate_to_width(&format!("path={path}, {lines} lines, {bytes} B"), max_width)
        }
        "edit_file" => {
            let path = args["path"].as_str().unwrap_or("?");
            let old_lines = args["old_string"]
                .as_str()
                .map(|s| s.lines().count().max(1))
                .unwrap_or(0);
            let new_lines = args["new_string"]
                .as_str()
                .map(|s| s.lines().count().max(1))
                .unwrap_or(0);
            truncate_to_width(
                &format!("path={path}, -{old_lines}/+{new_lines} lines"),
                max_width,
            )
        }
        "bash" => {
            let cmd = args["command"].as_str().unwrap_or("");
            let first_line = cmd.lines().next().unwrap_or(cmd);
            truncate_to_width(first_line, max_width)
        }
        "repl" => {
            let code = args["code"].as_str().unwrap_or("");
            let lines = code.lines().count().max(1);
            let preview = code.lines().next().unwrap_or("");
            if lines > 1 {
                truncate_to_width(&format!("{preview} … ({lines} lines)"), max_width)
            } else {
                truncate_to_width(preview, max_width)
            }
        }
        _ => format_tool_args(args, max_width),
    }
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

    #[test]
    fn summarize_write_file_elides_content_shows_size() {
        let huge = "fn main() {}\n".repeat(200);
        let args = serde_json::json!({"path": "src/main.rs", "content": huge});
        let summary = summarize_tool_args("write_file", &args, 120);
        assert!(summary.contains("path=src/main.rs"));
        assert!(summary.contains("200 lines"));
        assert!(
            !summary.contains("fn main"),
            "content body must not leak into summary: {summary}"
        );
    }

    #[test]
    fn summarize_edit_file_shows_line_delta() {
        let args = serde_json::json!({
            "path": "foo.rs",
            "old_string": "line1\nline2\nline3",
            "new_string": "replacement\nwith\ntwo extra\nlines\nhere",
        });
        let summary = summarize_tool_args("edit_file", &args, 120);
        assert!(summary.contains("path=foo.rs"));
        assert!(summary.contains("-3/+5 lines"));
        assert!(!summary.contains("line1"));
        assert!(!summary.contains("replacement"));
    }

    #[test]
    fn summarize_bash_shows_first_line_only() {
        let args = serde_json::json!({"command": "echo hello\nrm -rf /\n# not shown"});
        let summary = summarize_tool_args("bash", &args, 120);
        assert_eq!(summary, "echo hello");
    }

    #[test]
    fn summarize_repl_shows_line_count_when_multiline() {
        let args = serde_json::json!({"code": "x = 1\ny = 2\nprint(x + y)"});
        let summary = summarize_tool_args("repl", &args, 120);
        assert!(summary.contains("x = 1"));
        assert!(summary.contains("(3 lines)"));
    }

    #[test]
    fn summarize_unknown_tool_falls_back_to_generic_format() {
        let args = serde_json::json!({"pattern": "**/*.rs"});
        let summary = summarize_tool_args("glob_search", &args, 120);
        assert_eq!(summary, "pattern=**/*.rs");
    }

    #[test]
    fn verbose_flag_round_trip() {
        // Not thread-safe with other tests that touch VERBOSE, but we
        // only set it here and restore. Run serially.
        let prior = is_verbose();
        set_verbose(true);
        assert!(is_verbose());
        set_verbose(false);
        assert!(!is_verbose());
        set_verbose(prior);
    }
}
