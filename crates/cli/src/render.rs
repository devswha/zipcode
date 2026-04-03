use serde_json::Value;
use termimad::MadSkin;

/// Create a styled MadSkin for markdown rendering.
#[allow(dead_code)]
pub fn create_skin() -> MadSkin {
    let mut skin = MadSkin::default();
    // Bold headers
    skin.headers[0].set_fg(termimad::crossterm::style::Color::Cyan);
    skin.headers[1].set_fg(termimad::crossterm::style::Color::Blue);
    skin.headers[2].set_fg(termimad::crossterm::style::Color::Green);
    skin.bold.set_fg(termimad::crossterm::style::Color::Yellow);
    skin.italic
        .set_fg(termimad::crossterm::style::Color::Magenta);
    skin
}

/// Render a full markdown block to the terminal.
#[allow(dead_code)]
pub fn render_markdown(text: &str) {
    let skin = create_skin();
    skin.print_text(text);
}

/// Print a tool invocation line: yellow ">" + tool name + args summary.
pub fn print_tool_start(name: &str, args: &Value) {
    let args_str = match args {
        Value::Object(map) => map
            .iter()
            .map(|(k, v): (&String, &Value)| {
                let val = match v {
                    Value::String(s) => {
                        let s = s.trim();
                        if s.len() > 60 {
                            format!("{}...", &s[..60])
                        } else {
                            s.to_string()
                        }
                    }
                    other => other.to_string(),
                };
                format!("{k}={val}")
            })
            .collect::<Vec<_>>()
            .join(", "),
        other => other.to_string(),
    };

    eprintln!("\x1b[33m>\x1b[0m \x1b[1m{name}\x1b[0m({args_str})");
}

/// Print a tool result line: green "<" + tool name + truncated preview (200 chars).
pub fn print_tool_result(name: &str, result: &str) {
    let preview = result.trim();
    let preview = if preview.len() > 200 {
        format!("{}...", &preview[..200])
    } else {
        preview.to_string()
    };
    // Replace newlines with spaces for single-line display
    let preview = preview.replace('\n', " ");
    eprintln!("\x1b[32m<\x1b[0m \x1b[1m{name}\x1b[0m: {preview}");
}
