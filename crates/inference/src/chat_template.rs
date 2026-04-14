use crate::types::{ChatMessage, Role, ToolCallParsed};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Format a single message in Gemma 4 turn format
pub fn format_message(msg: &ChatMessage, tools: &[ToolSpec]) -> String {
    match msg.role {
        Role::System => {
            format!("<start_of_turn>user\n{}<end_of_turn>\n", msg.content)
        }
        Role::User => {
            let mut parts = String::new();
            if !tools.is_empty() {
                let tools_json = serde_json::to_string_pretty(tools).unwrap_or_default();
                parts.push_str(&format!(
                    "You have access to the following tools:\n{tools_json}\n\n"
                ));
            }
            parts.push_str(&msg.content);
            format!("<start_of_turn>user\n{parts}<end_of_turn>\n")
        }
        Role::Model => {
            format!("<start_of_turn>model\n{}<end_of_turn>\n", msg.content)
        }
        Role::Tool => {
            format!("<start_of_turn>tool\n{}<end_of_turn>\n", msg.content)
        }
    }
}

/// Format an entire conversation history into a single prompt string.
/// Tools are injected into the first user turn only.
/// Ends with `<start_of_turn>model\n` to prime generation.
pub fn format_conversation(messages: &[ChatMessage], tools: &[ToolSpec]) -> String {
    let mut prompt = String::new();
    let mut tools_injected = false;

    for msg in messages {
        if msg.role == Role::User && !tools_injected && !tools.is_empty() {
            prompt.push_str(&format_message(msg, tools));
            tools_injected = true;
        } else {
            prompt.push_str(&format_message(msg, &[]));
        }
    }

    // Add model turn prefix to prime generation
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

/// Parse tool calls from model output text.
/// Extracts all `<tool_call>...</tool_call>` blocks and parses their JSON.
/// Handles nested `</tool_call>` within JSON strings by trying progressively
/// larger slices until valid JSON is found.
pub fn parse_tool_calls(output: &str) -> Vec<ToolCallParsed> {
    let mut calls = Vec::new();
    let mut search_from = 0;

    while let Some(start_offset) = output[search_from..].find("<tool_call>") {
        let json_start = search_from + start_offset + "<tool_call>".len();

        // Find closing tag, but if JSON is invalid, try the next </tool_call>
        // to handle cases where </tool_call> appears inside JSON string values.
        let mut inner_search = 0;
        let mut found = false;

        while let Some(end_offset) = output[json_start + inner_search..].find("</tool_call>") {
            let actual_end = inner_search + end_offset;
            let json_str = output[json_start..json_start + actual_end].trim();

            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                let name = parsed["name"].as_str().unwrap_or("").to_string();
                if name.is_empty() {
                    tracing::warn!("Tool call has empty name, skipping");
                    search_from = json_start + actual_end + "</tool_call>".len();
                    found = true;
                    break;
                }
                let arguments = parsed["arguments"].clone();
                calls.push(ToolCallParsed {
                    id: format!("call_{}", calls.len()),
                    name,
                    arguments,
                });
                search_from = json_start + actual_end + "</tool_call>".len();
                found = true;
                break;
            }

            // JSON invalid — try the next </tool_call> occurrence
            tracing::debug!(
                "Tool call JSON invalid at offset {actual_end}, trying next closing tag"
            );
            inner_search = actual_end + "</tool_call>".len();
        }

        if !found {
            tracing::warn!(
                "Malformed tool call block: no valid JSON found between <tool_call> tags (offset {json_start})"
            );
            search_from = json_start;
        }
    }

    calls
}

/// Extract plain text from model output, stripping all `<tool_call>` blocks.
/// Handles nested `</tool_call>` within JSON strings by validating JSON before stripping.
pub fn extract_text_content(output: &str) -> String {
    let mut result = output.to_string();
    loop {
        let Some(start) = result.find("<tool_call>") else {
            break;
        };
        let after_tag = start + "<tool_call>".len();
        let mut inner_search = 0;
        let mut found = false;

        while let Some(end_offset) = result[after_tag + inner_search..].find("</tool_call>") {
            let actual_end = inner_search + end_offset;
            let json_str = result[after_tag..after_tag + actual_end].trim();

            // Accept either valid JSON or skip to the next closing tag
            if json_str.is_empty() || serde_json::from_str::<serde_json::Value>(json_str).is_ok() {
                let block_end = after_tag + actual_end + "</tool_call>".len();
                result = format!("{}{}", &result[..start], &result[block_end..]);
                found = true;
                break;
            }
            inner_search = actual_end + "</tool_call>".len();
        }

        if !found {
            let next_open = result[after_tag..]
                .find("<tool_call>")
                .map(|offset| after_tag + offset);
            let next_close = result[after_tag..]
                .find("</tool_call>")
                .map(|offset| after_tag + offset);

            match (next_open, next_close) {
                (None, Some(close)) => {
                    let block_end = close + "</tool_call>".len();
                    result = format!("{}{}", &result[..start], &result[block_end..]);
                }
                (Some(open), Some(close)) if close < open => {
                    let block_end = close + "</tool_call>".len();
                    result = format!("{}{}", &result[..start], &result[block_end..]);
                }
                _ => {
                    result = format!("{}{}", &result[..start], &result[after_tag..]);
                }
            }
        }
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn test_format_user_turn() {
        let msg = ChatMessage::user("Hello");
        let formatted = format_message(&msg, &[]);
        assert!(formatted.contains("<start_of_turn>user"));
        assert!(formatted.contains("Hello"));
        assert!(formatted.contains("<end_of_turn>"));
    }

    #[test]
    fn test_format_with_tools_in_system() {
        let tools = vec![ToolSpec {
            name: "bash".to_string(),
            description: "Execute shell commands".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
        }];
        let msg = ChatMessage::user("run ls");
        let formatted = format_message(&msg, &tools);
        assert!(formatted.contains("bash"));
        assert!(formatted.contains("Execute shell commands"));
    }

    #[test]
    fn test_parse_tool_call_from_output() {
        let output = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"src/main.rs\"}}\n</tool_call>";
        let parsed = parse_tool_calls(output);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "read_file");
        assert_eq!(parsed[0].arguments["file_path"], "src/main.rs");
    }

    #[test]
    fn test_parse_no_tool_call() {
        let output = "Here is the file content.";
        let parsed = parse_tool_calls(output);
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_format_tool_result_turn() {
        let msg = ChatMessage::tool_result("call_1", "{\"content\": \"hello\"}");
        let formatted = format_message(&msg, &[]);
        assert!(formatted.contains("<start_of_turn>tool"));
        assert!(formatted.contains("hello"));
    }

    #[test]
    fn test_format_conversation() {
        let messages = vec![
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello!"),
            ChatMessage::user("read main.rs"),
        ];
        let formatted = format_conversation(&messages, &[]);
        assert_eq!(formatted.matches("<start_of_turn>").count(), 4); // 3 messages + 1 model prefix
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let output = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>\nsome text\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"a.rs\"}}\n</tool_call>";
        let parsed = parse_tool_calls(output);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "bash");
        assert_eq!(parsed[1].name, "read_file");
    }

    #[test]
    fn test_extract_text_content() {
        let output = "Hello <tool_call>{\"name\": \"bash\", \"arguments\": {}}</tool_call> world";
        let text = extract_text_content(output);
        assert_eq!(text, "Hello  world");
    }

    #[test]
    fn test_parse_nested_tool_call_in_json_string() {
        // Model asks bash to echo a string containing </tool_call>
        let output = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"echo '</tool_call>'\"}}\n</tool_call>";
        let parsed = parse_tool_calls(output);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "bash");
        assert_eq!(
            parsed[0].arguments["command"].as_str().unwrap(),
            "echo '</tool_call>'"
        );
    }

    #[test]
    fn test_extract_text_with_nested_closing_tag() {
        let output =
            "Hello <tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"echo '</tool_call>'\"}}</tool_call> world";
        let text = extract_text_content(output);
        assert_eq!(text, "Hello  world");
    }

    #[test]
    fn test_parse_malformed_json_skipped_with_warning() {
        let output = "<tool_call>\nnot valid json\n</tool_call>";
        let parsed = parse_tool_calls(output);
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_parse_empty_name_skipped() {
        let output = "<tool_call>\n{\"arguments\": {\"foo\": \"bar\"}}\n</tool_call>";
        let parsed = parse_tool_calls(output);
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_parse_unclosed_tool_call_tag() {
        let output = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}";
        let parsed = parse_tool_calls(output);
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_parse_empty_tool_call() {
        let output = "<tool_call></tool_call>";
        let parsed = parse_tool_calls(output);
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_parse_malformed_block_does_not_hide_later_valid_call() {
        let output = concat!(
            "<tool_call>\n",
            "{not valid json}\n",
            "</tool_call>\n",
            "still talking\n",
            "<tool_call>\n",
            "{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"src/main.rs\"}}\n",
            "</tool_call>"
        );
        let parsed = parse_tool_calls(output);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "read_file");
        assert_eq!(parsed[0].arguments["file_path"], "src/main.rs");
    }

    #[test]
    fn test_parse_unclosed_malformed_block_does_not_hide_later_valid_call() {
        let output = concat!(
            "<tool_call>\n",
            "{\"name\":\n",
            "plain text between blocks\n",
            "<tool_call>\n",
            "{\"name\": \"bash\", \"arguments\": {\"command\": \"pwd\"}}\n",
            "</tool_call>"
        );
        let parsed = parse_tool_calls(output);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "bash");
        assert_eq!(parsed[0].arguments["command"], "pwd");
    }

    #[test]
    fn test_extract_text_strips_single_closed_malformed_block() {
        let output = "Hello <tool_call>{not json}</tool_call> world";
        let text = extract_text_content(output);
        assert_eq!(text, "Hello  world");
    }

    #[test]
    fn test_extract_text_skips_closed_malformed_block_and_later_valid_call() {
        let output = concat!(
            "<tool_call>\n",
            "{not valid json}\n",
            "</tool_call>\n",
            "still talking\n",
            "<tool_call>\n",
            "{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"src/main.rs\"}}\n",
            "</tool_call>"
        );
        let text = extract_text_content(output);
        assert_eq!(text, "still talking");
    }

    #[test]
    fn test_extract_text_recovers_later_valid_call_after_unclosed_malformed_block() {
        let output = concat!(
            "<tool_call>\n",
            "{\"name\":\n",
            "plain text between blocks\n",
            "<tool_call>\n",
            "{\"name\": \"bash\", \"arguments\": {\"command\": \"pwd\"}}\n",
            "</tool_call>"
        );
        let text = extract_text_content(output);
        assert!(text.contains("plain text between blocks"));
        assert!(!text.contains("<tool_call>"));
        assert!(!text.contains("\"command\": \"pwd\""));
    }

    #[test]
    fn test_extract_text_preserves_unclosed_content_without_tag_markup() {
        let output = "Hello <tool_call>partial content";
        let text = extract_text_content(output);
        assert_eq!(text, "Hello partial content");
    }
}
