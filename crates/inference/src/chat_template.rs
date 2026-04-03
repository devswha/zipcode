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
pub fn parse_tool_calls(output: &str) -> Vec<ToolCallParsed> {
    let mut calls = Vec::new();
    let mut search_from = 0;

    while let Some(start_offset) = output[search_from..].find("<tool_call>") {
        let json_start = search_from + start_offset + "<tool_call>".len();
        if let Some(end_offset) = output[json_start..].find("</tool_call>") {
            let json_str = output[json_start..json_start + end_offset].trim();
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                let name = parsed["name"].as_str().unwrap_or("").to_string();
                let arguments = parsed["arguments"].clone();
                calls.push(ToolCallParsed {
                    id: format!("call_{}", calls.len()),
                    name,
                    arguments,
                });
            }
            search_from = json_start + end_offset + "</tool_call>".len();
        } else {
            break;
        }
    }

    calls
}

/// Extract plain text from model output, stripping all `<tool_call>` blocks.
pub fn extract_text_content(output: &str) -> String {
    let mut result = output.to_string();
    while let Some(start) = result.find("<tool_call>") {
        if let Some(end_offset) = result[start..].find("</tool_call>") {
            result = format!(
                "{}{}",
                &result[..start],
                &result[start + end_offset + "</tool_call>".len()..]
            );
        } else {
            break;
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
}
