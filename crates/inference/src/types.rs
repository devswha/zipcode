use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Model,
    Tool,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallParsed>>,
}

impl ChatMessage {
    pub fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: Role::Model,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant_with_tool_calls(content: &str, calls: Vec<ToolCallParsed>) -> Self {
        Self {
            role: Role::Model,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: Some(calls),
        }
    }

    pub fn tool_result(call_id: &str, content: &str) -> Self {
        Self {
            role: Role::Tool,
            content: content.to_string(),
            tool_call_id: Some(call_id.to_string()),
            tool_calls: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallParsed {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone)]
pub enum TokenEvent {
    Token(String),
    /// A chunk of the model's private reasoning channel (Gemma 4
    /// `<|channel>thought<channel|>`). The UI may render this as dimmed /
    /// collapsible text. It must NOT be appended to assistant history —
    /// Gemma 4 `chat_template_caps.supports_preserve_reasoning` is `false`
    /// and the GGUF-embedded `strip_thinking` macro drops it on re-injection.
    Thinking(String),
    ToolCall(ToolCallParsed),
    Done(FinishReason),
    Error(InferenceError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    MaxTokens,
    ToolUse,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum InferenceError {
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    #[error("CUDA out of memory")]
    OutOfMemory,
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
    #[error("Generation error: {0}")]
    GenerationError(String),
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub max_tokens: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    /// Enable Gemma 4's thinking channel. When `true`, the backend asks
    /// llama-server to stream `delta.reasoning_content` alongside regular
    /// content — the runtime surfaces it via `TokenEvent::Thinking` so the
    /// UI can render the model's reasoning. When `false`, the template
    /// `<|think|>` token is not emitted and the model produces direct
    /// answers only. Defaults to `true` because the visible reasoning is
    /// a core part of the Claude-Code-like experience zipcode targets.
    pub enable_thinking: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 4096,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            enable_thinking: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_user() {
        let msg = ChatMessage::user("hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "hello");
    }

    #[test]
    fn test_chat_message_tool_result() {
        let msg = ChatMessage::tool_result("call_1", "file contents here");
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn test_generation_config_defaults() {
        let config = GenerationConfig::default();
        assert!((config.temperature - 0.7).abs() < f64::EPSILON);
        assert_eq!(config.max_tokens, 4096);
    }

    #[test]
    fn test_assistant_with_tool_calls() {
        let call = ToolCallParsed {
            id: "1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls"}),
        };
        let msg = ChatMessage::assistant_with_tool_calls("", vec![call]);
        assert_eq!(msg.tool_calls.unwrap().len(), 1);
    }
}
