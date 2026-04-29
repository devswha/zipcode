use serde::{Deserialize, Serialize};

/// Role of a participant in the conversation.
///
/// Maps to the Gemma 4 chat-template turn markers (`<start_of_turn>user`,
/// `<start_of_turn>model`, etc.). Serialized as lowercase strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Human user or system prompt.
    User,
    /// Model / assistant response.
    Model,
    /// Tool execution result.
    Tool,
    /// System-level instruction injected before user turns.
    System,
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_false(b: &bool) -> bool {
    !b
}

/// A single message in the conversation, flowing between all crates.
///
/// This is the primary wire type — it carries user input, model responses,
/// tool invocations, and tool results. The `tool_calls` field is populated
/// when the model requests tool execution; `tool_call_id` is set on tool
/// result messages to correlate them back to the originating call.
///
/// The `agent_invisible` flag supports the two-tier context compaction system:
/// messages marked invisible are excluded from inference but retained in the
/// session file for replay fidelity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallParsed>>,
    /// When `true`, this message is hidden from the inference provider and
    /// excluded from token estimates.  Used by the two-tier context compaction
    /// system to mark original tool-call / tool-result messages after they
    /// have been summarised, without physically removing them (preserving
    /// session replay and debugging fidelity).
    #[serde(default, skip_serializing_if = "is_false")]
    pub agent_invisible: bool,
}

impl ChatMessage {
    /// Create a user-role message.
    #[must_use]
    pub fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
            agent_invisible: false,
        }
    }

    /// Create a system-role message (injected as the first turn).
    #[must_use]
    pub fn system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
            agent_invisible: false,
        }
    }

    /// Create an assistant (model) message without tool calls.
    #[must_use]
    pub fn assistant(content: &str) -> Self {
        Self {
            role: Role::Model,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
            agent_invisible: false,
        }
    }

    /// Create an assistant message that carries one or more parsed tool calls.
    #[must_use]
    pub fn assistant_with_tool_calls(content: &str, calls: Vec<ToolCallParsed>) -> Self {
        Self {
            role: Role::Model,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: Some(calls),
            agent_invisible: false,
        }
    }

    /// Create a tool-result message, correlated to a specific tool call by `call_id`.
    #[must_use]
    pub fn tool_result(call_id: &str, content: &str) -> Self {
        Self {
            role: Role::Tool,
            content: content.to_string(),
            tool_call_id: Some(call_id.to_string()),
            tool_calls: None,
            agent_invisible: false,
        }
    }
}

/// A parsed tool call extracted from model output.
///
/// Contains the unique call id, tool name, and JSON arguments as produced
/// by the chat-template parser (e.g. `parse_tool_calls`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallParsed {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Streaming event emitted by inference backends during generation.
///
/// Consumers read these from the `mpsc::Receiver` returned by
/// [`InferenceProvider::generate_stream`]. The stream always ends with
/// either `Done` or `Error`.
#[derive(Debug, Clone)]
pub enum TokenEvent {
    /// A decoded text token from the model.
    Token(String),
    /// A chunk of the model's private reasoning channel (Gemma 4
    /// `<|channel>thought<channel|>`). The UI may render this as dimmed /
    /// collapsible text. It must NOT be appended to assistant history —
    /// Gemma 4 `chat_template_caps.supports_preserve_reasoning` is `false`
    /// and the GGUF-embedded `strip_thinking` macro drops it on re-injection.
    Thinking(String),
    /// A fully parsed tool call emitted mid-stream.
    ToolCall(ToolCallParsed),
    /// Generation finished with the given reason.
    Done(FinishReason),
    /// An error occurred during generation.
    Error(InferenceError),
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Model emitted a stop token (end of turn).
    Stop,
    /// Reached the configured `max_tokens` limit.
    MaxTokens,
    /// Model produced a tool call — the runtime should dispatch tools.
    ToolUse,
}

/// Errors that can occur during model loading or generation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum InferenceError {
    /// The specified model file does not exist or is unreadable.
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    /// CUDA ran out of GPU memory during generation.
    #[error("CUDA out of memory")]
    OutOfMemory,
    /// Tokenizer failed to load or encode/decode.
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
    /// Generic generation failure (e.g. backend crash, malformed output).
    #[error("Generation error: {0}")]
    GenerationError(String),
}

/// Sampling and generation parameters passed to inference backends.
///
/// Controls temperature, top-p/top-k filtering, repeat penalty, and
/// the maximum number of tokens per generation. The `enable_thinking`
/// flag toggles Gemma 4's reasoning channel.
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
            // Agentic workflows need room for thinking tokens + tool-call
            // arguments that carry entire file contents. 4096 was too small
            // — a 300-line write_file argument alone can exceed 3000 tokens,
            // and thinking eats another ~300. Setting this to the context
            // window size (8192) lets the model use whatever budget the
            // prompt leaves free; llama-server enforces the real ceiling.
            max_tokens: 8192,
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
        assert_eq!(config.max_tokens, 8192);
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

    // ── Serde roundtrip tests ─────────────────────────────────────

    #[test]
    fn test_role_serde_roundtrip() {
        for (variant, expected_json) in [
            (Role::User, "\"user\""),
            (Role::Model, "\"model\""),
            (Role::Tool, "\"tool\""),
            (Role::System, "\"system\""),
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            assert_eq!(
                json, expected_json,
                "Role::{variant:?} serialization mismatch"
            );
            let back: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(
                back, variant,
                "Role deserialization roundtrip failed for {variant:?}"
            );
        }
    }

    #[test]
    fn test_chat_message_serde_roundtrip_user() {
        let msg = ChatMessage::user("hello world");
        let json = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, Role::User);
        assert_eq!(back.content, "hello world");
        assert!(back.tool_call_id.is_none());
        assert!(back.tool_calls.is_none());
    }

    #[test]
    fn test_chat_message_serde_roundtrip_tool_result() {
        let msg = ChatMessage::tool_result("call_42", "file contents here");
        let json = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, Role::Tool);
        assert_eq!(back.content, "file contents here");
        assert_eq!(back.tool_call_id.as_deref(), Some("call_42"));
        assert!(back.tool_calls.is_none());
    }

    #[test]
    fn test_chat_message_serde_roundtrip_with_tool_calls() {
        let call = ToolCallParsed {
            id: "call_99".to_string(),
            name: "write_file".to_string(),
            arguments: serde_json::json!({"path": "test.txt", "content": "hi"}),
        };
        let msg = ChatMessage::assistant_with_tool_calls("writing file", vec![call]);
        let json = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, Role::Model);
        assert_eq!(back.content, "writing file");
        assert!(back.tool_call_id.is_none());
        let calls = back.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_99");
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(calls[0].arguments["path"], "test.txt");
    }

    #[test]
    fn test_chat_message_skip_serializing_none() {
        let msg = ChatMessage::user("hello");
        let json = serde_json::to_string(&msg).unwrap();
        // tool_call_id and tool_calls are None — should NOT appear in JSON
        assert!(
            !json.contains("tool_call_id"),
            "tool_call_id should be absent when None: {json}"
        );
        assert!(
            !json.contains("tool_calls"),
            "tool_calls should be absent when None: {json}"
        );
        // But role and content must be present
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"hello\""));
    }

    #[test]
    fn test_tool_call_parsed_serde_roundtrip() {
        let call = ToolCallParsed {
            id: "call_abc".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({
                "command": "echo hello",
                "timeout": 5000
            }),
        };
        let json = serde_json::to_string(&call).unwrap();
        let back: ToolCallParsed = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "call_abc");
        assert_eq!(back.name, "bash");
        assert_eq!(back.arguments["command"], "echo hello");
        assert_eq!(back.arguments["timeout"], 5000);
    }

    #[test]
    fn test_finish_reason_equality() {
        assert_eq!(FinishReason::Stop, FinishReason::Stop);
        assert_eq!(FinishReason::MaxTokens, FinishReason::MaxTokens);
        assert_eq!(FinishReason::ToolUse, FinishReason::ToolUse);
        assert_ne!(FinishReason::Stop, FinishReason::MaxTokens);
        assert_ne!(FinishReason::MaxTokens, FinishReason::ToolUse);
        assert_ne!(FinishReason::ToolUse, FinishReason::Stop);
    }

    #[test]
    fn test_inference_error_display() {
        assert_eq!(
            InferenceError::ModelNotFound("model.gguf".to_string()).to_string(),
            "Model file not found: model.gguf"
        );
        assert_eq!(
            InferenceError::OutOfMemory.to_string(),
            "CUDA out of memory"
        );
        assert!(InferenceError::TokenizerError("bad token".to_string())
            .to_string()
            .contains("bad token"));
        assert!(InferenceError::GenerationError("overflow".to_string())
            .to_string()
            .contains("overflow"));
    }

    #[test]
    fn test_generation_config_custom_values() {
        let config = GenerationConfig {
            temperature: 0.3,
            top_p: 0.5,
            top_k: 10,
            max_tokens: 4096,
            repeat_penalty: 1.2,
            repeat_last_n: 32,
            enable_thinking: false,
        };
        assert!((config.temperature - 0.3).abs() < f64::EPSILON);
        assert!((config.top_p - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.top_k, 10);
        assert_eq!(config.max_tokens, 4096);
        assert!((config.repeat_penalty - 1.2).abs() < f32::EPSILON);
        assert_eq!(config.repeat_last_n, 32);
        assert!(!config.enable_thinking);
    }

    #[test]
    fn test_generation_config_enable_thinking_defaults_to_true() {
        let config = GenerationConfig::default();
        assert!(
            config.enable_thinking,
            "enable_thinking should default to true"
        );
    }

    #[test]
    fn test_chat_message_system_constructor() {
        let msg = ChatMessage::system("you are helpful");
        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.content, "you are helpful");
        assert!(msg.tool_call_id.is_none());
        assert!(msg.tool_calls.is_none());
    }

    #[test]
    fn test_chat_message_assistant_constructor() {
        let msg = ChatMessage::assistant("here is the answer");
        assert_eq!(msg.role, Role::Model);
        assert_eq!(msg.content, "here is the answer");
        assert!(msg.tool_call_id.is_none());
        assert!(msg.tool_calls.is_none());
    }

    #[test]
    fn test_chat_message_tool_result_roundtrip_preserves_fields() {
        let msg = ChatMessage::tool_result("call_abc", "42 files found");
        let json = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, Role::Tool);
        assert_eq!(back.content, "42 files found");
        assert_eq!(back.tool_call_id.as_deref(), Some("call_abc"));
        assert!(back.tool_calls.is_none());
    }

    #[test]
    fn test_token_event_thinking_clone() {
        let event = TokenEvent::Thinking("reasoning step".to_string());
        let cloned = event;
        match cloned {
            TokenEvent::Thinking(text) => assert_eq!(text, "reasoning step"),
            _ => panic!("expected Thinking variant"),
        }
    }
}
