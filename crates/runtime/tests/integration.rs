use std::sync::mpsc;

use tempfile::TempDir;
use zipcode_inference::{
    ChatMessage, FinishReason, InferenceError, InferenceProvider, MockInferenceProvider,
    MockResponse, Role, TokenEvent, ToolCallParsed, ToolSpec,
};
use zipcode_runtime::{
    CompactPolicy, ConversationLoop, PermissionPolicy, Session, StreamCallback,
    COMPACTED_SUMMARY_MARKER,
};
use zipcode_tools::PermissionMode;

// ---------------------------------------------------------------------------
// Test callback
// ---------------------------------------------------------------------------

struct TestCallback {
    tokens: Vec<String>,
    thinking: Vec<String>,
    tool_calls: Vec<(String, serde_json::Value)>,
    tool_results: Vec<(String, String)>,
    permission_prompts: Vec<String>,
    approve_prompts: bool,
    errors: Vec<String>,
}

impl TestCallback {
    fn new() -> Self {
        Self {
            tokens: Vec::new(),
            thinking: Vec::new(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            permission_prompts: Vec::new(),
            approve_prompts: false,
            errors: Vec::new(),
        }
    }

    fn with_permission_response(approve_prompts: bool) -> Self {
        Self {
            approve_prompts,
            ..Self::new()
        }
    }

    fn all_tokens(&self) -> String {
        self.tokens.concat()
    }

    fn all_thinking(&self) -> String {
        self.thinking.concat()
    }
}

impl StreamCallback for TestCallback {
    fn on_token(&mut self, text: &str) {
        self.tokens.push(text.to_string());
    }

    fn on_thinking(&mut self, text: &str) {
        self.thinking.push(text.to_string());
    }

    fn on_tool_start(&mut self, name: &str, args: &serde_json::Value) {
        self.tool_calls.push((name.to_string(), args.clone()));
    }

    fn on_tool_result(&mut self, name: &str, result: &str) {
        self.tool_results
            .push((name.to_string(), result.to_string()));
    }

    /// Deny all permission prompts by default.
    fn on_permission_prompt(&mut self, message: &str) -> bool {
        self.permission_prompts.push(message.to_string());
        self.approve_prompts
    }

    fn on_error(&mut self, error: &str) {
        self.errors.push(error.to_string());
    }
}

// ---------------------------------------------------------------------------
// Helper: build a ConversationLoop with MockInferenceProvider
// ---------------------------------------------------------------------------

fn build_test_loop(
    dir: &TempDir,
    mock: MockInferenceProvider,
    permission: PermissionMode,
) -> ConversationLoop {
    build_test_loop_with_engine(dir, mock, permission)
}

fn build_test_loop_with_engine(
    dir: &TempDir,
    engine: impl InferenceProvider + 'static,
    permission: PermissionMode,
) -> ConversationLoop {
    use zipcode_tools::{
        bash::BashTool, edit_file::EditFileTool, glob_search::GlobSearchTool,
        grep_search::GrepSearchTool, read_file::ReadFileTool, write_file::WriteFileTool,
        ToolRegistry,
    };

    let mut registry = ToolRegistry::new();
    registry.register(Box::new(BashTool));
    registry.register(Box::new(ReadFileTool));
    registry.register(Box::new(WriteFileTool));
    registry.register(Box::new(EditFileTool));
    registry.register(Box::new(GlobSearchTool));
    registry.register(Box::new(GrepSearchTool));

    let tool_specs: Vec<ToolSpec> = registry
        .specs()
        .into_iter()
        .map(|s| ToolSpec {
            name: s.name,
            description: s.description,
            parameters: s.parameters,
        })
        .collect();

    ConversationLoop {
        engine: Box::new(engine),
        tools: registry,
        session: Session::new(),
        permission: PermissionPolicy::new(permission),
        system_prompt: "You are a test assistant.".to_string(),
        tool_specs,
        cwd: dir.path().to_path_buf(),
        last_sent_idx: 0,
    }
}

struct ToolCallMarkupProvider {
    content: String,
    call: ToolCallParsed,
    emitted_tool_call: bool,
}

impl InferenceProvider for ToolCallMarkupProvider {
    fn generate_stream(
        &mut self,
        _messages: &[ChatMessage],
        _tools: &[ToolSpec],
    ) -> mpsc::Receiver<TokenEvent> {
        let (tx, rx) = mpsc::channel();
        if self.emitted_tool_call {
            let _ = tx.send(TokenEvent::Token("done".to_string()));
            let _ = tx.send(TokenEvent::Done(FinishReason::Stop));
        } else {
            self.emitted_tool_call = true;
            let _ = tx.send(TokenEvent::Token(self.content.clone()));
            let _ = tx.send(TokenEvent::ToolCall(self.call.clone()));
            let _ = tx.send(TokenEvent::Done(FinishReason::ToolUse));
        }
        rx
    }
}

// ---------------------------------------------------------------------------
// Test 1: text-only response
// ---------------------------------------------------------------------------

#[test]
fn text_only_response() {
    let dir = TempDir::new().unwrap();
    let mock = MockInferenceProvider::new(vec![MockResponse::Text("Hello world!".to_string())]);
    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    let mut cb = TestCallback::new();

    conv.run_turn("hi", &mut cb).unwrap();

    assert!(
        cb.all_tokens().contains("Hello world!"),
        "expected token 'Hello world!' in {:?}",
        cb.tokens
    );
    assert!(cb.errors.is_empty(), "unexpected errors: {:?}", cb.errors);

    // system + user + assistant = 3 messages
    assert_eq!(
        conv.session.messages.len(),
        3,
        "expected 3 messages (system + user + assistant)"
    );
}

// ---------------------------------------------------------------------------
// Test 2: single tool call (read_file)
// ---------------------------------------------------------------------------

#[test]
fn single_tool_call() {
    let dir = TempDir::new().unwrap();
    let test_file = dir.path().join("test.txt");
    std::fs::write(&test_file, "hello").unwrap();
    let abs_path = test_file.to_str().unwrap().to_string();

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "read_file".to_string(),
            args: serde_json::json!({ "path": abs_path }),
        },
        MockResponse::Text("I read the file.".to_string()),
    ]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    let mut cb = TestCallback::new();

    conv.run_turn("read the file", &mut cb).unwrap();

    assert_eq!(cb.tool_calls.len(), 1, "expected 1 tool call");
    assert_eq!(cb.tool_calls[0].0, "read_file");

    assert_eq!(cb.tool_results.len(), 1, "expected 1 tool result");
    assert!(
        cb.tool_results[0].1.contains("hello"),
        "expected result to contain 'hello', got: {}",
        cb.tool_results[0].1
    );
}

// ---------------------------------------------------------------------------
// Test 3: multiple tool calls across turns
// ---------------------------------------------------------------------------

#[test]
fn multi_tool_turn() {
    let dir = TempDir::new().unwrap();

    // Three mock responses: two tool calls then final text.
    // Each generate_stream call returns one response from the queue.
    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({ "command": "echo test1" }),
        },
        MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({ "command": "echo test2" }),
        },
        MockResponse::Text("Both done.".to_string()),
    ]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    let mut cb = TestCallback::new();

    conv.run_turn("run both", &mut cb).unwrap();

    assert_eq!(cb.tool_calls.len(), 2, "expected 2 tool calls");
    assert_eq!(cb.tool_results.len(), 2, "expected 2 tool results");

    let combined = cb
        .tool_results
        .iter()
        .map(|(_, r)| r.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        combined.contains("test1"),
        "expected 'test1' in tool results: {combined}"
    );
    assert!(
        combined.contains("test2"),
        "expected 'test2' in tool results: {combined}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: permission denied (ReadOnly blocks write_file)
// ---------------------------------------------------------------------------

#[test]
fn permission_denied() {
    let dir = TempDir::new().unwrap();

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "write_file".to_string(),
            args: serde_json::json!({ "path": "test.txt", "content": "bad" }),
        },
        MockResponse::Text("OK".to_string()),
    ]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::ReadOnly);
    let mut cb = TestCallback::new();

    conv.run_turn("write something", &mut cb).unwrap();

    // The callback should NOT have received a tool result (denial goes to session only)
    assert!(
        cb.tool_results.is_empty(),
        "expected no callback tool results for denied tool, got: {:?}",
        cb.tool_results
    );

    // The denial message must appear in session as a tool_result message
    let has_denial = conv.session.messages.iter().any(|m| {
        let json = serde_json::to_string(m).unwrap_or_default();
        json.contains("not allowed in read-only mode") || json.contains("denied")
    });
    assert!(
        has_denial,
        "expected a denial tool_result in session messages"
    );
}

#[test]
fn workspace_write_permission_prompt_executes_bash_when_approved() {
    let dir = TempDir::new().unwrap();

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({ "command": "echo approved" }),
        },
        MockResponse::Text("done".to_string()),
    ]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::WorkspaceWrite);
    let mut cb = TestCallback::with_permission_response(true);

    conv.run_turn("run bash", &mut cb).unwrap();

    assert_eq!(
        cb.permission_prompts.len(),
        1,
        "expected one permission prompt"
    );
    assert!(cb.permission_prompts[0].contains("requires approval"));
    assert_eq!(cb.tool_calls.len(), 1, "expected approved tool execution");
    assert_eq!(cb.tool_calls[0].0, "bash");
    assert_eq!(
        cb.tool_results.len(),
        1,
        "expected tool result after approval"
    );
    assert!(cb.tool_results[0].1.contains("approved"));
}

#[test]
fn workspace_write_permission_prompt_records_denial_when_rejected() {
    let dir = TempDir::new().unwrap();

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({ "command": "echo should-not-run" }),
        },
        MockResponse::Text("done".to_string()),
    ]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::WorkspaceWrite);
    let mut cb = TestCallback::with_permission_response(false);

    conv.run_turn("run bash", &mut cb).unwrap();

    assert_eq!(
        cb.permission_prompts.len(),
        1,
        "expected one permission prompt"
    );
    assert!(
        cb.tool_calls.is_empty(),
        "tool should not execute after rejection"
    );
    assert!(
        cb.tool_results.is_empty(),
        "denial should not surface as callback tool result"
    );

    let has_denial = conv.session.messages.iter().any(|m| {
        let json = serde_json::to_string(m).unwrap_or_default();
        json.contains("User denied permission for this action.")
    });
    assert!(
        has_denial,
        "expected denied permission message in session history"
    );
}

// ---------------------------------------------------------------------------
// Test 5: tool call loop cap (MAX_TOOL_ITERATIONS = 25)
// ---------------------------------------------------------------------------

#[test]
fn tool_call_loop_cap() {
    let dir = TempDir::new().unwrap();

    // 30 bash tool calls followed by a final text response
    let mut responses: Vec<MockResponse> = (0..30)
        .map(|i| MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({ "command": format!("echo iter{i}") }),
        })
        .collect();
    responses.push(MockResponse::Text("done".to_string()));

    let mock = MockInferenceProvider::new(responses);
    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    let mut cb = TestCallback::new();

    let error = conv
        .run_turn("loop forever", &mut cb)
        .unwrap_err()
        .to_string();

    assert!(
        cb.tool_calls.len() <= 25,
        "expected at most 25 tool calls due to MAX_TOOL_ITERATIONS, got {}",
        cb.tool_calls.len()
    );
    assert!(error.contains("Stopped after 25 tool iterations"));
}

// ---------------------------------------------------------------------------
// Test 6: path traversal blocked
// ---------------------------------------------------------------------------

#[test]
fn path_traversal_blocked() {
    let dir = TempDir::new().unwrap();

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "read_file".to_string(),
            args: serde_json::json!({ "path": "../../etc/passwd" }),
        },
        MockResponse::Text("Error handled".to_string()),
    ]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    let mut cb = TestCallback::new();

    conv.run_turn("read passwd", &mut cb).unwrap();

    // The tool error should appear either in the callback result or in session messages
    let in_callback = cb
        .tool_results
        .iter()
        .any(|(_, r)| r.contains("outside the workspace") || r.contains("Tool error"));

    let in_session = conv.session.messages.iter().any(|m| {
        let json = serde_json::to_string(m).unwrap_or_default();
        json.contains("outside the workspace")
    });

    assert!(
        in_callback || in_session,
        "expected 'outside the workspace' error in callback results or session messages.\ncallback results: {:?}",
        cb.tool_results
    );
}

// ---------------------------------------------------------------------------
// Test 7: failed inference turns are persisted before returning the error
// ---------------------------------------------------------------------------

#[test]
fn failed_turn_is_saved_to_session_file() {
    let dir = TempDir::new().unwrap();
    let mock = MockInferenceProvider::new(vec![MockResponse::Error(
        InferenceError::GenerationError("backend exploded".to_string()),
    )]);
    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    let session_path = conv.session.path();
    let session_id = conv.session.id.clone();
    let mut cb = TestCallback::new();

    let error = conv
        .run_turn("please fail", &mut cb)
        .unwrap_err()
        .to_string();

    assert!(error.contains("backend exploded"));
    assert!(
        session_path.is_file(),
        "expected failed turn to save the session to {}",
        session_path.display()
    );

    let saved = Session::load(&session_id).unwrap();
    assert!(
        saved
            .messages
            .iter()
            .any(|message| message.content == "please fail"),
        "expected failed user turn to be persisted, got {:?}",
        saved.messages
    );

    std::fs::remove_file(session_path).ok();
}

// ---------------------------------------------------------------------------
// Test 8: tool-call turns strip raw markup before persisting assistant history
// ---------------------------------------------------------------------------

#[test]
fn tool_call_turn_strips_raw_markup_from_saved_assistant_content() {
    let dir = TempDir::new().unwrap();
    let provider = ToolCallMarkupProvider {
        content:
            "Before <tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"test.txt\"}}</tool_call> after"
                .to_string(),
        call: ToolCallParsed {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({ "path": "test.txt" }),
        },
        emitted_tool_call: false,
    };
    std::fs::write(dir.path().join("test.txt"), "hello").unwrap();

    let mut conv = build_test_loop_with_engine(&dir, provider, PermissionMode::FullAccess);
    let mut cb = TestCallback::new();

    conv.run_turn("use a tool", &mut cb).unwrap();

    let assistant = conv
        .session
        .messages
        .iter()
        .find(|message| message.role == Role::Model && message.tool_calls.is_some())
        .expect("expected assistant tool-call message");

    assert_eq!(assistant.content, "Before  after");
    assert!(
        !assistant.content.contains("<tool_call>"),
        "assistant history should not persist raw tool markup: {:?}",
        assistant
    );
    assert!(
        cb.all_tokens().contains("<tool_call>"),
        "sanity check: provider should have streamed raw markup into the callback"
    );
}

// ---------------------------------------------------------------------------
// Test 8b: Gemma 4 thinking channel is surfaced to the callback but NOT
// persisted into assistant session history.
// ---------------------------------------------------------------------------

#[test]
fn thinking_channel_routes_to_callback_but_not_to_history() {
    let dir = TempDir::new().unwrap();

    // Mock a turn that emits reasoning deltas, a visible token, and a done
    // event. The conversation loop must route thinking through `on_thinking`
    // (accumulated in `cb.thinking`) while persisting only the visible token
    // in the assistant message — Gemma 4 strips reasoning on re-injection,
    // so letting it leak into history would desync the session.
    let mock = MockInferenceProvider::new(vec![MockResponse::Events(vec![
        TokenEvent::Thinking("I should greet the user. ".to_string()),
        TokenEvent::Thinking("They just said hi.".to_string()),
        TokenEvent::Token("Hello world!".to_string()),
        TokenEvent::Done(FinishReason::Stop),
    ])]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    let mut cb = TestCallback::new();

    conv.run_turn("hi", &mut cb).unwrap();

    assert_eq!(
        cb.all_thinking(),
        "I should greet the user. They just said hi.",
        "thinking deltas must reach the callback",
    );
    assert_eq!(
        cb.all_tokens(),
        "Hello world!",
        "visible tokens must reach the callback unmixed with reasoning",
    );

    // system + user + assistant
    assert_eq!(conv.session.messages.len(), 3);
    let assistant = conv
        .session
        .messages
        .iter()
        .find(|m| m.role == Role::Model)
        .expect("expected assistant turn");

    assert_eq!(
        assistant.content, "Hello world!",
        "assistant history must contain only visible content, not reasoning"
    );
    assert!(
        !assistant.content.contains("greet"),
        "reasoning text must not leak into persisted assistant content: {:?}",
        assistant.content
    );
}

// ---------------------------------------------------------------------------
// Test 9: resumed sessions do not duplicate the system prompt
// ---------------------------------------------------------------------------

#[test]
fn resumed_session_does_not_duplicate_system_prompt() {
    let dir = TempDir::new().unwrap();
    let mock = MockInferenceProvider::new(vec![MockResponse::Text("resumed".to_string())]);
    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    conv.session.messages = vec![
        ChatMessage::system("saved system prompt"),
        ChatMessage::user("earlier"),
        ChatMessage::assistant("earlier response"),
    ];
    let mut cb = TestCallback::new();

    conv.run_turn("continue", &mut cb).unwrap();

    let system_count = conv
        .session
        .messages
        .iter()
        .filter(|message| message.role == Role::System)
        .count();
    assert_eq!(system_count, 1, "system prompt should not be duplicated");
    assert_eq!(conv.session.messages[0].content, "saved system prompt");
    assert!(
        cb.all_tokens().contains("resumed"),
        "expected resumed response, got {:?}",
        cb.tokens
    );
}

// ---------------------------------------------------------------------------
// Test 10: compacted sessions roundtrip and continue correctly
// ---------------------------------------------------------------------------

#[test]
fn compacted_session_roundtrip_can_continue_turns() {
    let dir = TempDir::new().unwrap();
    let tool_call = ToolCallParsed {
        id: "call_1".to_string(),
        name: "read_file".to_string(),
        arguments: serde_json::json!({"file_path":"src/main.rs"}),
    };
    let mut session = Session::new();
    let session_path = session.path();
    session.messages = vec![
        ChatMessage::system("saved system prompt"),
        ChatMessage::user("first"),
        ChatMessage::assistant("first response"),
        ChatMessage::user("second"),
        ChatMessage::assistant_with_tool_calls("checking", vec![tool_call]),
        ChatMessage::tool_result("call_1", "fn main() {}"),
        ChatMessage::assistant("second response"),
        ChatMessage::user("third"),
        ChatMessage::assistant("third response"),
    ];

    let compact_result = session.compact(CompactPolicy {
        retain_user_turns: 2,
        ..CompactPolicy::default()
    });
    assert!(compact_result.changed);
    session.save().unwrap();

    let loaded = Session::load(&session.id).unwrap();
    assert!(loaded
        .messages
        .iter()
        .any(|message| message.content.starts_with(COMPACTED_SUMMARY_MARKER)));
    assert_eq!(
        loaded
            .messages
            .iter()
            .filter(|message| message.role == Role::Tool)
            .count(),
        1
    );

    let mock = MockInferenceProvider::new(vec![MockResponse::Text("continued".to_string())]);
    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    conv.session = loaded;
    let mut cb = TestCallback::new();
    conv.run_turn("after compact", &mut cb).unwrap();

    let system_count = conv
        .session
        .messages
        .iter()
        .filter(|message| message.role == Role::System)
        .count();
    assert_eq!(system_count, 1);
    assert!(conv
        .session
        .messages
        .iter()
        .any(|message| message.content.starts_with(COMPACTED_SUMMARY_MARKER)));
    assert!(
        cb.all_tokens().contains("continued"),
        "expected continued response after compaction, got {:?}",
        cb.tokens
    );

    std::fs::remove_file(session_path).ok();
}

// ---------------------------------------------------------------------------
// Test 11: auto-retry nudges the model when it gives an empty turn after
// a tool result that contains errors
// ---------------------------------------------------------------------------

#[test]
fn auto_retry_nudges_model_after_empty_turn_on_error() {
    let dir = TempDir::new().unwrap();

    // Sequence:
    //  1. Model calls bash(echo "error: compilation failed")
    //  2. Tool result contains "error" keyword
    //  3. Model gives up → empty turn (Done with no tokens/tool_calls)
    //  4. Auto-retry injects "Let me re-read the error…" nudge
    //  5. Model tries again → produces "Fixed!" text
    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({ "command": "echo 'error: compilation failed'" }),
        },
        // Model gives up: empty turn
        MockResponse::Events(vec![TokenEvent::Done(FinishReason::Stop)]),
        // After auto-retry nudge, model recovers
        MockResponse::Text("Fixed the issue.".to_string()),
    ]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    let mut cb = TestCallback::new();

    conv.run_turn("build and test", &mut cb).unwrap();

    // The nudge message should appear in session history
    let has_nudge = conv
        .session
        .messages
        .iter()
        .any(|m| m.role == Role::Model && m.content.contains("re-read the error"));
    assert!(
        has_nudge,
        "expected auto-retry nudge in session history, got: {:?}",
        conv.session
            .messages
            .iter()
            .map(|m| format!(
                "{}:{}",
                m.role == Role::Model,
                &m.content[..m.content.len().min(60)]
            ))
            .collect::<Vec<_>>()
    );

    // The final answer should be present
    assert!(
        cb.all_tokens().contains("Fixed"),
        "expected model to produce 'Fixed' after auto-retry, got: {:?}",
        cb.tokens
    );
}

// ---------------------------------------------------------------------------
// Test 12: auto-retry does NOT fire when the tool result has no errors
// ---------------------------------------------------------------------------

#[test]
fn auto_retry_skips_when_tool_result_has_no_errors() {
    let dir = TempDir::new().unwrap();

    // Tool result is clean (no error keywords) → empty model turn should
    // NOT trigger auto-retry; the turn should just end silently.
    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({ "command": "echo 'all good'" }),
        },
        // Model gives an empty turn — but tool result was clean
        MockResponse::Events(vec![TokenEvent::Done(FinishReason::Stop)]),
    ]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    let mut cb = TestCallback::new();

    conv.run_turn("run something", &mut cb).unwrap();

    // No nudge should appear
    let has_nudge = conv
        .session
        .messages
        .iter()
        .any(|m| m.role == Role::Model && m.content.contains("re-read the error"));
    assert!(
        !has_nudge,
        "auto-retry must NOT fire when tool result has no error indicators"
    );
}

// ---------------------------------------------------------------------------
// Test 13: session save failure does not mask inference error (issue #63-E)
//
// When both inference fails AND session.save() fails, the returned error
// must describe the *inference* failure — not the session save failure.
// The session save error is logged but must not become the primary error.
// ---------------------------------------------------------------------------

#[test]
fn inference_error_is_not_masked_by_session_save_failure() {
    let dir = TempDir::new().unwrap();
    let mock = MockInferenceProvider::new(vec![MockResponse::Error(
        InferenceError::GenerationError("CUDA out of memory".to_string()),
    )]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);

    // Corrupt the session id so save() will fail — but only after messages
    // are pushed (push_message doesn't validate, only save does).
    // We set an invalid id that contains a path-traversal character.
    conv.session.id = "../bad/id".to_string();

    let mut cb = TestCallback::new();

    let error = conv
        .run_turn("please fail", &mut cb)
        .unwrap_err()
        .to_string();

    // The primary error must be the inference error, NOT a session save error.
    assert!(
        error.contains("CUDA out of memory"),
        "expected inference error in message, got: {error}"
    );
    assert!(
        !error.contains("save session") && !error.contains("Failed to save"),
        "session save failure must NOT mask the inference error, got: {error}"
    );
}

// ---------------------------------------------------------------------------
// Test 14: FinishReason::MaxTokens regression test (see issue #40)
//
// ConversationLoop detects MaxTokens and appends a truncation notice to the
// assistant text so the user is informed that output was cut short.
// ---------------------------------------------------------------------------

#[test]
fn max_tokens_finish_appends_truncation_notice() {
    let dir = TempDir::new().unwrap();

    let mock = MockInferenceProvider::new(vec![MockResponse::Events(vec![
        TokenEvent::Token("partial answer".to_string()),
        TokenEvent::Done(FinishReason::MaxTokens),
    ])]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    let mut cb = TestCallback::new();

    // run_turn should still succeed (not an error), but annotate the text
    conv.run_turn("tell me something long", &mut cb).unwrap();

    // The partial tokens must reach the callback
    assert!(
        cb.all_tokens().contains("partial answer"),
        "expected partial tokens in callback, got: {:?}",
        cb.tokens
    );

    // The persisted assistant message must contain the truncation notice
    let assistant = conv
        .session
        .messages
        .iter()
        .find(|m| m.role == Role::Model)
        .expect("expected an assistant message in session");

    assert!(
        assistant.content.contains("partial answer"),
        "expected partial text in assistant message, got: {:?}",
        assistant.content
    );
    assert!(
        assistant
            .content
            .contains("[...output truncated due to token limit]"),
        "expected truncation notice in assistant message, got: {:?}",
        assistant.content
    );
    assert!(
        assistant.tool_calls.is_none(),
        "MaxTokens turn should not produce tool calls"
    );
}

// ---------------------------------------------------------------------------
// Test 15: mixed permissions across tool calls in a single turn
//
// In WorkspaceWrite mode, the model calls read_file (always allowed),
// then bash (needs approval — approved by callback), then read_file again
// (always allowed). All three should execute and session should have all
// results.
// ---------------------------------------------------------------------------

#[test]
fn mixed_permission_partial_flow() {
    let dir = TempDir::new().unwrap();
    let test_file = dir.path().join("data.txt");
    std::fs::write(&test_file, "hello from file").unwrap();
    let abs_path = test_file.to_str().unwrap().to_string();

    // First call: read_file; Second call: bash; Third call: read_file; Fourth: text
    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "read_file".to_string(),
            args: serde_json::json!({ "path": abs_path.clone() }),
        },
        MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({ "command": "echo approved-bash" }),
        },
        MockResponse::ToolCall {
            name: "read_file".to_string(),
            args: serde_json::json!({ "path": abs_path }),
        },
        MockResponse::Text("All done.".to_string()),
    ]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::WorkspaceWrite);
    let mut cb = TestCallback::with_permission_response(true);

    conv.run_turn("read and run", &mut cb).unwrap();

    // bash needs approval, read_file doesn't
    assert_eq!(
        cb.permission_prompts.len(),
        1,
        "expected exactly 1 permission prompt (for bash), got {:?}",
        cb.permission_prompts
    );

    // All 3 tool calls should have been dispatched
    assert_eq!(
        cb.tool_calls.len(),
        3,
        "expected 3 tool calls, got {}",
        cb.tool_calls.len()
    );
    assert_eq!(cb.tool_calls[0].0, "read_file");
    assert_eq!(cb.tool_calls[1].0, "bash");
    assert_eq!(cb.tool_calls[2].0, "read_file");

    // Final text response
    assert!(cb.all_tokens().contains("All done."));

    // Session should have: system + user + 3*(assistant+tool_result) + final assistant = 9
    assert_eq!(
        conv.session.messages.len(),
        9,
        "expected 9 messages in session, got {}: {:?}",
        conv.session.messages.len(),
        conv.session
            .messages
            .iter()
            .map(|m| format!("{:?}", m.role))
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Test 16: denied tool still saves session and model continues
//
// Model calls bash (needs approval in WorkspaceWrite), user denies it,
// then model responds with text. Verify session is saved correctly with
// the denial and final text.
// ---------------------------------------------------------------------------

#[test]
fn denied_tool_still_saves_session_and_continues() {
    let dir = TempDir::new().unwrap();

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({ "command": "rm -rf /" }),
        },
        MockResponse::Text("I understand, I won't delete anything.".to_string()),
    ]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::WorkspaceWrite);
    let mut cb = TestCallback::with_permission_response(false); // deny

    conv.run_turn("delete everything", &mut cb).unwrap();

    // Permission prompt was issued and denied
    assert_eq!(cb.permission_prompts.len(), 1);
    assert!(cb.tool_calls.is_empty(), "denied tool should not execute");

    // Session should have denial message
    let has_denial = conv
        .session
        .messages
        .iter()
        .any(|m| m.role == Role::Tool && m.content.contains("User denied permission"));
    assert!(
        has_denial,
        "expected denial message in session, got: {:?}",
        conv.session.messages
    );

    // Final text response should be in session
    let has_final_text = conv
        .session
        .messages
        .iter()
        .any(|m| m.role == Role::Model && m.content.contains("I won't delete"));
    assert!(
        has_final_text,
        "expected final model text in session, got: {:?}",
        conv.session
            .messages
            .iter()
            .map(|m| format!("{:?}: {}", m.role, &m.content[..m.content.len().min(50)]))
            .collect::<Vec<_>>()
    );

    // Session file should exist on disk
    assert!(
        conv.session.path().is_file(),
        "expected session file to be saved"
    );
    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 17: tool error does not break session save
//
// Model calls bash with a command that will fail (non-zero exit), then
// model responds with text. Verify session is saved with the tool error
// in history and the final text response.
// ---------------------------------------------------------------------------

#[test]
fn tool_error_does_not_break_session_save() {
    let dir = TempDir::new().unwrap();

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({ "command": "exit 42" }),
        },
        MockResponse::Text("The command failed, but I'll handle it.".to_string()),
    ]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    let mut cb = TestCallback::new();

    conv.run_turn("run failing command", &mut cb).unwrap();

    // Tool should have been executed (even though it failed)
    assert_eq!(cb.tool_calls.len(), 1);
    assert_eq!(cb.tool_calls[0].0, "bash");

    // Tool result should contain error indicator
    assert_eq!(cb.tool_results.len(), 1);
    assert!(
        cb.tool_results[0].1.contains("42"),
        "expected exit code 42 in result, got: {}",
        cb.tool_results[0].1
    );

    // Session should have the tool error in history
    let has_tool_error = conv
        .session
        .messages
        .iter()
        .any(|m| m.role == Role::Tool && m.content.contains("42"));
    assert!(has_tool_error, "expected tool error in session history");

    // Final text should be in session
    let has_final = conv
        .session
        .messages
        .iter()
        .any(|m| m.role == Role::Model && m.content.contains("handle it"));
    assert!(has_final, "expected final model text in session");

    // Session file should exist
    assert!(
        conv.session.path().is_file(),
        "session file must be saved even after tool error"
    );
    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 18: empty text response saves correctly
//
// Model generates only a Done(Stop) event with no Token events, producing
// an empty full_text. Verify session saves the empty assistant message
// without crashing.
// ---------------------------------------------------------------------------

#[test]
fn empty_text_response_saves_correctly() {
    let dir = TempDir::new().unwrap();

    let mock = MockInferenceProvider::new(vec![MockResponse::Events(vec![TokenEvent::Done(
        FinishReason::Stop,
    )])]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    let mut cb = TestCallback::new();

    conv.run_turn("say nothing", &mut cb).unwrap();

    // No tokens received by callback
    assert!(
        cb.all_tokens().is_empty(),
        "expected no tokens, got: {:?}",
        cb.tokens
    );

    // Session should have empty assistant message
    let assistant_msgs: Vec<_> = conv
        .session
        .messages
        .iter()
        .filter(|m| m.role == Role::Model)
        .collect();
    assert_eq!(
        assistant_msgs.len(),
        1,
        "expected exactly 1 assistant message"
    );
    assert!(
        assistant_msgs[0].content.is_empty(),
        "expected empty assistant content, got: {:?}",
        assistant_msgs[0].content
    );
    assert!(
        assistant_msgs[0].tool_calls.is_none(),
        "empty response should have no tool calls"
    );

    // Total: system + user + assistant = 3
    assert_eq!(conv.session.messages.len(), 3);

    // Session file should exist
    assert!(
        conv.session.path().is_file(),
        "session file must be saved for empty response"
    );
    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 19: consecutive turns accumulate messages correctly
//
// Two consecutive run_turn calls on the same ConversationLoop.
// First turn: text response. Second turn: tool call then text.
// Verify system prompt is only added once and messages accumulate.
// ---------------------------------------------------------------------------

#[test]
fn consecutive_turns_accumulate_messages() {
    let dir = TempDir::new().unwrap();
    let test_file = dir.path().join("info.txt");
    std::fs::write(&test_file, "important data").unwrap();
    let abs_path = test_file.to_str().unwrap().to_string();

    let mock = MockInferenceProvider::new(vec![
        // Turn 1: plain text
        MockResponse::Text("Hello! How can I help?".to_string()),
        // Turn 2: tool call then text
        MockResponse::ToolCall {
            name: "read_file".to_string(),
            args: serde_json::json!({ "path": abs_path }),
        },
        MockResponse::Text("I read the file.".to_string()),
    ]);

    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);
    let mut cb1 = TestCallback::new();

    // First turn
    conv.run_turn("greet me", &mut cb1).unwrap();
    assert!(cb1.all_tokens().contains("Hello!"));
    assert!(cb1.errors.is_empty());

    // After turn 1: system + user + assistant = 3
    assert_eq!(
        conv.session.messages.len(),
        3,
        "after turn 1, expected 3 messages, got {}",
        conv.session.messages.len()
    );

    // System prompt should appear exactly once
    let system_count = conv
        .session
        .messages
        .iter()
        .filter(|m| m.role == Role::System)
        .count();
    assert_eq!(
        system_count, 1,
        "system prompt should appear exactly once after turn 1"
    );

    // Second turn with a fresh callback
    let mut cb2 = TestCallback::new();
    conv.run_turn("read the file", &mut cb2).unwrap();

    // Tool call should have been executed
    assert_eq!(cb2.tool_calls.len(), 1);
    assert_eq!(cb2.tool_calls[0].0, "read_file");
    assert!(cb2.all_tokens().contains("I read the file."));

    // After turn 2:
    // system + user1 + assistant1 + user2 + assistant_with_tool_calls + tool_result + assistant2 = 7
    assert_eq!(
        conv.session.messages.len(),
        7,
        "after turn 2, expected 7 messages, got {}: {:?}",
        conv.session.messages.len(),
        conv.session
            .messages
            .iter()
            .enumerate()
            .map(|(i, m)| format!("{}:{:?}", i, m.role))
            .collect::<Vec<_>>()
    );

    // System prompt still appears exactly once
    let system_count_after = conv
        .session
        .messages
        .iter()
        .filter(|m| m.role == Role::System)
        .count();
    assert_eq!(
        system_count_after, 1,
        "system prompt must not be duplicated across turns"
    );
    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 20: full history sent when provider does not manage context (default)
//
// Two consecutive run_turn calls with manages_own_context = false (default).
// On the 2nd turn's generate_stream call, the captured messages slice must
// include the system message AND the first user turn — i.e. the full
// accumulated history.
// ---------------------------------------------------------------------------

#[test]
fn conversation_loop_sends_full_history_when_provider_does_not_manage_context() {
    let dir = TempDir::new().unwrap();

    let mock = MockInferenceProvider::new(vec![
        MockResponse::Text("First response.".to_string()),
        MockResponse::Text("Second response.".to_string()),
    ])
    .with_manages_own_context(false); // explicit default

    let captured = mock.captured_messages.clone();

    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);

    conv.run_turn("first user message", &mut TestCallback::new())
        .unwrap();
    conv.run_turn("second user message", &mut TestCallback::new())
        .unwrap();

    let calls = captured.lock().unwrap().clone();
    // Two generate_stream calls (one per turn, no tool iterations)
    assert_eq!(calls.len(), 2, "expected 2 generate_stream calls");

    // On the 2nd call, full history must be present: system + user1 + assistant1 + user2 = 4
    let second_call_msgs = &calls[1];
    assert!(
        second_call_msgs.len() >= 4,
        "expected at least 4 messages on 2nd call (full history), got {}: {:?}",
        second_call_msgs.len(),
        second_call_msgs
            .iter()
            .map(|m| format!("{:?}", m.role))
            .collect::<Vec<_>>()
    );

    // System message must be present
    assert!(
        second_call_msgs.iter().any(|m| m.role == Role::System),
        "2nd call must include system message for non-managing provider"
    );

    // First user message must be present
    assert!(
        second_call_msgs
            .iter()
            .any(|m| m.role == Role::User && m.content.contains("first user message")),
        "2nd call must include first user turn for non-managing provider"
    );

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 21: only new segment sent when provider manages its own context
//
// Two consecutive run_turn calls with manages_own_context = true.
// On the 2nd turn's generate_stream call, the captured slice must contain
// ONLY the second user message — the system prompt and first turn were
// already sent on turn 1 and must not be re-sent.
// ---------------------------------------------------------------------------

#[test]
fn conversation_loop_sends_only_new_segment_when_provider_manages_context() {
    let dir = TempDir::new().unwrap();

    let mock = MockInferenceProvider::new(vec![
        MockResponse::Text("First response.".to_string()),
        MockResponse::Text("Second response.".to_string()),
    ])
    .with_manages_own_context(true);

    let captured = mock.captured_messages.clone();

    let mut conv = build_test_loop(&dir, mock, PermissionMode::FullAccess);

    conv.run_turn("first user message", &mut TestCallback::new())
        .unwrap();
    conv.run_turn("second user message", &mut TestCallback::new())
        .unwrap();

    let calls = captured.lock().unwrap().clone();
    assert_eq!(calls.len(), 2, "expected 2 generate_stream calls");

    // Turn 1: provider gets system + user1 (everything from idx 0)
    let first_call_msgs = &calls[0];
    assert!(
        first_call_msgs.iter().any(|m| m.role == Role::System),
        "1st call must include system message"
    );
    assert!(
        first_call_msgs
            .iter()
            .any(|m| m.role == Role::User && m.content.contains("first user message")),
        "1st call must include first user message"
    );

    // Turn 2: provider gets ONLY the new segment — the assistant reply from
    // turn 1 plus the second user message (everything added since the last
    // generate_stream call). System prompt and user1 must NOT be re-sent.
    let second_call_msgs = &calls[1];
    assert_eq!(
        second_call_msgs.len(),
        2,
        "expected exactly 2 new messages on 2nd call (assistant1 + user2), got {}: {:?}",
        second_call_msgs.len(),
        second_call_msgs
            .iter()
            .map(|m| format!("{:?}: {}", m.role, &m.content))
            .collect::<Vec<_>>()
    );
    // The last message in the slice must be the new user turn
    let last = second_call_msgs.last().unwrap();
    assert_eq!(last.role, Role::User);
    assert!(
        last.content.contains("second user message"),
        "last message on 2nd call must be second user turn, got: {:?}",
        last.content
    );

    // System message must NOT be re-sent
    assert!(
        !second_call_msgs.iter().any(|m| m.role == Role::System),
        "system message must not be re-sent when provider manages context"
    );

    // First user message must NOT be re-sent
    assert!(
        !second_call_msgs
            .iter()
            .any(|m| m.content.contains("first user message")),
        "first user turn must not be re-sent when provider manages context"
    );

    std::fs::remove_file(conv.session.path()).ok();
}
