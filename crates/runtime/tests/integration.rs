use tempfile::TempDir;
use zipcode_inference::{MockInferenceProvider, MockResponse, ToolSpec};
use zipcode_runtime::{ConversationLoop, PermissionPolicy, Session, StreamCallback};
use zipcode_tools::PermissionMode;

// ---------------------------------------------------------------------------
// Test callback
// ---------------------------------------------------------------------------

struct TestCallback {
    tokens: Vec<String>,
    tool_calls: Vec<(String, serde_json::Value)>,
    tool_results: Vec<(String, String)>,
    errors: Vec<String>,
}

impl TestCallback {
    fn new() -> Self {
        Self {
            tokens: Vec::new(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            errors: Vec::new(),
        }
    }

    fn all_tokens(&self) -> String {
        self.tokens.concat()
    }
}

impl StreamCallback for TestCallback {
    fn on_token(&mut self, text: &str) {
        self.tokens.push(text.to_string());
    }

    fn on_tool_start(&mut self, name: &str, args: &serde_json::Value) {
        self.tool_calls.push((name.to_string(), args.clone()));
    }

    fn on_tool_result(&mut self, name: &str, result: &str) {
        self.tool_results
            .push((name.to_string(), result.to_string()));
    }

    /// Deny all permission prompts by default.
    fn on_permission_prompt(&mut self, _message: &str) -> bool {
        false
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
        engine: Box::new(mock),
        tools: registry,
        session: Session::new(),
        permission: PermissionPolicy::new(permission),
        system_prompt: "You are a test assistant.".to_string(),
        tool_specs,
        cwd: dir.path().to_path_buf(),
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
