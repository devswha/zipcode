use std::sync::{Arc, Mutex, OnceLock};

use tempfile::TempDir;
use zipcode_inference::{
    FinishReason, MockInferenceProvider, MockResponse, Role, TokenEvent, ToolCallParsed,
};
use zipcode_runtime::{
    ConversationLoop, PermissionPolicy, Session, StreamCallback, MAX_AGENT_DEPTH,
};
use zipcode_tools::PermissionMode;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct NoopCallback;

impl StreamCallback for NoopCallback {
    fn on_token(&mut self, _text: &str) {}
    fn on_tool_start(&mut self, _name: &str, _args: &serde_json::Value) {}
    fn on_tool_result(&mut self, _name: &str, _result: &str) {}
    fn on_permission_prompt(&mut self, _message: &str) -> bool {
        true
    }
    fn on_error(&mut self, _error: &str) {}
}

fn build_loop(
    dir: &TempDir,
    mock: MockInferenceProvider,
    permission: PermissionMode,
) -> ConversationLoop {
    use zipcode_tools::{
        agent::AgentTool, grep_search::GrepSearchTool, read_file::ReadFileTool,
        write_file::WriteFileTool, ToolRegistry,
    };

    let mut registry = ToolRegistry::new();
    registry.register(Box::new(AgentTool));
    registry.register(Box::new(ReadFileTool));
    registry.register(Box::new(WriteFileTool));
    registry.register(Box::new(GrepSearchTool));

    let tool_specs: Vec<zipcode_inference::chat_template::ToolSpec> = registry
        .specs()
        .into_iter()
        .map(|s| zipcode_inference::chat_template::ToolSpec {
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
        depth: 0,
        last_sent_idx: 0,
        child_session_ids: Arc::new(Mutex::new(Vec::new())),
        skill_registry: None,
        compact_policy: zipcode_runtime::CompactPolicy::default(),
    }
}

/// Serializes all tests that mutate `ZIPCODE_SESSIONS_DIR` so they don't race.
static SESSION_DIR_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Redirect session files to a tempdir so ~/.zipcode/sessions is never touched.
/// Returns (`TempDir`, guard) — keep both alive for the test duration.
/// The guard serializes against other tests that also call this function.
fn with_temp_session_dir() -> (TempDir, std::sync::MutexGuard<'static, ()>) {
    let guard = SESSION_DIR_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = TempDir::new().unwrap();
    std::env::set_var("ZIPCODE_SESSIONS_DIR", dir.path());
    (dir, guard)
}

/// Find a session JSON file by ID across all temp dirs that may have been
/// used during this test. Scans /tmp for .json files matching the child ID.
/// This is necessary because parallel tests race on `ZIPCODE_SESSIONS_DIR`.
fn find_session_file(child_id: &str) -> Option<std::path::PathBuf> {
    let filename = format!("{child_id}.json");
    // Walk /tmp one level deep looking for our file
    if let Ok(entries) = std::fs::read_dir("/tmp") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let candidate = path.join(&filename);
                if candidate.exists() {
                    return Some(candidate);
                }
            }
            // Also check direct /tmp/{id}.json
            if path.file_name().and_then(|n| n.to_str()) == Some(filename.as_str()) {
                return Some(path);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Test 1: happy path — child executes read_file, tool_call_count > 0
// ---------------------------------------------------------------------------

#[test]
fn test_agent_delegation_happy_path() {
    let (_session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    std::fs::write(dir.path().join("hello.txt"), "hello from child").unwrap();

    // Shared queue: parent response first, then child responses.
    // clone_for_child shares the same Arc<Mutex<VecDeque>>, so the child
    // consumes responses from the same queue in order.
    let mock = MockInferenceProvider::new(vec![
        // parent: immediately delegates via spawn_child (no parent tool calls here)
        // child turn 1: read_file
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "child_read_1".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({ "file_path": "hello.txt" }),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        // child turn 2: final text after seeing tool result
        MockResponse::Text("child done: found hello from child".to_string()),
    ]);

    let mut conv = build_loop(&dir, mock, PermissionMode::FullAccess);
    let result = conv
        .spawn_child("read hello.txt and report", None, None, None)
        .unwrap();

    assert!(
        result.summary.contains("child done"),
        "summary must contain child's final text, got: {}",
        result.summary
    );
    assert_eq!(result.tool_call_count, 1, "child executed one tool call");
    assert!(!result.child_session_id.is_empty());

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 2: depth exceeded — spawn_child at MAX_AGENT_DEPTH returns Err
// ---------------------------------------------------------------------------

#[test]
fn test_agent_delegation_depth_exceeded() {
    let (_session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    let mock = MockInferenceProvider::new(vec![]);

    let mut conv = ConversationLoop {
        engine: Box::new(mock),
        tools: zipcode_tools::ToolRegistry::new(),
        session: Session::new(),
        permission: PermissionPolicy::new(PermissionMode::FullAccess),
        system_prompt: String::new(),
        tool_specs: Vec::new(),
        cwd: dir.path().to_path_buf(),
        depth: MAX_AGENT_DEPTH, // at the limit
        last_sent_idx: 0,
        child_session_ids: Arc::new(Mutex::new(Vec::new())),
        skill_registry: None,
        compact_policy: zipcode_runtime::CompactPolicy::default(),
    };

    let err = conv.spawn_child("go deeper", None, None, None).unwrap_err();
    assert!(
        err.to_string().contains("maximum agent depth"),
        "error must mention max depth, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: permission downgrade — FullAccess parent → ReadOnly child write denied
// ---------------------------------------------------------------------------

#[test]
fn test_agent_delegation_permission_downgrade() {
    let (session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    // Child tries write_file but has ReadOnly permission from downgrade
    let mock = MockInferenceProvider::new(vec![
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "child_write_1".to_string(),
                name: "write_file".to_string(),
                arguments: serde_json::json!({
                    "file_path": "out.txt",
                    "content": "should be denied"
                }),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        MockResponse::Text("write denied as expected".to_string()),
    ]);

    let mut conv = build_loop(&dir, mock, PermissionMode::FullAccess);
    conv.spawn_child(
        "try to write a file",
        None,
        Some(PermissionMode::ReadOnly),
        None,
    )
    .unwrap();

    // The child had ReadOnly permission — write_file must have been denied.
    // The file must not exist.
    assert!(
        !dir.path().join("out.txt").exists(),
        "write_file must be denied under ReadOnly child permission"
    );

    // Also verify PermissionPolicy enforces the downgrade correctly.
    let parent_policy = PermissionPolicy::new(PermissionMode::FullAccess);
    let child_policy = parent_policy.inherit_for_child(Some(PermissionMode::ReadOnly));
    assert_eq!(child_policy.mode(), PermissionMode::ReadOnly);

    std::fs::remove_file(conv.session.path()).ok();
    let _ = session_dir;
}

// ---------------------------------------------------------------------------
// Test 4: escalation blocked — ReadOnly parent → FullAccess request still ReadOnly
// ---------------------------------------------------------------------------

#[test]
fn test_agent_delegation_permission_escalation_blocked() {
    use zipcode_tools::{read_file::ReadFileTool, write_file::WriteFileTool, ToolRegistry};
    let (session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    // Child tries write_file; escalation to FullAccess must be refused
    let mock = MockInferenceProvider::new(vec![
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "child_write_2".to_string(),
                name: "write_file".to_string(),
                arguments: serde_json::json!({
                    "file_path": "secret.txt",
                    "content": "escalated"
                }),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        MockResponse::Text("done".to_string()),
    ]);

    let mut reg = ToolRegistry::new();
    reg.register(Box::new(ReadFileTool));
    reg.register(Box::new(WriteFileTool));
    let tool_specs = reg
        .specs()
        .into_iter()
        .map(|s| zipcode_inference::chat_template::ToolSpec {
            name: s.name,
            description: s.description,
            parameters: s.parameters,
        })
        .collect();

    let mut conv = ConversationLoop {
        engine: Box::new(mock),
        tools: reg,
        session: Session::new(),
        permission: PermissionPolicy::new(PermissionMode::ReadOnly),
        system_prompt: "test".to_string(),
        tool_specs,
        cwd: dir.path().to_path_buf(),
        depth: 0,
        last_sent_idx: 0,
        child_session_ids: Arc::new(Mutex::new(Vec::new())),
        skill_registry: None,
        compact_policy: zipcode_runtime::CompactPolicy::default(),
    };

    conv.spawn_child(
        "write secret.txt",
        None,
        Some(PermissionMode::FullAccess), // escalation attempt
        None,
    )
    .unwrap();

    // Escalation refused → child stays ReadOnly → write_file denied → file not created.
    assert!(
        !dir.path().join("secret.txt").exists(),
        "escalation must be refused: secret.txt must not exist"
    );

    // Also verify PermissionPolicy blocks escalation directly.
    let parent_policy = PermissionPolicy::new(PermissionMode::ReadOnly);
    let child_policy = parent_policy.inherit_for_child(Some(PermissionMode::FullAccess));
    assert_eq!(child_policy.mode(), PermissionMode::ReadOnly);
    let ws_policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite);
    let child_ws = ws_policy.inherit_for_child(Some(PermissionMode::FullAccess));
    assert_eq!(child_ws.mode(), PermissionMode::WorkspaceWrite);

    std::fs::remove_file(conv.session.path()).ok();
    let _ = session_dir;
}

// ---------------------------------------------------------------------------
// Test 5: budget — compute_child_budget floor/cap + max_tokens passes through
// ---------------------------------------------------------------------------

#[test]
fn test_agent_delegation_budget_exhaustion() {
    use zipcode_runtime::compute_child_budget;

    assert_eq!(compute_child_budget(0), 4096, "zero floors to 4096");
    assert_eq!(compute_child_budget(4096), 4096, "small floors to 4096");
    assert_eq!(compute_child_budget(10_000), 5_000, "mid is halved");
    assert_eq!(compute_child_budget(65_536), 32_768, "large caps at 32768");

    // Verify max_tokens passes through spawn_child without error
    let (_session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();
    let mock = MockInferenceProvider::new(vec![MockResponse::Text("child ok".to_string())]);
    let mut conv = build_loop(&dir, mock, PermissionMode::FullAccess);
    let result = conv
        .spawn_child("budget test", None, None, Some(4096))
        .unwrap();
    assert!(!result.child_session_id.is_empty());

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 6: allowlist — write_file excluded; file must not be created/modified
// ---------------------------------------------------------------------------

#[test]
fn test_agent_delegation_allowlist_filters_tools() {
    use zipcode_tools::{
        grep_search::GrepSearchTool, read_file::ReadFileTool, write_file::WriteFileTool,
        ToolRegistry,
    };
    let (_session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("data.txt"), "original").unwrap();

    // Child tries to write_file; it is NOT in the allowlist.
    // ToolRegistry::create_filtered strips it → execute_tool returns Unknown tool.
    // The child still completes (error stored as tool result) and returns a summary.
    let mock = MockInferenceProvider::new(vec![
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "child_filtered_write".to_string(),
                name: "write_file".to_string(),
                arguments: serde_json::json!({
                    "file_path": "data.txt",
                    "content": "overwritten"
                }),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        MockResponse::Text("write attempted".to_string()),
    ]);

    let mut conv = build_loop(&dir, mock, PermissionMode::FullAccess);
    let allowlist = vec!["read_file".to_string(), "grep_search".to_string()];
    let result = conv
        .spawn_child("try to write", Some(&allowlist), None, None)
        .unwrap();

    // write_file was attempted but the tool is not in the child registry.
    // The file must remain unmodified.
    assert_eq!(
        std::fs::read_to_string(dir.path().join("data.txt")).unwrap(),
        "original",
        "file must not have been modified — write_file not in allowlist"
    );
    // The child still ran to completion and returned a summary.
    assert!(!result.child_session_id.is_empty());

    let mut reg = ToolRegistry::new();
    reg.register(Box::new(ReadFileTool));
    reg.register(Box::new(WriteFileTool));
    reg.register(Box::new(GrepSearchTool));
    let filtered = reg.create_filtered(&["read_file".to_string(), "grep_search".to_string()]);
    assert!(
        filtered.get("read_file").is_some(),
        "read_file must be in filtered registry"
    );
    assert!(
        filtered.get("grep_search").is_some(),
        "grep_search must be in filtered registry"
    );
    assert!(
        filtered.get("write_file").is_none(),
        "write_file must NOT be in filtered registry"
    );

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 7: "agent" always stripped from child registry by create_filtered
// ---------------------------------------------------------------------------

#[test]
fn test_agent_delegation_blocks_nested_agent_call() {
    use zipcode_tools::{
        agent::AgentTool, grep_search::GrepSearchTool, read_file::ReadFileTool, ToolRegistry,
    };
    let (_session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    // Child tries to call "agent"; it must not be in the child registry.
    // spawn_child runs to completion even when tool is unknown (error stored
    // as tool result, child returns final text).
    let mock = MockInferenceProvider::new(vec![
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "child_nested_agent".to_string(),
                name: "agent".to_string(),
                arguments: serde_json::json!({ "task": "nested task" }),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        MockResponse::Text("nested agent done".to_string()),
    ]);

    let mut conv = build_loop(&dir, mock, PermissionMode::FullAccess);
    // Include "agent" in allowlist — create_filtered always strips it
    let allowlist = vec![
        "read_file".to_string(),
        "agent".to_string(),
        "grep_search".to_string(),
    ];
    let result = conv
        .spawn_child("try nested agent", Some(&allowlist), None, None)
        .unwrap();
    assert!(!result.child_session_id.is_empty());

    // Verify via ToolRegistry that create_filtered always removes "agent".
    let mut reg = ToolRegistry::new();
    reg.register(Box::new(AgentTool));
    reg.register(Box::new(ReadFileTool));
    reg.register(Box::new(GrepSearchTool));
    let allowlist_with_agent = vec![
        "read_file".to_string(),
        "agent".to_string(),
        "grep_search".to_string(),
    ];
    let filtered = reg.create_filtered(&allowlist_with_agent);
    assert!(
        filtered.get("agent").is_none(),
        "\"agent\" must always be stripped from filtered registry"
    );
    assert!(
        filtered.get("read_file").is_some(),
        "read_file must remain in filtered registry"
    );
    assert!(
        filtered.get("grep_search").is_some(),
        "grep_search must remain in filtered registry"
    );

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 8: e2e — parent read_file + agent call; child grep_search returns results
// ---------------------------------------------------------------------------

#[test]
fn test_parent_read_then_child_grep_e2e() {
    let (_session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    std::fs::write(
        dir.path().join("notes.txt"),
        "TODO: fix the bug\nfoo bar\nTODO: add tests",
    )
    .unwrap();

    let mock = MockInferenceProvider::new(vec![
        // parent turn 1: read_file
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "c_read".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({ "file_path": "notes.txt" }),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        // parent turn 2: delegate grep to child agent
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "c_agent".to_string(),
                name: "agent".to_string(),
                arguments: serde_json::json!({
                    "task": "grep for TODOs in notes.txt",
                    "tool_allowlist": ["grep_search"]
                }),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        // parent final response
        MockResponse::Text("Found TODOs via child agent.".to_string()),
        // child turn 1: grep_search (consumed by child's clone of the shared queue)
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "child_grep".to_string(),
                name: "grep_search".to_string(),
                arguments: serde_json::json!({ "pattern": "TODO", "path": "." }),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        // child final response after seeing grep results
        MockResponse::Text("Found 2 TODOs: fix the bug and add tests".to_string()),
    ]);

    let mut conv = build_loop(&dir, mock, PermissionMode::FullAccess);
    conv.run_turn("find all TODOs", &mut NoopCallback).unwrap();

    let msgs = &conv.session.messages;

    let has_read_result = msgs
        .iter()
        .any(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("c_read"));
    assert!(
        has_read_result,
        "read_file result must be in parent session"
    );

    let has_agent_result = msgs
        .iter()
        .any(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("c_agent"));
    assert!(
        has_agent_result,
        "agent tool result must be in parent session"
    );

    // The agent result content must contain the child's grep summary
    let agent_content = msgs
        .iter()
        .find(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("c_agent"))
        .map_or("", |m| m.content.as_str());
    assert!(
        agent_content.contains("TODOs") || agent_content.contains("TODO"),
        "agent result should reference child's grep output, got: {agent_content}"
    );

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 9: child session file has parent_id == parent session id
// ---------------------------------------------------------------------------

#[test]
fn test_child_session_file_has_parent_id() {
    let (session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    let mock = MockInferenceProvider::new(vec![MockResponse::Text("child ok".to_string())]);
    let mut conv = build_loop(&dir, mock, PermissionMode::FullAccess);
    let parent_id = conv.session.id.clone();

    let result = conv.spawn_child("simple task", None, None, None).unwrap();
    let child_id = &result.child_session_id;

    // Scan /tmp for the child session file — necessary because parallel tests
    // race on ZIPCODE_SESSIONS_DIR, so the actual save path may differ from
    // the path our set_var pointed to.
    let child_path = find_session_file(child_id)
        .unwrap_or_else(|| panic!("child session file not found for id {child_id}"));

    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&child_path).unwrap()).unwrap();
    assert_eq!(
        json["parent_id"].as_str(),
        Some(parent_id.as_str()),
        "child session parent_id must match parent session id"
    );

    std::fs::remove_file(conv.session.path()).ok();
    let _ = session_dir;
}

// ---------------------------------------------------------------------------
// Test 10: orphan child sessions cleaned on parent ConversationLoop drop
// ---------------------------------------------------------------------------

#[test]
fn test_orphan_child_session_cleaned_on_parent_drop() {
    let (session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    let mock = MockInferenceProvider::new(vec![MockResponse::Text("child ok".to_string())]);

    let child_session_path;
    {
        let mut conv = build_loop(&dir, mock, PermissionMode::FullAccess);
        let result = conv.spawn_child("temp task", None, None, None).unwrap();
        let child_id = &result.child_session_id;

        // Scan /tmp for the actual file path — parallel tests race on
        // ZIPCODE_SESSIONS_DIR so the save path may differ from our set_var.
        child_session_path = find_session_file(child_id)
            .unwrap_or_else(|| panic!("child session file not found for id {child_id}"));
        assert!(
            child_session_path.exists(),
            "child session must exist before parent drop at {}",
            child_session_path.display()
        );
        std::fs::remove_file(conv.session.path()).ok();
        // conv drops here → Drop impl removes child session files
    }

    // After the parent drops, the child session file must be gone.
    assert!(
        !child_session_path.exists(),
        "child session file must be removed after parent ConversationLoop is dropped, path: {}",
        child_session_path.display()
    );
    let _ = session_dir;
}
