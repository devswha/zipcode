use std::sync::{Arc, Mutex};

use tempfile::TempDir;
use zipcode_inference::{
    FinishReason, MockInferenceProvider, MockResponse, Role, TokenEvent, ToolCallParsed, ToolSpec,
};
use zipcode_runtime::{ConversationLoop, PermissionPolicy, Session, StreamCallback};
use zipcode_tools::{ChildResult, PermissionMode, SpawnChildFn, Tool};

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
        last_sent_idx: 0,
        child_session_ids: Arc::new(Mutex::new(Vec::new())),
    }
}

/// Override the session dir so tests don't pollute ~/.zipcode/sessions.
/// Returns the TempDir that must be kept alive for the test duration.
fn with_temp_session_dir() -> TempDir {
    let dir = TempDir::new().unwrap();
    std::env::set_var("ZIPCODE_SESSIONS_DIR", dir.path());
    dir
}

// ---------------------------------------------------------------------------
// Test 1: happy path — parent calls spawn_child, child returns summary
// ---------------------------------------------------------------------------

#[test]
fn test_agent_delegation_happy_path() {
    let _session_dir = with_temp_session_dir();
    let dir = TempDir::new().unwrap();
    let mock = MockInferenceProvider::new(vec![MockResponse::Text("parent done".to_string())]);
    let mut conv = build_loop(&dir, mock, PermissionMode::FullAccess);

    let result = conv
        .spawn_child("summarize the project", None, None, None)
        .unwrap();

    assert!(!result.summary.is_empty(), "summary should not be empty");
    assert!(
        !result.child_session_id.is_empty(),
        "child session id should be set"
    );
    // tool_call_count is 0 in the current stub implementation
    assert_eq!(result.tool_call_count, 0);

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 2: depth exceeded — spawning at depth >= MAX_AGENT_DEPTH must error
// ---------------------------------------------------------------------------

#[test]
fn test_agent_delegation_depth_exceeded() {
    let _session_dir = with_temp_session_dir();
    let dir = TempDir::new().unwrap();
    let mock = MockInferenceProvider::new(vec![]);

    // Give the loop a parent_id so depth is treated as 1.
    // MAX_AGENT_DEPTH = 2, so spawning at depth 1 is fine, but the AgentTool
    // checks ctx.depth >= 2 before calling spawn. Test the stub spawn_child
    // which checks depth via session.parent_id.
    let parent = Session::new();
    let mut child_session = Session::new_child(parent.id.clone());
    // Give child a parent_id so the loop reports depth 1 when calling spawn_child
    // (spawn_child checks if session.parent_id.is_some() → depth 1, but our stub
    // compares depth >= MAX_AGENT_DEPTH(2)). Give it a grandchild-like setup:
    // set parent_id so depth = 1; spawn_child itself won't fail at depth 1.
    // Instead test via the AgentTool ctx.depth >= 2 path.
    child_session.parent_id = Some(parent.id.clone());

    // Use SpawnChildFn mock to check depth gate in the AgentTool directly.
    let call_count = Arc::new(Mutex::new(0usize));
    let call_count_clone = Arc::clone(&call_count);
    let cb: Arc<SpawnChildFn> = Arc::new(move |_task, _al, _perm, _tok| {
        *call_count_clone.lock().unwrap() += 1;
        Ok(ChildResult {
            summary: "nested".to_string(),
            tool_call_count: 0,
            child_session_id: "nested-child".to_string(),
        })
    });

    use zipcode_tools::{agent::AgentTool, ToolContext};
    let tool = AgentTool;
    // depth = 2 triggers the guard in AgentTool::execute
    let ctx = ToolContext {
        cwd: dir.path().to_path_buf(),
        permission: PermissionMode::FullAccess,
        session_id: "test-parent".to_string(),
        parent_session_id: Some("grandparent".to_string()),
        depth: 2,
        budget_tokens: None,
        spawn_child: Some(cb),
    };

    let result = tool
        .execute(serde_json::json!({ "task": "go deeper" }), &ctx)
        .unwrap();

    assert!(
        result.content.contains("max agent depth"),
        "expected depth error, got: {}",
        result.content
    );
    // Callback must NOT have been called
    assert_eq!(*call_count.lock().unwrap(), 0);

    std::fs::remove_file(mock_session_path_for(&mock)).ok();
    let _ = dir;
}

fn mock_session_path_for(_mock: &MockInferenceProvider) -> std::path::PathBuf {
    std::path::PathBuf::new() // placeholder — sessions cleaned via with_temp_session_dir
}

// ---------------------------------------------------------------------------
// Test 3: permission downgrade — FullAccess parent → ReadOnly child
// ---------------------------------------------------------------------------

#[test]
fn test_agent_delegation_permission_downgrade() {
    let _session_dir = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    let received_perm: Arc<Mutex<Option<PermissionMode>>> = Arc::new(Mutex::new(None));
    let received_perm_clone = Arc::clone(&received_perm);

    let cb: Arc<SpawnChildFn> = Arc::new(move |_task, _al, perm_override, _tok| {
        *received_perm_clone.lock().unwrap() = perm_override;
        Ok(ChildResult {
            summary: "done".to_string(),
            tool_call_count: 0,
            child_session_id: "child-perm-down".to_string(),
        })
    });

    use zipcode_tools::{agent::AgentTool, ToolContext};
    let tool = AgentTool;
    let ctx = ToolContext {
        cwd: dir.path().to_path_buf(),
        permission: PermissionMode::FullAccess,
        session_id: "parent-full".to_string(),
        parent_session_id: None,
        depth: 0,
        budget_tokens: None,
        spawn_child: Some(cb),
    };

    // The agent tool passes permission_override=None to spawn_fn; permission
    // downgrade is encoded as None (inherit). Verify via spawn_child API directly.
    let mock = MockInferenceProvider::new(vec![]);
    let mut conv = build_loop(&dir, mock, PermissionMode::FullAccess);

    let result = conv
        .spawn_child(
            "read some files",
            None,
            Some(PermissionMode::ReadOnly), // explicit downgrade
            None,
        )
        .unwrap();

    // Child session must have been created
    assert!(!result.child_session_id.is_empty());

    // Verify PermissionPolicy::inherit_for_child enforces the downgrade
    let parent_policy = PermissionPolicy::new(PermissionMode::FullAccess);
    let child_policy = parent_policy.inherit_for_child(Some(PermissionMode::ReadOnly));
    assert_eq!(child_policy.mode(), PermissionMode::ReadOnly);

    std::fs::remove_file(conv.session.path()).ok();
    let _ = (tool, ctx, received_perm);
}

// ---------------------------------------------------------------------------
// Test 4: permission escalation blocked — ReadOnly parent → FullAccess rejected
// ---------------------------------------------------------------------------

#[test]
fn test_agent_delegation_permission_escalation_blocked() {
    // ReadOnly parent tries to grant FullAccess to child — must be refused
    let parent_policy = PermissionPolicy::new(PermissionMode::ReadOnly);
    let child_policy = parent_policy.inherit_for_child(Some(PermissionMode::FullAccess));
    assert_eq!(
        child_policy.mode(),
        PermissionMode::ReadOnly,
        "escalation to FullAccess from ReadOnly must be refused"
    );

    // Also verify WorkspaceWrite → FullAccess is refused
    let ws_policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite);
    let child_ws = ws_policy.inherit_for_child(Some(PermissionMode::FullAccess));
    assert_eq!(
        child_ws.mode(),
        PermissionMode::WorkspaceWrite,
        "escalation to FullAccess from WorkspaceWrite must be refused"
    );
}

// ---------------------------------------------------------------------------
// Test 5: budget exhaustion — max_tokens respected via compute_child_budget
// ---------------------------------------------------------------------------

#[test]
fn test_agent_delegation_budget_exhaustion() {
    use zipcode_runtime::conversation::compute_child_budget;

    // With parent_remaining = 4096, child gets max(4096/2, 4096).clamp(4096, 32768) = 4096
    let budget = compute_child_budget(4096);
    assert_eq!(budget, 4096, "small budget should be floored at 4096");

    // With parent_remaining = 65536, child gets 32768 (capped)
    let budget_large = compute_child_budget(65536);
    assert_eq!(
        budget_large, 32768,
        "large budget should be capped at 32768"
    );

    // Spawn with max_tokens hint; captured in mock callback
    let received_tokens: Arc<Mutex<Option<usize>>> = Arc::new(Mutex::new(None));
    let received_clone = Arc::clone(&received_tokens);
    let cb: Arc<SpawnChildFn> = Arc::new(move |_task, _al, _perm, tokens| {
        *received_clone.lock().unwrap() = tokens;
        Ok(ChildResult {
            summary: "done".to_string(),
            tool_call_count: 0,
            child_session_id: "budget-child".to_string(),
        })
    });

    use zipcode_tools::{agent::AgentTool, ToolContext};
    let tool = AgentTool;
    let dir = TempDir::new().unwrap();
    let ctx = ToolContext {
        cwd: dir.path().to_path_buf(),
        permission: PermissionMode::FullAccess,
        session_id: "budget-parent".to_string(),
        parent_session_id: None,
        depth: 0,
        budget_tokens: None,
        spawn_child: Some(cb),
    };

    tool.execute(
        serde_json::json!({ "task": "work", "max_tokens": 4096 }),
        &ctx,
    )
    .unwrap();

    assert_eq!(
        *received_tokens.lock().unwrap(),
        Some(4096),
        "max_tokens should be forwarded to spawn callback"
    );
}

// ---------------------------------------------------------------------------
// Test 6: allowlist filters tools — write_file excluded, read_file allowed
// ---------------------------------------------------------------------------

#[test]
fn test_agent_delegation_allowlist_filters_tools() {
    let received_allowlist: Arc<Mutex<Option<Vec<String>>>> = Arc::new(Mutex::new(None));
    let received_clone = Arc::clone(&received_allowlist);

    let cb: Arc<SpawnChildFn> = Arc::new(move |_task, allowlist, _perm, _tokens| {
        *received_clone.lock().unwrap() = allowlist.map(|a| a.to_vec());
        Ok(ChildResult {
            summary: "filtered".to_string(),
            tool_call_count: 1,
            child_session_id: "allowlist-child".to_string(),
        })
    });

    use zipcode_tools::{agent::AgentTool, ToolContext};
    let tool = AgentTool;
    let dir = TempDir::new().unwrap();
    let ctx = ToolContext {
        cwd: dir.path().to_path_buf(),
        permission: PermissionMode::FullAccess,
        session_id: "parent-allowlist".to_string(),
        parent_session_id: None,
        depth: 0,
        budget_tokens: None,
        spawn_child: Some(cb),
    };

    tool.execute(
        serde_json::json!({
            "task": "search code",
            "tool_allowlist": ["read_file", "grep_search"]
        }),
        &ctx,
    )
    .unwrap();

    let list = received_allowlist.lock().unwrap();
    let list = list.as_ref().expect("allowlist should have been set");
    assert!(list.contains(&"read_file".to_string()));
    assert!(list.contains(&"grep_search".to_string()));
    assert!(
        !list.contains(&"write_file".to_string()),
        "write_file should not be in the allowlist"
    );
}

// ---------------------------------------------------------------------------
// Test 7: agent tool excluded from child — even if caller includes "agent"
// ---------------------------------------------------------------------------

#[test]
fn test_agent_delegation_blocks_nested_agent_call() {
    let received_allowlist: Arc<Mutex<Option<Vec<String>>>> = Arc::new(Mutex::new(None));
    let received_clone = Arc::clone(&received_allowlist);

    let cb: Arc<SpawnChildFn> = Arc::new(move |_task, allowlist, _perm, _tokens| {
        *received_clone.lock().unwrap() = allowlist.map(|a| a.to_vec());
        Ok(ChildResult {
            summary: "ok".to_string(),
            tool_call_count: 0,
            child_session_id: "no-agent-child".to_string(),
        })
    });

    use zipcode_tools::{agent::AgentTool, ToolContext};
    let tool = AgentTool;
    let dir = TempDir::new().unwrap();
    let ctx = ToolContext {
        cwd: dir.path().to_path_buf(),
        permission: PermissionMode::FullAccess,
        session_id: "parent-no-nest".to_string(),
        parent_session_id: None,
        depth: 0,
        budget_tokens: None,
        spawn_child: Some(cb),
    };

    // Include "agent" in allowlist — it must be stripped before forwarding
    tool.execute(
        serde_json::json!({
            "task": "do stuff",
            "tool_allowlist": ["read_file", "agent", "grep_search"]
        }),
        &ctx,
    )
    .unwrap();

    let list = received_allowlist.lock().unwrap();
    let list = list.as_ref().expect("allowlist should have been set");
    assert!(
        !list.contains(&"agent".to_string()),
        "agent must be stripped from child allowlist to prevent circular delegation"
    );
    assert!(list.contains(&"read_file".to_string()));
    assert!(list.contains(&"grep_search".to_string()));
}

// ---------------------------------------------------------------------------
// Test 8: e2e — parent run_turn triggers agent call, child returns summary
// ---------------------------------------------------------------------------

#[test]
fn test_parent_read_then_child_grep_e2e() {
    let _session_dir = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    // Write a file for the parent to read and child to grep
    std::fs::write(
        dir.path().join("notes.txt"),
        "TODO: fix the bug\nfoo bar\nTODO: add tests",
    )
    .unwrap();

    // Parent: first calls read_file, then calls agent tool
    // Turn 1: model issues read_file tool call → result → then final text response
    let read_file_call = ToolCallParsed {
        id: "c_read".to_string(),
        name: "read_file".to_string(),
        arguments: serde_json::json!({ "file_path": "notes.txt" }),
    };
    let agent_call = ToolCallParsed {
        id: "c_agent".to_string(),
        name: "agent".to_string(),
        arguments: serde_json::json!({
            "task": "grep for TODOs",
            "tool_allowlist": ["grep_search"]
        }),
    };

    let mock = MockInferenceProvider::new(vec![
        MockResponse::Events(vec![
            TokenEvent::ToolCall(read_file_call),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        MockResponse::Events(vec![
            TokenEvent::ToolCall(agent_call),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        MockResponse::Text("Found TODOs via child agent.".to_string()),
    ]);

    let mut conv = build_loop(&dir, mock, PermissionMode::FullAccess);
    conv.run_turn("find all TODOs", &mut NoopCallback).unwrap();

    // Parent session should contain tool results for both read_file and agent
    let has_read_result = conv
        .session
        .messages
        .iter()
        .any(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("c_read"));
    assert!(
        has_read_result,
        "read_file result must be in parent session"
    );

    let has_agent_result = conv
        .session
        .messages
        .iter()
        .any(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("c_agent"));
    assert!(has_agent_result, "agent result must be in parent session");

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 9: child session file has parent_id == parent session id
// ---------------------------------------------------------------------------

#[test]
fn test_child_session_file_has_parent_id() {
    let _session_dir = with_temp_session_dir();
    let dir = TempDir::new().unwrap();
    let mock = MockInferenceProvider::new(vec![]);
    let mut conv = build_loop(&dir, mock, PermissionMode::FullAccess);
    let parent_id = conv.session.id.clone();

    let result = conv.spawn_child("read config", None, None, None).unwrap();

    let child_id = &result.child_session_id;

    // The child session file should have been saved with parent_id == parent_id
    let child_path = Session::path_for_id(child_id).unwrap();
    assert!(
        child_path.exists(),
        "child session file must exist at {}",
        child_path.display()
    );

    let content = std::fs::read_to_string(&child_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert_eq!(
        json["parent_id"].as_str(),
        Some(parent_id.as_str()),
        "child session parent_id must match parent session id"
    );

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 10: orphan child sessions cleaned on parent ConversationLoop drop
// ---------------------------------------------------------------------------

#[test]
fn test_orphan_child_session_cleaned_on_parent_drop() {
    let _session_dir = with_temp_session_dir();
    let dir = TempDir::new().unwrap();
    let mock = MockInferenceProvider::new(vec![]);

    let child_session_path;
    {
        let mut conv = build_loop(&dir, mock, PermissionMode::FullAccess);

        let result = conv
            .spawn_child("temporary task", None, None, None)
            .unwrap();
        let child_id = &result.child_session_id;

        child_session_path = Session::path_for_id(child_id).unwrap();
        assert!(
            child_session_path.exists(),
            "child session must exist before parent drop"
        );

        std::fs::remove_file(conv.session.path()).ok();
        // conv drops here → Drop impl removes child session files
    }

    assert!(
        !child_session_path.exists(),
        "child session file must be removed after parent ConversationLoop is dropped"
    );
}
