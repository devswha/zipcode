//! Integration tests for the two-tier context compaction system.
//!
//! Tests 1, 2, 4, 7 use `ConversationLoop` (which saves session files) and
//! therefore require `with_temp_session_dir()` + `SESSION_DIR_LOCK`.
//! Tests 3, 5, 6 operate directly on `Session` and never call `save()`, so
//! they need no filesystem isolation.

use std::sync::{Arc, Mutex, OnceLock};

use tempfile::TempDir;
use zipcode_inference::{
    ChatMessage, FinishReason, MockInferenceProvider, MockResponse, Role, TokenEvent,
    ToolCallParsed,
};
use zipcode_runtime::{
    CompactPolicy, ConversationLoop, PermissionPolicy, Session, StreamCallback,
    TOOL_PAIR_SUMMARY_MARKER,
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

/// Serialises all tests that mutate `ZIPCODE_SESSIONS_DIR` so they never race.
static SESSION_DIR_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Redirect session file I/O to a fresh temporary directory.
/// Returns `(TempDir, guard)` — keep both alive for the test duration.
fn with_temp_session_dir() -> (TempDir, std::sync::MutexGuard<'static, ()>) {
    let guard = SESSION_DIR_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let dir = TempDir::new().unwrap();
    std::env::set_var("ZIPCODE_SESSIONS_DIR", dir.path());
    (dir, guard)
}

/// Build a `ConversationLoop` with `AgentTool` + file tools registered and
/// a custom `CompactPolicy`.  Uses the supplied `TempDir` as cwd.
fn build_loop(
    dir: &TempDir,
    mock: MockInferenceProvider,
    policy: CompactPolicy,
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
        permission: PermissionPolicy::new(PermissionMode::FullAccess),
        system_prompt: "You are a test assistant.".to_string(),
        tool_specs,
        cwd: dir.path().to_path_buf(),
        depth: 0,
        last_sent_idx: 0,
        child_session_ids: Arc::new(Mutex::new(Vec::new())),
        skill_registry: None,
        compact_policy: policy,
    }
}

/// Push `count` synthetic `(assistant_with_tool_calls, tool_result)` pairs
/// into `session`, each message body padded to `content_chars` characters.
fn seed_tool_pairs(session: &mut Session, count: usize, content_chars: usize) {
    for i in 0..count {
        let call = ToolCallParsed {
            id: format!("seed_{i}"),
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": format!("file_{i}.rs")}),
        };
        session.push_message(ChatMessage::assistant_with_tool_calls(
            &"a".repeat(content_chars),
            vec![call],
        ));
        session.push_message(ChatMessage::tool_result(
            &format!("seed_{i}"),
            &"r".repeat(content_chars),
        ));
    }
}

// ---------------------------------------------------------------------------
// Test 1: 100-turn synthetic dialog stays under 32 K tokens
// ---------------------------------------------------------------------------

#[test]
fn test_compaction_100_turn_dialog_no_overflow_at_32k() {
    let (_session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    // prompt_eval_count = 30 000 > floor(0.8 × 32 768) = 26 214
    // → tier-2 fires on every run_turn call.
    let mock = MockInferenceProvider::new(vec![MockResponse::Text("turn 1 done".to_string())])
        .with_prompt_eval_count(30_000);

    // Default policy: context_window_tokens=32768, tier2_usage_threshold=0.8,
    // tier1_pair_cutoff=10.
    let policy = CompactPolicy::default();
    let mut conv = build_loop(&dir, mock, policy);

    // Seed with a system prompt and 100 synthetic tool-call/result pairs at
    // 800 chars each.  Without compaction the total is:
    //   100 × 2 messages × (800 chars / 4) = 40 000 tokens  (> 32 768).
    conv.session
        .push_message(ChatMessage::system("test assistant"));
    seed_tool_pairs(&mut conv.session, 100, 800);

    let initial_tokens = conv.session.estimated_tokens();
    assert!(
        initial_tokens > 32_768,
        "initial token estimate must exceed 32 K to validate the test: {initial_tokens}"
    );

    // Single run_turn triggers tier-2, which evicts tool-result messages
    // until estimated_tokens ≤ 26 214.
    conv.run_turn("verify session under 32 K", &mut NoopCallback)
        .unwrap();

    let final_tokens = conv.session.estimated_tokens();
    assert!(
        final_tokens < 32_768,
        "after tier-2 compaction, estimated_tokens must be < 32 768, got {final_tokens}"
    );

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 2: tier-1 summarises the oldest 10 tool-call pairs
// ---------------------------------------------------------------------------

#[test]
fn test_tier1_summarizes_oldest_10_pairs() {
    let (_session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    // Default policy: tier1_pair_cutoff=10, tier1_batch_size=10.
    // Seeding 20 pairs (> cutoff of 10) causes tier-1 to compact 10.
    let mock = MockInferenceProvider::new(vec![MockResponse::Text("after tier1".to_string())]);
    let policy = CompactPolicy::default();
    let mut conv = build_loop(&dir, mock, policy);

    conv.session.push_message(ChatMessage::system("test"));
    for i in 0..20usize {
        let call = ToolCallParsed {
            id: format!("c{i}"),
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": format!("echo {i}")}),
        };
        conv.session
            .push_message(ChatMessage::assistant_with_tool_calls(
                &format!("step {i}"),
                vec![call],
            ));
        conv.session.push_message(ChatMessage::tool_result(
            &format!("c{i}"),
            &format!("output {i}"),
        ));
    }

    assert_eq!(
        conv.session.tool_pair_count(),
        20,
        "should start with 20 visible pairs"
    );

    // run_turn: MockResponse::Text drives the turn; tier-1 fires afterward.
    conv.run_turn("trigger tier-1", &mut NoopCallback).unwrap();

    // After tier-1 compaction: oldest 10 pairs invisible, 10 pairs remain.
    assert_eq!(
        conv.session.tool_pair_count(),
        10,
        "10 pairs must remain visible after tier-1 compacts the oldest 10"
    );

    // A summary message prefixed with TOOL_PAIR_SUMMARY_MARKER must be visible.
    let has_summary = conv
        .session
        .messages
        .iter()
        .any(|m| m.content.starts_with(TOOL_PAIR_SUMMARY_MARKER) && !m.agent_invisible);
    assert!(
        has_summary,
        "tier-1 must produce a visible {TOOL_PAIR_SUMMARY_MARKER} message"
    );

    // Exactly 20 messages (10 pairs × 2) must be agent_invisible.
    let invisible_count = conv
        .session
        .messages
        .iter()
        .filter(|m| m.agent_invisible)
        .count();
    assert_eq!(
        invisible_count, 20,
        "oldest 10 pairs (20 messages) must be agent_invisible after tier-1"
    );

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 3: tier-2 progressive eviction stops at the first satisfied step
// ---------------------------------------------------------------------------

#[test]
fn test_tier2_progressive_eviction_to_threshold() {
    // 10 tool-result messages × 200 chars / 4 = 50 tokens each → 500 tokens total.
    // threshold = 449:
    //   10 % → 1 invisible → 9 × 50 = 450 > 449 → continue
    //   20 % → 2 invisible → 8 × 50 = 400 ≤ 449 → stop
    let mut session = Session::new();
    for i in 0..10usize {
        session.push_message(ChatMessage::tool_result(
            &format!("c{i}"),
            &"x".repeat(200), // 200 chars / 4 = 50 tokens
        ));
    }

    assert_eq!(
        session.estimated_tokens(),
        500,
        "initial: 10 × 50 = 500 tokens"
    );

    let evicted = session.evict_tool_responses_progressive(449);
    assert!(evicted, "must return true when eviction occurred");

    let invisible = session
        .messages
        .iter()
        .filter(|m| m.agent_invisible)
        .count();
    assert_eq!(
        invisible, 2,
        "20 % of 10 = 2 messages evicted to satisfy the 449-token threshold"
    );
    assert_eq!(
        session.estimated_tokens(),
        400,
        "8 remaining visible × 50 tokens = 400"
    );
}

// ---------------------------------------------------------------------------
// Test 4: a panic inside compact_tool_pairs is non-blocking
// ---------------------------------------------------------------------------

#[test]
fn test_tier1_failure_non_blocking() {
    let (_session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    // tier1_batch_size=0 causes compact_tool_pairs to panic when it tries to
    // index the empty batch slice (`batch[0]`).  The catch_unwind wrapper in
    // ConversationLoop must absorb the panic so run_turn returns Ok.
    let mock = MockInferenceProvider::new(vec![MockResponse::Text(
        "turn completes despite tier-1 panic".to_string(),
    )]);
    let policy = CompactPolicy {
        tier1_pair_cutoff: 0, // fires when there is at least 1 pair
        tier1_batch_size: 0,  // batch[0] panics on an empty batch slice
        ..CompactPolicy::default()
    };
    let mut conv = build_loop(&dir, mock, policy);

    conv.session.push_message(ChatMessage::system("test"));
    let call = ToolCallParsed {
        id: "c1".to_string(),
        name: "read_file".to_string(),
        arguments: serde_json::json!({"path": "foo.rs"}),
    };
    // 1 pair → tool_pair_count (1) > tier1_pair_cutoff (0) → tier-1 fires.
    conv.session
        .push_message(ChatMessage::assistant_with_tool_calls("read", vec![call]));
    conv.session
        .push_message(ChatMessage::tool_result("c1", "file contents"));

    // Must succeed even though compact_tool_pairs panics internally.
    let result = conv.run_turn("does tier-1 panic block the turn?", &mut NoopCallback);
    assert!(
        result.is_ok(),
        "run_turn must return Ok even when tier-1 compact_tool_pairs panics: {result:?}"
    );

    // The final text response must be in the session, proving the turn completed.
    let has_response = conv
        .session
        .messages
        .iter()
        .any(|m| m.role == Role::Model && m.content.contains("despite tier-1 panic"));
    assert!(
        has_response,
        "final response must be in session, proving the turn completed despite tier-1 panic"
    );

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 5: tier-2 does not double-evict on a second call
// ---------------------------------------------------------------------------

#[test]
fn test_tier2_progressive_does_not_double_evict() {
    // Same sizing as test 3: 10 × 50 = 500 tokens, threshold 449.
    // After the first call 2 messages are invisible and tokens drop to 400.
    // A second call must return false and leave the invisible count unchanged.
    let mut session = Session::new();
    for i in 0..10usize {
        session.push_message(ChatMessage::tool_result(
            &format!("c{i}"),
            &"x".repeat(200), // 50 tokens each
        ));
    }

    let first = session.evict_tool_responses_progressive(449);
    assert!(first, "first call must evict");
    let invisible_after_first = session
        .messages
        .iter()
        .filter(|m| m.agent_invisible)
        .count();
    assert_eq!(invisible_after_first, 2, "first call: 2 messages invisible");
    assert_eq!(session.estimated_tokens(), 400);

    // Second call: already at 400 ≤ 449 → returns false immediately.
    let second = session.evict_tool_responses_progressive(449);
    assert!(
        !second,
        "second call must return false (already under threshold)"
    );

    let invisible_after_second = session
        .messages
        .iter()
        .filter(|m| m.agent_invisible)
        .count();
    assert_eq!(
        invisible_after_second, 2,
        "no additional messages must be evicted on the second call"
    );
    assert_eq!(
        session.estimated_tokens(),
        400,
        "estimated_tokens must be unchanged after the no-op second call"
    );
}

// ---------------------------------------------------------------------------
// Test 6: compaction preserves the relative order of agent-visible messages
// ---------------------------------------------------------------------------

#[test]
fn test_compaction_preserves_message_order() {
    let mut session = Session::new();

    // Layout: user_a → 15 tool pairs → user_b
    session.push_message(ChatMessage::user("user_message_a"));
    for i in 0..15usize {
        let call = ToolCallParsed {
            id: format!("c{i}"),
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": format!("f{i}.rs")}),
        };
        session.push_message(ChatMessage::assistant_with_tool_calls(
            &format!("step_{i}"),
            vec![call],
        ));
        session.push_message(ChatMessage::tool_result(
            &format!("c{i}"),
            &format!("res{i}"),
        ));
    }
    session.push_message(ChatMessage::user("user_message_b"));

    let n = session.compact_tool_pairs(10, 120);
    assert_eq!(n, 10, "should compact exactly 10 pairs");

    // Collect visible messages in document order.
    let visible_contents: Vec<&str> = session
        .messages
        .iter()
        .filter(|m| !m.agent_invisible)
        .map(|m| m.content.as_str())
        .collect();

    // user_a must precede the summary.
    let user_a_pos = visible_contents
        .iter()
        .position(|&c| c == "user_message_a")
        .expect("user_message_a must be visible");
    let summary_pos = visible_contents
        .iter()
        .position(|c| c.starts_with(TOOL_PAIR_SUMMARY_MARKER))
        .expect("summary must be visible");
    assert!(user_a_pos < summary_pos, "user_a must precede the summary");

    // user_b must be the last visible message.
    assert_eq!(
        *visible_contents.last().unwrap(),
        "user_message_b",
        "user_message_b must be the last visible message"
    );

    // The 5 remaining visible tool-result messages must be in original order.
    let visible_tool_results: Vec<&str> = session
        .messages
        .iter()
        .filter(|m| m.role == Role::Tool && !m.agent_invisible)
        .map(|m| m.content.as_str())
        .collect();
    assert_eq!(
        visible_tool_results.len(),
        5,
        "exactly 5 pairs must remain visible"
    );
    for (i, &content) in visible_tool_results.iter().enumerate() {
        assert_eq!(
            content,
            format!("res{}", i + 10).as_str(),
            "tool-result order must be preserved (expected res{}, got {content})",
            i + 10
        );
    }
}

// ---------------------------------------------------------------------------
// Test 7: compaction does not interfere with Phase-1 child agent delegation
// ---------------------------------------------------------------------------

#[test]
fn test_compaction_does_not_affect_phase1_delegation() {
    let (_session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    // Queue order (shared between parent and child via cloned mock):
    //   1. Parent iteration 1 : agent tool call
    //   2. Child iteration    : final text ("child completed the task")
    //   3. Parent iteration 2 : final text after agent result
    let mock = MockInferenceProvider::new(vec![
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "agent_call_1".to_string(),
                name: "agent".to_string(),
                arguments: serde_json::json!({"task": "simple delegation task"}),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        MockResponse::Text("child completed the task".to_string()),
        MockResponse::Text("delegation complete".to_string()),
    ]);

    // tier1_pair_cutoff=5, tier1_batch_size=5 → tier-1 fires after seeding 6 pairs
    // (6 seeded + 1 agent pair added by the turn = 7 pairs > cutoff 5).
    let policy = CompactPolicy {
        tier1_pair_cutoff: 5,
        tier1_batch_size: 5,
        ..CompactPolicy::default()
    };
    let mut conv = build_loop(&dir, mock, policy);

    // Seed 6 tool pairs (> cutoff 5) with small content.
    conv.session
        .push_message(ChatMessage::system("test assistant"));
    seed_tool_pairs(&mut conv.session, 6, 40);
    assert_eq!(conv.session.tool_pair_count(), 6);

    // run_turn: parent → agent tool call → child runs → final response.
    // After the turn tier-1 fires (7 total pairs > cutoff 5) and compacts 5.
    conv.run_turn("delegate to child agent", &mut NoopCallback)
        .unwrap();

    // Child delegation must have occurred.
    {
        let child_ids = conv.child_session_ids.lock().unwrap();
        assert!(
            !child_ids.is_empty(),
            "child agent must have been spawned via Phase-1 delegation"
        );
    }

    // Agent tool result must still be visible in the parent session.
    let has_agent_result = conv.session.messages.iter().any(|m| {
        m.role == Role::Tool
            && m.tool_call_id.as_deref() == Some("agent_call_1")
            && !m.agent_invisible
    });
    assert!(
        has_agent_result,
        "agent tool result must be visible in parent session after compaction"
    );

    // Tier-1 compaction must have produced a summary (proving it ran).
    let has_summary = conv
        .session
        .messages
        .iter()
        .any(|m| m.content.starts_with(TOOL_PAIR_SUMMARY_MARKER));
    assert!(
        has_summary,
        "tier-1 must have produced a {TOOL_PAIR_SUMMARY_MARKER} in the parent session"
    );

    std::fs::remove_file(conv.session.path()).ok();
}
