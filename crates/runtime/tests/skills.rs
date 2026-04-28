use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use tempfile::TempDir;
use zipcode_inference::{
    FinishReason, MockInferenceProvider, MockResponse, Role, TokenEvent, ToolCallParsed,
};
use zipcode_runtime::{
    CompactPolicy, ConversationLoop, PermissionPolicy, Session, SkillRegistry, SkillTool,
    StreamCallback,
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

/// Serializes all tests that mutate `ZIPCODE_SESSIONS_DIR` so they don't race.
static SESSION_DIR_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn with_temp_session_dir() -> (TempDir, std::sync::MutexGuard<'static, ()>) {
    let guard = SESSION_DIR_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = TempDir::new().unwrap();
    std::env::set_var("ZIPCODE_SESSIONS_DIR", dir.path());
    (dir, guard)
}

/// Write a skill .md fixture to a directory.
fn write_skill(dir: &TempDir, filename: &str, content: &str) {
    std::fs::write(dir.path().join(filename), content).unwrap();
}

fn skill_md(name: &str, description: &str, allowlist: &[&str], body: &str) -> String {
    let al = if allowlist.is_empty() {
        String::new()
    } else {
        format!(
            "tool_allowlist:\n{}\n",
            allowlist
                .iter()
                .map(|t| format!("  - {t}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };
    format!("---\nname: {name}\ndescription: {description}\n{al}---\n{body}\n")
}

fn build_loop_with_skill_tool(
    dir: &TempDir,
    mock: MockInferenceProvider,
    skill_registry: &Arc<SkillRegistry>,
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
    registry.register(Box::new(SkillTool {
        registry: Arc::clone(skill_registry),
    }));

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
        skill_registry: Some(Arc::clone(skill_registry)),
        compact_policy: CompactPolicy::default(),
    }
}

// ---------------------------------------------------------------------------
// Test 1: SkillRegistry loads three skill files
// ---------------------------------------------------------------------------

#[test]
fn test_skill_registry_loads_three_skills() {
    let dir = TempDir::new().unwrap();

    write_skill(
        &dir,
        "review.md",
        &skill_md("review", "Review code changes", &[], "Review body"),
    );
    write_skill(
        &dir,
        "search.md",
        &skill_md("search", "Search codebase", &[], "Search body"),
    );
    write_skill(
        &dir,
        "test.md",
        &skill_md("test", "Run tests", &[], "Test body"),
    );
    // Non-.md file should be ignored
    std::fs::write(dir.path().join("readme.txt"), "ignored").unwrap();

    let registry = SkillRegistry::load_from(dir.path()).unwrap();
    let mut names = registry.names();
    names.sort_unstable();

    assert_eq!(names.len(), 3, "should load exactly 3 skills");
    assert_eq!(names, vec!["review", "search", "test"]);
    assert!(registry.get("review").is_some());
    assert!(registry.get("search").is_some());
    assert!(registry.get("test").is_some());
}

// ---------------------------------------------------------------------------
// Test 2: skill tool invokes child agent via spawn_child
// ---------------------------------------------------------------------------

#[test]
fn test_skill_tool_invokes_child_agent() {
    let (_session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    let skills_dir = TempDir::new().unwrap();
    write_skill(
        &skills_dir,
        "greet.md",
        &skill_md(
            "greet",
            "Greet a person",
            &["read_file"],
            "Hello {{ who }}!",
        ),
    );
    let registry = Arc::new(SkillRegistry::load_from(skills_dir.path()).unwrap());

    // The shared mock queue is consumed in order by both parent and child.
    // Order: parent tool call → child read_file → child final text → parent final text
    let mock = MockInferenceProvider::new(vec![
        // parent turn 1: invokes skill tool
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "skill_call_1".to_string(),
                name: "skill".to_string(),
                arguments: serde_json::json!({"name": "greet", "params": {"who": "world"}}),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        // child turn 1 (spawn_child runs while parent awaits skill result): reads a file
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "child_read_1".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"file_path": "hello.txt"}),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        // child final summary (after tool result)
        MockResponse::Text("child completed greeting task".to_string()),
        // parent final response (after skill tool result is stored)
        MockResponse::Text("skill invoked successfully".to_string()),
    ]);

    std::fs::write(dir.path().join("hello.txt"), "hello file content").unwrap();

    let mut conv = build_loop_with_skill_tool(&dir, mock, &registry, PermissionMode::FullAccess);
    conv.run_turn("invoke greet skill", &mut NoopCallback)
        .unwrap();

    // Parent session must contain a skill tool result
    let has_skill_result = conv
        .session
        .messages
        .iter()
        .any(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("skill_call_1"));
    assert!(
        has_skill_result,
        "skill tool result must appear in parent session"
    );

    // The skill result must contain the child's summary
    let skill_result_content = conv
        .session
        .messages
        .iter()
        .find(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("skill_call_1"))
        .map_or("", |m| m.content.as_str());
    assert!(
        skill_result_content.contains("child completed greeting"),
        "skill tool result must contain child summary, got: {skill_result_content}"
    );

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 3: skill catalog injected in system prompt when registry is non-empty,
//         absent when registry is empty
// ---------------------------------------------------------------------------

#[test]
fn test_skill_catalog_injected_in_system_prompt() {
    use zipcode_runtime::prompt::build_system_prompt;
    use zipcode_tools::ToolRegistry;

    let skills_dir = TempDir::new().unwrap();
    write_skill(
        &skills_dir,
        "review.md",
        &skill_md("review", "Review recently changed code", &[], "body"),
    );
    write_skill(
        &skills_dir,
        "search.md",
        &skill_md("search", "Search the codebase", &[], "body"),
    );

    let registry_with_skills = SkillRegistry::load_from(skills_dir.path()).unwrap();
    let catalog = registry_with_skills.catalog_for_prompt();
    assert!(!catalog.is_empty(), "catalog should be non-empty");

    let cwd = TempDir::new().unwrap();
    let tool_registry = ToolRegistry::new();

    // Non-empty catalog → skill section present
    let (prompt_with, _) =
        build_system_prompt(cwd.path(), &tool_registry, "full-access", Some(&catalog));
    assert!(
        prompt_with.contains("## Available Skills"),
        "prompt must contain skill section when catalog is non-empty"
    );
    assert!(
        prompt_with.contains("review: Review recently changed code"),
        "prompt must contain review skill entry"
    );
    assert!(
        prompt_with.contains("search: Search the codebase"),
        "prompt must contain search skill entry"
    );

    // None catalog → no skill section
    let (prompt_without, _) = build_system_prompt(cwd.path(), &tool_registry, "full-access", None);
    assert!(
        !prompt_without.contains("## Available Skills"),
        "prompt must not contain skill section when catalog is None"
    );

    // Empty catalog → no skill section
    let (prompt_empty, _) =
        build_system_prompt(cwd.path(), &tool_registry, "full-access", Some(""));
    assert!(
        !prompt_empty.contains("## Available Skills"),
        "prompt must not contain skill section when catalog is empty"
    );
}

// ---------------------------------------------------------------------------
// Test 4: {{ scope }} placeholder substituted end-to-end in actual child task_prompt
// ---------------------------------------------------------------------------

#[test]
fn test_skill_render_substitutes_params_e2e() {
    let (_session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    let skills_dir = TempDir::new().unwrap();
    write_skill(
        &skills_dir,
        "review.md",
        &skill_md(
            "review",
            "Review code changes",
            &["read_file"],
            "You are a reviewer. Focus on {{ scope }}.",
        ),
    );
    let registry = Arc::new(SkillRegistry::load_from(skills_dir.path()).unwrap());

    // Order: parent tool call → child response → parent final response
    let mock = MockInferenceProvider::new(vec![
        // parent turn 1: invokes skill tool with scope param
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "skill_call_review".to_string(),
                name: "skill".to_string(),
                arguments: serde_json::json!({
                    "name": "review",
                    "params": {"scope": "main..feature-branch"}
                }),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        // child consumes the rendered task prompt and returns a summary.
        MockResponse::Text("child review complete".to_string()),
        // parent final response
        MockResponse::Text("review done".to_string()),
    ]);

    let mut conv = build_loop_with_skill_tool(&dir, mock, &registry, PermissionMode::FullAccess);

    conv.run_turn("run the review skill", &mut NoopCallback)
        .unwrap();

    // The real e2e verification: the child session file on disk must contain
    // the rendered body with the placeholder replaced. If rendering or
    // spawn_child wiring ever regresses, the placeholder will leak through.
    let child_id = {
        let ids = conv.child_session_ids.lock().unwrap();
        assert!(
            !ids.is_empty(),
            "spawn_child must have been called, populating child_session_ids"
        );
        ids[0].0.clone()
    };

    let child_session = Session::load(&child_id).expect("child session file must exist on disk");
    let child_user_msg = child_session
        .messages
        .iter()
        .find(|m| m.role == Role::User)
        .map_or("", |m| m.content.as_str());

    assert!(
        child_user_msg.contains("main..feature-branch"),
        "child must have received the rendered task with {{{{ scope }}}} substituted, got: {child_user_msg}"
    );
    assert!(
        !child_user_msg.contains("{{"),
        "child prompt must not contain unresolved placeholders, got: {child_user_msg}"
    );
    assert!(
        child_user_msg.contains("reviewer"),
        "child prompt must contain the skill body text, got: {child_user_msg}"
    );

    // And the parent's skill tool result surfaces the child summary.
    let skill_result = conv
        .session
        .messages
        .iter()
        .find(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("skill_call_review"))
        .map_or("", |m| m.content.as_str());
    assert!(
        skill_result.contains("child review complete"),
        "skill tool result must surface child summary, got: {skill_result}"
    );

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 5: skill tool_allowlist scopes child registry (write_file denied)
// ---------------------------------------------------------------------------

#[test]
fn test_skill_tool_allowlist_scopes_child_registry() {
    let (_session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    let skills_dir = TempDir::new().unwrap();
    // Skill only allows read_file — write_file must be excluded from child
    write_skill(
        &skills_dir,
        "readonly.md",
        &skill_md(
            "readonly",
            "Read-only skill",
            &["read_file", "grep_search"],
            "Only read files.",
        ),
    );
    let registry = Arc::new(SkillRegistry::load_from(skills_dir.path()).unwrap());

    std::fs::write(dir.path().join("data.txt"), "original content").unwrap();

    // Order: parent tool call → child write attempt → child final → parent final
    let mock = MockInferenceProvider::new(vec![
        // parent turn 1: invokes skill tool
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "skill_readonly".to_string(),
                name: "skill".to_string(),
                arguments: serde_json::json!({"name": "readonly"}),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        // child tries write_file (not in allowlist) — runs during spawn_child
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "child_write".to_string(),
                name: "write_file".to_string(),
                arguments: serde_json::json!({
                    "file_path": "data.txt",
                    "content": "overwritten"
                }),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        MockResponse::Text("write attempted".to_string()),
        // parent final response
        MockResponse::Text("skill done".to_string()),
    ]);

    let mut conv = build_loop_with_skill_tool(&dir, mock, &registry, PermissionMode::FullAccess);
    conv.run_turn("run readonly skill", &mut NoopCallback)
        .unwrap();

    // write_file was attempted but not in child's allowlist → file unchanged
    assert_eq!(
        std::fs::read_to_string(dir.path().join("data.txt")).unwrap(),
        "original content",
        "write_file must be blocked: allowlist excludes it"
    );

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 6: skill tool unknown skill → error lists available skills
// ---------------------------------------------------------------------------

#[test]
fn test_skill_tool_unknown_skill_errors_lists_available() {
    let (_session_dir, _session_guard) = with_temp_session_dir();
    let dir = TempDir::new().unwrap();

    let skills_dir = TempDir::new().unwrap();
    write_skill(
        &skills_dir,
        "review.md",
        &skill_md("review", "Review code", &[], "body"),
    );
    write_skill(
        &skills_dir,
        "search.md",
        &skill_md("search", "Search code", &[], "body"),
    );
    let registry = Arc::new(SkillRegistry::load_from(skills_dir.path()).unwrap());

    let mock = MockInferenceProvider::new(vec![
        // parent invokes skill tool with unknown name
        MockResponse::Events(vec![
            TokenEvent::ToolCall(ToolCallParsed {
                id: "skill_unknown".to_string(),
                name: "skill".to_string(),
                arguments: serde_json::json!({"name": "nonexistent-skill"}),
            }),
            TokenEvent::Done(FinishReason::ToolUse),
        ]),
        MockResponse::Text("handled error".to_string()),
    ]);

    let mut conv = build_loop_with_skill_tool(&dir, mock, &registry, PermissionMode::FullAccess);
    conv.run_turn("invoke nonexistent skill", &mut NoopCallback)
        .unwrap();

    // The skill tool result must mention the unknown skill name and list available ones
    let skill_result = conv
        .session
        .messages
        .iter()
        .find(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("skill_unknown"))
        .map_or("", |m| m.content.as_str());

    assert!(
        skill_result.contains("nonexistent-skill"),
        "error must mention the unknown skill name, got: {skill_result}"
    );
    assert!(
        skill_result.contains("review") || skill_result.contains("search"),
        "error must list available skills, got: {skill_result}"
    );

    std::fs::remove_file(conv.session.path()).ok();
}

// ---------------------------------------------------------------------------
// Test 7: Library-level verification of the pieces `zipcode skill` composes
// (multi-skill discovery, render, parse_params shape). The CLI binary flow
// is exercised by crates/cli/tests/smoke.rs — this test guards the runtime
// primitives independently.
// ---------------------------------------------------------------------------

#[test]
fn test_skill_cli_building_blocks() {
    let skills_dir = TempDir::new().unwrap();
    write_skill(
        &skills_dir,
        "deploy.md",
        &skill_md(
            "deploy",
            "Deploy to an environment",
            &["bash"],
            "Deploy to {{ env }} using {{ strategy }}.",
        ),
    );
    write_skill(
        &skills_dir,
        "review.md",
        &skill_md(
            "review",
            "Review code",
            &["read_file", "grep_search"],
            "Review body",
        ),
    );

    let registry = SkillRegistry::load_from(skills_dir.path()).unwrap();

    // Verify 2 skills loaded
    let mut names = registry.names();
    names.sort_unstable();
    assert_eq!(names, vec!["deploy", "review"]);

    // Verify render with params
    let skill = registry.get("deploy").unwrap();
    let mut params = HashMap::new();
    params.insert("env".to_string(), "production".to_string());
    params.insert("strategy".to_string(), "blue-green".to_string());
    let rendered = skill.render(&params);
    assert!(
        rendered.contains("production"),
        "render must substitute env, got: {rendered}"
    );
    assert!(
        rendered.contains("blue-green"),
        "render must substitute strategy, got: {rendered}"
    );
    assert!(
        !rendered.contains("{{"),
        "render must leave no unresolved placeholders"
    );

    // Verify unknown skill: names listing includes both skills
    let unknown_result = registry.get("nonexistent");
    assert!(
        unknown_result.is_none(),
        "nonexistent skill must return None"
    );
    let mut available = registry.names();
    available.sort_unstable();
    assert!(
        available.contains(&"deploy"),
        "available list must include 'deploy'"
    );
    assert!(
        available.contains(&"review"),
        "available list must include 'review'"
    );

    // Verify parse_params logic (mirrors commands::parse_params)
    let raw = [
        "env=production".to_string(),
        "strategy=blue-green".to_string(),
    ];
    let parsed: HashMap<String, String> = raw
        .iter()
        .filter_map(|entry| {
            let (k, v) = entry.split_once('=')?;
            Some((k.to_string(), v.to_string()))
        })
        .collect();
    assert_eq!(parsed.get("env").map(String::as_str), Some("production"));
    assert_eq!(
        parsed.get("strategy").map(String::as_str),
        Some("blue-green")
    );

    // Verify invalid param (missing =) is detected
    let bad_raw = ["no-equals-sign".to_string()];
    let bad_result: Option<(&str, &str)> = bad_raw[0].split_once('=');
    assert!(
        bad_result.is_none(),
        "entry without '=' must not parse as key=value"
    );
}
