use std::io::{ErrorKind, Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use tempfile::TempDir;
use zipcode_inference::{MockInferenceProvider, MockResponse, ToolSpec};
use zipcode_runtime::{ConversationLoop, PermissionPolicy, Session, StreamCallback};
use zipcode_tools::PermissionMode;

struct ScenarioCallback {
    tokens: Vec<String>,
    tool_calls: Vec<String>,
    tool_results: Vec<(String, String)>,
    errors: Vec<String>,
}

impl ScenarioCallback {
    const fn new() -> Self {
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

    fn result_for(&self, name: &str) -> String {
        self.tool_results
            .iter()
            .filter(|(tool, _)| tool == name)
            .map(|(_, result)| result.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl StreamCallback for ScenarioCallback {
    fn on_token(&mut self, text: &str) {
        self.tokens.push(text.to_string());
    }

    fn on_tool_start(&mut self, name: &str, _args: &serde_json::Value) {
        self.tool_calls.push(name.to_string());
    }

    fn on_tool_result(&mut self, name: &str, result: &str) {
        self.tool_results
            .push((name.to_string(), result.to_string()));
    }

    fn on_permission_prompt(&mut self, _message: &str) -> bool {
        true
    }

    fn on_error(&mut self, error: &str) {
        self.errors.push(error.to_string());
    }
}

fn build_scenario_loop(cwd: &Path, engine: MockInferenceProvider) -> ConversationLoop {
    use zipcode_tools::{
        bash::BashTool, edit_file::EditFileTool, glob_search::GlobSearchTool,
        grep_search::GrepSearchTool, read_file::ReadFileTool, repl::ReplTool,
        todo_write::TodoWriteTool, tool_search::ToolSearchTool, write_file::WriteFileTool,
        ToolRegistry,
    };

    let mut registry = ToolRegistry::new();
    registry.register(Box::new(BashTool));
    registry.register(Box::new(ReadFileTool));
    registry.register(Box::new(WriteFileTool));
    registry.register(Box::new(EditFileTool));
    registry.register(Box::new(GlobSearchTool));
    registry.register(Box::new(GrepSearchTool));
    registry.register(Box::new(TodoWriteTool));
    registry.register(Box::new(ReplTool));
    registry.register(Box::new(ToolSearchTool::from_registry(&registry)));

    let tool_specs: Vec<ToolSpec> = registry
        .specs()
        .into_iter()
        .map(|spec| ToolSpec {
            name: spec.name,
            description: spec.description,
            parameters: spec.parameters,
        })
        .collect();

    ConversationLoop {
        engine: Box::new(engine),
        tools: registry,
        session: Session::new(),
        permission: PermissionPolicy::new(PermissionMode::FullAccess),
        system_prompt: "You are a deterministic coding-agent scenario test assistant.".to_string(),
        tool_specs,
        cwd: cwd.to_path_buf(),
        depth: 0,
        last_sent_idx: 0,
        child_session_ids: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        skill_registry: None,
        compact_policy: zipcode_runtime::CompactPolicy::default(),
    }
}

fn write_file(path: &std::path::Path, contents: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent directory");
    }
    std::fs::write(path, contents).expect("write fixture file");
}

fn serve_once(body: &'static str) -> (String, thread::JoinHandle<Result<(), String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind local fixture server");
    let url = format!("http://{}/sample.txt", listener.local_addr().unwrap());
    listener
        .set_nonblocking(true)
        .expect("configure local fixture server");
    let handle = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(2);
        let (mut stream, _) = loop {
            match listener.accept() {
                Ok(accepted) => break accepted,
                Err(error)
                    if error.kind() == ErrorKind::WouldBlock && Instant::now() < deadline =>
                {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    return Err("timed out waiting for fixture request".to_string());
                }
                Err(error) => return Err(format!("accept fixture request: {error}")),
            }
        };

        let mut request = [0_u8; 1024];
        stream
            .read(&mut request)
            .map_err(|error| format!("read fixture request: {error}"))?;

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream
            .write_all(response.as_bytes())
            .map_err(|error| format!("write fixture response: {error}"))?;
        Ok(())
    });

    (url, handle)
}

#[test]
fn bugfix_loop_reads_edits_and_runs_tests() {
    let dir = TempDir::new().unwrap();
    write_file(
        &dir.path().join("Cargo.toml"),
        r#"[package]
name = "agent_bugfix_fixture"
version = "0.1.0"
edition = "2021"
"#,
    );
    write_file(
        &dir.path().join("src/lib.rs"),
        r"pub fn add(a: i32, b: i32) -> i32 {
    a - b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_numbers() {
        assert_eq!(add(2, 3), 5);
    }
}
",
    );

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "read_file".to_string(),
            args: serde_json::json!({ "path": "src/lib.rs" }),
        },
        MockResponse::ToolCall {
            name: "edit_file".to_string(),
            args: serde_json::json!({
                "path": "src/lib.rs",
                "old_string": "    a - b",
                "new_string": "    a + b"
            }),
        },
        MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({ "command": "cargo test --offline --quiet" }),
        },
        MockResponse::Text("Fixed add and verified the tests pass.".to_string()),
    ]);

    let mut conv = build_scenario_loop(dir.path(), mock);
    let mut cb = ScenarioCallback::new();
    conv.run_turn("Fix the failing add test", &mut cb).unwrap();

    let lib = std::fs::read_to_string(dir.path().join("src/lib.rs")).unwrap();
    assert!(lib.contains("a + b"));
    assert_eq!(cb.tool_calls, ["read_file", "edit_file", "bash"]);
    assert!(
        !cb.result_for("bash").contains("Exit code"),
        "cargo test should pass: {}",
        cb.result_for("bash")
    );
    assert!(cb.all_tokens().contains("verified"));
    assert!(cb.errors.is_empty(), "unexpected errors: {:?}", cb.errors);
}

#[test]
fn grep_driven_rename_edits_code_and_docs() {
    let dir = TempDir::new().unwrap();
    write_file(
        &dir.path().join("src/main.rs"),
        "fn main() { println!(\"ZipCodeAgent\"); }\n",
    );
    write_file(
        &dir.path().join("src/config.rs"),
        "pub const AGENT_NAME: &str = \"ZipCodeAgent\";\n",
    );
    write_file(
        &dir.path().join("README.md"),
        "# ZipCodeAgent\n\nLocal coding agent fixture.\n",
    );

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "grep_search".to_string(),
            args: serde_json::json!({ "pattern": "ZipCodeAgent", "path": "." }),
        },
        MockResponse::ToolCall {
            name: "edit_file".to_string(),
            args: serde_json::json!({
                "path": "src/main.rs",
                "old_string": "ZipCodeAgent",
                "new_string": "LocalAgent"
            }),
        },
        MockResponse::ToolCall {
            name: "edit_file".to_string(),
            args: serde_json::json!({
                "path": "src/config.rs",
                "old_string": "ZipCodeAgent",
                "new_string": "LocalAgent"
            }),
        },
        MockResponse::ToolCall {
            name: "edit_file".to_string(),
            args: serde_json::json!({
                "path": "README.md",
                "old_string": "ZipCodeAgent",
                "new_string": "LocalAgent"
            }),
        },
        MockResponse::ToolCall {
            name: "grep_search".to_string(),
            args: serde_json::json!({ "pattern": "ZipCodeAgent", "path": "." }),
        },
        MockResponse::Text("Renamed ZipCodeAgent to LocalAgent everywhere.".to_string()),
    ]);

    let mut conv = build_scenario_loop(dir.path(), mock);
    let mut cb = ScenarioCallback::new();
    conv.run_turn("Rename ZipCodeAgent to LocalAgent", &mut cb)
        .unwrap();

    for relative in ["src/main.rs", "src/config.rs", "README.md"] {
        let contents = std::fs::read_to_string(dir.path().join(relative)).unwrap();
        assert!(
            contents.contains("LocalAgent"),
            "missing rename in {relative}"
        );
        assert!(
            !contents.contains("ZipCodeAgent"),
            "old name remains in {relative}"
        );
    }
    assert_eq!(
        cb.tool_calls,
        [
            "grep_search",
            "edit_file",
            "edit_file",
            "edit_file",
            "grep_search"
        ]
    );
    assert!(cb.result_for("grep_search").contains("No matches found."));
}

#[test]
fn test_first_repair_loop_uses_failure_output_before_patch() {
    let dir = TempDir::new().unwrap();
    write_file(
        &dir.path().join("Cargo.toml"),
        r#"[package]
name = "agent_test_first_fixture"
version = "0.1.0"
edition = "2021"
"#,
    );
    write_file(
        &dir.path().join("src/lib.rs"),
        r#"pub fn discount_cents(price_cents: u32, percent: u32) -> u32 {
    price_cents * percent / 100
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_discount() {
        assert_eq!(discount_cents(2000, 25), 1500);
    }
}
"#,
    );

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({ "command": "cargo test --offline --quiet" }),
        },
        MockResponse::ToolCall {
            name: "read_file".to_string(),
            args: serde_json::json!({ "path": "src/lib.rs" }),
        },
        MockResponse::ToolCall {
            name: "edit_file".to_string(),
            args: serde_json::json!({
                "path": "src/lib.rs",
                "old_string": "    price_cents * percent / 100",
                "new_string": "    price_cents - (price_cents * percent / 100)"
            }),
        },
        MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({ "command": "cargo test --offline --quiet" }),
        },
        MockResponse::Text(
            "Reproduced the failure, patched the calculation, and reran tests.".to_string(),
        ),
    ]);

    let mut conv = build_scenario_loop(dir.path(), mock);
    let mut cb = ScenarioCallback::new();
    conv.run_turn("Run tests first, then fix the discount bug", &mut cb)
        .unwrap();

    let lib = std::fs::read_to_string(dir.path().join("src/lib.rs")).unwrap();
    let bash_results: Vec<&str> = cb
        .tool_results
        .iter()
        .filter(|(tool, _)| tool == "bash")
        .map(|(_, result)| result.as_str())
        .collect();

    assert_eq!(cb.tool_calls, ["bash", "read_file", "edit_file", "bash"]);
    assert_eq!(bash_results.len(), 2);
    assert!(
        bash_results[0].contains("Exit code"),
        "first test run should fail: {}",
        bash_results[0]
    );
    assert!(
        !bash_results[1].contains("Exit code"),
        "second test run should pass: {}",
        bash_results[1]
    );
    assert!(lib.contains("price_cents - (price_cents * percent / 100)"));
    assert!(cb.all_tokens().contains("reran tests"));
}

#[test]
fn glob_driven_discovery_reads_routes_and_writes_inventory() {
    let dir = TempDir::new().unwrap();
    write_file(
        &dir.path().join("src/routes/admin.rs"),
        "pub const ROUTE: &str = \"/admin\";\n",
    );
    write_file(
        &dir.path().join("src/routes/users.rs"),
        "pub const ROUTE: &str = \"/users\";\n",
    );
    write_file(&dir.path().join("src/main.rs"), "fn main() {}\n");

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "glob_search".to_string(),
            args: serde_json::json!({ "pattern": "src/routes/*.rs" }),
        },
        MockResponse::ToolCall {
            name: "read_file".to_string(),
            args: serde_json::json!({ "path": "src/routes/admin.rs" }),
        },
        MockResponse::ToolCall {
            name: "read_file".to_string(),
            args: serde_json::json!({ "path": "src/routes/users.rs" }),
        },
        MockResponse::ToolCall {
            name: "write_file".to_string(),
            args: serde_json::json!({
                "path": "docs/routes.md",
                "content": "# Route Inventory\n\n- /admin from src/routes/admin.rs\n- /users from src/routes/users.rs\n"
            }),
        },
        MockResponse::Text("Discovered route files and wrote the inventory.".to_string()),
    ]);

    let mut conv = build_scenario_loop(dir.path(), mock);
    let mut cb = ScenarioCallback::new();
    conv.run_turn("Find route modules and document their routes", &mut cb)
        .unwrap();

    let inventory = std::fs::read_to_string(dir.path().join("docs/routes.md")).unwrap();

    assert_eq!(
        cb.tool_calls,
        ["glob_search", "read_file", "read_file", "write_file"]
    );
    assert!(cb.result_for("glob_search").contains("src/routes/admin.rs"));
    assert!(cb.result_for("glob_search").contains("src/routes/users.rs"));
    assert!(inventory.contains("/admin"));
    assert!(inventory.contains("/users"));
    assert!(cb.all_tokens().contains("inventory"));
}

#[test]
fn tool_search_guides_agent_to_grep_then_patch_todo() {
    let dir = TempDir::new().unwrap();
    write_file(
        &dir.path().join("src/lib.rs"),
        r#"pub fn greeting() -> &'static str {
    "TODO: choose greeting"
}
"#,
    );

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "tool_search".to_string(),
            args: serde_json::json!({ "query": "search" }),
        },
        MockResponse::ToolCall {
            name: "grep_search".to_string(),
            args: serde_json::json!({ "pattern": "TODO", "path": "src" }),
        },
        MockResponse::ToolCall {
            name: "read_file".to_string(),
            args: serde_json::json!({ "path": "src/lib.rs" }),
        },
        MockResponse::ToolCall {
            name: "edit_file".to_string(),
            args: serde_json::json!({
                "path": "src/lib.rs",
                "old_string": "TODO: choose greeting",
                "new_string": "hello from zipcode"
            }),
        },
        MockResponse::Text("Found the right search tool and replaced the TODO.".to_string()),
    ]);

    let mut conv = build_scenario_loop(dir.path(), mock);
    let mut cb = ScenarioCallback::new();
    conv.run_turn(
        "Find TODOs even if you need to discover the right tool",
        &mut cb,
    )
    .unwrap();

    let lib = std::fs::read_to_string(dir.path().join("src/lib.rs")).unwrap();

    assert_eq!(
        cb.tool_calls,
        ["tool_search", "grep_search", "read_file", "edit_file"]
    );
    assert!(cb.result_for("tool_search").contains("grep_search"));
    assert!(cb
        .result_for("grep_search")
        .contains("TODO: choose greeting"));
    assert!(lib.contains("hello from zipcode"));
    assert!(!lib.contains("TODO: choose greeting"));
}

#[test]
fn todo_write_file_and_repl_loop_creates_feature() {
    let dir = TempDir::new().unwrap();
    let slug_module = r#"import re

def slugify(value):
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")
"#;

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "todo_write".to_string(),
            args: serde_json::json!({
                "todos": [
                    { "id": "1", "content": "Create slug helper", "status": "in_progress" },
                    { "id": "2", "content": "Verify slug helper", "status": "pending" }
                ]
            }),
        },
        MockResponse::ToolCall {
            name: "write_file".to_string(),
            args: serde_json::json!({ "path": "slug.py", "content": slug_module }),
        },
        MockResponse::ToolCall {
            name: "repl".to_string(),
            args: serde_json::json!({
                "language": "python",
                "code": "from pathlib import Path\nns = {}\nexec(Path('slug.py').read_text(), ns)\nprint(ns['slugify']('Hello, Local AI!'))"
            }),
        },
        MockResponse::Text("Added slugify helper and verified it in Python.".to_string()),
    ]);

    let mut conv = build_scenario_loop(dir.path(), mock);
    let mut cb = ScenarioCallback::new();
    conv.run_turn("Create and verify a slug helper", &mut cb)
        .unwrap();

    assert!(dir.path().join(".zipcode-todos.json").is_file());
    assert!(dir.path().join("slug.py").is_file());
    assert!(cb.result_for("repl").contains("hello-local-ai"));
    assert_eq!(cb.tool_calls, ["todo_write", "write_file", "repl"]);
}

#[test]
fn path_traversal_safety_blocks_secret_read_and_write() {
    let parent = TempDir::new().unwrap();
    let workspace = parent.path().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let outside_secret = parent.path().join("outside-secret.txt");
    let outside_write = parent.path().join("owned.txt");
    let secret = "ZIPCODE_TEST_SECRET=redacted-test-token";
    std::fs::write(&outside_secret, secret).unwrap();

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "read_file".to_string(),
            args: serde_json::json!({ "path": "../outside-secret.txt" }),
        },
        MockResponse::ToolCall {
            name: "write_file".to_string(),
            args: serde_json::json!({ "path": "../owned.txt", "content": "bad" }),
        },
        MockResponse::Text("Workspace boundary blocked the attempted access.".to_string()),
    ]);

    let mut conv = build_scenario_loop(&workspace, mock);
    let mut cb = ScenarioCallback::new();
    conv.run_turn("Try to read and write outside the workspace", &mut cb)
        .unwrap();

    let combined_results = cb
        .tool_results
        .iter()
        .map(|(_, result)| result.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    assert!(combined_results.contains("outside the workspace"));
    assert!(!combined_results.contains(secret));
    assert!(!cb.all_tokens().contains(secret));
    assert!(!outside_write.exists());
}

#[test]
fn local_fixture_download_is_fetched_read_and_summarized() {
    let dir = TempDir::new().unwrap();
    let fixture = "zipcode fixture download\nversion: local-only\nsize: tiny\n";
    let (url, server) = serve_once(fixture);
    let download_command = format!(
        "python3 -c \"import urllib.request; urllib.request.urlretrieve('{url}', 'sample.txt')\""
    );

    let mock = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({ "command": download_command }),
        },
        MockResponse::ToolCall {
            name: "read_file".to_string(),
            args: serde_json::json!({ "path": "sample.txt" }),
        },
        MockResponse::ToolCall {
            name: "write_file".to_string(),
            args: serde_json::json!({
                "path": "summary.md",
                "content": "# Download Summary\n\nFetched local-only zipcode fixture sample.txt.\n"
            }),
        },
        MockResponse::Text("Downloaded sample.txt and wrote summary.md.".to_string()),
    ]);

    let mut conv = build_scenario_loop(dir.path(), mock);
    let mut cb = ScenarioCallback::new();
    conv.run_turn("Download the local fixture and summarize it", &mut cb)
        .unwrap();
    let server_result = server.join().expect("fixture server should not panic");
    assert!(
        server_result.is_ok(),
        "fixture server should finish: {}",
        server_result.unwrap_err()
    );

    let sample = std::fs::read_to_string(dir.path().join("sample.txt")).unwrap();
    let summary = std::fs::read_to_string(dir.path().join("summary.md")).unwrap();

    assert_eq!(sample, fixture);
    assert!(summary.contains("local-only zipcode fixture"));
    assert_eq!(cb.tool_calls, ["bash", "read_file", "write_file"]);
    assert!(
        !cb.result_for("bash").contains("Exit code"),
        "download command should pass: {}",
        cb.result_for("bash")
    );
    assert!(cb.result_for("read_file").contains("version: local-only"));
}
