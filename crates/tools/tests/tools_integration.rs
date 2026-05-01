//! Integration tests for the `zipcode-tools` crate.
//!
//! This file covers cross-cutting behavior that is not well-exercised by the
//! inline unit tests in `src/lib.rs` and individual tool modules:
//!
//! - `ToolRegistry::create_filtered()` allowlist semantics
//! - `execute_tool()` end-to-end with real tool implementations
//! - Cross-tool workflows (write → grep, write → glob)
//! - `PermissionMode` serde edge cases (case sensitivity, unknown strings)
//! - `ToolResult::truncate()` at byte boundaries with mixed content
//! - `recover_duplicated_workspace_prefix()` with varied directory layouts
//! - `resolve_and_validate_path()` with symlinks and edge-case paths

use tempfile::TempDir;
use zipcode_tools::agent::AgentTool;
use zipcode_tools::bash::BashTool;
use zipcode_tools::edit_file::EditFileTool;
use zipcode_tools::glob_search::GlobSearchTool;
use zipcode_tools::grep_search::GrepSearchTool;
use zipcode_tools::read_file::ReadFileTool;
use zipcode_tools::repl::ReplTool;
use zipcode_tools::todo_write::TodoWriteTool;
use zipcode_tools::tool_search::ToolSearchTool;
use zipcode_tools::write_file::WriteFileTool;
use zipcode_tools::{
    execute_tool, make_relative_path, recover_duplicated_workspace_prefix,
    resolve_and_validate_path, validate_glob_pattern, PermissionMode, ToolContext, ToolRegistry,
    ToolResult,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A minimal `ToolContext` for integration tests.
fn test_ctx(cwd: &std::path::Path) -> ToolContext {
    ToolContext {
        cwd: cwd.to_path_buf(),
        permission: PermissionMode::FullAccess,
        session_id: "integration-test".to_string(),
        parent_session_id: None,
        depth: 0,
        budget_tokens: None,
        spawn_child: None,
    }
}

/// Build a registry with all 10 built-in tools registered.
fn full_registry() -> ToolRegistry {
    let mut r = ToolRegistry::new();
    r.register(Box::new(BashTool));
    r.register(Box::new(ReadFileTool));
    r.register(Box::new(WriteFileTool));
    r.register(Box::new(EditFileTool));
    r.register(Box::new(GlobSearchTool));
    r.register(Box::new(GrepSearchTool));
    r.register(Box::new(TodoWriteTool));
    r.register(Box::new(ReplTool));
    r.register(Box::new(AgentTool));
    // ToolSearchTool needs registry specs for search functionality
    let search = ToolSearchTool::from_registry(&r);
    r.register(Box::new(search));
    r
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. ToolRegistry::create_filtered()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn create_filtered_includes_only_allowlisted_tools() {
    let registry = full_registry();
    let filtered = registry.create_filtered(&["read_file".to_string(), "grep_search".to_string()]);
    let mut names = filtered.names();
    names.sort_unstable();
    assert_eq!(names, vec!["grep_search", "read_file"]);
}

#[test]
fn create_filtered_empty_allowlist_includes_nothing() {
    let registry = full_registry();
    let filtered = registry.create_filtered(&[]);
    assert!(
        filtered.names().is_empty(),
        "empty allowlist should produce an empty registry"
    );
}

#[test]
fn create_filtered_ignores_unknown_tool_names() {
    let registry = full_registry();
    let filtered = registry.create_filtered(&[
        "read_file".to_string(),
        "nonexistent_tool".to_string(),
        "another_fake".to_string(),
    ]);
    let mut names = filtered.names();
    names.sort_unstable();
    assert_eq!(names, vec!["read_file"]);
}

#[test]
fn create_filtered_always_excludes_agent_tool() {
    let registry = full_registry();
    let filtered = registry.create_filtered(&["agent".to_string(), "bash".to_string()]);
    let mut names = filtered.names();
    names.sort_unstable();
    assert_eq!(names, vec!["bash"]);
}

#[test]
fn create_filtered_is_case_sensitive() {
    let registry = full_registry();
    let filtered = registry.create_filtered(&["Read_File".to_string(), "BASH".to_string()]);
    assert!(
        filtered.names().is_empty(),
        "allowlist matching should be case-sensitive"
    );
}

#[test]
fn create_filtered_preserves_tool_behavior() {
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());

    let registry = full_registry();
    let filtered = registry.create_filtered(&["bash".to_string()]);

    // The filtered bash tool should still work correctly
    let result = execute_tool(
        &filtered,
        "bash",
        serde_json::json!({"command": "echo hello"}),
        &ctx,
    )
    .unwrap();
    assert!(result.content.contains("hello"));
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. execute_tool() end-to-end
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn execute_tool_unknown_name_returns_error() {
    let registry = full_registry();
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let result = execute_tool(&registry, "does_not_exist", serde_json::json!({}), &ctx);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Unknown tool"),
        "error should mention 'Unknown tool': {err}"
    );
}

#[test]
fn execute_tool_bash_echo_end_to_end() {
    let registry = full_registry();
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let result = execute_tool(
        &registry,
        "bash",
        serde_json::json!({"command": "echo hello_world"}),
        &ctx,
    )
    .unwrap();
    assert!(result.content.contains("hello_world"));
    assert!(!result.truncated);
}

#[test]
fn execute_tool_write_then_read_roundtrip() {
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let registry = full_registry();

    // Write a file
    let write_result = execute_tool(
        &registry,
        "write_file",
        serde_json::json!({
            "path": "subdir/test.txt",
            "content": "line 1\nline 2\nline 3"
        }),
        &ctx,
    )
    .unwrap();
    assert!(!write_result.truncated);

    // Read it back
    let read_result = execute_tool(
        &registry,
        "read_file",
        serde_json::json!({"path": "subdir/test.txt"}),
        &ctx,
    )
    .unwrap();
    assert!(read_result.content.contains("line 1"));
    assert!(read_result.content.contains("line 2"));
    assert!(read_result.content.contains("line 3"));
}

#[test]
fn execute_tool_truncates_large_output() {
    // Generate output larger than MAX_TOOL_OUTPUT_BYTES (8192)
    let registry = full_registry();
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());

    let big_output = "x".repeat(20_000);
    let result = execute_tool(
        &registry,
        "bash",
        serde_json::json!({"command": format!("echo '{}'", big_output)}),
        &ctx,
    )
    .unwrap();

    // The result should be truncated and marked
    assert!(
        result.content.len() <= 8192,
        "output should be truncated to ≤8192 bytes, got {}",
        result.content.len()
    );
    assert!(result.truncated, "truncated flag should be set");
    assert!(
        result.content.contains("[truncated:"),
        "truncation marker should be present"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Cross-tool workflows
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn write_then_glob_finds_file() {
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let registry = full_registry();

    // Write multiple files
    execute_tool(
        &registry,
        "write_file",
        serde_json::json!({"path": "src/main.rs", "content": "fn main() {}"}),
        &ctx,
    )
    .unwrap();
    execute_tool(
        &registry,
        "write_file",
        serde_json::json!({"path": "src/lib.rs", "content": "// lib"}),
        &ctx,
    )
    .unwrap();
    execute_tool(
        &registry,
        "write_file",
        serde_json::json!({"path": "README.md", "content": "# hello"}),
        &ctx,
    )
    .unwrap();

    // Glob for *.rs files
    let glob_result = execute_tool(
        &registry,
        "glob_search",
        serde_json::json!({"pattern": "**/*.rs"}),
        &ctx,
    )
    .unwrap();

    assert!(glob_result.content.contains("main.rs"));
    assert!(glob_result.content.contains("lib.rs"));
    assert!(
        !glob_result.content.contains("README.md"),
        "should not match .md files"
    );
}

#[test]
fn write_then_grep_finds_content() {
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let registry = full_registry();

    execute_tool(
        &registry,
        "write_file",
        serde_json::json!({"path": "hello.txt", "content": "hello world\nfoo bar\nhello Rust"}),
        &ctx,
    )
    .unwrap();

    let grep_result = execute_tool(
        &registry,
        "grep_search",
        serde_json::json!({"pattern": "hello"}),
        &ctx,
    )
    .unwrap();

    assert!(grep_result.content.contains("hello world"));
    assert!(grep_result.content.contains("hello Rust"));
    assert!(
        !grep_result.content.contains("foo bar"),
        "should not match non-matching lines"
    );
}

#[test]
fn write_then_edit_then_read_reflects_change() {
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let registry = full_registry();

    execute_tool(
        &registry,
        "write_file",
        serde_json::json!({"path": "config.toml", "content": "version = \"1.0.0\"\ndebug = false"}),
        &ctx,
    )
    .unwrap();

    let edit_result = execute_tool(
        &registry,
        "edit_file",
        serde_json::json!({
            "path": "config.toml",
            "old_string": "debug = false",
            "new_string": "debug = true"
        }),
        &ctx,
    )
    .unwrap();
    assert!(
        edit_result.content.contains("edited"),
        "edit should report success, got: {}",
        edit_result.content
    );

    let read_result = execute_tool(
        &registry,
        "read_file",
        serde_json::json!({"path": "config.toml"}),
        &ctx,
    )
    .unwrap();
    assert!(read_result.content.contains("debug = true"));
    assert!(
        !read_result.content.contains("debug = false"),
        "old content should be gone"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. PermissionMode serde edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn permission_mode_rejects_uppercase_variant() {
    let result = serde_json::from_str::<PermissionMode>("\"READ-ONLY\"");
    assert!(result.is_err(), "uppercase variant should be rejected");
}

#[test]
fn permission_mode_rejects_camel_case_variant() {
    let result = serde_json::from_str::<PermissionMode>("\"workspaceWrite\"");
    assert!(result.is_err(), "camelCase variant should be rejected");
}

#[test]
fn permission_mode_rejects_empty_string() {
    let result = serde_json::from_str::<PermissionMode>("\"\"");
    assert!(result.is_err(), "empty string should be rejected");
}

#[test]
fn permission_mode_rejects_integer() {
    let result = serde_json::from_str::<PermissionMode>("42");
    assert!(result.is_err(), "integer should be rejected");
}

#[test]
fn permission_mode_rejects_null() {
    let result = serde_json::from_str::<PermissionMode>("null");
    assert!(result.is_err(), "null should be rejected");
}

#[test]
fn permission_mode_all_three_variants_roundtrip() {
    let variants = [
        ("read-only", PermissionMode::ReadOnly),
        ("workspace-write", PermissionMode::WorkspaceWrite),
        ("full-access", PermissionMode::FullAccess),
    ];
    for (json_str, expected) in variants {
        let deserialized: PermissionMode =
            serde_json::from_str(&format!("\"{json_str}\"")).unwrap();
        assert_eq!(deserialized, expected);
        let serialized = serde_json::to_string(&expected).unwrap();
        assert_eq!(serialized, format!("\"{json_str}\""));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. ToolResult::truncate edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn truncate_at_exact_boundary_no_truncation() {
    let content = "A".repeat(100);
    let result = ToolResult::new(content).truncate(100);
    assert!(!result.truncated);
    assert_eq!(result.content.len(), 100);
}

#[test]
fn truncate_one_byte_over_truncates() {
    let content = "A".repeat(101);
    let result = ToolResult::new(content).truncate(100);
    assert!(result.truncated);
    assert!(
        result.content.len() <= 100,
        "result should be ≤100 bytes, got {}",
        result.content.len()
    );
}

#[test]
fn truncate_with_mixed_ascii_and_multibyte() {
    // Mix ASCII and Korean (3-byte UTF-8) characters
    let content = format!("{}{}", "hello ".repeat(50), "안녕하세요".repeat(20));
    let result = ToolResult::new(content).truncate(200);
    // Content must be valid UTF-8
    assert!(std::str::from_utf8(result.content.as_bytes()).is_ok());
    if result.truncated {
        assert!(
            result.content.len() <= 200,
            "truncated result should be ≤200 bytes, got {}",
            result.content.len()
        );
    }
}

#[test]
fn truncate_error_result_does_not_corrupt_prefix() {
    let long_msg = "x".repeat(1000);
    let result = ToolResult::error(&long_msg);
    // "Error: " prefix is 7 bytes, so we need max > 7 for the prefix to survive.
    // Use 200 bytes — well above the 7-byte prefix, well below the 1007-byte content.
    let truncated = result.truncate(200);
    assert!(truncated.content.starts_with("Error: "));
    assert!(truncated.content.contains("[truncated:"));
    assert!(truncated.truncated);
}

#[test]
fn truncate_result_with_only_newlines() {
    let content = "\n".repeat(500);
    let result = ToolResult::new(content).truncate(100);
    if result.truncated {
        assert!(result.content.len() <= 100);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. resolve_and_validate_path edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn resolve_rejects_double_dot_at_start() {
    let dir = TempDir::new().unwrap();
    let result = resolve_and_validate_path("../../etc/shadow", dir.path());
    assert!(result.is_err(), "path traversal should be blocked");
}

#[test]
fn resolve_rejects_hidden_traversal() {
    let dir = TempDir::new().unwrap();
    let result = resolve_and_validate_path("foo/../../etc/passwd", dir.path());
    assert!(
        result.is_err(),
        "traversal hidden inside subdirectory should be blocked"
    );
}

#[test]
fn resolve_accepts_deep_nested_within_workspace() {
    let dir = TempDir::new().unwrap();
    let result = resolve_and_validate_path("a/b/c/d/e/f/g/h/file.txt", dir.path());
    assert!(
        result.is_ok(),
        "deep but safe nested path should be accepted"
    );
    assert_eq!(result.unwrap(), dir.path().join("a/b/c/d/e/f/g/h/file.txt"));
}

#[test]
fn resolve_accepts_dot_slash_prefix() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("test.txt"), "hello").unwrap();
    let result = resolve_and_validate_path("./test.txt", dir.path());
    assert!(result.is_ok());
}

#[test]
fn resolve_with_existing_symlink_within_workspace() {
    let dir = TempDir::new().unwrap();
    let real_file = dir.path().join("real.txt");
    std::fs::write(&real_file, "hello").unwrap();

    #[cfg(unix)]
    {
        let link_path = dir.path().join("link.txt");
        std::os::unix::fs::symlink(&real_file, &link_path).unwrap();
        let result = resolve_and_validate_path("link.txt", dir.path());
        assert!(result.is_ok(), "symlink within workspace should resolve");
    }
}

#[test]
fn resolve_rejects_symlink_escaping_workspace() {
    let dir = TempDir::new().unwrap();
    #[cfg(unix)]
    {
        let link_path = dir.path().join("escape.txt");
        // Symlink to /etc/passwd (outside workspace)
        std::os::unix::fs::symlink("/etc/passwd", &link_path).unwrap();
        let result = resolve_and_validate_path("escape.txt", dir.path());
        assert!(
            result.is_err(),
            "symlink pointing outside workspace should be rejected"
        );
    }
}

#[test]
fn resolve_rejects_absolute_path_outside_workspace() {
    let dir = TempDir::new().unwrap();
    let result = resolve_and_validate_path("/etc/passwd", dir.path());
    assert!(
        result.is_err(),
        "absolute path outside workspace should be rejected"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. recover_duplicated_workspace_prefix edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn recover_returns_none_for_absolute_path() {
    let dir = TempDir::new().unwrap();
    let workspace = dir.path().join("myproject");
    std::fs::create_dir_all(&workspace).unwrap();
    assert!(recover_duplicated_workspace_prefix("/absolute/path/file.txt", &workspace).is_none());
}

#[test]
fn recover_returns_none_when_first_component_differs_from_cwd_name() {
    let dir = TempDir::new().unwrap();
    let workspace = dir.path().join("myproject");
    std::fs::create_dir_all(&workspace).unwrap();
    assert!(recover_duplicated_workspace_prefix("other_project/README.md", &workspace).is_none());
}

#[test]
fn recover_returns_none_when_only_workspace_name_given() {
    let dir = TempDir::new().unwrap();
    let workspace = dir.path().join("myproject");
    std::fs::create_dir_all(&workspace).unwrap();
    // Just "myproject" without a remainder file — should return None
    assert!(recover_duplicated_workspace_prefix("myproject", &workspace).is_none());
}

#[test]
fn recover_returns_none_when_repaired_path_does_not_exist() {
    let dir = TempDir::new().unwrap();
    let workspace = dir.path().join("myproject");
    std::fs::create_dir_all(&workspace).unwrap();
    // "myproject/nonexistent.rs" — the file doesn't exist
    assert!(recover_duplicated_workspace_prefix("myproject/nonexistent.rs", &workspace).is_none());
}

#[test]
fn recover_succeeds_for_existing_file_under_duplicated_prefix() {
    let dir = TempDir::new().unwrap();
    let workspace = dir.path().join("zipcode");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::write(workspace.join("Cargo.toml"), "[package]").unwrap();

    let (relative, absolute) =
        recover_duplicated_workspace_prefix("zipcode/Cargo.toml", &workspace).unwrap();
    assert_eq!(relative, "Cargo.toml");
    assert_eq!(absolute, workspace.join("Cargo.toml"));
}

#[test]
fn recover_handles_nested_subdirectory() {
    let dir = TempDir::new().unwrap();
    let workspace = dir.path().join("project");
    let nested = workspace.join("src/bin");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(nested.join("main.rs"), "fn main() {}").unwrap();

    let (relative, absolute) =
        recover_duplicated_workspace_prefix("project/src/bin/main.rs", &workspace).unwrap();
    assert_eq!(relative, "src/bin/main.rs");
    assert_eq!(absolute, nested.join("main.rs"));
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. validate_glob_pattern edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn glob_allows_star_extension() {
    assert!(validate_glob_pattern("*.rs").is_ok());
}

#[test]
fn glob_allows_double_star_recursive() {
    assert!(validate_glob_pattern("src/**/*.rs").is_ok());
}

#[test]
fn glob_allows_question_mark_single_char() {
    assert!(validate_glob_pattern("test?.rs").is_ok());
}

#[test]
fn glob_allows_brace_expansion() {
    assert!(validate_glob_pattern("*.{rs,toml}").is_ok());
}

#[test]
fn glob_rejects_leading_double_dot() {
    assert!(validate_glob_pattern("../secrets/*").is_err());
}

#[test]
fn glob_rejects_double_dot_in_middle() {
    assert!(validate_glob_pattern("foo/../bar/*").is_err());
}

#[test]
fn glob_rejects_absolute_path_pattern() {
    assert!(validate_glob_pattern("/etc/passwd").is_err());
}

#[test]
fn glob_allows_plain_filename() {
    assert!(validate_glob_pattern("README.md").is_ok());
}

#[test]
fn glob_allows_deep_specific_path() {
    assert!(validate_glob_pattern("crates/tools/src/lib.rs").is_ok());
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. make_relative_path edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn make_relative_works_for_file_in_subdirectory() {
    let dir = TempDir::new().unwrap();
    let subdir = dir.path().join("src");
    std::fs::create_dir_all(&subdir).unwrap();
    let file = subdir.join("main.rs");
    std::fs::write(&file, "").unwrap();

    let relative = make_relative_path(&file, dir.path());
    assert_eq!(relative, "src/main.rs");
}

#[test]
fn make_relative_works_for_file_at_root() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("README.md");
    std::fs::write(&file, "hello").unwrap();

    let relative = make_relative_path(&file, dir.path());
    assert_eq!(relative, "README.md");
}

#[test]
fn make_relative_deeply_nested() {
    let dir = TempDir::new().unwrap();
    let deep = dir.path().join("a/b/c/d");
    std::fs::create_dir_all(&deep).unwrap();
    let file = deep.join("file.txt");
    std::fs::write(&file, "").unwrap();

    let relative = make_relative_path(&file, dir.path());
    assert_eq!(relative, "a/b/c/d/file.txt");
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. ToolSpec / registry introspection
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn full_registry_has_10_tools() {
    let registry = full_registry();
    let mut names = registry.names();
    names.sort_unstable();
    assert_eq!(names.len(), 10, "should have exactly 10 registered tools");
    assert_eq!(
        names,
        vec![
            "agent",
            "bash",
            "edit_file",
            "glob_search",
            "grep_search",
            "read_file",
            "repl",
            "todo_write",
            "tool_search",
            "write_file"
        ]
    );
}

#[test]
fn all_tools_have_valid_specs() {
    let registry = full_registry();
    let specs = registry.specs();
    assert_eq!(specs.len(), 10);

    for spec in &specs {
        assert!(!spec.name.is_empty(), "tool name should not be empty");
        assert!(
            !spec.description.is_empty(),
            "{} should have a description",
            spec.name
        );
        // parameters must be a valid JSON object
        assert!(
            spec.parameters.is_object(),
            "{} parameters_schema should be a JSON object",
            spec.name
        );
        assert!(
            spec.parameters["type"] == "object",
            "{} parameters_schema should have type=object",
            spec.name
        );
    }
}

#[test]
fn each_tool_name_matches_impl() {
    // Verify that each struct's name() matches the expected registration name
    let registry = full_registry();
    let expected = [
        "bash",
        "read_file",
        "write_file",
        "edit_file",
        "glob_search",
        "grep_search",
        "todo_write",
        "repl",
        "agent",
        "tool_search",
    ];
    for name in &expected {
        let tool = registry
            .get(name)
            .unwrap_or_else(|| panic!("{name} should be registered"));
        assert_eq!(tool.name(), *name);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. tool_search end-to-end via execute_tool()
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn tool_search_finds_bash_by_name() {
    let registry = full_registry();
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let result = execute_tool(
        &registry,
        "tool_search",
        serde_json::json!({ "query": "bash" }),
        &ctx,
    )
    .unwrap();
    assert!(
        result.content.contains("bash"),
        "should find bash, got: {}",
        result.content
    );
    assert!(
        result.content.contains("Execute a shell command"),
        "should include bash description, got: {}",
        result.content
    );
    assert!(
        !result.content.contains("read_file"),
        "should not include read_file, got: {}",
        result.content
    );
}

#[test]
fn tool_search_finds_by_description_keyword() {
    let registry = full_registry();
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let result = execute_tool(
        &registry,
        "tool_search",
        serde_json::json!({ "query": "regex" }),
        &ctx,
    )
    .unwrap();
    assert!(
        result.content.contains("grep_search"),
        "should find grep_search via 'regex' keyword, got: {}",
        result.content
    );
    assert!(
        !result.content.contains("bash"),
        "should not include bash, got: {}",
        result.content
    );
    assert!(
        !result.content.contains("write_file"),
        "should not include write_file, got: {}",
        result.content
    );
}

#[test]
fn tool_search_no_match_returns_not_found() {
    let registry = full_registry();
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let result = execute_tool(
        &registry,
        "tool_search",
        serde_json::json!({ "query": "nonexistent_tool_xyz" }),
        &ctx,
    )
    .unwrap();
    assert!(
        result.content.contains("No tools found"),
        "should report no matches, got: {}",
        result.content
    );
}

#[test]
fn tool_search_empty_query_rejected_via_execute_tool() {
    let registry = full_registry();
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let result = execute_tool(
        &registry,
        "tool_search",
        serde_json::json!({ "query": "" }),
        &ctx,
    );
    assert!(result.is_err(), "empty query should be rejected");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("empty"),
        "error should mention empty, got: {err}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. todo_write roundtrip end-to-end
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn todo_write_creates_json_file_and_bash_reads_it() {
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let registry = full_registry();

    // Write a todo list
    let write_result = execute_tool(
        &registry,
        "todo_write",
        serde_json::json!({
            "todos": [
                { "id": "1", "content": "test task alpha", "status": "pending" }
            ]
        }),
        &ctx,
    )
    .unwrap();
    assert!(
        !write_result.content.contains("Error"),
        "todo_write should succeed, got: {}",
        write_result.content
    );

    // Verify the file exists via bash
    let ls_result = execute_tool(
        &registry,
        "bash",
        serde_json::json!({ "command": "ls .zipcode-todos.json" }),
        &ctx,
    )
    .unwrap();
    assert!(
        ls_result.content.contains(".zipcode-todos.json"),
        "todo file should exist, got: {}",
        ls_result.content
    );

    // Read the file and verify content
    let read_result = execute_tool(
        &registry,
        "read_file",
        serde_json::json!({ "path": ".zipcode-todos.json" }),
        &ctx,
    )
    .unwrap();
    assert!(
        read_result.content.contains("test task alpha"),
        "file should contain the todo content, got: {}",
        read_result.content
    );
}

#[test]
fn todo_write_overwrite_updates_file() {
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let registry = full_registry();

    // Write initial todo with one item
    execute_tool(
        &registry,
        "todo_write",
        serde_json::json!({
            "todos": [
                { "id": "1", "content": "initial task", "status": "pending" }
            ]
        }),
        &ctx,
    )
    .unwrap();

    // Overwrite with two items
    execute_tool(
        &registry,
        "todo_write",
        serde_json::json!({
            "todos": [
                { "id": "1", "content": "initial task", "status": "completed" },
                { "id": "2", "content": "second task", "status": "pending" }
            ]
        }),
        &ctx,
    )
    .unwrap();

    // Verify both items are in the file
    let read_result = execute_tool(
        &registry,
        "read_file",
        serde_json::json!({ "path": ".zipcode-todos.json" }),
        &ctx,
    )
    .unwrap();
    assert!(
        read_result.content.contains("initial task"),
        "should contain first item, got: {}",
        read_result.content
    );
    assert!(
        read_result.content.contains("second task"),
        "should contain second item, got: {}",
        read_result.content
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. read_file with offset/limit end-to-end
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn read_file_with_offset_and_limit_via_execute_tool() {
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let registry = full_registry();

    // Create a 5-line file
    execute_tool(
        &registry,
        "write_file",
        serde_json::json!({
            "path": "multiline.txt",
            "content": "line one\nline two\nline three\nline four\nline five"
        }),
        &ctx,
    )
    .unwrap();

    // Read with offset=1, limit=2 (should show lines 2-3 only)
    let result = execute_tool(
        &registry,
        "read_file",
        serde_json::json!({
            "path": "multiline.txt",
            "offset": 1,
            "limit": 2
        }),
        &ctx,
    )
    .unwrap();

    assert!(
        result.content.contains("line two"),
        "should contain line two, got: {}",
        result.content
    );
    assert!(
        result.content.contains("line three"),
        "should contain line three, got: {}",
        result.content
    );
    assert!(
        !result.content.contains("line one"),
        "should NOT contain line one (before offset), got: {}",
        result.content
    );
    assert!(
        !result.content.contains("line four"),
        "should NOT contain line four (past limit), got: {}",
        result.content
    );
    assert!(
        !result.content.contains("line five"),
        "should NOT contain line five (past limit), got: {}",
        result.content
    );
}

#[test]
fn read_file_full_content_no_offset() {
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let registry = full_registry();

    execute_tool(
        &registry,
        "write_file",
        serde_json::json!({
            "path": "full.txt",
            "content": "alpha\nbeta\ngamma"
        }),
        &ctx,
    )
    .unwrap();

    let result = execute_tool(
        &registry,
        "read_file",
        serde_json::json!({ "path": "full.txt" }),
        &ctx,
    )
    .unwrap();

    assert!(
        result.content.contains("alpha"),
        "should contain alpha, got: {}",
        result.content
    );
    assert!(
        result.content.contains("beta"),
        "should contain beta, got: {}",
        result.content
    );
    assert!(
        result.content.contains("gamma"),
        "should contain gamma, got: {}",
        result.content
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. bash creates file then glob/grep finds it
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bash_creates_file_then_glob_finds_it() {
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let registry = full_registry();

    // Use bash to create a file
    execute_tool(
        &registry,
        "bash",
        serde_json::json!({
            "command": "mkdir -p src && echo 'fn main() {}' > src/generated.rs"
        }),
        &ctx,
    )
    .unwrap();

    // Glob should find the generated file
    let glob_result = execute_tool(
        &registry,
        "glob_search",
        serde_json::json!({ "pattern": "src/*.rs" }),
        &ctx,
    )
    .unwrap();

    assert!(
        glob_result.content.contains("generated.rs"),
        "glob should find generated.rs, got: {}",
        glob_result.content
    );
}

#[test]
fn bash_creates_file_then_grep_finds_content() {
    let dir = TempDir::new().unwrap();
    let ctx = test_ctx(dir.path());
    let registry = full_registry();

    // Use bash to create a file with specific content
    execute_tool(
        &registry,
        "bash",
        serde_json::json!({
            "command": "echo 'SEARCH_TARGET_123' > marker.txt"
        }),
        &ctx,
    )
    .unwrap();

    // Grep should find the content
    let grep_result = execute_tool(
        &registry,
        "grep_search",
        serde_json::json!({ "pattern": "SEARCH_TARGET_123" }),
        &ctx,
    )
    .unwrap();

    assert!(
        grep_result.content.contains("marker.txt"),
        "grep should find marker.txt, got: {}",
        grep_result.content
    );
    assert!(
        grep_result.content.contains("SEARCH_TARGET_123"),
        "grep should show the search target, got: {}",
        grep_result.content
    );
}
