use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, ExitStatus};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use wait_timeout::ChildExt;

/// Resolve a path and ensure it stays within the workspace root.
/// Returns error if the resolved path escapes the workspace.
///
/// # Errors
///
/// Returns an error if the resolved path escapes the workspace directory.
pub fn resolve_and_validate_path(
    file_path: &str,
    cwd: &std::path::Path,
) -> anyhow::Result<std::path::PathBuf> {
    let p = std::path::Path::new(file_path);
    let resolved = if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    };

    let cwd_canonical = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    let canonical = canonicalize_even_if_missing(&resolved)?;

    if !canonical.starts_with(&cwd_canonical) {
        anyhow::bail!(
            "Path '{}' resolves to '{}' which is outside the workspace '{}'",
            file_path,
            canonical.display(),
            cwd_canonical.display()
        );
    }

    Ok(canonical)
}

/// Strip the `base` prefix from `path`, returning a relative path string.
///
/// Falls back to the original display if stripping fails (should not happen
/// when path validation is correct).  Canonicalizes both sides so that
/// non-canonical prefixes (e.g. symlinks) still match.
#[must_use]
pub fn make_relative_path(path: &std::path::Path, base: &std::path::Path) -> String {
    let canon_base = base.canonicalize().unwrap_or_else(|_| base.to_path_buf());
    path.canonicalize().map_or_else(
        |_| path.to_string_lossy().into_owned(),
        |canon_path| {
            canon_path.strip_prefix(&canon_base).map_or_else(
                |_| path.to_string_lossy().into_owned(),
                |rel| rel.to_string_lossy().into_owned(),
            )
        },
    )
}

#[must_use]
pub fn recover_duplicated_workspace_prefix(
    file_path: &str,
    cwd: &std::path::Path,
) -> Option<(String, PathBuf)> {
    let requested = std::path::Path::new(file_path);
    if requested.is_absolute() {
        return None;
    }

    let cwd_name = cwd.file_name()?;
    let mut components = requested.components();
    let first = components.next()?;
    let Component::Normal(first_segment) = first else {
        return None;
    };
    if first_segment != cwd_name {
        return None;
    }

    let remainder = components.as_path();
    if remainder.as_os_str().is_empty() {
        return None;
    }

    let repaired_relative = remainder.to_string_lossy().to_string();
    let repaired_absolute = cwd.join(remainder);
    repaired_absolute
        .exists()
        .then_some((repaired_relative, repaired_absolute))
}

fn canonicalize_even_if_missing(path: &std::path::Path) -> anyhow::Result<PathBuf> {
    if path.exists() {
        return path.canonicalize().map_err(Into::into);
    }

    let mut suffix = Vec::new();
    let mut ancestor = path;

    while !ancestor.exists() {
        let Some(name) = ancestor.file_name() else {
            break;
        };
        suffix.push(name.to_os_string());
        ancestor = ancestor
            .parent()
            .ok_or_else(|| anyhow::anyhow!("failed to resolve path '{}'", path.display()))?;
    }

    let mut canonical = ancestor.canonicalize()?;
    for segment in suffix.iter().rev() {
        canonical.push(segment);
    }

    Ok(normalize_path(&canonical))
}

/// Validate that a glob pattern stays within the workspace.
/// Rejects absolute paths and any component that escapes the workspace.
///
/// # Errors
///
/// Returns an error if the pattern is absolute or contains path-traversal components.
pub fn validate_glob_pattern(pattern: &str) -> Result<()> {
    let path = Path::new(pattern);
    if path.is_absolute() {
        anyhow::bail!("Glob pattern must stay within the workspace");
    }

    if path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        anyhow::bail!("Glob pattern must not escape the workspace");
    }

    Ok(())
}

fn normalize_path(path: &std::path::Path) -> PathBuf {
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
        }
    }

    normalized
}

fn send_signal_to_process_tree(root_pid: u32, signal: &str) {
    let pids = process_tree_pids(root_pid);
    if pids.is_empty() {
        return;
    }

    let kill_bin = ["/bin/kill", "/usr/bin/kill"]
        .into_iter()
        .map(Path::new)
        .find(|path| path.is_file());

    if let Some(kill_bin) = kill_bin {
        let mut args = Vec::with_capacity(pids.len() + 1);
        args.push(signal.to_string());
        args.extend(pids.iter().map(u32::to_string));
        let _ = std::process::Command::new(kill_bin).args(&args).status();
    }
}

pub(crate) fn terminate_child_tree(child: &mut Child) {
    let root_pid = child.id();

    for signal in ["-TERM", "-KILL"] {
        send_signal_to_process_tree(root_pid, signal);

        if wait_for_exit(child, Duration::from_millis(250)) {
            return;
        }
    }

    let _ = child.kill();
    let _ = child.wait();
}

pub(crate) struct CollectedChildOutput {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

fn spawn_pipe_reader<T: Read + Send + 'static>(
    mut pipe: T,
) -> JoinHandle<std::io::Result<Vec<u8>>> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        pipe.read_to_end(&mut buf)?;
        Ok(buf)
    })
}

fn join_pipe_reader(
    handle: Option<JoinHandle<std::io::Result<Vec<u8>>>>,
    stream_name: &str,
) -> Result<Vec<u8>> {
    let Some(handle) = handle else {
        return Ok(Vec::new());
    };

    handle
        .join()
        .map_err(|_| anyhow::anyhow!("failed to join {stream_name} reader thread"))?
        .with_context(|| format!("failed to read child {stream_name}"))
}

pub(crate) fn wait_with_output_timeout(
    mut child: Child,
    timeout: Duration,
) -> Result<Option<CollectedChildOutput>> {
    let stdout_reader = child.stdout.take().map(spawn_pipe_reader);
    let stderr_reader = child.stderr.take().map(spawn_pipe_reader);

    let Some(status) = child.wait_timeout(timeout)? else {
        terminate_child_tree(&mut child);
        let _ = join_pipe_reader(stdout_reader, "stdout");
        let _ = join_pipe_reader(stderr_reader, "stderr");
        return Ok(None);
    };

    Ok(Some(CollectedChildOutput {
        status,
        stdout: join_pipe_reader(stdout_reader, "stdout")?,
        stderr: join_pipe_reader(stderr_reader, "stderr")?,
    }))
}

fn process_tree_pids(root_pid: u32) -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut stack = vec![root_pid];
    let mut ordered = Vec::new();

    while let Some(pid) = stack.pop() {
        if !seen.insert(pid) {
            continue;
        }
        ordered.push(pid);
        stack.extend(read_child_pids(pid));
    }

    ordered.reverse();
    ordered
}

fn read_child_pids(pid: u32) -> Vec<u32> {
    let path = format!("/proc/{pid}/task/{pid}/children");
    let Ok(children) = std::fs::read_to_string(path) else {
        return Vec::new();
    };

    children
        .split_whitespace()
        .filter_map(|value| value.parse::<u32>().ok())
        .collect()
}

fn wait_for_exit(child: &mut Child, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return true,
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Ok(None) | Err(_) => return false,
        }
    }
}

pub mod agent;
pub mod bash;
pub mod edit_file;
pub mod glob_search;
pub mod grep_search;
pub mod read_file;
pub mod repl;
pub mod todo_write;
pub mod tool_search;
pub mod write_file;

/// Permission modes for tool execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionMode {
    ReadOnly,
    WorkspaceWrite,
    FullAccess,
}

/// Result returned when a child agent completes.
#[derive(Debug, Clone)]
pub struct ChildResult {
    pub summary: String,
    pub tool_call_count: usize,
    pub child_session_id: String,
}

/// Callback type for spawning a child agent from within a tool.
///
/// Parameters: task_prompt, tool_allowlist, permission_override, max_tokens
pub type SpawnChildFn = dyn Fn(
        &str,
        Option<&[String]>,
        Option<PermissionMode>,
        Option<usize>,
    ) -> anyhow::Result<ChildResult>
    + Send
    + Sync;

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("cwd", &self.cwd)
            .field("permission", &self.permission)
            .field("session_id", &self.session_id)
            .field("parent_session_id", &self.parent_session_id)
            .field("depth", &self.depth)
            .field("budget_tokens", &self.budget_tokens)
            .field("spawn_child", &self.spawn_child.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

/// Context passed to every tool execution
#[derive(Clone)]
pub struct ToolContext {
    pub cwd: PathBuf,
    pub permission: PermissionMode,
    pub session_id: String,
    /// Parent session ID when running inside a sub-agent spawned from
    /// another session. None at the top level.
    pub parent_session_id: Option<String>,
    /// Nesting depth of the owning conversation. 0 = top-level user
    /// session, 1 = sub-agent; deeper nesting is reserved.
    pub depth: u32,
    /// Soft token budget for the owning conversation. None means no
    /// explicit cap (inherit from parent or use provider default).
    pub budget_tokens: Option<usize>,
    /// Callback injected by the runtime to spawn a child conversation loop.
    /// None when the tool is executed outside a full ConversationLoop context
    /// (e.g., unit tests, CLI one-shot mode).
    pub spawn_child: Option<Arc<SpawnChildFn>>,
}

/// Result from a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: String,
    pub truncated: bool,
}

impl ToolResult {
    #[must_use]
    pub const fn new(content: String) -> Self {
        Self {
            content,
            truncated: false,
        }
    }

    #[must_use]
    pub fn error(msg: &str) -> Self {
        Self {
            content: format!("Error: {msg}"),
            truncated: false,
        }
    }

    #[must_use]
    pub fn truncate(self, max_bytes: usize) -> Self {
        // Pre-compute the suffix template to measure its length exactly,
        // then reserve space so the final string stays within max_bytes.
        // Worst-case suffix length: "\n\n[truncated: showing first {max_digits} bytes of {total_digits}]"
        // where max_digits ≤ total_digits ≤ 10 (for up to 10 billion bytes).
        // Using a conservative fixed bound avoids a double-format allocation.
        const SUFFIX_OVERHEAD: usize = 80; // "\n\n[truncated: showing first … bytes of …]" worst case

        if self.content.len() <= max_bytes {
            return self;
        }

        let available = max_bytes.saturating_sub(SUFFIX_OVERHEAD);
        // Find a safe char boundary at or before `available`
        let mut end = available;
        while end > 0 && !self.content.is_char_boundary(end) {
            end -= 1;
        }
        let suffix = format!(
            "\n\n[truncated: showing first {} bytes of {}]",
            end,
            self.content.len()
        );
        let truncated_content = format!("{}{}", &self.content[..end], suffix);
        Self {
            content: truncated_content,
            truncated: true,
        }
    }
}

/// Spec for a single tool — used to inject tool schemas into the model prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Trait every tool must implement
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn parameters_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given arguments and context.
    ///
    /// # Errors
    ///
    /// Returns an error if the tool execution fails.
    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult>;
}

/// Registry holding all available tools
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(AsRef::as_ref)
    }

    #[must_use]
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .map(|t| ToolSpec {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters_schema(),
            })
            .collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_TOOL_OUTPUT_BYTES: usize = 8192;

/// Execute a tool by name with automatic truncation.
///
/// # Errors
///
/// Returns an error if the tool is not found or if tool execution fails.
pub fn execute_tool(
    registry: &ToolRegistry,
    name: &str,
    args: serde_json::Value,
    ctx: &ToolContext,
) -> Result<ToolResult> {
    let tool = registry
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Unknown tool: {name}"))?;
    let result = tool.execute(args, ctx)?;
    Ok(result.truncate(MAX_TOOL_OUTPUT_BYTES))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool;

    impl Tool for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn description(&self) -> &'static str {
            "Echoes input"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string" }
                },
                "required": ["text"]
            })
        }
        fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> Result<ToolResult> {
            let text = args["text"].as_str().unwrap_or("");
            Ok(ToolResult::new(text.to_string()))
        }
    }

    fn test_ctx() -> ToolContext {
        ToolContext {
            cwd: std::path::PathBuf::from("/tmp"),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
            parent_session_id: None,
            depth: 0,
            budget_tokens: None,
            spawn_child: None,
        }
    }

    #[test]
    fn test_registry_add_and_get() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        assert!(registry.get("echo").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_specs() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        let specs = registry.specs();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "echo");
    }

    #[test]
    fn test_tool_result_truncation() {
        let long_content = "x".repeat(10_000);
        let result = ToolResult::new(long_content);
        let truncated = result.truncate(8192);
        // The final content must stay within max_bytes (not max_bytes + suffix overhead)
        assert!(
            truncated.content.len() <= 8192,
            "truncated content should not exceed max_bytes, got {} bytes",
            truncated.content.len()
        );
        assert!(truncated.truncated);
    }

    #[test]
    fn test_truncate_never_exceeds_max_bytes() {
        // Regression test: the old implementation could exceed max_bytes by ~60
        // bytes because it appended the suffix after truncating at max_bytes.
        for max in [50, 100, 200, 500, 1000, 8192] {
            let content = "x".repeat(max * 3);
            let result = ToolResult::new(content).truncate(max);
            assert!(
                result.content.len() <= max,
                "truncate({max}) produced {} bytes — must not exceed {max}",
                result.content.len()
            );
            assert!(result.truncated);
        }
    }

    #[test]
    fn test_tool_execution() {
        let tool = EchoTool;
        let ctx = test_ctx();
        let args = serde_json::json!({"text": "hello"});
        let result = tool.execute(args, &ctx).unwrap();
        assert_eq!(result.content, "hello");
    }

    #[test]
    fn test_execute_unknown_tool() {
        let registry = ToolRegistry::new();
        let ctx = test_ctx();
        let result = execute_tool(&registry, "unknown", serde_json::json!({}), &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_path_traversal_blocked() {
        let cwd = std::env::temp_dir();
        let result = resolve_and_validate_path("../../etc/passwd", &cwd);
        assert!(result.is_err());
    }

    #[test]
    fn test_valid_path_allowed() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("test.txt"), "hello").unwrap();
        let result = resolve_and_validate_path("test.txt", dir.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_path_traversal_blocked_for_missing_parent() {
        let dir = tempfile::TempDir::new().unwrap();
        let result = resolve_and_validate_path("../escape/new.txt", dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_nested_new_file_allowed() {
        let dir = tempfile::TempDir::new().unwrap();
        let result = resolve_and_validate_path("nested/dir/new.txt", dir.path()).unwrap();
        assert_eq!(result, dir.path().join("nested/dir/new.txt"));
    }

    #[test]
    fn test_recover_duplicated_workspace_prefix_finds_existing_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let workspace = dir.path().join("zipcode");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::write(workspace.join("README.md"), "hello").unwrap();

        let recovered =
            recover_duplicated_workspace_prefix("zipcode/README.md", &workspace).unwrap();
        assert_eq!(recovered.0, "README.md");
        assert_eq!(recovered.1, workspace.join("README.md"));
    }

    #[test]
    fn test_recover_duplicated_workspace_prefix_ignores_unrelated_paths() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(recover_duplicated_workspace_prefix("README.md", dir.path()).is_none());
        assert!(recover_duplicated_workspace_prefix("../README.md", dir.path()).is_none());
    }

    // ── validate_glob_pattern tests ──────────────────────────────────

    #[test]
    fn test_glob_valid_simple_wildcard() {
        assert!(validate_glob_pattern("*.rs").is_ok());
    }

    #[test]
    fn test_glob_valid_nested_pattern() {
        assert!(validate_glob_pattern("src/**/*.ts").is_ok());
    }

    #[test]
    fn test_glob_valid_single_filename() {
        assert!(validate_glob_pattern("main.rs").is_ok());
    }

    #[test]
    fn test_glob_valid_deeply_nested() {
        assert!(validate_glob_pattern("crates/tools/src/lib.rs").is_ok());
    }

    #[test]
    fn test_glob_valid_empty_pattern() {
        // Empty string has no components — nothing to escape through
        assert!(validate_glob_pattern("").is_ok());
    }

    #[test]
    fn test_glob_rejects_absolute_path() {
        assert!(validate_glob_pattern("/etc/passwd").is_err());
    }

    #[test]
    fn test_glob_rejects_parent_traversal() {
        assert!(validate_glob_pattern("../../etc/passwd").is_err());
    }

    #[test]
    fn test_glob_rejects_single_parent() {
        assert!(validate_glob_pattern("../secret").is_err());
    }

    #[test]
    fn test_glob_rejects_parent_in_middle() {
        assert!(validate_glob_pattern("foo/../bar/../../etc/passwd").is_err());
    }

    #[test]
    fn test_glob_rejects_trailing_parent() {
        assert!(validate_glob_pattern("foo/..").is_err());
    }

    // ── normalize_path tests ─────────────────────────────────────────

    #[test]
    fn test_normalize_removes_curdir() {
        assert_eq!(
            normalize_path(std::path::Path::new("foo/./bar")),
            PathBuf::from("foo/bar")
        );
    }

    #[test]
    fn test_normalize_resolves_parentdir() {
        assert_eq!(
            normalize_path(std::path::Path::new("foo/bar/../baz")),
            PathBuf::from("foo/baz")
        );
    }

    #[test]
    fn test_normalize_already_clean() {
        assert_eq!(
            normalize_path(std::path::Path::new("foo/bar/baz")),
            PathBuf::from("foo/bar/baz")
        );
    }

    #[test]
    fn test_normalize_mixed_dots() {
        assert_eq!(
            normalize_path(std::path::Path::new("foo/./bar/../baz/./qux")),
            PathBuf::from("foo/baz/qux")
        );
    }

    #[test]
    fn test_normalize_empty_path() {
        assert_eq!(normalize_path(std::path::Path::new("")), PathBuf::new());
    }

    #[test]
    fn test_normalize_leading_curdir() {
        assert_eq!(
            normalize_path(std::path::Path::new("./foo")),
            PathBuf::from("foo")
        );
    }

    #[test]
    fn test_normalize_double_parentdir() {
        assert_eq!(
            normalize_path(std::path::Path::new("a/b/c/../../d")),
            PathBuf::from("a/d")
        );
    }

    #[test]
    fn test_normalize_parentdir_beyond_root_relative() {
        // Popping beyond the start results in an empty PathBuf
        assert_eq!(
            normalize_path(std::path::Path::new("foo/../../bar")),
            PathBuf::from("bar")
        );
    }

    // ── ToolResult::truncate edge cases ──────────────────────────────

    #[test]
    fn test_truncate_exact_byte_boundary() {
        let content = "abcd".to_string(); // exactly 4 bytes
        let result = ToolResult::new(content).truncate(4);
        assert!(!result.truncated);
        assert_eq!(result.content, "abcd");
    }

    #[test]
    fn test_truncate_one_byte_over() {
        let content = "abcde".to_string(); // 5 bytes
        let result = ToolResult::new(content).truncate(100);
        // With max_bytes >> content, no truncation occurs
        assert!(!result.truncated);
        assert_eq!(result.content, "abcde");
    }

    #[test]
    fn test_truncate_small_overrun() {
        // Content slightly exceeds max_bytes — the suffix must still fit
        let content = "x".repeat(110);
        let result = ToolResult::new(content).truncate(100);
        assert!(result.truncated);
        assert!(
            result.content.len() <= 100,
            "result was {} bytes",
            result.content.len()
        );
        assert!(result.content.contains("[truncated:"));
    }

    #[test]
    fn test_truncate_splits_multibyte_utf8() {
        // "안녕" is 6 bytes (3+3 in UTF-8). Truncate at 200 bytes.
        // Content is only 6 bytes, so it should not be truncated.
        let content = "안녕".to_string();
        assert_eq!(content.len(), 6);
        let result = ToolResult::new(content).truncate(200);
        assert!(!result.truncated);
        assert_eq!(result.content, "안녕");
    }

    #[test]
    fn test_truncate_multibyte_utf8_actually_truncated() {
        // "안녕하세요" is 15 bytes (5×3). Truncate at 50 bytes.
        // The suffix overhead means we keep only the first few chars.
        let content = "안녕하세요".repeat(10); // 150 bytes
        let result = ToolResult::new(content).truncate(50);
        assert!(result.truncated);
        assert!(
            result.content.len() <= 50,
            "result was {} bytes",
            result.content.len()
        );
        assert!(result.content.contains("[truncated:"));
    }

    #[test]
    fn test_truncate_empty_content() {
        let result = ToolResult::new(String::new()).truncate(100);
        assert!(!result.truncated);
        assert!(result.content.is_empty());
    }

    #[test]
    fn test_truncate_very_large_ratio() {
        // 1 MB content truncated to 200 bytes — must fit within budget
        let content = "x".repeat(1_000_000);
        let result = ToolResult::new(content).truncate(200);
        assert!(result.truncated);
        assert!(
            result.content.len() <= 200,
            "result was {} bytes",
            result.content.len()
        );
        assert!(result.content.contains("1000000"));
    }

    #[test]
    fn test_truncate_preserves_content_under_limit() {
        let content = "hello world".to_string();
        let result = ToolResult::new(content.clone()).truncate(100);
        assert!(!result.truncated);
        assert_eq!(result.content, content);
    }

    #[test]
    fn test_truncate_zero_max() {
        // Edge case: truncate with max_bytes = 0
        let content = "hello".to_string();
        let result = ToolResult::new(content).truncate(0);
        // With 0 bytes, the suffix message still appears but no prefix
        assert!(result.truncated);
        assert!(result.content.contains("[truncated:"));
    }

    // ── ToolResult::error tests ──────────────────────────────────────

    #[test]
    fn test_error_result_has_prefix() {
        let result = ToolResult::error("something went wrong");
        assert!(result.content.starts_with("Error: "));
        assert!(result.content.contains("something went wrong"));
        assert!(!result.truncated);
    }

    #[test]
    fn test_error_result_empty_message() {
        let result = ToolResult::error("");
        assert_eq!(result.content, "Error: ");
        assert!(!result.truncated);
    }

    // ── ToolRegistry::names tests ────────────────────────────────────

    #[test]
    fn test_registry_names_empty() {
        let registry = ToolRegistry::new();
        assert!(registry.names().is_empty());
    }

    #[test]
    fn test_registry_names_after_register() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        let names = registry.names();
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "echo");
    }

    #[test]
    fn test_registry_overwrite_on_duplicate_name() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        registry.register(Box::new(EchoTool)); // same name overwrites
        assert_eq!(registry.names().len(), 1);
    }

    #[test]
    fn test_registry_default() {
        let registry = ToolRegistry::default();
        assert!(registry.names().is_empty());
    }

    // ── PermissionMode serde roundtrip tests ─────────────────────

    #[test]
    fn test_permission_mode_serde_roundtrip() {
        for (variant, expected_json) in [
            (PermissionMode::ReadOnly, "\"read-only\""),
            (PermissionMode::WorkspaceWrite, "\"workspace-write\""),
            (PermissionMode::FullAccess, "\"full-access\""),
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            assert_eq!(
                json, expected_json,
                "PermissionMode::{variant:?} serialization mismatch"
            );
            let back: PermissionMode = serde_json::from_str(&json).unwrap();
            assert_eq!(
                back, variant,
                "PermissionMode deserialization roundtrip failed for {variant:?}"
            );
        }
    }

    #[test]
    fn test_permission_mode_deserialize_rejects_danger_full_access() {
        // "danger-full-access" is NOT a serde variant — it's mapped only by
        // parse_permission_mode() in the runtime crate, not by serde.
        let result = serde_json::from_str::<PermissionMode>("\"danger-full-access\"");
        assert!(result.is_err(), "serde should reject 'danger-full-access'");
    }

    #[test]
    fn test_tool_result_new_not_truncated() {
        let result = ToolResult::new("hello".to_string());
        assert_eq!(result.content, "hello");
        assert!(!result.truncated);
    }

    #[test]
    fn test_tool_result_error_format() {
        let result = ToolResult::error("something failed");
        assert_eq!(result.content, "Error: something failed");
        assert!(!result.truncated);
    }

    #[test]
    fn test_tool_result_truncate_preserves_short_content() {
        let result = ToolResult::new("short".to_string());
        let truncated = result.truncate(100);
        assert_eq!(truncated.content, "short");
        assert!(!truncated.truncated);
    }

    #[test]
    fn test_tool_result_truncate_at_char_boundary() {
        // Korean characters are 3 bytes each in UTF-8.
        // Create content that is mostly multi-byte to force a mid-character boundary.
        let korean = "안녕하세요".repeat(200); // ~3000 bytes
        let result = ToolResult::new(korean);
        // Truncate to a size that will land in the middle of a multi-byte char
        let truncated = result.truncate(500);
        assert!(
            truncated.truncated,
            "content should be truncated for oversized Korean text"
        );
        // Must not panic — the content must be valid UTF-8
        assert!(std::str::from_utf8(truncated.content.as_bytes()).is_ok());
        assert!(
            truncated.content.len() <= 500,
            "truncated content must not exceed max_bytes, got {}",
            truncated.content.len()
        );
    }

    // ── make_relative_path tests ──────────────────────────────────

    #[test]
    fn test_make_relative_path_strips_cwd() {
        let dir = tempfile::TempDir::new().unwrap();
        let file = dir.path().join("src/main.rs");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, "").unwrap();
        let relative = make_relative_path(&file, dir.path());
        assert_eq!(relative, "src/main.rs");
    }

    #[test]
    fn test_make_relative_path_already_relative_fallback() {
        // When path doesn't start with base (shouldn't happen in normal flow),
        // falls back to original display
        let relative = make_relative_path(
            std::path::Path::new("/some/other/path.rs"),
            std::path::Path::new("/home/user/project"),
        );
        assert_eq!(relative, "/some/other/path.rs");
    }

    #[test]
    fn test_make_relative_path_file_at_root() {
        let dir = tempfile::TempDir::new().unwrap();
        let file = dir.path().join("README.md");
        std::fs::write(&file, "hello").unwrap();
        let relative = make_relative_path(&file, dir.path());
        assert_eq!(relative, "README.md");
    }
}
