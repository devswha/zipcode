# AGENTS.md - zipcode-tools

**Generated:** 2026-04-03  
**Purpose:** 10 built-in tool implementations behind the Tool trait  
**Parent:** ../AGENTS.md

---

## Crate Purpose

`zipcode-tools` provides the core tool system for the zipcode agent. It implements 10 built-in tools for file manipulation, shell execution, code search, and REPL interaction. Every tool conforms to the `Tool` trait and operates within a permission-based security boundary.

**Key Design:**
- Trait-based tool interface for extensibility
- `ToolRegistry` for centralized tool management
- Path traversal prevention via `resolve_and_validate_path()`
- Automatic output truncation at 8 KB
- Permission-based execution gating (ReadOnly, WorkspaceWrite, FullAccess)

---

## Architecture Overview

### Tool Trait

All tools implement the `Tool` trait:

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult>;
}
```

**Interface contract:**
- `name()` — unique identifier for tool lookup
- `description()` — brief human-readable description
- `parameters_schema()` — JSON Schema describing expected arguments
- `execute()` — main entry point; receives args + context

### ToolContext

Context passed to every tool execution:

```rust
pub struct ToolContext {
    pub cwd: PathBuf,              // Current working directory
    pub permission: PermissionMode, // Read-only, workspace-write, or full-access
    pub session_id: String,         // Session UUID for audit trails
}
```

### ToolResult

Result type with truncation support:

```rust
pub struct ToolResult {
    pub content: String,  // Tool output
    pub truncated: bool,  // True if output exceeded MAX_TOOL_OUTPUT_BYTES (8 KB)
}
```

**Methods:**
- `new(content)` — create result
- `error(msg)` — error result
- `truncate(max_bytes)` — truncate at safe char boundary

---

## Built-In Tools (10 Total)

| Tool Name | File | Description | Permission Required | Key Feature |
|-----------|------|-------------|---------------------|-------------|
| `bash` | bash.rs | Execute shell commands | FullAccess | Timeout support (120s default), captures stderr |
| `read_file` | read_file.rs | Read file with line numbers | ReadOnly | Offset/limit for partial reads, line-numbered output |
| `write_file` | write_file.rs | Write content, creates dirs | WorkspaceWrite | Auto-creates parent directories |
| `edit_file` | edit_file.rs | Replace unique string | WorkspaceWrite | Fails if target appears 0 or 2+ times |
| `glob_search` | glob_search.rs | Find files by pattern | ReadOnly | Sorted results, supports recursive patterns |
| `grep_search` | grep_search.rs | Search file contents | ReadOnly | Regex matching, skips binary files, line-numbered |
| `repl` | repl.rs | Execute code (python/node) | FullAccess | Python 3 or Node.js only |
| `todo_write` | todo_write.rs | Write todos to JSON | WorkspaceWrite | Persists to `.zipcode-todos.json` |
| `tool_search` | tool_search.rs | Search tool registry | ReadOnly | Keyword search by name/description |
| `agent` | agent.rs | Delegate to sub-agent | FullAccess | Stub; not yet implemented |

---

## Key Files

| File | Purpose | Exports |
|------|---------|---------|
| `lib.rs` | Trait definitions, registry, path validation | `Tool`, `ToolRegistry`, `ToolContext`, `ToolResult`, `resolve_and_validate_path()` |
| `bash.rs` | Shell command execution | `BashTool` struct |
| `read_file.rs` | File reading with line numbers | `ReadFileTool` struct |
| `write_file.rs` | File writing | `WriteFileTool` struct |
| `edit_file.rs` | String replacement in files | `EditFileTool` struct |
| `glob_search.rs` | Glob-based file search | `GlobSearchTool` struct |
| `grep_search.rs` | Regex-based content search | `GrepSearchTool` struct |
| `repl.rs` | Python/Node REPL execution | `ReplTool` struct |
| `todo_write.rs` | Todo JSON persistence | `TodoWriteTool` struct |
| `tool_search.rs` | Tool registry search | `ToolSearchTool` struct |
| `agent.rs` | Agent delegation stub | `AgentTool` struct |

---

## Tool Registry

The `ToolRegistry` manages all available tools:

```rust
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self { ... }
    pub fn register(&mut self, tool: Box<dyn Tool>) { ... }
    pub fn get(&self, name: &str) -> Option<&dyn Tool> { ... }
    pub fn specs(&self) -> Vec<ToolSpec> { ... }
    pub fn names(&self) -> Vec<&str> { ... }
}
```

**Public function:**
```rust
pub fn execute_tool(
    registry: &ToolRegistry,
    name: &str,
    args: serde_json::Value,
    ctx: &ToolContext,
) -> Result<ToolResult>
```

Returns error if tool not found. Automatically truncates output at `MAX_TOOL_OUTPUT_BYTES` (8192 bytes).

---

## Path Security

All file-manipulation tools use `resolve_and_validate_path()` to prevent directory traversal:

```rust
pub fn resolve_and_validate_path(
    file_path: &str,
    cwd: &std::path::Path,
) -> anyhow::Result<std::path::PathBuf>
```

**Security guarantees:**
1. Resolves relative paths against `cwd`
2. Resolves absolute paths as-is
3. Canonicalizes existing paths and parent dirs
4. **Fails if resolved path escapes `cwd`**

**Example:**
```rust
// Safe: resolves to cwd/foo.txt
resolve_and_validate_path("foo.txt", cwd)?

// Blocked: resolves outside cwd
resolve_and_validate_path("../../etc/passwd", cwd)?  // Error

// Safe: any relative path within cwd
resolve_and_validate_path("src/lib.rs", cwd)?
resolve_and_validate_path("./nested/file.txt", cwd)?
```

---

## Permission Model

| Mode | Tools Allowed | Use Case |
|------|---------------|----------|
| `ReadOnly` | read_file, glob_search, grep_search, tool_search | Safe browsing, no modifications |
| `WorkspaceWrite` | + write_file, edit_file, todo_write | Edit files within workspace |
| `FullAccess` | + bash, repl, agent | Execute arbitrary commands |

Permission enforcement is implemented at the runtime/inference layer (not in this crate).

---

## Tool Specifications

Each tool exposes a `ToolSpec` for inclusion in the model's system prompt:

```rust
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,  // JSON Schema
}
```

**Usage:**
```rust
let specs = registry.specs();
// Convert to JSON Schema for LLM context
```

---

## Important Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_TOOL_OUTPUT_BYTES` | 8192 | Output truncation limit (8 KB) |
| `BASH_DEFAULT_TIMEOUT` | 120_000 ms | Default shell command timeout |

---

## Tool Details

### bash.rs

**Execute shell commands with timeout support.**

Parameters:
- `command` (required): Shell command string
- `timeout` (optional): Milliseconds (default 120,000 = 2 minutes)

Behavior:
- Runs command via `bash -c`
- Captures stdout and stderr
- Returns exit code if non-zero
- Kills process on timeout
- Works in `ctx.cwd` directory

Test coverage: 4 tests
- Echo command
- Stderr capture
- Exit codes
- Working directory

---

### read_file.rs

**Read files with line numbers and optional offset/limit.**

Parameters:
- `path` (required): File path (relative to cwd)
- `offset` (optional): Starting line (0-based)
- `limit` (optional): Max lines to read

Output format: `{line_number}\t{line_content}` (1-based line numbering)

Test coverage: 3 tests
- Read full file
- Read with offset and limit
- Nonexistent file error

---

### write_file.rs

**Write content to file, auto-creating parent directories.**

Parameters:
- `path` (required): File path
- `content` (required): File content

Behavior:
- Creates parent directories if missing
- Overwrites existing file
- Returns success message with path

Test coverage: 2 tests
- Write new file
- Create nested directories

---

### edit_file.rs

**Replace a unique string occurrence in a file.**

Parameters:
- `path` (required): File path
- `old_string` (required): Exact string to find
- `new_string` (required): Replacement string

Validation:
- Fails if `old_string` appears 0 times → "not found"
- Fails if `old_string` appears 2+ times → "must be unique"
- Succeeds only if exactly 1 match

Test coverage: 3 tests
- Replace unique string
- Error on not found
- Error on duplicate

---

### glob_search.rs

**Find files matching glob pattern.**

Parameters:
- `pattern` (required): Glob pattern (e.g., `*.rs`, `**/*.ts`)
- `path` (optional): Base directory (defaults to cwd)

Output: Sorted file paths, one per line

Test coverage: 5 tests
- Find by extension
- No matches message
- Recursive patterns (`**/*`)
- Custom base path
- Result sorting

---

### grep_search.rs

**Search file contents with regex, skipping binaries.**

Parameters:
- `pattern` (required): Regex pattern
- `path` (optional): File or directory to search
- `glob` (optional): Filter files by glob pattern (when path is directory)

Output format: `filepath:line_num: line_content`

Behavior:
- Walks directory recursively
- Skips binary files (detects null bytes in first 8 KB)
- Silent error handling for unreadable files
- Case-sensitive by default (regex controls case)

Test coverage: 6 tests
- Regex matching in directory
- No matches message
- Single file search
- Glob filtering
- Output format verification
- Binary file skipping

---

### repl.rs

**Execute Python 3 or Node.js code.**

Parameters:
- `language` (required): `"python"` or `"node"`
- `code` (required): Code to execute

Behavior:
- Runs `python3 -c` or `node -e`
- Captures stdout and stderr separately
- Returns "(no output)" if empty

Test coverage: Minimal (unit tests not in file, but integration tested)

---

### todo_write.rs

**Write structured todo list to `.zipcode-todos.json`.**

Parameters:
- `todos` (required): Array of todo objects
  - `id` (string)
  - `content` (string)
  - `status` (string): e.g., "pending", "in_progress", "completed"

Output: `Wrote N todo(s) to {path}`

File format: Pretty-printed JSON

Test coverage: 3 tests
- Create new todo file
- Verify todo structure
- Write empty todos

---

### tool_search.rs

**Search registered tools by name or description keyword.**

Parameters:
- `query` (required): Keyword to search

Behavior:
- Case-insensitive search
- Matches tool name OR description
- Returns `{name}: {description}` format

Initialization:
```rust
// From specs
let tool = ToolSearchTool::from_specs(vec![...]);

// From registry
let tool = ToolSearchTool::from_registry(&registry);
```

Test coverage: 4 tests
- Search by name
- Search by description
- Case-insensitive matching
- No results message

---

### agent.rs

**Delegate task to sub-agent (stub implementation).**

Parameters:
- `task` (required): Task description

Current behavior: Returns "not yet implemented" message

Planned feature: Multi-agent orchestration for complex tasks

---

## Testing

**Total test count:** 37 tests (unit + integration)

**Testing patterns:**
- Use `tempfile` crate for file operations
- Use `tempdir()` for directory-based tests
- Create mock `ToolContext` with test cwd/permissions
- Verify both success and error paths

**Test organization:**
- Each tool file includes `#[cfg(test)]` module
- Tests use local `ctx()` helper for consistency
- All file tests use `tempfile::NamedTempFile` or `TempDir`

**Key test patterns:**

File operations:
```rust
let dir = TempDir::new().unwrap();
let tool = WriteFileTool;
let ctx = ToolContext {
    cwd: dir.path().to_path_buf(),
    permission: PermissionMode::FullAccess,
    session_id: "test".to_string(),
};
let args = serde_json::json!({ "path": "file.txt", "content": "test" });
let result = tool.execute(args, &ctx).unwrap();
```

Shell execution:
```rust
let ctx = ToolContext {
    cwd: PathBuf::from("/tmp"),
    permission: PermissionMode::FullAccess,
    session_id: "test".to_string(),
};
let args = serde_json::json!({ "command": "echo hello" });
let result = tool.execute(args, &ctx).unwrap();
assert!(result.content.contains("hello"));
```

---

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| `serde` | workspace | Serialization/deserialization |
| `serde_json` | workspace | JSON handling |
| `anyhow` | workspace | Error handling |
| `glob` | crates/tools | File globbing |
| `regex` | crates/tools | Regex pattern matching |
| `grep-regex` | crates/tools | Advanced grep support |
| `grep-searcher` | crates/tools | Efficient file searching |
| `wait-timeout` | crates/tools | Process timeout enforcement |
| `tempfile` | (dev) | Testing file operations |

---

## Adding New Tools

To add a new tool to the registry:

1. **Create a new file** `crates/tools/src/my_tool.rs`:
```rust
use crate::{Tool, ToolContext, ToolResult};
use anyhow::Result;
use serde_json::Value;

pub struct MyTool;

impl Tool for MyTool {
    fn name(&self) -> &str { "my_tool" }
    fn description(&self) -> &str { "Does something useful" }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "param": { "type": "string" }
            },
            "required": ["param"]
        })
    }
    fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let param = args["param"].as_str().ok_or_else(
            || anyhow::anyhow!("missing param")
        )?;
        // Implement logic
        Ok(ToolResult::new("result".to_string()))
    }
}
```

2. **Export in `lib.rs`**:
```rust
pub mod my_tool;
```

3. **Register in runtime** (crates/runtime or crates/cli):
```rust
let mut registry = ToolRegistry::new();
registry.register(Box::new(my_tool::MyTool));
```

4. **Add tests** in the same file:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_my_tool() {
        let tool = MyTool;
        let ctx = ToolContext { /* ... */ };
        let args = serde_json::json!({ "param": "value" });
        let result = tool.execute(args, &ctx).unwrap();
        assert!(result.content.contains("expected"));
    }
}
```

5. **Check permissions** — If tool needs specific permissions, document in tool description and validate in execute():
```rust
if ctx.permission == PermissionMode::ReadOnly {
    anyhow::bail!("This tool requires WorkspaceWrite permission");
}
```

---

## Testing Patterns

### File Operation Tests

Use `tempfile::NamedTempFile` for single files:
```rust
use tempfile::NamedTempFile;
use std::io::Write;

#[test]
fn test_read_file() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "hello").unwrap();
    
    let tool = ReadFileTool;
    let args = serde_json::json!({ "path": f.path().to_str().unwrap() });
    let result = tool.execute(args, &ctx()).unwrap();
    assert!(result.content.contains("hello"));
}
```

Use `tempfile::TempDir` for directory structures:
```rust
use tempfile::TempDir;
use std::fs;

#[test]
fn test_write_nested() {
    let dir = TempDir::new().unwrap();
    let tool = WriteFileTool;
    let path = dir.path().join("a").join("b").join("c.txt");
    
    let args = serde_json::json!({
        "path": path.to_str().unwrap(),
        "content": "nested"
    });
    let result = tool.execute(args, &ctx(&dir)).unwrap();
    
    let content = fs::read_to_string(&path).unwrap();
    assert_eq!(content, "nested");
}
```

### Shell Execution Tests

```rust
#[test]
fn test_bash_with_timeout() {
    let tool = BashTool;
    let ctx = ToolContext { /* with FullAccess */ };
    let args = serde_json::json!({
        "command": "echo hello",
        "timeout": 5000
    });
    let result = tool.execute(args, &ctx).unwrap();
    assert!(result.content.contains("hello"));
}
```

### Search Tests

```rust
#[test]
fn test_glob_search() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("foo.rs"), "").unwrap();
    fs::write(dir.path().join("bar.txt"), "").unwrap();
    
    let tool = GlobSearchTool;
    let ctx = ToolContext { cwd: dir.path().to_path_buf(), /* ... */ };
    let args = serde_json::json!({ "pattern": "*.rs" });
    let result = tool.execute(args, &ctx).unwrap();
    
    assert!(result.content.contains("foo.rs"));
    assert!(!result.content.contains("bar.txt"));
}
```

---

## Output Truncation

All tool outputs are automatically truncated at 8 KB:

```rust
pub fn execute_tool(
    registry: &ToolRegistry,
    name: &str,
    args: serde_json::Value,
    ctx: &ToolContext,
) -> Result<ToolResult> {
    let tool = registry.get(name).ok_or_else(|| anyhow::anyhow!("Unknown tool: {name}"))?;
    let result = tool.execute(args, ctx)?;
    Ok(result.truncate(MAX_TOOL_OUTPUT_BYTES))  // 8192 bytes
}
```

**Truncation behavior:**
- Finds safe UTF-8 char boundary
- Appends message: `[truncated: showing first N bytes of M]`
- Sets `truncated: true` flag

---

## Known Issues & Limitations

1. **Agent Tool Stub** — `agent` tool returns "not yet implemented"
2. **REPL Binary Dependency** — Requires python3 and node binaries on system
3. **Regex Performance** — Large file searches may be slow with complex patterns
4. **Binary Detection** — Only checks first 8 KB for null bytes (may miss sparse binaries)

---

## Error Handling

All tools return `Result<ToolResult>`:
- **Success:** `Ok(ToolResult { content, truncated: false })`
- **Error:** `Err(anyhow::Error)` with descriptive message

Common error patterns:
```rust
// Missing parameter
.ok_or_else(|| anyhow::anyhow!("missing 'field' argument"))?

// File I/O
.with_context(|| format!("failed to read {}", path.display()))?

// Path validation
crate::resolve_and_validate_path(path_str, &ctx.cwd)?  // Returns error if escapes cwd
```

---

## Integration with Runtime

The `zipcode-runtime` crate:
1. Creates a `ToolRegistry` instance
2. Registers all 10 tools
3. Injects tool specs into the LLM system prompt
4. Calls `execute_tool()` when the model generates tool calls
5. Applies permission checks (enforced at runtime level)

---

## AI Instructions

When modifying this crate:

1. **Understand the trait boundary** — New tools must implement `Tool` with all 4 methods
2. **Validate paths** — Always use `resolve_and_validate_path()` for file operations
3. **Handle permissions** — Check `ctx.permission` if tool needs specific access level
4. **Respect output limits** — Account for 8 KB truncation in tool design
5. **Test thoroughly** — Use tempfile for all file-based tests
6. **Error messages matter** — Tools run in an offline agent; clear errors are critical
7. **Document parameters** — JSON Schema in `parameters_schema()` is the tool's API contract
8. **Binary safety** — Grep tool must skip binary files; follow its pattern
9. **Path handling** — Use `resolve_and_validate_path()` consistently across all tools
10. **No unsafe code** — Workspace forbids `unsafe` blocks (clippy: unsafe_code = "forbid")

---
