# tools — `Tool` trait, registry, 10 implementations

The `zipcode-tools` crate. Home of the `Tool` + `ToolRegistry` god node.

**Crate path:** [`crates/tools/`](../../crates/tools/)
**Depends on:** nothing from this workspace.

---

## Module layout

**EXTRACTED** `crates/tools/src/`

| File | Tool / responsibility |
|------|----------------------|
| `lib.rs` | `Tool` trait, `ToolRegistry`, `ToolContext`, `ToolResult`, `PermissionMode`, `resolve_and_validate_path()`, `execute_tool()`, `MAX_TOOL_OUTPUT_BYTES` |
| `bash.rs` | `BashTool` |
| `read_file.rs` | `ReadFileTool` |
| `write_file.rs` | `WriteFileTool` |
| `edit_file.rs` | `EditFileTool` |
| `glob_search.rs` | `GlobSearchTool` |
| `grep_search.rs` | `GrepSearchTool` |
| `repl.rs` | `ReplTool` (python3 / node) |
| `todo_write.rs` | `TodoWriteTool` |
| `tool_search.rs` | `ToolSearchTool` |
| `agent.rs` | `AgentTool` — **STUB** |

---

## `Tool` trait

**EXTRACTED** `crates/tools/src/lib.rs:160-165`

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult>;
}
```

### `ToolRegistry`

**EXTRACTED** `lib.rs:168-207`

```rust
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}
```

Methods: `register(Box<dyn Tool>)`, `get(&str)`, `specs() -> Vec<ToolSpec>`, `names() -> Vec<&str>`.

The registry is populated once by the CLI at REPL startup ([`crates/cli/src/repl.rs`](cli.md#repl-initialization)) and moved into [`ConversationLoop`](conversation-loop.md).

### `execute_tool()` — the single entry point

**EXTRACTED** `lib.rs:212-223`

```rust
pub fn execute_tool(
    registry: &ToolRegistry,
    name: &str,
    args: serde_json::Value,
    ctx: &ToolContext,
) -> Result<ToolResult>
```

Looks up the tool, calls `tool.execute()`, and **auto-truncates** the result to `MAX_TOOL_OUTPUT_BYTES` (8 192 bytes). Called from `ConversationLoop::run_turn()` after permission check.

---

## `ToolContext`

**EXTRACTED** `lib.rs:100-105`

```rust
pub struct ToolContext {
    pub cwd: PathBuf,
    pub permission: PermissionMode,
    pub session_id: String,
}
```

Passed to every `execute()`. Tools that touch the filesystem rebase paths onto `cwd` and validate via `resolve_and_validate_path()`.

---

## `ToolResult`

**EXTRACTED** `lib.rs:108-149`

```rust
pub struct ToolResult {
    pub content: String,
    pub truncated: bool,
}
```

Method `truncate(max_bytes)` (lines 129-149) finds a safe UTF-8 char boundary, trims, and appends `[truncated: showing first X bytes of Y]`. **GOTCHA:** the model is told how many bytes were kept but NOT what was cut, so `ls -la` on a huge directory becomes an information black hole.

---

## The 10 tools

**EXTRACTED** — one line per impl, with absolute file reference.

| # | Name | File | Behavior (short) |
|---|------|------|------------------|
| 1 | `BashTool` | `bash.rs` | `bash -c <command>` with cwd; 120 s timeout via `wait_timeout`; captures stdout/stderr; preserves exit code; timeout cleanup walks `/proc` and kills descendant processes too |
| 2 | `ReadFileTool` | `read_file.rs` | Offset/limit, line numbering, 10 MB `MAX_READ_SIZE`, binary detection |
| 3 | `WriteFileTool` | `write_file.rs` | Creates parent dirs; path traversal blocked |
| 4 | `EditFileTool` | `edit_file.rs` | Exact string replacement; **fails if 0 or >1 match** (enforced uniqueness) |
| 5 | `GlobSearchTool` | `glob_search.rs` | `glob::glob()` sorted, filters to files, rejects absolute / `..` escape patterns, and re-validates each match against workspace boundaries |
| 6 | `GrepSearchTool` | `grep_search.rs` | `grep-regex` + `grep-searcher`, optional glob filter, rejects absolute / `..` traversal globs, re-validates matched files against the workspace, `file:line` output |
| 7 | `ReplTool` | `repl.rs` | Spawns `python3 -c` or `node -e`; captures stdout/stderr; timeout cleanup kills descendant interpreters/processes too |
| 8 | `TodoWriteTool` | `todo_write.rs` | Writes JSON to `.zipcode-todos.json` in cwd |
| 9 | `ToolSearchTool` | `tool_search.rs` | Case-insensitive search over registered `ToolSpec`s |
| 10 | `AgentTool` | `agent.rs` | **STUB** — returns "not yet implemented" |

**Permission-gated subset** (per [permissions](permissions.md)):
- Read-only: `read_file`, `glob_search`, `grep_search`, `tool_search`
- Workspace-write requires approval: `bash`, `repl`
- Full-access: everything including `bash`, `repl`, `agent` (if it existed)

---

## Path safety: `resolve_and_validate_path()`

**EXTRACTED** `lib.rs:10-78`

Single gate that every file-touching tool routes through.

Algorithm:
1. Rebase relative paths onto `ctx.cwd`.
2. Canonicalize (via `canonicalize_even_if_missing()` at `:36-60` for paths that don't exist yet).
3. Assert `canonical.starts_with(&cwd_canonical)`.

**GOTCHA:** Only works because every file tool remembers to call it. `glob_search` now also rejects absolute/parent-directory escape patterns before globbing and re-validates resolved matches, but a new file-touching tool that skips `resolve_and_validate_path()` can still bypass the boundary. Tests at `lib.rs:305-309` cover the blocking case but won't catch a brand-new tool that forgets the helper.

---

## Output truncation

**EXTRACTED** `lib.rs:209` → `MAX_TOOL_OUTPUT_BYTES = 8192` — applied in `execute_tool()` at `:212-223`.

**Why 8 KB?** Fits comfortably into Gemma's 8192-token default context without crowding the conversation. Overridable would require a config field (not present today).

---

## Tests

**EXTRACTED** — ~40 unit tests across the crate.

| File | Tests | Covers |
|------|-------|--------|
| `lib.rs` | 8 | registry ops, specs, truncation, path traversal (blocked + allowed + missing parent) |
| `bash.rs` | 4 | echo, stderr capture, nonzero exit, cwd usage |
| `read_file.rs` | ~8 | offset, limit, binary detection, too large |
| `write_file.rs` | 2 | new file, nested dirs |
| `edit_file.rs` | ~6 | unique replacement, multi-match error, missing string |
| `glob_search.rs` | ~3 | pattern match, no matches |
| `grep_search.rs` | ~4 | regex, glob filter, no matches |

---

## Related pages

- [conversation-loop](conversation-loop.md) — the consumer of `ToolRegistry::execute_tool()`
- [permissions](permissions.md) — the gate in front of every call
- [recipes › Add a new tool](recipes.md#add-a-new-tool)
- [gotchas](gotchas.md) — truncation, path safety, and stub tool
