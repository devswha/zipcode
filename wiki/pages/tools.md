# tools — `Tool` trait, registry, 11 implementations

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
| `fetch_repo.rs` | `FetchRepoTool` |
| `glob_search.rs` | `GlobSearchTool` |
| `grep_search.rs` | `GrepSearchTool` |
| `repl.rs` | `ReplTool` (python3 / node) |
| `todo_write.rs` | `TodoWriteTool` |
| `tool_search.rs` | `ToolSearchTool` |
| `agent.rs` | `AgentTool` — **STUB** |

---

## `Tool` trait

**EXTRACTED** `crates/tools/src/lib.rs:431-442`

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult>;
}
```

### `ToolRegistry`

**EXTRACTED** `lib.rs:445-497`

```rust
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}
```

Methods: `register(Box<dyn Tool>)`, `get(&str)`, `specs() -> Vec<ToolSpec>`, `names() -> Vec<&str>`.

The registry is populated once by the CLI at REPL startup ([`crates/cli/src/repl.rs`](cli.md#repl-initialization)) and moved into [`ConversationLoop`](conversation-loop.md).

### `execute_tool()` — the single entry point

**EXTRACTED** `lib.rs:509-519`

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

**EXTRACTED** `lib.rs:347-366`

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

**EXTRACTED** `lib.rs:368-420`

```rust
pub struct ToolResult {
    pub content: String,
    pub truncated: bool,
}
```

Method `truncate(max_bytes)` (`lib.rs:391-418`) finds a safe UTF-8 char boundary, trims, and appends `[truncated: showing first X bytes of Y]`. **GOTCHA:** the model is told how many bytes were kept but NOT what was cut, so `ls -la` on a huge directory becomes an information black hole.

---

## The 11 tools

**EXTRACTED** — one line per impl, with absolute file reference.

| # | Name | File | Behavior (short) |
|---|------|------|------------------|
| 1 | `BashTool` | `bash.rs` | `bash -c <command>` with cwd; 120 s timeout; captures stdout/stderr concurrently so large output does not false-timeout; timeout cleanup walks `/proc` and kills descendant processes too |
| 2 | `ReadFileTool` | `read_file.rs` | Offset/limit, line numbering, 10 MB `MAX_READ_SIZE`, binary detection |
| 3 | `WriteFileTool` | `write_file.rs` | Creates parent dirs; path traversal blocked |
| 4 | `EditFileTool` | `edit_file.rs` | Exact string replacement; **fails if 0 or >1 match** (enforced uniqueness) |
| 5 | `FetchRepoTool` | `fetch_repo.rs` | Shallow-clones repository-root `https://github.com/<owner>/<repo>` URLs into `.zipcode-remote/<owner>__<repo>`; reuses an existing scratch clone; rejects nested/non-GitHub URLs |
| 6 | `GlobSearchTool` | `glob_search.rs` | `glob::glob()` sorted, filters to files, rejects absolute / `..` escape patterns, and re-validates each match against workspace boundaries |
| 7 | `GrepSearchTool` | `grep_search.rs` | `grep-regex` + `grep-searcher`, optional glob filter, rejects absolute / `..` traversal globs, re-validates matched files against the workspace, `file:line` output |
| 8 | `ReplTool` | `repl.rs` | Spawns `python3 -c` or `node -e`; captures stdout/stderr concurrently so large output does not false-timeout; timeout cleanup kills descendant interpreters/processes too |
| 9 | `TodoWriteTool` | `todo_write.rs` | Writes JSON to `.zipcode-todos.json` in cwd; validates each item before writing: rejects empty/whitespace `id` or `content`, enforces `status` enum (`pending`, `in_progress`, `completed`, `cancelled`); entire batch is rejected on first validation failure |
| 10 | `ToolSearchTool` | `tool_search.rs` | Case-insensitive search over registered `ToolSpec`s |
| 11 | `AgentTool` | `agent.rs` | **STUB** — returns "not yet implemented" |

**Permission-gated subset** (per [permissions](permissions.md)):
- Read-only: `read_file`, `glob_search`, `grep_search`, `tool_search`
- Workspace-write allowed without approval: `write_file`, `edit_file`, `todo_write`, `fetch_repo`
- Workspace-write requires approval: `bash`, `repl`
- Full-access: everything including `bash`, `repl`, `fetch_repo`, `agent` (if it existed)

---

## Path safety: `resolve_and_validate_path()`

**EXTRACTED** `lib.rs:15-93`

Single gate that every file-touching tool routes through.

Algorithm:
1. Rebase relative paths onto `ctx.cwd`.
2. Canonicalize (via `canonicalize_even_if_missing()` at `:71-95` for paths that don't exist yet).
3. Assert `canonical.starts_with(&cwd_canonical)`.

**GOTCHA:** Only works because every file tool remembers to call it. `glob_search` now also rejects absolute/parent-directory escape patterns before globbing and re-validates resolved matches, but a new file-touching tool that skips `resolve_and_validate_path()` can still bypass the boundary. Shared path-safety regression coverage lives in `crates/tools/src/lib.rs:627+`, but it still won't catch a brand-new tool that forgets the helper.

---

## Output truncation

**EXTRACTED** `lib.rs:502` → `MAX_TOOL_OUTPUT_BYTES = 8192` — applied in `execute_tool()` at `:509-519`.

**Why 8 KB?** Fits comfortably into Gemma's 8192-token default context without crowding the conversation. Overridable would require a config field (not present today).

---

## Tests

**EXTRACTED** — 248 tests total in `zipcode-tools` (186 unit + 62 integration; `cargo test -p zipcode-tools` on 2026-05-02).

Recent regression coverage that materially changed the crate since the previous wiki snapshot includes:

- `bash` timeout descendant cleanup and large-stdout false-timeout protection. `crates/tools/src/bash.rs:133`, `crates/tools/src/bash.rs:163`
- `repl` timeout descendant cleanup and large-stdout false-timeout protection. `crates/tools/src/repl.rs:237`, `crates/tools/src/repl.rs:275`
- `glob_search` rejects absolute and parent-directory escape patterns. `crates/tools/src/glob_search.rs:177`, `crates/tools/src/glob_search.rs:193`
- `grep_search` rejects absolute and parent-directory traversal globs and stops scanning once the output budget is full. `crates/tools/src/grep_search.rs:355`, `crates/tools/src/grep_search.rs:372`, `crates/tools/src/grep_search.rs:317`
- `tool_search` end-to-end via `execute_tool()` — keyword search by name/description returns correct matches, empty query rejected. `crates/tools/tests/tools_integration.rs`
- `todo_write` roundtrip end-to-end — write creates JSON file that `bash` can list and `read_file` can read back; overwrite updates content. `crates/tools/tests/tools_integration.rs`
- `read_file` offset/limit via `execute_tool()` — partial reads verified through the integration dispatch path. `crates/tools/tests/tools_integration.rs`
- `bash` creates file then `glob_search`/`grep_search` finds it — cross-tool chain starting from bash (the most common real-world pattern). `crates/tools/tests/tools_integration.rs`

The rest of the crate still has broad per-tool coverage for:

| Area | Example coverage anchors |
|------|---------------------------|
| Registry / truncation / shared path safety | `crates/tools/src/lib.rs:509`, `crates/tools/src/lib.rs:484`, `crates/tools/src/lib.rs:627` |
| File readers / writers / editor behaviors | `crates/tools/src/read_file.rs`, `crates/tools/src/write_file.rs`, `crates/tools/src/edit_file.rs` |
| Search behavior and budget limits | `crates/tools/src/glob_search.rs`, `crates/tools/src/grep_search.rs` |
| Execution tools (`bash`, `repl`) | `crates/tools/src/bash.rs`, `crates/tools/src/repl.rs` |

---

## Related pages

- [conversation-loop](conversation-loop.md) — the consumer of `ToolRegistry::execute_tool()`
- [permissions](permissions.md) — the gate in front of every call
- [recipes › Add a new tool](recipes.md#add-a-new-tool)
- [gotchas](gotchas.md) — truncation, path safety, and stub tool
