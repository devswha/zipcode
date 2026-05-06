# permissions — 3-tier access control for tool execution

Permission state is **split across two crates** — GOTCHA flagged in [`GRAPH_REPORT`](../GRAPH_REPORT.md#cross-crate-connections-surprising-edges) — because the enum lives with the tools and the policy lives with the runtime.

| Concept | Where | File |
|---------|-------|------|
| `PermissionMode` enum | `zipcode-tools` | `crates/tools/src/lib.rs:306` |
| `PermissionPolicy` + `check()` | `zipcode-runtime` | `crates/runtime/src/permission.rs:5` |
| CLI flag `--permission-mode` | `zipcode` (cli) | `crates/cli/src/main.rs` |
| Config field `permission_mode` | `zipcode-runtime` | `crates/runtime/src/config.rs` (default: `workspace-write`) |

---

## `PermissionMode`

**EXTRACTED** `crates/tools/src/lib.rs:306-310`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionMode {
    ReadOnly,
    WorkspaceWrite,
    FullAccess,
}
```

Serialization: kebab-case (`read-only`, `workspace-write`, `full-access`). The parser also accepts `danger-full-access` as an alias (`permission.rs:60-69`).

---

## `PermissionPolicy::check()`

**EXTRACTED** `crates/runtime/src/permission.rs:32-54`

Returns a 3-variant enum:

```rust
pub enum PermissionCheck {
    Allowed,
    NeedsApproval(String),   // message shown to user via StreamCallback::on_permission_prompt
    Denied(String),          // tool never runs; denial message surfaced to model
}
```

### Matrix

| Tool | Read-only | Workspace-write | Full-access |
|------|:---------:|:---------------:|:-----------:|
| `read_file` | Allowed | Allowed | Allowed |
| `glob_search` | Allowed | Allowed | Allowed |
| `grep_search` | Allowed | Allowed | Allowed |
| `tool_search` | Allowed | Allowed | Allowed |
| `write_file` | Denied | Allowed | Allowed |
| `edit_file` | Denied | Allowed | Allowed |
| `todo_write` | Denied | Allowed | Allowed |
| `fetch_repo` | Denied | Allowed | Allowed |
| `bash` | Denied | **Approval** | Allowed |
| `repl` | Denied | **Approval** | Allowed |
| `agent` | Denied | Denied | Allowed |

**EXTRACTED** from `permission.rs:28-54` — read-only's allowlist is hardcoded; workspace-write allows file-writing tools plus the explicit GitHub-fetch exception `fetch_repo`; workspace-write's approval list is `{bash, repl}`; `agent` falls through the `WorkspaceWrite` wildcard arm and is **Denied** (`permission.rs:31-47`).

### Approval flow

When `check()` returns `NeedsApproval(msg)`, [`ConversationLoop`](conversation-loop.md) calls `StreamCallback::on_permission_prompt(msg)`. The CLI prompts the user interactively; if the user returns `false`, the tool is skipped and the denial is surfaced to the model as the tool result.

---

## Tests

**EXTRACTED** `permission.rs` — 9 inline tests as of 2026-04-15:

- `test_full_access_allows_everything`
- `test_read_only_blocks_writes`
- `test_read_only_blocks_repl`
- `test_workspace_write_needs_approval_for_bash`
- `test_workspace_write_needs_approval_for_repl`
- `test_workspace_write_denies_agent`
- `test_workspace_write_denies_unknown_tools`
- `test_parse_permission_mode_accepts_known_values`
- `test_parse_permission_mode_rejects_unknown_values`

---

## Adding a new tier or tool

1. **New tier:** add a variant to `PermissionMode` in `tools/lib.rs:282` **and** a match arm in `PermissionPolicy::check()` in `runtime/permission.rs:28`. Both crates must compile together.
2. **New tool into existing tier:** add its name to the appropriate allowlist/approval list in `permission.rs:28-49`. Also add a test in `permission.rs`.
3. **CLI flag:** `--permission-mode` is parsed via `parse_permission_mode()` (`permission.rs:60-69`). Any new variant must round-trip through that function.

---

## GOTCHAs

- **GOTCHA** — `read-only` mode's allowlist is a hardcoded set; any new read-safe tool you add is **implicitly denied** in read-only mode unless you remember to update `permission.rs:28`.
- **GOTCHA** — `AgentTool` is a stub that returns "not yet implemented", but its permission behavior IS defined: `Denied` in `read-only` and `workspace-write` (falls through the wildcard arm), `Allowed` in `full-access`. Test: `test_workspace_write_denies_agent` at `permission.rs:166`. Cross-reference [tools › the 11 tools](tools.md#the-11-tools).

---

## Related pages

- [tools](tools.md) — the executors gated by this policy
- [conversation-loop](conversation-loop.md) — calls `check()` on every tool invocation
- [cli](cli.md) — parses `--permission-mode` and handles the approval UI
