# conversation-loop — The agentic loop driver

**File:** [`crates/runtime/src/conversation.rs`](../../crates/runtime/src/conversation.rs)
**God node.** Holds references to every other god node: [`InferenceProvider`](inference.md#inferenceprovider-trait), [`ToolRegistry`](tools.md#toolregistry), [`PermissionPolicy`](permissions.md), [`Session`](session.md).

---

## `ConversationLoop` struct

**EXTRACTED** `conversation.rs:21-29`

```rust
pub struct ConversationLoop {
    pub engine: Box<dyn InferenceProvider>,
    pub tools: ToolRegistry,
    pub session: Session,
    pub permission: PermissionPolicy,
    pub system_prompt: String,
    pub tool_specs: Vec<ToolSpec>,
    pub cwd: std::path::PathBuf,
}
```

Generic over inference backend via the trait object. Owns everything needed to drive one user turn to completion.

---

## `run_turn()` flow

**EXTRACTED** `conversation.rs:32-147`

```
┌─ first turn only: push system prompt to session (prompt.rs:build)
├─ push user message to session
│
├─ loop up to MAX_TOOL_ITERATIONS (= 25, conversation.rs:41)
│     │
│     ├─ engine.generate_stream(messages, tool_specs)
│     │     ↓ (receiver of TokenEvent)
│     ├─ collect tokens → stream via StreamCallback::on_token
│     ├─ collect tool_calls if any
│     │
│     ├─ push assistant message (content + optional tool_calls) to session
│     │
│     ├─ if no tool calls → break (turn complete)
│     │
│     ├─ for each tool call:
│     │     ├─ permission.check(name, args) →
│     │     │     Allowed              → run
│     │     │     NeedsApproval(msg)   → StreamCallback::on_permission_prompt → if false, skip
│     │     │     Denied(msg)          → append denial message, skip
│     │     ├─ execute_tool(&registry, name, args, &ctx)       (tools/lib.rs:212)
│     │     │     ↑ auto-truncated to MAX_TOOL_OUTPUT_BYTES (= 8192)
│     │     ├─ push tool-result message to session
│     │     └─ StreamCallback::on_tool_result
│     │
│     └─ (model now sees results, loops back to generate_stream)
│
├─ save session to ~/.zipcode/sessions/{uuid}.json
└─ if iteration counter == MAX_TOOL_ITERATIONS → return Err
```

The loop alternates: model turn → tool calls → tool results → model turn. It exits cleanly when the model produces a turn with zero tool calls (a final natural-language answer).

---

## `StreamCallback` trait

**EXTRACTED** `conversation.rs:12-19`

```rust
pub trait StreamCallback {
    fn on_token(&mut self, text: &str);
    fn on_tool_start(&mut self, name: &str, args: &serde_json::Value);
    fn on_tool_result(&mut self, name: &str, result: &str);
    fn on_permission_prompt(&mut self, message: &str) -> bool;
    fn on_error(&mut self, error: &str);
}
```

Implemented by the CLI's REPL / TUI to pipe events to the terminal. `on_permission_prompt` returns `true` if the user approved — this is how [workspace-write](permissions.md) mode lets `bash` / `repl` run only after confirmation.

---

## Bounds & invariants

| Bound | Constant | Location | What happens at the limit |
|-------|----------|----------|--------------------------|
| Max tool iterations per turn | `MAX_TOOL_ITERATIONS = 25` | `conversation.rs:41` | Returns `Err`; turn does not resume gracefully |
| Max tool output per call | `MAX_TOOL_OUTPUT_BYTES = 8192` | `tools/lib.rs:209` | Output silently trimmed with `[truncated: ...]` note — see [tools](tools.md#output-truncation) |
| Session file size | none | `session.rs` | Grows unbounded per message |

**GOTCHA:** hitting the 25-iteration cap throws an `Err` but the session is still saved, so the partial conversation is persisted. The next turn starts with a model that saw its own loop getting killed mid-thought, which can produce confused output.

---

## Tests

**EXTRACTED** — integration tests in `crates/runtime/tests/integration.rs`.

Drive `ConversationLoop` with a `MockInferenceProvider` (and one custom event-stream provider) that covers the success and failure edges in `crates/runtime/tests/integration.rs`.

Current named coverage includes:

1. `text_only_response` — single natural-language turn exits after one iteration. `crates/runtime/tests/integration.rs:147`
2. `single_tool_call` — tool call → tool execution → model follow-up → final answer. `crates/runtime/tests/integration.rs:175`
3. `multi_tool_turn` — more than one tool call in the same turn. `crates/runtime/tests/integration.rs:210`
4. `permission_denied` — `read_only` mode blocks `bash`. `crates/runtime/tests/integration.rs:256`
5. `tool_call_loop_cap` — repeated tool requests hit the 25-iteration guard. `crates/runtime/tests/integration.rs:295`
6. `path_traversal_blocked` — `read_file ../../etc/passwd` is rejected by tool path validation. `crates/runtime/tests/integration.rs:329`
7. `failed_turn_is_saved_to_session_file` — inference errors still persist the attempted turn. `crates/runtime/tests/integration.rs:368`
8. `tool_call_turn_strips_raw_markup_from_saved_assistant_content` — raw `<tool_call>` markup is stripped before assistant history is saved. `crates/runtime/tests/integration.rs:408`
9. `resumed_session_does_not_duplicate_system_prompt` — resume path keeps the prompt boundary stable. `crates/runtime/tests/integration.rs:452`
10. `compacted_session_roundtrip_can_continue_turns` — compacted sessions still resume correctly. `crates/runtime/tests/integration.rs:485`

10 integration tests total (**EXTRACTED** — `cargo test -p zipcode-runtime --test integration -- --list` on 2026-04-14).

---

## Related pages

- [inference](inference.md) — the producer of `TokenEvent`s consumed here
- [tools](tools.md) — `execute_tool()` sink
- [permissions](permissions.md) — `PermissionPolicy::check()` call site
- [session](session.md) — where messages persist after each iteration
- [cli](cli.md) — `StreamCallback` implementor
