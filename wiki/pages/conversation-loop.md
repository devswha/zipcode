# conversation-loop — The agentic loop driver

**File:** [`crates/runtime/src/conversation.rs`](../../crates/runtime/src/conversation.rs)
**God node.** Holds references to every other god node: [`InferenceProvider`](inference.md#inferenceprovider-trait), [`ToolRegistry`](tools.md#toolregistry), [`PermissionPolicy`](permissions.md), [`Session`](session.md).

---

## `ConversationLoop` struct

**EXTRACTED** `conversation.rs:58-81`

```rust
pub struct ConversationLoop {
    pub engine: Box<dyn InferenceProvider>,
    pub tools: ToolRegistry,
    pub session: Session,
    pub permission: PermissionPolicy,
    pub system_prompt: String,
    pub tool_specs: Vec<ToolSpec>,
    pub cwd: std::path::PathBuf,
    pub depth: u32,                        // nesting: 0 = top-level, 1 = sub-agent
    pub last_sent_idx: usize,              // context-window send boundary
    pub child_session_ids: Arc<Mutex<Vec<(String, PathBuf)>>>,
    pub skill_registry: Option<Arc<SkillRegistry>>,
    pub compact_policy: CompactPolicy,     // tier-1/tier-2 compaction triggers
}
```

Generic over inference backend via the trait object. Owns everything needed to drive one user turn to completion. The `depth`, `child_session_ids`, and `skill_registry` fields support the [`spawn_child()`](#child-loop-creation--build_and_run_child) agent-delegation path; `compact_policy` controls automatic context compaction; `last_sent_idx` enables incremental context sending when the engine manages its own context window.

---

## `run_turn()` flow

**EXTRACTED** `conversation.rs:93-200`

```
┌─ first turn only: push system prompt to session (prompt.rs:build)
├─ push user message to session
│
├─ loop up to MAX_TOOL_ITERATIONS (= 25, conversation.rs:94)
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
│     │     ├─ execute_tool(&registry, name, args, &ctx)       (tools/lib.rs:419)
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

**EXTRACTED** `conversation.rs:30-46`

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
| Max tool iterations per turn | `MAX_TOOL_ITERATIONS = 25` | `conversation.rs:94` | Returns `Err`; turn does not resume gracefully |
| Max tool output per call | `MAX_TOOL_OUTPUT_BYTES = 8192` | `tools/lib.rs:502` | Output silently trimmed with `[truncated: ...]` note — see [tools](tools.md#output-truncation) |
| Session file size | none | `session.rs` | Grows unbounded per message |

**GOTCHA:** hitting the 25-iteration cap throws an `Err` but the session is still saved, so the partial conversation is persisted. The next turn starts with a model that saw its own loop getting killed mid-thought, which can produce confused output.

---

## Tests

**EXTRACTED** — integration tests in `crates/runtime/tests/integration.rs`.

Drive `ConversationLoop` with a `MockInferenceProvider` (and one custom event-stream provider) that covers the success and failure edges in `crates/runtime/tests/integration.rs`.

Current named coverage includes:

1. `text_only_response` — single natural-language turn exits after one iteration. `crates/runtime/tests/integration.rs:175`
2. `single_tool_call` — tool call → tool execution → model follow-up → final answer. `crates/runtime/tests/integration.rs:203`
3. `multi_tool_turn` — more than one tool call in the same turn. `crates/runtime/tests/integration.rs:238`
4. `permission_denied` — `read_only` mode blocks `write_file`. `crates/runtime/tests/integration.rs:284`
5. `workspace_write_permission_prompt_executes_bash_when_approved` — approval prompt is emitted and `bash` executes only after acceptance. `crates/runtime/tests/integration.rs:319`
6. `workspace_write_permission_prompt_records_denial_when_rejected` — rejected approval leaves a denial tool-result message in session history without executing the tool. `crates/runtime/tests/integration.rs:352`
7. `tool_call_loop_cap` — repeated tool requests hit the 25-iteration guard. `crates/runtime/tests/integration.rs:475`
8. `path_traversal_blocked` — `read_file ../../etc/passwd` is rejected by tool path validation. `crates/runtime/tests/integration.rs:509`
9. `failed_turn_is_saved_to_session_file` — inference errors still persist the attempted turn. `crates/runtime/tests/integration.rs:548`
10. `tool_call_turn_strips_raw_markup_from_saved_assistant_content` — raw `({...`})` markup is stripped before assistant history is saved. `crates/runtime/tests/integration.rs:588`
11. `thinking_channel_routes_to_callback_but_not_to_history` — thinking-mode tokens reach the `StreamCallback` but are not saved to session history. `crates/runtime/tests/integration.rs:632`
12. `resumed_session_does_not_duplicate_system_prompt` — resume path keeps the prompt boundary stable. `crates/runtime/tests/integration.rs:688`
13. `compacted_session_roundtrip_can_continue_turns` — compacted sessions still resume correctly. `crates/runtime/tests/integration.rs:721`
14. `auto_retry_nudges_model_after_empty_turn_on_error` — empty assistant turn with tool errors triggers a retry nudge. `crates/runtime/tests/integration.rs:796`
15. `auto_retry_skips_when_tool_result_has_no_errors` — no retry nudge when tool results are clean. `crates/runtime/tests/integration.rs:854`
16. `inference_error_is_not_masked_by_session_save_failure` — inference errors propagate even if session save also fails. `crates/runtime/tests/integration.rs:894`
17. `max_tokens_finish_appends_truncation_notice` — `FinishReason::MaxTokens` causes a truncation notice to be appended to the assistant message. `crates/runtime/tests/integration.rs:933`
18. `mixed_permission_partial_flow` — mix of allowed and denied tools in one turn; allowed tools run, denied tools get denial messages. `crates/runtime/tests/integration.rs:990`
19. `denied_tool_still_saves_session_and_continues` — a denied tool result is saved and the loop continues. `crates/runtime/tests/integration.rs:1063`
20. `tool_error_does_not_break_session_save` — a tool execution error produces an error result message and still saves the session. `crates/runtime/tests/integration.rs:1128`
21. `empty_text_response_saves_correctly` — an empty string assistant response is saved without truncation. `crates/runtime/tests/integration.rs:1189`
22. `consecutive_turns_accumulate_messages` — multiple user turns accumulate messages in session history in order. `crates/runtime/tests/integration.rs:1250`
23. `conversation_loop_sends_full_history_when_provider_does_not_manage_context` — full message history is sent to the provider when it does not manage context itself. `crates/runtime/tests/integration.rs:1343`
24. `conversation_loop_sends_only_new_segment_when_provider_manages_context` — only the new segment is sent to the provider when it manages its own context. `crates/runtime/tests/integration.rs:1404`
25. `workspace_write_permission_prompt_executes_repl_when_approved` — approval prompt is emitted and `repl` executes only after acceptance. `crates/runtime/tests/integration.rs:397`
26. `workspace_write_permission_prompt_records_denial_when_rejected_for_repl` — rejected approval for `repl` leaves a denial tool-result message in session history without executing the tool. `crates/runtime/tests/integration.rs:430`

26 integration tests total (**EXTRACTED** — `cargo test -p zipcode-runtime --test integration -- --list` on 2026-05-03).

---

## Child-loop creation: `build_and_run_child`

**EXTRACTED** `conversation.rs:347-426`

Refactored in commit `8259d42` — the duplicated child-loop creation logic was extracted from `make_spawn_child_callback` and `spawn_child` into a shared private helper:

```rust
fn build_and_run_child(
    engine: Box<dyn InferenceProvider>,
    task: &str,
    allowlist: Option<&[String]>,
    parent_depth: u32,
    parent_session: &Session,
    parent_permission: &PermissionPolicy,
    parent_registry: &ToolRegistry,
    child_session_ids: Arc<Mutex<Vec<(String, PathBuf)>>>,
) -> Result<ChildResult>
```

Both `spawn_child()` (public API, `conversation.rs:478`) and `make_spawn_child_callback()` (Agent tool callback, `conversation.rs:427`) now delegate to this single canonical implementation. The helper handles:
- Depth guard (`MAX_AGENT_DEPTH` check)
- Permission inheritance via `inherit_for_child()`
- Tool registry filtering (allowlist handling)
- Child session creation (`Session::new_child()`)
- Child session ID tracking
- `ConversationLoop` construction and `run_turn()` execution

**INFERRED:** Any future change to child-loop creation (e.g., adding a field, changing session handling) now requires editing only one location instead of two.

---

## Related pages

- [inference](inference.md) — the producer of `TokenEvent`s consumed here
- [tools](tools.md) — `execute_tool()` sink
- [permissions](permissions.md) — `PermissionPolicy::check()` call site
- [session](session.md) — where messages persist after each iteration
- [cli](cli.md) — `StreamCallback` implementor
