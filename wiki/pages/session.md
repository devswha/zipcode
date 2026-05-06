# session — Conversation persistence

**File:** [`crates/runtime/src/session.rs`](../../crates/runtime/src/session.rs)

---

## Storage layout

**EXTRACTED** `session.rs:438` (`session_path()` helper)

```
~/.zipcode/sessions/{uuid}.json
```

One JSON file per session. UUID v4 generated at creation.

---

## `Session` struct

**EXTRACTED** `session.rs`

```rust
pub struct Session {
    pub id: String,                       // UUID v4
    pub messages: Vec<ChatMessage>,
    pub created_at: String,               // RFC3339
    pub updated_at: String,               // RFC3339
}
```

### Methods

| Method | Behavior |
|--------|----------|
| `Session::new()` | New UUID + timestamps; empty messages vec (`session.rs:100`) |
| `session.push_message(msg)` | Append to `messages`; refresh `updated_at` (`session.rs:234`) |
| `session.save()` | Serialize to `~/.zipcode/sessions/{id}.json`, creating parent dirs if needed (`session.rs:128`) |
| `Session::load(id)` | Read + deserialize (`session.rs:147`) |
| `session.path()` | Returns the resolved `PathBuf` for this session's JSON file (`session.rs:168`) |
| `session.compact(CompactPolicy) -> CompactResult` | Compacts message history in-place according to policy; preserves system prefix and a safe suffix of recent turns (`session.rs:182`) |

---

## Lifecycle

**EXTRACTED** — traced from `crates/cli/src/repl.rs` into `crates/runtime/src/conversation.rs`.

1. CLI creates `Session::new()` at REPL startup.
2. [`ConversationLoop`](conversation-loop.md) holds it by value (`pub session: Session`).
3. On the first turn, the system prompt is pushed via `push_message()` (`conversation.rs:100`).
4. Every subsequent user / model / tool message goes through `push_message()` inside `run_turn()`.
5. `session.save()` is called at the end of each turn.

---

## GOTCHAs

- **GOTCHA** — no rotation, no retention policy. `~/.zipcode/sessions/` grows unbounded. A long-running user may accumulate thousands of small JSON files over weeks.
- **GOTCHA** — resuming a session that hit the 25-iteration cap means the model starts its next turn seeing its own loop getting killed. Session persistence happens regardless of whether `run_turn()` returned Ok or Err (see [conversation-loop › bounds](conversation-loop.md#bounds--invariants)).
- **GOTCHA** — no locking. Two concurrent zipcode instances pointed at the same session id would race on `save()`.

---

## Tests

**EXTRACTED** — 14 inline tests in `session.rs` as of 2026-04-15.

Current coverage includes:
- basic lifecycle: `test_new_session`, `test_push_message`, `test_session_roundtrip`
- load-path validation: mismatched ids, invalid ids, traversal, slash, backslash, null-byte, and empty-id rejection
- compaction behavior: `test_compact_preserves_system_message_and_safe_boundary_suffix` and `test_compact_is_idempotent_without_new_user_turns`
- save hardening: `test_save_with_malicious_id_blocked`
- positive path validation: `test_valid_uuid_session_id_accepted`

---

## Related pages

- [conversation-loop](conversation-loop.md) — primary consumer (holds `Session` by value, calls `push_message` / `save` on every turn)
- [cli](cli.md) — also a direct consumer: `Session::new()`, `Session::load()`, `.compact()`, and `.save()` are called from `crates/cli/src/repl.rs` for startup, `/session`, `/clear`, and `/compact` flows
- [inference › ChatMessage](inference.md#chatmessage-lines-13-67) — the element type of `messages`
- [gotchas](gotchas.md) — retention, concurrency
