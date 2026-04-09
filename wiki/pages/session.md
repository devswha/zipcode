# session — Conversation persistence

**File:** [`crates/runtime/src/session.rs`](../../crates/runtime/src/session.rs)

---

## Storage layout

**EXTRACTED** `session.rs:57-61`

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
| `Session::new()` | New UUID + timestamps; empty messages vec |
| `session.push_message(msg)` | Append to `messages`; refresh `updated_at` |
| `session.save()` | Serialize to `~/.zipcode/sessions/{id}.json`, creating parent dirs if needed |
| `Session::load(id)` | Read + deserialize |

---

## Lifecycle

**EXTRACTED** — traced from `crates/cli/src/repl.rs` into `crates/runtime/src/conversation.rs`.

1. CLI creates `Session::new()` at REPL startup.
2. [`ConversationLoop`](conversation-loop.md) holds it by value (`pub session: Session`).
3. On the first turn, the system prompt is pushed via `push_message()` (`conversation.rs:35-37`).
4. Every subsequent user / model / tool message goes through `push_message()` inside `run_turn()`.
5. `session.save()` is called at the end of each turn.

---

## GOTCHAs

- **GOTCHA** — no rotation, no retention policy. `~/.zipcode/sessions/` grows unbounded. A long-running user may accumulate thousands of small JSON files over weeks.
- **GOTCHA** — resuming a session that hit the 25-iteration cap means the model starts its next turn seeing its own loop getting killed. Session persistence happens regardless of whether `run_turn()` returned Ok or Err (see [conversation-loop › bounds](conversation-loop.md#bounds--invariants)).
- **GOTCHA** — no locking. Two concurrent zipcode instances pointed at the same session id would race on `save()`.

---

## Tests

**EXTRACTED** — 3 inline tests:
- `session_new_has_uuid_and_timestamps`
- `push_message_updates_updated_at`
- `save_load_roundtrip` (uses `tempfile::TempDir`)

---

## Related pages

- [conversation-loop](conversation-loop.md) — only consumer
- [inference › ChatMessage](inference.md#chatmessage-lines-13-67) — the element type of `messages`
- [gotchas](gotchas.md) — retention, concurrency
