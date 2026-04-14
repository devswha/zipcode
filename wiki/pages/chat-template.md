# chat-template — Gemma 4 format & tool-call parsing

**File:** [`crates/inference/src/chat_template.rs`](../../crates/inference/src/chat_template.rs)
**God node adjacent:** sits between [`ChatMessage`](inference.md#core-types) and every [`InferenceProvider`](inference.md#inferenceprovider-trait).

Every prompt flows through this module before reaching a backend. Every model response flows back through `parse_tool_calls()` / `extract_text_content()` before the conversation loop sees it.

---

## Format (hardcoded for Gemma 4)

**EXTRACTED** `chat_template.rs:11-56`

Per-turn wrapper:
```
<start_of_turn>user
{content}
<end_of_turn>
<start_of_turn>model
...
```

Roles recognized: `user`, `model`, `tool`, `system`. The template is hardcoded — **there is no alternate template path**.

### `format_conversation()` (lines 40-56)

1. Iterates messages.
2. Injects the tool schema into the **first user turn only** (lines 44-50).
3. Appends `<start_of_turn>model\n` as the generation prefix.

### `format_message()` (lines 12-35)

Single-message wrapper. Handles per-role rendering and attaches `<tool_call>` JSON blocks for `Model` messages that have `tool_calls: Some(...)`.

---

## Tool-call parsing

**EXTRACTED** `chat_template.rs:60-84`

The model is expected to emit JSON wrapped in `<tool_call>...</tool_call>` blocks. `parse_tool_calls()` scans for the start tag, then tries successive closing tags until it finds valid JSON:

```
<tool_call>
{"name": "read_file", "arguments": {"path": "foo.rs"}}
</tool_call>
```

Produces a `Vec<ToolCallParsed>`. Each parsed call gets a generated `id` for tracking through the conversation loop.

**Recovery behavior that matters:** malformed or unclosed `<tool_call>` blocks are skipped without discarding later valid blocks in the same model turn. This lets the conversation loop keep tool use that appears after a broken block instead of losing the whole turn's tool-call set.

### `extract_text_content()` (lines 87-101)

Strips `<tool_call>...</tool_call>` blocks from the raw output so that the "model text" shown to the user is just natural language.

---

## Stop-token detection

**EXTRACTED** — candle path: `engine.rs:108-112`; llama-cpp path: `llama_cpp_backend.rs:146` uses `model.is_eog_token()`; llama-server path: SSE `finish_reason` field.

All three recognize `<eos>` and `<end_of_turn>` as stops.

---

## GOTCHA: single-model coupling

**GOTCHA** — the entire chat template is hardcoded for Gemma 4. Consequences:

1. **Qwen/Llama 3/OpenAI-compatible models silently appear tool-blind.** They don't emit `<tool_call>` blocks, so `parse_tool_calls()` always returns an empty vec, so [`ConversationLoop`](conversation-loop.md) thinks the model finished without wanting tools and exits. The user sees the model's raw text, including any "I would call X tool" prose, but no tools actually fire.

2. **Tool schema injection is tied to Gemma's first-turn idiom** (`chat_template.rs:44-50`). Other model families expect tool schemas in the system message or a dedicated `tools` API field.

3. **Turn markers (`<start_of_turn>`, `<end_of_turn>`) are Gemma-specific.** Other models use `<|im_start|>`, `<|im_end|>`, `[INST]`, etc. Tokenizing those against a Gemma model → gibberish.

### Path forward (INFERRED)

A proper fix needs a `ChatTemplate` trait that each `InferenceProvider` pairs with, plus per-model template detection at load time. Not yet implemented. See [recipes › Swap chat template](recipes.md#swap-or-add-a-chat-template).

---

## Tests

**EXTRACTED** — 7 tests inline in `chat_template.rs`:
- `format_message()` for each role
- `format_conversation()` full flow
- `parse_tool_calls()` on well-formed blocks
- `extract_text_content()` removes blocks cleanly

---

## Related pages

- [inference](inference.md) — provides `ChatMessage` and calls this module
- [conversation-loop](conversation-loop.md) — consumes the `ToolCallParsed` output
- [gotchas](gotchas.md) — Gemma hardcoding is gotcha #1
