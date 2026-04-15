# chat-template — legacy Gemma 3 format, off the Gemma 4 hot path

**File:** [`crates/inference/src/chat_template.rs`](../../crates/inference/src/chat_template.rs)
**Status:** legacy. Kept compilable for feature-gated `candle` / `llama-cpp-rs` backends, **not invoked** on the current Gemma 4 llama-server path.
**Authoritative Gemma 4 reference:** [gemma4-format-spec](gemma4-format-spec.md).

Historically this module was labelled "Gemma 4" but the tokens it emits are actually Gemma 3 (`<start_of_turn>` / `<end_of_turn>`, JSON-wrapped `<tool_call>` blocks). The mismatch was invisible during operation because the production backend uses `llama-server --jinja`, which applies the GGUF-embedded Jinja template and parses tool calls back to OpenAI JSON — see [gemma4-format-spec § Empirical wire contract](gemma4-format-spec.md#empirical-wire-contract-llama-server-path--phase-0-answered).

**Only the feature-gated candle and llama-cpp-rs backends call into this module**, and both are known broken for Gemma 4 today per [`CLAUDE.md`](../../CLAUDE.md) "Current Limitations" #1–#2. For production Gemma 4 work, this module is effectively dead code.

---

## Format (hardcoded — actually Gemma 3 tokens)

**EXTRACTED** `chat_template.rs:11-56`

Per-turn wrapper:
```
<start_of_turn>user
{content}
<end_of_turn>
<start_of_turn>model
...
```

**GOTCHA** — these turn markers are the Gemma 3 format. The real Gemma 4 tokens are `<|turn>` / `<turn|>`; see [gemma4-format-spec § Control tokens](gemma4-format-spec.md#control-tokens-gemma-4).

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

## GOTCHA: single-model coupling + Gemma 3 vs Gemma 4 mislabel

**GOTCHA** — the entire chat template is hardcoded for **Gemma 3** despite historical "Gemma 4" labeling. Two distinct problems:

1. **Gemma 3 tokens instead of Gemma 4.** Real Gemma 4 uses `<|turn>` / `<turn|>` / `<|tool_call>call:FUNC{…}<tool_call|>` (a structured mini-language, **not JSON**). This module still emits `<start_of_turn>` / `<end_of_turn>` / `<tool_call>{json}</tool_call>`. The divergence stayed invisible because `llama-server --jinja` bypasses this module and uses the GGUF-embedded template directly — see [gemma4-format-spec](gemma4-format-spec.md).

2. **Qwen/Llama 3/OpenAI-compatible models silently appear tool-blind** on the native (non-llama-server) path. They don't emit `<tool_call>` blocks, so `parse_tool_calls()` always returns an empty vec, so [`ConversationLoop`](conversation-loop.md) thinks the model finished without wanting tools and exits. The user sees the model's raw text, including any "I would call X tool" prose, but no tools actually fire. This only manifests when the feature-gated candle / llama-cpp-rs backends are active.

3. **Tool schema injection is tied to Gemma's first-turn idiom** (`chat_template.rs:44-50`). Other model families expect tool schemas in the system message or a dedicated `tools` API field.

4. **Turn markers are Gemma-3-specific.** Other models use `<|im_start|>`, `<|im_end|>`, `[INST]`, etc. Tokenizing those against any Gemma model → gibberish.

### Path forward (INFERRED)

For the current Gemma 4 production path, **no change is needed here** — llama-server handles templating and tool-call parsing. If candle / llama-cpp-rs backends are revived, this module must be rewritten against the real Gemma 4 wire format using [gemma4-format-spec § What this means for the zipcode rewrite](gemma4-format-spec.md#what-this-means-for-the-zipcode-rewrite) as the implementation spec. Doing that properly also requires the long-planned `ChatTemplate` trait so multiple model families can coexist. See [recipes › Swap chat template](recipes.md#swap-or-add-a-chat-template).

---

## Tests

**EXTRACTED** — 20 inline tests in `chat_template.rs` as of 2026-04-15.

Coverage now includes:
- `format_message()` / `format_conversation()` role rendering and first-turn tool injection
- `parse_tool_calls()` on well-formed, empty, malformed, missing-name, nested-closing-tag, and unclosed-block cases
- explicit regression guards that malformed or unclosed `<tool_call>` blocks do **not** hide later valid calls in the same model turn
- `extract_text_content()` behavior for closed malformed blocks, later valid blocks after malformed prefixes, nested closing tags inside JSON strings, and preservation of partial unclosed content without raw tag markup

---

## Related pages

- [inference](inference.md) — provides `ChatMessage` and calls this module
- [conversation-loop](conversation-loop.md) — consumes the `ToolCallParsed` output
- [gotchas](gotchas.md) — Gemma hardcoding is gotcha #1
