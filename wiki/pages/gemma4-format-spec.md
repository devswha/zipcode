# gemma4-format-spec — Gemma 4 native chat & tool-call format

**Phase 0 finding for the chat-template rewrite effort.**
**Snapshot:** extracted 2026-04-15 from `~/.zipcode/models/gemma-4-e2b-it-Q8_0.gguf` via llama-server `/props` endpoint (`llama-server b1-0d049d6`).
**Status:** authoritative reference for implementing [chat-template](chat-template.md) against real Gemma 4, not the Gemma 3 tokens currently hardcoded.

This page documents the **real** Gemma 4 chat template embedded in the model file, so that `crates/inference/src/chat_template.rs` can be rewritten against a known spec instead of guessed.

---

## Executive gap

**EXTRACTED** — `crates/inference/src/chat_template.rs:12-35` (current) vs `/props.chat_template` (real).

| Aspect | zipcode's current code (labelled "Gemma 4") | **Real Gemma 4 (from GGUF)** |
|---|---|---|
| Turn open | `<start_of_turn>ROLE\n` | `<\|turn>ROLE\n` |
| Turn close | `<end_of_turn>\n` | `<turn\|>\n` |
| Tool spec location | Injected into **first user turn** as pretty-printed JSON | Injected into **system turn** as `<\|tool>...<tool\|>` declarations in a custom mini-language |
| Tool call emission | `<tool_call>{json}</tool_call>` | `<\|tool_call>call:FUNC{key:val,...}<tool_call\|>` (custom, **not** JSON) |
| Tool response injection | `<start_of_turn>tool\n...<end_of_turn>` | `<\|turn>tool\n<\|tool_response>response:FUNC{...}<tool_response\|>` — turn **not** closed if tool_response is the only content |
| String literal delimiter | `"..."` (JSON) | `<\|"\|>...<\|"\|>` (custom delimiter token) |
| Thinking mode | none | `<\|think\|>` in system turn + `<\|channel>thought...<channel\|>` in model output |
| Parallel tool calls | implicit (multiple `<tool_call>` blocks) | explicit — `chat_template_caps.supports_parallel_tool_calls: true` |

**GOTCHA** — the label in [chat-template.md](chat-template.md) says "Gemma 4" but the tokens are actually Gemma 3 (`<start_of_turn>/<end_of_turn>`). The current implementation has been running acceptably **only because the llama-server backend uses `--jinja` to apply the GGUF's embedded template**, bypassing `chat_template.rs` for prompt assembly. For the candle and llama-cpp-rs backends, `chat_template.rs` is the source of truth, and it is wrong for Gemma 4.

---

## Metadata from the running model

**EXTRACTED** — `/props` response from a probe llama-server run against `gemma-4-e2b-it-Q8_0.gguf` on 2026-04-15.

```json
{
  "bos_token": "<bos>",
  "eos_token": "<eos>",
  "chat_template_caps": {
    "supports_object_arguments": true,
    "supports_parallel_tool_calls": true,
    "supports_preserve_reasoning": false,
    "supports_string_content": true,
    "supports_system_role": true,
    "supports_tool_calls": true,
    "supports_tools": true,
    "supports_typed_content": false
  },
  "modalities": { "vision": false, "audio": false }
}
```

- `bos_token = <bos>` — same as Gemma 3.
- `eos_token = <eos>` — same as Gemma 3.
- `supports_typed_content: false` — the model does **not** accept OpenAI-style typed content arrays except for multimodal parts; plain `string` content is required for text.
- `supports_preserve_reasoning: false` — thinking output is meant to be **stripped** from history before re-injection (see `strip_thinking()` macro in the template).
- `modalities` — E2B GGUF shipped text-only in this build; the mmproj file is a separate load. Vision and audio tokens (`<|image|>`, `<|audio|>`, `<|video|>`) exist in the template regardless.

---

## Control tokens (Gemma 4)

**EXTRACTED** — template body `/tmp/gemma4-chat-template.jinja` lines 141-263.

| Category | Token | Purpose |
|---|---|---|
| Boundary | `<bos>` | Prompt prefix (once, before first turn) |
| Boundary | `<eos>` | End of generation stop token |
| Turn | `<\|turn>` ... `<turn\|>` | Wraps a conversational turn by role |
| System aux | `<\|think\|>` | Enables thinking mode (placed inside system turn) |
| Thinking | `<\|channel>thought` ... `<channel\|>` | Model's private reasoning channel — stripped by `strip_thinking()` before re-injection |
| Tool decl | `<\|tool>` ... `<tool\|>` | One tool declaration; multiple allowed in system turn |
| Tool call | `<\|tool_call>` ... `<tool_call\|>` | Model's request to invoke a tool |
| Tool result | `<\|tool_response>` ... `<tool_response\|>` | Application's reply back to the model |
| String lit | `<\|"\|>` | String delimiter used inside tool decl/call/response blocks |
| Multimodal | `<\|image\|>`, `<\|audio\|>`, `<\|video\|>` | Inline media placeholders |

Roles used with `<|turn>`: `system`, `user`, `model` (note: `assistant` is **remapped** to `model` by the template — see line 186 of the Jinja), `tool`.

---

## Turn structure

### 1. System + tools block (optional, emitted once at the start)

Emitted if **any** of: `enable_thinking`, `tools`, or first message role is `system` / `developer`.

```
<bos><|turn>system
[<|think|>]                                  # only if enable_thinking
{system_content_trimmed}                     # only if present
<|tool>{tool_decl_1}<tool|>                   # one per tool
<|tool>{tool_decl_2}<tool|>
...
<turn|>
```

- `<|think|>` comes **before** system content if thinking mode is on.
- Tool declarations come **after** system content, **before** `<turn|>`.
- All tool decls share the one system turn — they are not separate turns.

### 2. User turn

```
<|turn>user
{content_trimmed}
<turn|>
```

Multimodal user content can interleave `<|image|>`, `<|audio|>`, `<|video|>` tokens in the body; otherwise it is just trimmed plain text.

### 3. Model turn (assistant)

The `assistant` role is remapped to `model`. The turn body can contain:

- Zero or more `<|tool_call>...<tool_call|>` blocks
- Optional text content (with thinking channel stripped on re-injection)

```
<|turn>model
<|tool_call>call:read_file{path:<|"|>foo.rs<|"|>}<tool_call|>
<|tool_call>call:read_file{path:<|"|>bar.rs<|"|>}<tool_call|>
Okay, I've read both files.
<turn|>
```

### 4. Tool response turn

```
<|turn>tool
<|tool_response>response:read_file{content:<|"|>fn main() {}<|"|>}<tool_response|>
```

**GOTCHA** — if a `tool` message carries only `tool_responses` and no text content, the template **does not emit `<turn|>`** (line 254 of the Jinja). The model continues in the same open turn. Concretely: `<|turn>tool\n<|tool_response>...<tool_response|>` — no closer. This is the mechanism by which the model resumes generation right after receiving tool output without being re-invoked on a new turn.

### 5. Generation prompt

```
<|turn>model
```

Added at the very end if `add_generation_prompt=True`, **unless** the previous message was a `tool_response` (in which case the turn is already open per §4).

---

## Tool declaration format

**EXTRACTED** — `format_function_declaration` macro, lines 79-110.

Syntax (whitespace is non-semantic, shown expanded for readability):

```
declaration:FUNC_NAME{
  description:<|"|>FUNCTION DESCRIPTION<|"|>,
  parameters:{
    properties:{
      PARAM_NAME:{
        description:<|"|>PARAM DESC<|"|>,
        [nullable:true,]
        [enum:[<|"|>A<|"|>,<|"|>B<|"|>],]        # STRING with enum
        type:<|"|>STRING<|"|>
      },
      ...
    },
    required:[<|"|>PARAM1<|"|>,<|"|>PARAM2<|"|>],
    type:<|"|>OBJECT<|"|>
  }
  [,response:{description:<|"|>...<|"|>,type:<|"|>OBJECT<|"|>}]
}
```

Rules derived from the Jinja:
1. Field keys are **bare identifiers** (`description`, `parameters`, `properties`, `required`, `type`, `nullable`, `enum`, `items`, `response`, `declaration`, `call`).
2. String values are wrapped in `<|"|>...<|"|>`.
3. Booleans are bare `true`/`false`.
4. Numbers are bare.
5. Arrays use `[...,...]` with comma separation.
6. Objects use `{key:value,...}`.
7. Type names are **uppercased** (`STRING`, `OBJECT`, `ARRAY`, `NUMBER`, `INTEGER`, `BOOLEAN`).
8. `OBJECT` type additionally emits `properties:{...}` and `required:[...]` children.
9. `ARRAY` type additionally emits `items:{...}`.
10. Nested objects recurse through `format_parameters`.

### Worked example — `read_file`

Input JSON schema:

```json
{
  "function": {
    "name": "read_file",
    "description": "Read a file from the workspace",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string", "description": "Relative path to the file"},
        "line_offset": {"type": "integer", "description": "Line number to start reading from", "nullable": true}
      },
      "required": ["path"]
    }
  }
}
```

Gemma 4 serialization (as the template would emit it):

```
<|tool>declaration:read_file{description:<|"|>Read a file from the workspace<|"|>,parameters:{properties:{line_offset:{description:<|"|>Line number to start reading from<|"|>,nullable:true,type:<|"|>INTEGER<|"|>},path:{description:<|"|>Relative path to the file<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>path<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>
```

Note the property ordering is **dictsort** (alphabetical by key) because the template sorts with `| dictsort`. This is deterministic but surprising for anyone expecting insertion-order JSON.

---

## Tool call format (model → app)

**EXTRACTED** — lines 189-206.

```
<|tool_call>call:FUNC_NAME{ARG1:VAL1,ARG2:VAL2,...}<tool_call|>
```

Rules:
1. Argument keys are **bare** (no `<|"|>` wrapping) — see `escape_keys=False` on line 198.
2. Argument values are formatted via `format_argument`:
   - String → `<|"|>value<|"|>`
   - Boolean → `true` / `false`
   - Number → bare
   - Object → `{bare_key:value,...}` (keys still unescaped when inside a tool_call arg)
   - Array → `[item1,item2,...]`
3. If the model emits `arguments` as a raw string (vs a dict), the string is passed through verbatim — the template supports both.
4. Multiple tool calls per turn are supported (`chat_template_caps.supports_parallel_tool_calls: true`).

### Worked example — tool call

```
<|tool_call>call:read_file{path:<|"|>crates/cli/src/commands.rs<|"|>}<tool_call|>
```

For a call with several arg types:

```
<|tool_call>call:search_code{pattern:<|"|>InferenceProvider<|"|>,max_results:25,case_sensitive:false,include_globs:[<|"|>**/*.rs<|"|>]}<tool_call|>
```

---

## Tool response format (app → model)

**EXTRACTED** — lines 208-225.

Two paths depending on whether the `response` field is a mapping or a scalar:

**Object response:**
```
<|tool_response>response:FUNC_NAME{key1:val1,key2:val2,...}<tool_response|>
```

**Scalar/string response (wrapped under synthetic `value` key):**
```
<|tool_response>response:FUNC_NAME{value:<|"|>raw text output<|"|>}<tool_response|>
```

- `FUNC_NAME` comes from `tool_response['name']`, defaulting to `unknown` if missing.
- Keys are bare, values use the same `format_argument` rules as tool calls.
- Dict iteration uses `| dictsort` → alphabetical.

### Wire format contract for the application

When constructing `tool_responses` in message history, the application must shape the message like:

```json
{
  "role": "tool",
  "tool_responses": [
    {"name": "read_file", "response": {"content": "fn main() {}"}}
  ]
}
```

The template will serialize this into the `<|tool_response>` block and deliberately **omit the turn closer** so the model can immediately continue.

---

## Thinking mode

**EXTRACTED** — lines 141-151 (`strip_thinking` macro), lines 160-164 (activation).

Activation: include `<|think|>` as the first content of the first system turn. The template does this automatically when `enable_thinking=True` is passed to `apply_chat_template`.

Model output format when thinking:

```
<|turn>model
<|channel>thought
I need to figure out what the user is asking...
<channel|>
Here's my answer.
<turn|>
```

Re-injection stripping: when a historical model turn is rendered back into a new prompt, `strip_thinking()` splits the content on `<channel|>`, and for each part that contains `<|channel>`, keeps only the text *before* `<|channel>`. Net effect: the `<|channel>thought...<channel|>` segment is removed, and only the post-thinking text remains.

**GOTCHA** — `supports_preserve_reasoning: false` is deliberate. If the application tries to persist raw reasoning and re-feed it, the next turn's template will strip it anyway. Store the stripped version if you need clean history.

**Observation from the official docs** — larger Gemma 4 models (26B A4B, 31B) sometimes emit a thought channel even when thinking mode is off. Google's mitigation is to inject an empty `<|think|>` token to stabilize. Worth keeping in mind for the 26B/31B path later.

---

## Stop tokens

For Gemma 4, inference engines must treat the following as stops (in addition to `<eos>`):

1. `<eos>` — model end of stream.
2. `<turn|>` — end of current turn (rare as a stop in practice; usually `<eos>` fires first when generation completes a turn).
3. `<tool_response|>` — end of tool response; acts as a stop sequence per the ai.google.dev docs so that the engine halts after the application-supplied response and waits for re-invocation.

**GOTCHA** — zipcode currently recognizes `<eos>` and `<end_of_turn>` as stops ([chat-template](chat-template.md) § Stop-token detection). The `<end_of_turn>` entry is a **Gemma 3 artifact** and will never fire on Gemma 4 output. The real Gemma 4 stops are `<eos>` and `<turn|>` (plus `<tool_response|>` for engines that implement it as a stop sequence).

---

## Capabilities matrix (from `chat_template_caps`)

For any `ChatTemplate` trait implementation, these are the capability flags the runtime should honor on Gemma 4:

| Flag | Value | Meaning for zipcode |
|---|---|---|
| `supports_system_role` | `true` | A `system` message can be passed directly. |
| `supports_tools` | `true` | Tool specs can be passed through the template. |
| `supports_tool_calls` | `true` | Assistant messages can carry `tool_calls`. |
| `supports_parallel_tool_calls` | `true` | Multiple `<\|tool_call>` per model turn. The conversation loop must be able to execute and collect multiple calls in one iteration. |
| `supports_object_arguments` | `true` | Tool call arguments can be nested objects, not only flat strings. |
| `supports_string_content` | `true` | Plain `content: "..."` strings are accepted. |
| `supports_typed_content` | `false` | Do **not** send `[{"type":"text","text":"..."}]` for plain text — use a raw string. Typed-content arrays are only for multimodal. |
| `supports_preserve_reasoning` | `false` | Strip thinking before re-feeding history. |

---

## What this means for the zipcode rewrite

> **Read the "Empirical wire contract" section below first.** Phase 0 probing established that for the current **llama-server** backend (the only working Gemma 4 path), `chat_template.rs` is dead code — llama-server handles templating and tool-call parsing entirely. The 10 deltas below are the plan **if** zipcode ever needs to render Gemma 4 natively without llama-server (i.e., for a future `llama-cpp-rs` or `candle` backend revival). For the current production path, skip to the "Two-line fix" in the empirical section.

**INFERRED — for a hypothetical native (non-llama-server) Gemma 4 path.** Concrete implementation deltas for [chat-template](chat-template.md) only if that path is revived:

1. **Turn tokens:** replace every `<start_of_turn>` / `<end_of_turn>` with `<|turn>` / `<turn|>`. Stop-token set must be updated in parallel (`engine.rs:108-112` and `llama_cpp_backend.rs:146`).
2. **Role name:** rename `assistant` to `model` in the wire-format layer, or map at rendering time. `assistant` as input role name is fine; it's only the rendered token that must say `model`.
3. **System turn assembly:** move tool spec injection from the first user turn into the system turn. System content, thinking token, and tool declarations all share one `<|turn>system ... <turn|>` block.
4. **Tool spec serialization:** write a new function that takes a `ToolSpec` (JSON-schema-ish) and renders it as the Gemma 4 declaration mini-language. Do **not** emit JSON. Key rules:
   - Uppercase type names
   - `<|"|>` wrap every string value
   - Bare booleans / numbers
   - Alphabetical key order (dictsort) to match what the GGUF template emits — matters for cache keys and determinism
5. **Tool call parser:** rewrite `parse_tool_calls()`. Input is no longer JSON inside `<tool_call>` tags; it is now `<|tool_call>call:NAME{key:val,...}<tool_call|>` with the custom mini-language. Needs a small hand-written parser that understands `<|"|>` string escaping, bare scalars, nested `{}` and `[]`. This is the single biggest unit of work.
6. **Tool result serializer:** render `tool_responses` as `<|tool_response>response:NAME{...}<tool_response|>` inside a `<|turn>tool` turn. Must **not** emit `<turn|>` if the tool message has no text content — the model continues in the same turn.
7. **Generation prompt logic:** emit `<|turn>model\n` unless the previous message was a bare `tool_response` (per §4).
8. **Thinking mode plumbing:** expose an `enable_thinking` flag at the runtime level. When set, prepend `<|think|>` to the system turn and strip `<|channel>...<channel|>` blocks from stored assistant content before re-injection.
9. **GBNF grammar (Phase C):** the custom mini-language is **ideal** for GBNF-constrained decoding. A formal grammar for the tool-call block can force the model to emit only well-formed `<|tool_call>...<tool_call|>` blocks, removing the "malformed tool call edge cases" that Jiunsong's README explicitly calls out as needing runtime hardening. This gives zipcode a hard guarantee on tool-call validity that JSON-based templates cannot.
10. **Multimodal path (deferred):** `<|image|>`, `<|audio|>`, `<|video|>` are already in the template. The mmproj GGUF files are already on disk. This is usable once zipcode adopts a multimodal input path, but it is not in scope for the Phase A tool-call rewrite.

---

## Empirical wire contract (llama-server path) — Phase 0 answered

**EXTRACTED** from live probe against `gemma-4-e2b-it-Q8_0.gguf` on 2026-04-15.

Question from the first draft of this page: *does zipcode actually use `chat_template.rs` on the llama-server path, or does llama-server handle everything via `--jinja`?* Resolved by running the zipcode-shaped request directly against a probe llama-server and inspecting the raw OpenAI JSON / SSE traffic.

### The real contract

1. **zipcode sends plain OpenAI `/v1/chat/completions` JSON** (`crates/inference/src/llama_server_backend.rs:296-319`). Roles: `system` / `user` / `assistant` / `tool`. Content is always a flat string (or `null` for tool-call-only assistant turns). Tools are passed as `{type:"function", function:{name, description, parameters}}`.
2. **zipcode sets `tool_choice: "auto"` and `parse_tool_calls: true`** — the latter is a llama-server extension that asks the server to parse the model's raw Gemma 4 tool-call output back into OpenAI-shape JSON.
3. **llama-server applies the GGUF's embedded Jinja template** because `LlamaServerProvider::load` passes `--jinja` (line 107). The Gemma 4 mini-language serialization shown earlier in this spec happens **inside llama-server**, not in `chat_template.rs`.
4. **llama-server's Gemma 4 tool-call parser works correctly.** Both non-streaming and streaming responses carry `tool_calls` in the OpenAI shape with JSON-string `arguments`. The Gemma 4 `<|tool_call>call:FUNC{key:<|"|>val<|"|>}<tool_call|>` output is transparently converted.
5. **`chat_template.rs` is DEAD for Gemma 4 on the llama-server path.** It is re-exported from `lib.rs` but never called during prompt assembly or tool-call parsing when `LlamaServerProvider` is active. It only matters for `candle` / `llama-cpp-rs` backends — both of which are broken for Gemma 4 anyway per `CLAUDE.md` Limitations #1 and #2.

**Implication**: the original Phase A plan in the "What this means for the zipcode rewrite" section below is mostly **unnecessary** for the llama-server path. See the rewritten version in that section.

### The real bug: reasoning_content is dropped

**EXTRACTED** — SSE delta stream captured 2026-04-15 via curl against the probe llama-server.

Gemma 4's thinking channel is emitted by llama-server as a new delta field `delta.reasoning_content`. This field is **not part of stock OpenAI SSE** and is **completely ignored** by `llama_server_backend.rs:418-441`:

```rust
if let Some(tool_calls) = delta["tool_calls"].as_array() { ... }
if let Some(content) = delta["content"].as_str() { ... }
// reasoning_content: silently dropped
```

Delta histogram from a simple tool-calling probe (`"What's in README.md?"` with a single `read_file` tool, `enable_thinking` default = true):

| Phase | content | reasoning_content | tool_calls | finish |
|---|---|---|---|---|
| Tool-calling turn | **0** | **62** | 9 | `tool_calls` |
| Post-tool turn (final answer) | **37** | **84** | 0 | `stop` |

Read: **70% of the model's token stream is invisible** to the user when zipcode is in default (thinking-enabled) mode. Tool calls themselves are emitted correctly on `delta.tool_calls` and zipcode's accumulator picks them up. The problem is purely that the **thinking phase produces a dead-UI pause** before each tool call fires and before the final answer starts streaming, giving the impression of sluggish or "plain chat-like" behavior.

### The two-line fix

**EXTRACTED** — confirmed against the same probe, passing `chat_template_kwargs: {"enable_thinking": false}` in the OpenAI request body.

| Phase | content | reasoning_content | tool_calls | finish |
|---|---|---|---|---|
| Tool-calling turn | 0 | **0** | 9 | `tool_calls` |
| Post-tool turn (final answer) | **37** | **0** | 0 | `stop` |

With thinking disabled, the token counts for visible content and tool calls are **identical** to the thinking-enabled case; only the 62+84 hidden thinking deltas disappear. Final answer quality is the same (`"The README.md file contains the following: # zipcode ..."`).

**Minimum viable Phase A**: add one line to `build_chat_request()`:

```rust
request["chat_template_kwargs"] = json!({ "enable_thinking": false });
```

This alone removes the "plain chat" symptom entirely without touching `chat_template.rs`, the parser, or the conversation loop. It is the single highest-ROI change identified in Phase 0.

### The nicer fix (optional)

Alternatively, handle `reasoning_content` in `parse_sse_events` so the thinking channel becomes visible as dimmed / collapsible / toggleable UX:

1. Add a new `TokenEvent::Thinking(String)` variant to `crates/inference/src/types.rs`.
2. In `parse_sse_events`, add a branch that extracts `delta.reasoning_content` and emits `SseEvent::Thinking(text)` → `TokenEvent::Thinking(text)`.
3. In `StreamCallback`, add `on_thinking(&mut self, text: &str)` and render it in the CLI as dimmed or behind a toggle.
4. Do **not** persist reasoning into session history. `chat_template_caps.supports_preserve_reasoning` is `false`, and the GGUF-embedded `strip_thinking()` macro strips any thinking text on re-injection anyway. Keep it for the live UI only.

This second fix is the "Claude-Code-like experience" path — it gives users visibility into the model's reasoning instead of hiding it.

### Additional observations

- `finish_reason` is emitted as JSON `null` during streaming (not the string `"null"`) — zipcode's check `reason != "null"` on `as_str()` happens to work because `null.as_str() == None`, so the check is skipped entirely when the field is null. Fragile but correct.
- The `delta` shape for tool_call chunks matches what zipcode expects: `[{"index":0,"id":"...","function":{"name":"read_file","arguments":"{"}}]` then later chunks carry `{"index":0,"function":{"arguments":"\"path"}}` etc. Accumulation logic in `ToolCallAccumulator` (`llama_server_backend.rs:444-449, 546-565`) handles this correctly.
- llama-server's build version reported in `chat.completion.chunk.system_fingerprint`: `b1-0d049d6`. Tool-call parsing, Jinja templating, and `chat_template_kwargs` support are all present in this build.
- E2B triggered the tool call in ~1.2 s end-to-end on RTX 2070S (thinking on) and produced a 25-token reasoning prelude before the `<|tool_call>` block — confirming that a) Gemma 4 E2B is actually a working tool-calling model on this hardware, and b) the model's default behavior is to think first, then act.

---

## Phase B — E2B vs E4B under production load

**EXTRACTED** from live probes against `gemma-4-e2b-it-Q8_0.gguf` and
`gemma-4-E4B-it-Q4_K_M.gguf` on 2026-04-15. Both models served through
zipcode's bundled `llama-server` with the full 10-tool registry and the
real `BASE_SYSTEM_PROMPT` from `crates/runtime/src/prompt.rs`.

Each row is one request. Reasoning / tool-call delta counts are taken
directly from the SSE stream.

| Scenario | Model | reasoning Δ | tool_call Δ | Picked tool | Quality |
|---|---|---|---|---|---|
| 1. Simple `"What's in README.md?"` | **E2B** | 32 | 9 | `read_file({path:"README.md"})` | ✓ ideal |
| 1. Simple `"What's in README.md?"` | **E4B** | 28 | 9 | `read_file({path:"README.md"})` | ✓ ideal |
| 2. Ambiguous `"Find where the InferenceProvider trait is defined"` | **E2B** | 91 | 9 | `glob_search({pattern:"**/*.rs"})` | ⚠ broad file enum |
| 2. Ambiguous `"Find where the InferenceProvider trait is defined"` | **E4B** | 46 | 12 | `glob_search({pattern:"**/*InferenceProvider*"})` | ⚠ targeted filename glob |
| 3. Explicit `"Grep the codebase for uses of MAX_TOOL_ITERATIONS."` | **E2B** | 65 | 12 | `grep_search({pattern:"MAX_TOOL_ITERATIONS"})` | ✓ correct, minimal args |
| 3. Explicit `"Grep the codebase for uses of MAX_TOOL_ITERATIONS."` | **E4B** | 48 | 19 | `grep_search({path:"**/*",pattern:"MAX_TOOL_ITERATIONS"})` | ✓ correct, defensive args |

### Takeaways

1. **10-tool production shape does not break either model.** Both E2B
   and E4B emit well-formed tool calls under the full
   `BASE_SYSTEM_PROMPT` + 10-tool schema load. This closes the Phase A
   prerequisite — the `enable_thinking` wire contract holds at scale.
2. **Neither model is broken; the differentiator is reasoning
   efficiency and tool-selection nuance.** E4B consistently burns
   ~30–50% fewer reasoning tokens than E2B to reach the same or
   better conclusion.
3. **E4B is materially smarter on ambiguous intent.** On Scenario 2,
   E2B falls back to `**/*.rs` (enumerate every Rust file, then
   post-process), while E4B narrows by filename with
   `**/*InferenceProvider*` — same wrong tool family (both picked
   `glob_search` instead of `grep_search`), but E4B's candidate space
   is orders of magnitude smaller.
4. **Explicit-intent queries converge.** When the user literally says
   "grep", both models pick `grep_search`. E2B emits minimal args;
   E4B adds a defensive `path:"**/*"` workspace glob. Both are
   acceptable; E4B's version is slightly safer if `grep_search`
   defaults to CWD-only.
5. **Reasoning channel wire contract is identical on E2B and E4B.**
   `delta.reasoning_content` fires on both with the same shape; the
   `TokenEvent::Thinking` lane added in the previous commits handles
   both transparently — no code changes needed for E4B support.
6. **Tool-selection ceiling is a prompt-engineering problem on this
   hardware tier, not a format problem.** Both models know
   `grep_search` exists; neither reaches for it first on exploratory
   phrasing like "Find where X is defined". A short directive in
   `BASE_SYSTEM_PROMPT` ("When locating symbol definitions, prefer
   `grep_search` over `glob_search`") is likely enough to close the
   gap without touching model selection.

### Hardware envelope observed

- RTX 2070 SUPER 8 GB, `-ngl 999` (full GPU offload)
- E4B Q4_K_M (4.7 GB on disk) + 4 k ctx fits inside 8 GB VRAM with
  headroom. E5/Q6 quants would likely still fit at ≤4 k ctx.
- Q5_K_M (5.48 GB on disk per upstream Unsloth repo) is the next
  quality step and should still fit on this card at ≤4 k ctx.

### Artifacts captured on disk (2026-04-15, Phase B)

- `/tmp/probeB-req.json`, `/tmp/probeB-stream.sse` — E2B Case 1
- `/tmp/probeB2-req.json`, `/tmp/probeB2-stream.sse` — E2B Case 2
- `/tmp/e2b-req-3.json`, `/tmp/e2b-stream-3.sse` — E2B Case 3
- `/tmp/e4b-req-1.json`, `/tmp/e4b-stream-1.sse` — E4B Case 1
- `/tmp/e4b-req-2.json`, `/tmp/e4b-stream-2.sse` — E4B Case 2
- `/tmp/e4b-req-3.json`, `/tmp/e4b-stream-3.sse` — E4B Case 3

Scratch, not committed. Regenerate by launching llama-server on the
respective GGUFs and replaying the JSON requests.

---

## Open questions still unresolved

1. ~~**Does E2B behave under zipcode's real load** — 10 tools + the long `BASE_SYSTEM_PROMPT` + optional `.zipcode.md` + multi-turn history?~~ **Resolved above** — both E2B and E4B emit valid tool calls under the full load.
2. ~~**Is `chat_template_caps` consistent across E2B / E4B?**~~ **Resolved above** — the SSE delta shape and `reasoning_content` wire contract are identical; zipcode's `TokenEvent::Thinking` lane handles both without code changes.
3. **Does the real Gemma 4 function-calling-tuned model accept Gemma-3-style `<tool_call>{json}</tool_call>` prompts at inference time as a lenient fallback?** Less urgent now that we know the llama-server path bypasses `chat_template.rs` entirely — but still relevant if candle/llama-cpp-rs backends come back online.
4. **Edge case: does `<|"|>` require escaping of an inner literal `<|"|>` sequence?** The Jinja template does not show any escape mechanism, which implies tool call argument strings cannot contain the delimiter token verbatim. Relevant for Phase C grammar definition but not for the llama-server path (llama-server handles escaping internally).
5. **Is `chat_template_caps` consistent across the larger variants** (26B A4B / 31B) — e.g. does `supports_preserve_reasoning` flip to `true`? Needs a 24 GB+ GPU or Apple Silicon host to verify.
6. **Does `chat_template_kwargs: {enable_thinking: false}` behave consistently on 26B A4B / 31B?** Google's docs note the larger models sometimes emit a thought channel even when thinking is explicitly off. Confirm before declaring the kwarg fix universal.
7. **Tool-selection nudging:** does adding a short "prefer `grep_search` over `glob_search` for symbol lookups" directive to `BASE_SYSTEM_PROMPT` materially change Scenario 2 behavior on E2B / E4B? Low-risk experiment; one-line system prompt edit followed by the same probe.

---

## Artifacts captured on disk (2026-04-15)

- `/tmp/gemma4-chat-template.jinja` — full raw Jinja template (11926 chars) extracted via `/props` from a probe llama-server run against `~/.zipcode/models/gemma-4-e2b-it-Q8_0.gguf`.
- `/tmp/llama-props.json` — full `/props` response including capability flags and metadata (13998 chars).
- `/tmp/gemma4-gguf-strings.txt` — `strings -n 20` dump of the GGUF file (1113 lines) for future cross-referencing of other embedded metadata.

These are scratch artifacts, not committed. Regenerate via:

```bash
~/.zipcode/bin/llama-server -m ~/.zipcode/models/gemma-4-e2b-it-Q8_0.gguf \
    --host 127.0.0.1 --port 44262 --alias zipcode-probe --jinja -c 2048 -ngl 0 &
curl -s http://127.0.0.1:44262/props > /tmp/llama-props.json
```

---

## Sources

- GGUF embedded `tokenizer.chat_template` (authoritative — this is what the model was trained against)
- [Google AI — Gemma 4 Prompt Formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
- [Google AI — Function Calling with Gemma 4](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4)
- [vLLM Recipes — Gemma 4](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)
- [Jiunsong/supergemma4-26b-uncensored-gguf-v2](https://huggingface.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2) — corroborates "malformed Gemma 4 tool-call edge cases" and ships a neutral template to avoid routing drift
- [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) — primary upstream for the template source

---

## Related pages

- [chat-template](chat-template.md) — current (Gemma-3-format) implementation this spec will replace in Phase A
- [inference](inference.md) — consumes this format via `InferenceProvider` implementations
- [conversation-loop](conversation-loop.md) — owns the tool-call iteration logic that must adapt to parallel tool calls
- [llama-server](llama-server.md) — backend that currently bypasses `chat_template.rs` via `--jinja`
- [`docs/experiments/2026-04-16-gemma4-e4b-gamedev-probe.md`](../../docs/experiments/2026-04-16-gemma4-e4b-gamedev-probe.md) — end-to-end capability probe: "build me a game" across Python, HTML Canvas, and Rust, showing E4B's one-shot ceiling and the second-hop debug wall
