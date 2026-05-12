# gotchas — Non-obvious couplings, silent failures, hardcoded invariants

Consolidated list of things that will bite you. Each entry links to its detailed page.

---

## Backend reality check

**EXTRACTED** from `CLAUDE.md` + code.

The workspace ships with 3 inference backends, but **only one is production-viable for Gemma 4**:

| Backend | Status | File | Problem |
|---------|--------|------|---------|
| `llama-server` | ✅ Working | `llama_server_backend.rs` | Subprocess + SSE; the path you want |
| `llama-cpp` | ❌ Blocked upstream | `llama_cpp_backend.rs` | `llama-cpp-rs` 0.1.141 bundles an old llama.cpp without Gemma 4 arch support |
| `candle` (default feature) | ❌ Placeholder only | `engine.rs:1-4` | candle 0.8 has no `quantized_gemma` module — uses `quantized_llama` as stand-in; compiles but won't correctly load real Gemma GGUF |

**What this means practically:** current CLI startup now tries to auto-select `llama-server` for Gemma 4 when a helper is available, but the practical production path is still helper-backed execution rather than native `llama-cpp`. If no helper is available, Gemma 4 remains blocked. See [project-direction](project-direction.md).

---

## #1 — Chat template is Gemma-only

**Location:** [`chat-template.md`](chat-template.md)

Every prompt flows through a hardcoded `<start_of_turn>/<end_of_turn>` + `<tool_call>` format. Models that emit tool calls differently (Qwen, Llama 3, OpenAI-compatible) will look like they finished their turn without requesting any tools, so [`ConversationLoop`](conversation-loop.md) exits. The user sees text output but no tool execution.

**Symptom:** "I asked for help editing a file and it just described what it would do without actually calling `edit_file`."

**Fix path:** needs a `ChatTemplate` trait per backend. Not implemented.

---

## #2 — Candle is a stub

**Location:** `crates/inference/src/engine.rs:1-4` (TODO comment)

candle 0.8 has no `quantized_gemma`. `InferenceEngine` currently uses `quantized_llama` loader which compiles against Gemma GGUFs but won't actually produce correct output. Good for compile-time contract testing, useless at runtime.

**Symptom:** `--backend candle` runs without crashing and streams tokens, but the tokens are nonsense.

---

## #3 — Tool output silently truncated at 8 KB

**Location:** `crates/tools/src/lib.rs:503` + `:391-418`

`MAX_TOOL_OUTPUT_BYTES = 8192`. If a tool returns more, `ToolResult::truncate()` finds a safe UTF-8 boundary, trims, and appends `[truncated: showing first X bytes of Y]`. The model sees the note but has no way to request "show me the rest".

**Symptom:** `bash "ls -laR /"` returns the first 8 KB and the model plans as if that's the whole tree.

**Fix path:** add a `follow_up` mechanism or a per-tool config override. Not implemented.

---

## #4 — 25-iteration cap returns an error mid-turn

**Location:** `crates/runtime/src/conversation.rs:94` (`MAX_TOOL_ITERATIONS` declaration)

When the model chains tools > 25 times in one turn, `run_turn()` returns `Err`. The session IS saved with all the partial messages, so the NEXT turn begins with a model that saw its own loop get killed — a confused-Claude effect.

**Symptom:** "It stopped mid-task and gave a weird apology."

**Fix path:** graceful summarization before the error, or a higher cap, or a resumable bookmark. Not implemented.

---

## #5 — Path traversal prevention depends on every tool calling it

**Location:** `crates/tools/src/lib.rs:19-97` — `resolve_and_validate_path()`

Single gate that every file-touching tool routes paths through. Works today because all 5 file tools (`read_file`, `write_file`, `edit_file`, `glob_search`, `grep_search`) remember to call it. Tests at `lib.rs:470+` verify the rejection path.

**Silent-failure mode:** a new tool that forgets to call it will be accepted without CI catching it. There's no type-level guarantee.

**Fix path:** newtype wrapper (`ValidatedPath`) that only `resolve_and_validate_path()` can produce. Not implemented.

---

## #6 — llama-server subprocess lifecycle

**Location:** `crates/inference/src/llama_server_backend.rs:339-360`

- `Drop` kills the child. Fine for normal exit.
- Health check polls `/health` with a hardcoded timeout. Slow machines loading huge models can time out even though the server will eventually come up.
- Port reservation (`reserve_local_port()`) is race-prone: bind to `:0`, read the port, drop the listener, assume the port is still free when the child binds. On busy machines this can collide.

**Symptom:** "llama-server sometimes fails to start and doctor reports it as broken even though running it manually works."

---

## #7 — Agent tool is a stub

**Location:** `crates/tools/src/agent.rs`

Returns "not yet implemented". Included in the registry, listed in `tool_search`, permission-gated like any other tool, but calling it is a no-op. Permission behavior IS defined: `Denied` in `read-only` and `workspace-write`, `Allowed` in `full-access` (falls through the `WorkspaceWrite` wildcard arm — `permission.rs:31-47`). Test: `test_workspace_write_denies_agent`.

---

## #8 — Session directory grows forever

**Location:** `crates/runtime/src/session.rs`

`~/.zipcode/sessions/{uuid}.json` per session. No rotation, no retention, no index. After a few weeks of heavy use you'll have thousands of JSON files and `ls` will get slow.

**Fix path:** periodic cleanup, or a SQLite index with TTL. Not implemented.

---

## #9 — GPU env vars are case-sensitive

**Location:** env var read in `crates/cli/src/repl.rs` + `crates/runtime/src/config.rs`

`ZIPCODE_GPU_LAYERS` — exact spelling. `zipcode_gpu_layers`, `Zipcode_Gpu_Layers` all silently ignored, and the model runs CPU-only.

**Symptom:** "I set the env var but it's still slow."

**Fix path:** could accept case-insensitive by reading both uppercase + exact. Not implemented.

---

## #10 — Project root detection is order-sensitive

**Location:** `crates/runtime/src/config.rs:248-259`

`find_project_root()` walks ancestors looking for `.zipcode.json`, then `.zipcode.md`, then `.git`. First match wins. If you have a nested git submodule, the nested `.git` will anchor project root at the submodule instead of the outer workspace.

**Symptom:** "Why is it reading config from the wrong directory?"

**Fix path:** prefer the outermost `.zipcode.*` before falling back to the innermost `.git`. Not implemented.

---

## #11 — Permission state split across two crates

**Location:** `crates/tools/src/lib.rs:306` + `crates/runtime/src/permission.rs:5`

Adding a permission tier requires editing both crates. Adding a new tool requires remembering to slot it into the permission matrix at `permission.rs:31`. Missing that step means the tool is silently denied in read-only mode.

**Fix path:** declare permission level as part of `Tool` trait metadata. Not implemented.

---

## Related pages

- All community pages cross-link to this list.
- See [`GRAPH_REPORT`](../GRAPH_REPORT.md#gotcha-summary-full-list-in-pagesgotchas) for the short table.
