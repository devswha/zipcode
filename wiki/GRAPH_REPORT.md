# GRAPH_REPORT — zipcode knowledge graph

Graphify-style summary of the zipcode `crates/` workspace as of 2026-05-03 (provenance anchors refreshed).

> This report is the "map" view of the wiki: god nodes ranked, surprising cross-crate connections, questions the graph can answer, and a consolidated gotcha list. Community pages live under [`pages/`](pages/).

---

## God Nodes (by degree, approximate)

Ranked by how many other concepts route through them.

| Rank | Node | Location | Why it's central |
|------|------|----------|------------------|
| 1 | `InferenceProvider` trait | `crates/inference/src/lib.rs:47` | 4 implementors; single point where `runtime` talks to inference. Swapping backends = implement one trait. |
| 2 | `ConversationLoop` | `crates/runtime/src/conversation.rs:58` | Holds `Box<dyn InferenceProvider>` + `ToolRegistry` + `PermissionPolicy` + `Session` + `system_prompt`. Every turn flows through `run_turn()`. |
| 3 | `Tool` trait + `ToolRegistry` | `crates/tools/src/lib.rs:431,445` | 10 implementors; `execute_tool()` (`:509`) is the single entry point from `runtime`. |
| 4 | `ChatMessage` | `crates/inference/src/types.rs:36` | Wire format carried by every component: REPL → conversation loop → inference → chat template → back. |
| 5 | `PermissionMode` | `crates/tools/src/lib.rs:305` | Referenced by `ToolContext`, `PermissionPolicy`, CLI args, config. Gates every tool call. |
| 6 | Gemma chat template | `crates/inference/src/chat_template.rs:11` | Every prompt is formatted through `format_conversation()` before reaching any backend. |
| 7 | `resolve_and_validate_path()` | `crates/tools/src/lib.rs:19` | Called by every file-touching tool (read, write, edit, glob, grep). Single path-safety gate. |

See [`pages/`](pages/) for detail on each.

---

## Cross-crate connections (surprising edges)

1. **`runtime` never imports `tools::*` for tool logic — only the trait interface.** `ConversationLoop` holds `tools: ToolRegistry` but only ever calls `execute_tool()`. The 10 concrete tool types stay behind the `Box<dyn Tool>` wall. **Why it matters:** adding a tool doesn't require changes in `runtime`.

2. **The conversation loop's iteration cap lives in `runtime`, but the tool-result truncation cap lives in `tools`.** `MAX_TOOL_ITERATIONS = 25` (`crates/runtime/src/conversation.rs:94`) and `MAX_TOOL_OUTPUT_BYTES = 8192` (`crates/tools/src/lib.rs:502`). Two independent safety limits, two independent crates. **Why it matters:** a runaway agent is bounded by both — model iterations AND per-tool output size.

3. **Chat template parsing is upstream of the conversation loop.** The model returns raw text; `chat_template::parse_tool_calls()` (`crates/inference/src/chat_template.rs:842`) extracts `<tool_call>` JSON before `ConversationLoop` ever sees a `ToolCallParsed`. **Why it matters:** a model that doesn't emit `<tool_call>` blocks (Qwen, Llama 3, OpenAI-compatible) will be silently tool-blind.

4. **`cli` owns model-path resolution, not `runtime`.** `crates/cli/src/repl.rs` resolves `--model` flag > project config > global config > `~/.zipcode/models` scan. **Why it matters:** automated test drivers need to pass `--model` or set config; `runtime` can't find a model by itself.

5. **Permission policy is split across two crates.** `PermissionMode` enum lives in `tools` (`:305`); `PermissionPolicy::check()` lives in `runtime` (`permission.rs:32`). **Why it matters:** adding a new permission tier requires editing both crates.

6. **Session persistence is JSON-per-file, not a database.** `~/.zipcode/sessions/{uuid}.json` (`crates/runtime/src/session.rs:100`). **Why it matters:** `ls ~/.zipcode/sessions/ | wc -l` scales linearly forever; no retention policy.

7. **The `llama-server` backend is the only fully-functional path today.** `llama-cpp-rs` 0.1.141 lacks Gemma 4 arch support; candle has no `quantized_gemma`. See [`gotchas`](pages/gotchas.md#backend-reality-check).

---

## Suggested questions the graph can answer

Drop these into a code search or walk the links — the wiki is pre-wired for them.

1. **"Where does a tool call actually execute?"** → `crates/tools/src/lib.rs:509` (`execute_tool`) called from `crates/runtime/src/conversation.rs:105-143`. Both pages cross-link.
2. **"What stops the agent from looping forever?"** → Two bounds: 25 tool iterations (`conversation.rs:94`) and 8 KB per tool result (`tools/lib.rs:502`).
3. **"Why doesn't model X work?"** → Almost always the hardcoded Gemma chat template. See [`chat-template`](pages/chat-template.md).
4. **"Where is `~/.zipcode/config.json` read?"** → `crates/runtime/src/config.rs:6` (`ZipcodeConfig::load`), merged with `.zipcode.json` from project root.
5. **"Which tools are blocked in read-only mode?"** → Everything except `read_file`, `glob_search`, `grep_search`, `tool_search`. `crates/runtime/src/permission.rs:32-49`.
6. **"How does llama-server get its GPU flags?"** → `ZIPCODE_GPU_LAYERS` / `ZIPCODE_FLASH_ATTENTION` env vars override `config.gpu_layers` / `config.flash_attention`, passed to subprocess as `-ngl` and `--flash-attn`. See [`llama-server`](pages/llama-server.md).
7. **"How is the system prompt built?"** → `crates/runtime/src/prompt.rs:28-60`: base prompt + permission line + cwd + `.zipcode.md` content.

---

## Gotcha summary (full list in [`pages/gotchas`](pages/gotchas.md))

| # | Gotcha | Location |
|---|--------|----------|
| 1 | Chat template is Gemma-only; other models appear tool-blind | `chat_template.rs:11-56` |
| 2 | Candle backend uses `quantized_llama` as a Gemma placeholder; compiles but won't load real Gemma | `engine.rs:1-4` |
| 3 | `llama-cpp-rs` 0.1.141 lacks Gemma 4 arch → zipcode falls back to `llama-server` subprocess | `llama_cpp_backend.rs` + CLAUDE.md |
| 4 | Tool output silently truncated at 8 KB; model isn't told how much was cut | `tools/lib.rs:502,391-410` |
| 5 | Tool iteration cap of 25 returns an error, not a graceful handoff | `conversation.rs:94` (declaration) |
| 6 | Agent tool is a stub — returns "not yet implemented" | `agent.rs` |
| 7 | Path traversal prevention depends on every file tool calling `resolve_and_validate_path()` | `tools/lib.rs:14-78` |
| 8 | `llama-server` subprocess killed in `Drop`; if the process crashes during health check, no explicit error path | `llama_server_backend.rs:143-148` |
| 9 | Session files grow unbounded in `~/.zipcode/sessions/` (no rotation) | `session.rs:100` |
| 10 | Env var names for GPU are case-sensitive; typos silently ignored | `config.rs` + CLI env read |

---

## Test surface (1,228 passing + 2 ignored across the workspace)

| Crate | Unit / inline | Integration | Notable |
|-------|---------------|-------------|---------|
| `inference` | 284 passing | 7 (template) | `chat_template.rs` alone carries 50+ parsing/formatting robustness tests; `llama_server_backend.rs` has parser/streaming tests covering thinking-mode request defaults, `reasoning_content` SSE handling, reasoning/tool-call stream separation, block-array content extraction, and default server options |
| `tools` | 186 | 62 (tools_integration) | Every tool has focused unit coverage; workspace-escape regressions exist for both glob and grep search |
| `runtime` | 324 unit + 7 doc-tests | 43 (integration + compaction + agent_delegation + skills) | `MockInferenceProvider` drives loop tests including read-only denial, workspace-write approval accept/reject, traversal blocking, persistence, and compaction resume |
| `cli` | 268 unit | 47 smoke tests (`tests/smoke.rs`) | Unit tests live across `commands`, `repl`, `render`, `tui`, `tui_composer`, and `width`; smoke tests spawn the real binary in temp `HOME` |

**EXTRACTED** from `cargo test --workspace` on 2026-05-03 plus per-file test inventories in the source tree.

---

## Next entry points

- New to the code? Read in this order: [inference](pages/inference.md) → [tools](pages/tools.md) → [conversation-loop](pages/conversation-loop.md).
- Debugging a user-visible bug? Start at [cli](pages/cli.md), follow the stream path into [conversation-loop](pages/conversation-loop.md).
- Adding a feature? See [recipes](pages/recipes.md).
- Something's weird? Check [gotchas](pages/gotchas.md) first.
