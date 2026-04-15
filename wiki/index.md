# zipcode Wiki

Developer reference for the `crates/` workspace. Structured as a knowledge graph — navigate by **community** (a tightly-related module cluster) and jump between pages via **god node** links.

**Scope:** `/home/devswha/workspace/zipcode/crates/` only.
**Snapshot:** extracted from code on 2026-04-15.
**Provenance rule:** every factual claim below carries a `file:line` reference.

> Graphify-inspired layout. Start at [god nodes](#god-nodes) if you're new to the codebase — that's where everything else hangs off.

---

## Dependency Graph

```
cli ──▶ runtime ──▶ inference
          │
          └─▶ tools
```

| Crate | Role | Cargo | Path |
|-------|------|-------|------|
| `zipcode-inference` | GGUF loading, sampling, chat template, backend trait | `crates/inference/Cargo.toml` | [`crates/inference/`](../crates/inference/) |
| `zipcode-tools` | `Tool` trait + 10 tool implementations + path safety | `crates/tools/Cargo.toml` | [`crates/tools/`](../crates/tools/) |
| `zipcode-runtime` | Conversation loop, permissions, config, sessions, system prompt | `crates/runtime/Cargo.toml` | [`crates/runtime/`](../crates/runtime/) |
| `zipcode` (cli) | clap entrypoint, REPL, fullscreen TUI, doctor, setup | `crates/cli/Cargo.toml` | [`crates/cli/`](../crates/cli/) |

---

## Communities

| Page | What it covers |
|------|----------------|
| [inference](pages/inference.md) | `InferenceProvider` trait, 4 backends (candle, llama-cpp, llama-server, mock), sampling, `GenerationConfig` |
| [chat-template](pages/chat-template.md) | Current implementation — labelled "Gemma 4" but actually Gemma 3 tokens; `<tool_call>` JSON parsing; single-model coupling |
| [gemma4-format-spec](pages/gemma4-format-spec.md) | **Phase 0 spec** — real Gemma 4 chat template extracted from the GGUF: `<\|turn>` tokens, custom tool-call mini-language, thinking channel, stop tokens, implementation deltas |
| [tools](pages/tools.md) | `Tool` trait, `ToolRegistry`, all 10 tools, path safety, 8 KB truncation |
| [conversation-loop](pages/conversation-loop.md) | `ConversationLoop`, 25-iteration cap, `StreamCallback`, agentic loop flow |
| [permissions](pages/permissions.md) | `PermissionMode`, 3-tier matrix, approval-gated tools |
| [config](pages/config.md) | `ZipcodeConfig`, global + project override, path expansion |
| [session](pages/session.md) | `~/.zipcode/sessions/{uuid}.json`, `Session` struct, roundtrip save/load |
| [cli](pages/cli.md) | clap parser, REPL, fullscreen TUI, `doctor`, `setup`, wrapper script |
| [llama-server](pages/llama-server.md) | Subprocess lifecycle, `/health`, SSE streaming, GPU offload env vars |
| [gotchas](pages/gotchas.md) | Non-obvious couplings, hardcoded invariants, silent failure modes |
| [recipes](pages/recipes.md) | "How do I..." playbook: add a tool, swap chat template, tune GPU offload |
| [project-direction](pages/project-direction.md) | Current backend strategy, product priorities, and why `llama-server` is the practical Gemma 4 path |

---

## God Nodes

The highest-degree concepts — almost every code path touches one of these. Read these pages first.

1. **[`InferenceProvider`](pages/inference.md#inferenceprovider-trait)** — abstracts all inference backends. `crates/inference/src/lib.rs:28`. 4 implementors: `LlamaCppProvider`, `LlamaServerProvider`, `InferenceEngine` (candle), `MockInferenceProvider`.
2. **[`Tool` + `ToolRegistry`](pages/tools.md#tool-trait)** — abstracts every tool invocation. `crates/tools/src/lib.rs:328` (trait) + `:336` (registry). 10 implementors.
3. **[`ConversationLoop`](pages/conversation-loop.md)** — drives the agentic loop. Holds both god nodes above plus `PermissionPolicy` and `Session`. `crates/runtime/src/conversation.rs:23`.
4. **[`ChatMessage`](pages/inference.md#core-types)** — the wire format that crosses every crate. `crates/inference/src/types.rs:13`.
5. **[`PermissionMode`](pages/permissions.md)** — gates every tool invocation. `crates/tools/src/lib.rs:262` (enum) + `crates/runtime/src/permission.rs:28` (policy).

---

## Where do I go to change X?

| Goal | Start here |
|------|-----------|
| Add a new inference backend | [inference › Adding a backend](pages/inference.md#adding-a-backend) → factory in `crates/inference/src/lib.rs:71` |
| Add a new tool | [recipes › Add a new tool](pages/recipes.md#add-a-new-tool) → `crates/tools/src/lib.rs` |
| Support a non-Gemma model | [chat-template](pages/chat-template.md) — currently single-model hardcoded |
| Rewrite chat template to real Gemma 4 | [gemma4-format-spec](pages/gemma4-format-spec.md) — authoritative spec + implementation deltas |
| Change the tool-loop iteration cap | `MAX_TOOL_ITERATIONS` in `crates/runtime/src/conversation.rs:43` |
| Change the tool output size limit | `MAX_TOOL_OUTPUT_BYTES` in `crates/tools/src/lib.rs:377` |
| Add a config field | [config › Fields](pages/config.md#fields) → `crates/runtime/src/config.rs` |
| Tune GPU offload | [llama-server › GPU offload](pages/llama-server.md#gpu-offload) — env vars or config |
| Understand the intended backend roadmap | [project-direction](pages/project-direction.md) |
| Change permission tiers | [permissions](pages/permissions.md) → `crates/runtime/src/permission.rs:28` |
| Change what's injected into the system prompt | `crates/runtime/src/prompt.rs:27-59` (`build_system_prompt`, including `.zipcode.md` injection at `:39-45`) |

---

## Audit Tag Legend

Pages use graphify-style provenance tags on claims:

- **EXTRACTED** — directly verified against code with `file:line`
- **INFERRED** — reasoned from code but not a literal quote
- **GOTCHA** — non-obvious coupling, silent failure mode, or invariant worth flagging
- **STUB** — code exists but isn't implemented yet

---

## Navigation

- You are here: `wiki/index.md`
- Graph-level summary: [`GRAPH_REPORT.md`](GRAPH_REPORT.md)
- All community pages: [`pages/`](pages/)
