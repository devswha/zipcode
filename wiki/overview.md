# zipcode — Project Overview

zipcode is a Rust-based local-only AI coding agent. It runs GGUF models via llama.cpp (Rust bindings) or candle, providing Claude Code-like functionality (file editing, shell execution, code search) entirely offline.

## Architecture

Four crates: `cli → runtime → inference`, with `runtime → tools`.

- **inference** — Three backends: candle (pure Rust, placeholder for Gemma), llama-cpp (native bindings, Gemma 4 blocked upstream), llama-server (HTTP subprocess, fully functional). SSE streaming enabled for real-time token delivery.
- **tools** — 10 tools behind `Tool` trait + `ToolRegistry`. Path traversal prevention, 8KB output truncation, permission-gated execution.
- **runtime** — Agentic conversation loop with 25-iteration cap. Hierarchical config (global + project), three permission modes, UUID-based session persistence.
- **cli** — REPL (rustyline), one-shot prompts, doctor health checks, setup wizard. Smart default entrypoint routes to setup/repair/REPL.

## Current State (2026-04-08)

- **Production-ready**: llama-server backend with SSE streaming, GPU offload (-ngl, --flash-attn), KV cache reuse
- **120 tests**, zero clippy warnings
- **GPU performance**: ~20x speedup over CPU (7.6s vs 150s on RTX 2070 SUPER)
- **Health check**: Uses /health endpoint (waits for model load complete)

## Known Limitations

1. Gemma 4 on native llama-cpp-rs 0.1.141 still blocked — falls back to llama-server
2. Candle has no quantized_gemma module — uses quantized_llama as placeholder
3. Chat template hardcoded for Gemma format
4. Agent tool is a stub

## Open Questions

- Multi-model chat template support (Qwen, Llama 3)
- Agent tool implementation strategy
- Candle Gemma 4 support timeline (depends on upstream)
