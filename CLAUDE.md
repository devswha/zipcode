# CLAUDE.md

## Project Overview

zipcode is a Rust-based local-only AI coding agent. It runs GGUF models via llama.cpp (Rust bindings) or candle, providing Claude Code-like functionality (file editing, shell execution, code search) entirely offline. Designed for air-gapped environments — ships as a single ZIP.

## Wiki

Project knowledge base at `wiki/`. Read `wiki/index.md` for page catalog.
When learning something new about the project, update or create wiki pages.

## Stack

- Language: Rust 2021
- Workspace: 4 crates (`inference`, `tools`, `runtime`, `cli`)
- Inference backends: `llama-cpp-2` (default feature: `candle`, optional: `llama-cpp`)
- CLI: clap, rustyline, termimad

## Build & Test

```bash
# Build (candle backend only, default)
cargo build --release

# Build with llama-cpp backend (requires cmake + libclang-dev)
cargo build --release -p zipcode-inference --features llama-cpp

# Build CLI with llama-cpp
# First: set features = ["llama-cpp"] in crates/cli/Cargo.toml for zipcode-inference dep
cargo build --release -p zipcode

# Test
cargo test --workspace

# Lint
cargo clippy --workspace --all-targets -- -D warnings

# Format
cargo fmt --all -- --check
```

## Architecture

```
cli → runtime → inference
         ↓
       tools
```

- `inference` — GGUF model loading, tokenization, streaming generation, Gemma chat template
- `tools` — 10 tool implementations behind `Tool` trait + `ToolRegistry`
- `runtime` — Agentic conversation loop, config, permissions, sessions
- `cli` — REPL, one-shot mode, doctor command

## Key Patterns

- `InferenceProvider` trait abstracts inference backends (candle, llama-cpp, mock)
- `ConversationLoop` is generic over `Box<dyn InferenceProvider>`
- `MockInferenceProvider` queues predetermined responses for testing
- Tool output auto-truncated at 8KB (`MAX_TOOL_OUTPUT_BYTES`)
- Path traversal prevention via `resolve_and_validate_path()` in all file tools
- Tool-call loop capped at 25 iterations (`MAX_TOOL_ITERATIONS`)
- Chat template hardcoded for Gemma format (`<start_of_turn>/<end_of_turn>`, `<tool_call>` tags)

## Environment Variables (llama-server backend)

- `ZIPCODE_LLAMA_SERVER_BIN` — Path to llama-server binary
- `ZIPCODE_LLAMA_SERVER_ALIAS` — Model alias (default: "zipcode")
- `ZIPCODE_LLAMA_SERVER_CTX` — Context window size (default: 131072, i.e. Gemma 4 128K native)
- `ZIPCODE_GPU_LAYERS` — Number of layers to offload to GPU (e.g., 99)
- `ZIPCODE_FLASH_ATTENTION` — Enable flash attention ("1" or "true")

## Current Limitations (known issues to address)

1. **Gemma 4 on native bindings is still blocked** — `llama-cpp-rs` 0.1.141 still lacks `gemma4` architecture support, so zipcode now falls back to a recent external `llama-server` binary when `ZIPCODE_LLAMA_SERVER_BIN` is set or `llama-server` is on PATH.
2. **candle backend has no quantized Gemma** — candle 0.8 has no `quantized_gemma` module. Uses `quantized_llama` as stand-in. Only compiles, does not actually load Gemma GGUF.
3. **Chat template is Gemma-only** — Other models (Qwen, Llama) use different tool-calling formats. Models that don't understand `<tool_call>` tags won't trigger tool execution.
4. **Agent tool is a stub** — Returns "not yet implemented".

## File Conventions

- `.zipcode.md` in project root = project-specific instructions injected into system prompt
- `.zipcode.json` in project root = project config (overrides `~/.zipcode/config.json`)
- `.zipcode-todos.json` = todo list created by TodoWrite tool
- `~/.zipcode/models/` = GGUF model storage
- `~/.zipcode/sessions/` = conversation history persistence

## Working Agreements

- Tests first when possible. Currently 75 tests across all crates.
- `cargo clippy -- -D warnings` must pass before commit.
- Commit messages in English, conventional format (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Do not commit `.env` or model files.
- Keep inference backends behind feature flags to avoid forcing heavy deps.
