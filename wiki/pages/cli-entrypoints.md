---
title: CLI Entrypoints
tags: [modules]
sources: [session-2026-04-08]
updated: 2026-04-08
---

# CLI Entrypoints

The `zipcode` (cli) crate provides the user-facing binary.

## Subcommands

| Command | Function | Description |
|---------|----------|-------------|
| (default) | `run_default()` | Routes to setup if not ready, repair if broken, REPL if ready |
| `repl` | `run_interactive()` | Interactive REPL with rustyline |
| `prompt TEXT` | `run_oneshot()` | Single prompt, then exit |
| `doctor` | `run_doctor()` | Health check: model file, llama-server, CUDA, permissions |
| `setup` | `run_setup()` | Config wizard + optional smoke test |

## Global Flags

- `--model PATH` — Override model directory or .gguf file
- `--backend BACKEND` — llama-cpp, llama-server, candle
- `--permission-mode MODE` — read-only, workspace-write, full-access

## REPL Slash Commands

`/help`, `/status`, `/clear`, `/quit`

## Key Functions

- `build_registry()` — Creates ToolRegistry with all 10 tools
- `create_loop()` — Builds ConversationLoop from config + CLI args
- `resolve_model_path()` — Resolves model file from CLI flag, config, or auto-discovery

## See Also
- [[runtime-loop]]
- [[inference-backends]]
