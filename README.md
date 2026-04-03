<p align="center">
  <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Rust" />
  <img src="https://img.shields.io/badge/Gemma_4-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemma 4" />
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/Air--Gapped-FF6B6B?style=for-the-badge&logoColor=white" alt="Air-Gapped" />
</p>

<h1 align="center">zipcode</h1>

<p align="center">
  <strong>Local-only AI coding agent. No API keys. No network. Just code.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#air-gapped-deployment">Air-Gapped Deploy</a> &bull;
  <a href="#tools">Tools</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#configuration">Config</a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/devswha/zipcode?style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/platform-linux%20x86__64-blue?style=flat-square" alt="Platform" />
  <img src="https://img.shields.io/badge/tests-60%20passing-brightgreen?style=flat-square" alt="Tests" />
  <img src="https://img.shields.io/badge/clippy-zero%20warnings-brightgreen?style=flat-square" alt="Clippy" />
</p>

---

**zipcode** is a Rust-based coding agent that runs **entirely on your machine** using local LLM inference. Powered by [Gemma 4](https://ai.google.dev/gemma) through the [candle](https://github.com/huggingface/candle) ML framework, it provides Claude Code-like functionality — file editing, shell execution, code search — without ever touching the network.

Ship it on a USB stick. Run it in a SCIF. It just works.

```
$ zipcode
zipcode v0.1.0 -- local AI coding agent
Type /help for commands, Ctrl+D to exit

> read main.rs and add error handling to the database connection

  Reading src/main.rs...
  > edit_file {"file_path": "src/main.rs", ...}
  < Successfully edited src/main.rs

  I've wrapped the database connection in a proper error handler with
  retry logic and connection pooling...
```

## Why zipcode?

| Problem | zipcode Solution |
|---------|-----------------|
| API keys expire, leak, or get rate-limited | No API. Model runs locally. |
| Corporate networks block LLM endpoints | No network needed after setup. |
| Sensitive code can't leave the machine | Everything stays on disk. |
| Cloud LLMs add latency | GPU inference on your hardware. |
| Setup requires `npm`, `pip`, Docker... | Single static binary. |

---

## Quick Start

### 1. Build

```bash
git clone https://github.com/devswha/zipcode.git
cd zipcode
cargo build --release
```

### 2. Get a model

```bash
# Automated (requires internet + huggingface-cli)
./scripts/download_model.sh

# Or manual: download Gemma 4 27B GGUF from HuggingFace
# Place in ~/.zipcode/models/
```

### 3. Check your environment

```bash
./target/release/zipcode doctor
```

```
zipcode doctor
--------------
  Binary:   zipcode v0.1.0 (linux-x86_64)
  CUDA:     Available
  Model:    gemma-4-27b-it-Q8_0.gguf (28.3 GB)
  Ready:    All checks passed
```

### 4. Run

```bash
# Interactive REPL
zipcode

# One-shot
zipcode prompt "explain this codebase"

# Custom model
zipcode --model ./my-model.gguf

# Read-only mode (safe exploration)
zipcode --permission-mode read-only
```

---

## Air-Gapped Deployment

zipcode was designed from the ground up for **isolated networks**. Package everything into a ZIP, move it on a USB drive, and run.

### Package

```bash
./scripts/package.sh
# Creates: dist/zipcode-v0.1.0-linux-x86_64-cuda.zip
```

### Archive contents

```
zipcode-v0.1.0-linux-x86_64-cuda.zip
 |- zipcode                     # single binary (~30 MB)
 |- libcudart.so.12             # CUDA runtime (optional)
 |- install.sh                  # one-command setup
 |- download_model.sh           # model download helper
 |- README.md
 '- models/
     '- PLACE_MODEL_HERE.txt
```

### Transfer workflow

```
  Internet Machine                    Air-Gapped Machine
  ================                    ==================

  1. Download zipcode.zip
  2. Download gemma-4-27b.gguf
  3. Copy to USB
                        ---- USB ---->
                                      4. Unzip
                                      5. ./install.sh
                                      6. cp *.gguf ~/.zipcode/models/
                                      7. zipcode doctor
                                      8. zipcode
```

---

## Tools

zipcode comes with **10 built-in tools** that the model can invoke autonomously:

| Tool | Description | Permission |
|------|-------------|------------|
| **Bash** | Execute shell commands via `bash -c` | Needs approval in `workspace-write` |
| **ReadFile** | Read file contents with line numbers, offset/limit | Always allowed |
| **WriteFile** | Create or overwrite files, auto-creates parent dirs | Blocked in `read-only` |
| **EditFile** | Targeted string replacement in existing files | Blocked in `read-only` |
| **GlobSearch** | Find files by glob pattern (`**/*.rs`, `src/*.py`) | Always allowed |
| **GrepSearch** | Search file contents with regex, returns `file:line:` | Always allowed |
| **TodoWrite** | Structured todo list persistence (`.zipcode-todos.json`) | Blocked in `read-only` |
| **REPL** | Execute Python/Node.js code snippets | Blocked in `read-only` |
| **Agent** | Spawn sub-agent for delegated tasks | _Stub in v0.1_ |
| **ToolSearch** | Search available tools by keyword | Always allowed |

All tool output is automatically truncated at **8 KB** with a byte-count summary.

### Permission Modes

| Mode | Bash | Writes | Reads | Use case |
|------|------|--------|-------|----------|
| `read-only` | Denied | Denied | Allowed | Safe exploration |
| `workspace-write` | Approval required | Allowed | Allowed | **Default** |
| `full-access` | Allowed | Allowed | Allowed | Trusted automation |

---

## Architecture

### System Overview

```
+-------------------------------------------------+
|                  zipcode CLI                     |
|            REPL / one-shot / doctor              |
+-------------------------------------------------+
|                Agentic Runtime                   |
|   +--------+  +----------+  +---------+         |
|   |Session |  |Permission|  | Config  |         |
|   |Manager |  |  Policy  |  | Loader  |         |
|   +--------+  +----------+  +---------+         |
|          +-----------------+                     |
|          | Conversation    |<--- tool results    |
|          |     Loop        |                     |
|          +-------+---------+                     |
|                  | tool calls                    |
|          +-------v---------+                     |
|          |  Tool Router    |                     |
|          +--+---------+----+                     |
|             |         |                          |
|       +-----+    +----+----+                     |
|       |Bash |    |FileOps  |  ...10 tools        |
|       +-----+    +---------+                     |
+-------------------------------------------------+
|             Inference Engine                     |
|   candle GGUF loader + CUDA acceleration         |
|   Tokenizer | KV Cache | Streaming Generation   |
+-------------------------------------------------+
|              Model Store                         |
|   ~/.zipcode/models/*.gguf                       |
+-------------------------------------------------+
```

### Workspace Structure

```
zipcode/
 |- crates/
 |   |- inference/    Candle GGUF engine, Gemma 4 chat template, sampler
 |   |- tools/        10 tool implementations + Tool trait + registry
 |   |- runtime/      Agentic loop, config, permissions, sessions
 |   '- cli/          REPL, one-shot, doctor, slash commands
 |- scripts/          install.sh, download_model.sh, package.sh
 |- models/           .gguf model files (gitignored)
 '- docs/             Design specs + implementation plans
```

### Crate Dependencies

```
cli --> runtime --> inference
           |
           '--> tools
```

| Crate | Responsibility |
|-------|---------------|
| **inference** | GGUF loading, tokenization, KV cache, streaming generation. Pure computation. |
| **tools** | 10 tool implementations. `Tool` trait, `ToolRegistry`, `ToolResult` truncation. |
| **runtime** | Agentic conversation loop. Config hierarchy, permission policy, session persistence. |
| **cli** | Binary entry point. REPL (rustyline), ANSI rendering (termimad), clap argument parsing. |

---

## CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `zipcode` | Start interactive REPL |
| `zipcode prompt "..."` | One-shot prompt, then exit |
| `zipcode doctor` | Check CUDA, model, binary version |

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model <PATH>` | `~/.zipcode/models/` | Path to `.gguf` file |
| `--permission-mode` | `workspace-write` | `read-only` / `workspace-write` / `full-access` |

### Slash Commands (REPL)

| Command | Action |
|---------|--------|
| `/help` | Show commands |
| `/status` | Session ID, message count, cwd |
| `/clear` | Reset conversation |
| `/quit` | Exit |

---

## Configuration

### Project config (`.zipcode.json`)

Loaded from working directory. Overrides global `~/.zipcode/config.json`.

```json
{
  "permission_mode": "workspace-write",
  "model_dir": "~/.zipcode/models",
  "generation": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 4096
  }
}
```

### Project instructions (`.zipcode.md`)

Place in any project root. Contents are injected into the system prompt.

```markdown
# My Project
- Language: Rust 2021
- Build: cargo build --release
- Test: cargo test --workspace
- Never modify files under vendor/
```

### Storage layout

```
~/.zipcode/
 |- config.json          # global config
 |- models/              # GGUF model files
 '- sessions/            # conversation history
     '- <uuid>.json
```

---

## Model Setup

### Requirements

| Component | Minimum |
|-----------|---------|
| Model | Gemma 4 27B IT (GGUF format) |
| VRAM | 24 GB for Q8_0 quantization |
| CUDA | 12.0+ (optional, CPU fallback available) |
| Disk | ~28 GB for Q8_0 model file |

### Download

```bash
# Option 1: helper script
./scripts/download_model.sh

# Option 2: manual
# Download from: huggingface.co/google/gemma-4-27b-it-GGUF
# Place in: ~/.zipcode/models/
```

CPU inference works but is significantly slower. zipcode auto-detects CUDA availability at startup.

---

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| [candle](https://github.com/huggingface/candle) | Rust-native ML framework for GGUF inference |
| [tokenizers](https://github.com/huggingface/tokenizers) | HuggingFace tokenizer |
| [clap](https://github.com/clap-rs/clap) | CLI argument parsing |
| [rustyline](https://github.com/kkawakam/rustyline) | REPL line editing |
| [termimad](https://github.com/Canop/termimad) | Markdown to ANSI rendering |
| [tokio](https://tokio.rs) | Async runtime |

---

## Contributing

Contributions welcome. Please open an issue first for major changes.

```bash
# Development workflow
cargo test --workspace          # run all tests
cargo clippy --workspace        # lint
cargo fmt --all                 # format
cargo build --release -p zipcode  # build binary
```

---

## Acknowledgments

Inspired by [claw-code](https://github.com/instructkr/claw-code) by [@instructkr](https://github.com/instructkr).

Built with [candle](https://github.com/huggingface/candle) by Hugging Face.

## License

[MIT](LICENSE)
