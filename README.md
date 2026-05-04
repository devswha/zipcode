<p align="center">
  <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Rust" />
  <img src="https://img.shields.io/badge/Gemma_4-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemma 4" />
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/Air--Gapped-FF6B6B?style=for-the-badge&logoColor=white" alt="Air-Gapped" />
</p>

<h1 align="center">zipcode</h1>

<p align="center">
  <strong>Local-only AI coding agent. No API keys. No cloud dependency. Just code.</strong>
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
  <img src="https://img.shields.io/badge/tests-265%20passing-brightgreen?style=flat-square" alt="Tests" />
  <img src="https://img.shields.io/badge/clippy-zero%20warnings-brightgreen?style=flat-square" alt="Clippy" />
</p>

---

**zipcode** is a Rust-based coding agent that runs **entirely on your machine** using local LLM inference. It can run Gemma-family GGUF models through candle, native `llama-cpp`, or a `llama-server` fallback for newer Gemma 4 models that outpace the current Rust bindings. Responses stream token-by-token via SSE for instant feedback.

The intended entrypoint is plain `zipcode`: if the machine is ready, you land in the REPL; if it is fresh or broken, zipcode points you at setup or repair steps instead of dropping you into backend jargon.

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
| Corporate networks block LLM endpoints | No network needed after setup unless you explicitly ask zipcode to fetch a GitHub repo URL. |
| Sensitive code can't leave the machine | Everything stays on disk. |
| Cloud LLMs add latency | GPU inference on your hardware. |
| Setup requires `npm`, `pip`, Docker... | Single static binary. |

### Performance

| Setup | Response Time | Tokens/sec |
|-------|--------------|------------|
| CPU only (12 threads) | ~150s | ~0.8 |
| **GPU (RTX 2070 SUPER)** | **~7.6s** | **~15** |

> SSE streaming delivers tokens to your terminal as they're generated — no waiting for the full response.

---

## Quick Start

### 1. Install from a clone

```bash
git clone https://github.com/devswha/zipcode.git
cd zipcode
./install.sh
```

The clone installer builds zipcode, installs a `zipcode` launcher into `~/.local/bin`,
copies any model artifacts you point it at, and when nothing is ready it walks you
through a setup interview: download in the terminal now, show the download links,
paste the local paths after you fetch the files, or skip for now.

Common flags:

```bash
./install.sh --model /path/to/model.gguf --tokenizer /path/to/tokenizer.json
./install.sh --llama-server /path/to/llama-server
./install.sh --skip-build --binary ./target/release/zipcode
./scripts/download_model.sh --yes --dir ~/.zipcode/models
./scripts/download_model.sh --yes --preset 31b --dir ~/.zipcode/models
./scripts/build_llama_server.sh ~/.zipcode
```

The online download path may require Hugging Face login / access approval for the
Gemma GGUF repository.

### 2. Get a model + tokenizer

```bash
# Automated (requires internet + hf or huggingface-cli)
./scripts/download_model.sh

# Or manual: download a Gemma GGUF + matching tokenizer.json
# Place both files in ~/.zipcode/models/
```

### 3. First run

```bash
# Start here. Ready machines drop into the REPL.
zipcode

# Scripted first-run / repair path
zipcode setup --skip-smoke

# Detailed readiness + repair report
zipcode doctor
```

`zipcode` is the default entrypoint. Use `setup` when you want zipcode to discover the model,
write or refresh the global config, and optionally run a smoke prompt. Use `doctor` when you
want the repair summary without entering the agent.

### 4. Power-user flows

```bash
# Interactive entrypoint
zipcode

# One-shot
zipcode prompt "explain this codebase"

# Guided config + smoke test
zipcode setup

# Guided config only
zipcode setup --skip-smoke

# Readiness / repair report
zipcode doctor

# Custom model
zipcode --model ./my-model.gguf

# Gemma 4 via the llama-server helper
ZIPCODE_LLAMA_SERVER_BIN=/path/to/llama-server \
zipcode --backend llama-server --model ~/.zipcode/models/gemma-4-e2b-it-Q8_0.gguf prompt "hello"

# GPU acceleration (CUDA)
ZIPCODE_GPU_LAYERS=99 ZIPCODE_FLASH_ATTENTION=1 zipcode

# Build llama-server with CUDA (auto-detects GPU)
./scripts/build_llama_server.sh

# If CUDA 11.x + newer GCC fails, prefer gcc-10/g++-10 for GPU builds
CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 \
ZIPCODE_LLAMA_SERVER_CMAKE_ARGS="-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10" \
./scripts/build_llama_server.sh

# Read-only mode (safe exploration)
zipcode --permission-mode read-only
```

---

## Air-Gapped Deployment

zipcode was designed from the ground up for **isolated networks**. Package everything into a ZIP,
move it on a USB drive, and let bare `zipcode` handle first-run versus recovery on the target
machine.

### Package

```bash
# Smallest bundle
./scripts/package.sh

# Or bundle llama-server so Gemma 4 stays fully offline on first run / repair
./scripts/package.sh --with-llama-server=/path/to/llama-server

# Creates: dist/zipcode-v0.1.0-linux-x86_64-cuda.zip
```

### Archive contents

```
zipcode-v0.1.0-linux-x86_64-cuda.zip
 |- zipcode                     # single binary (~30 MB)
 |- install.sh                  # extracted-bundle installer
 |- install_llama_server.sh     # repair helper for adding llama-server later
 |- build_llama_server.sh       # build llama-server from source when needed
 |- download_model.sh           # model download helper
 |- README.md
 |- llama-server                # optional, when bundled during packaging
 |- scripts/lib/install_common.sh
 '- models/
     |- PLACE_MODEL_HERE.txt
     '- PLACE_TOKENIZER_HERE.txt
```

### Transfer workflow

```
  Internet Machine                    Air-Gapped Machine
  ================                    ==================

  1. Build or download zipcode.zip
  2. Download model.gguf + tokenizer.json
  3. Optional: bundle llama-server for Gemma 4
  4. Copy to USB
                        ---- USB ---->
                                      5. Unzip
                                      6. ./install.sh
                                      7. cp model.gguf tokenizer.json ~/.zipcode/models/
                                      8. zipcode
                                      9. If repair is needed:
                                         zipcode doctor
                                         zipcode setup --skip-smoke
```

---

## Tools

zipcode comes with **11 built-in tools** that the model can invoke autonomously:

| Tool | Description | Permission |
|------|-------------|------------|
| **Bash** | Execute shell commands via `bash -c` | Needs approval in `workspace-write` |
| **ReadFile** | Read file contents with line numbers, offset/limit | Always allowed |
| **WriteFile** | Create or overwrite files, auto-creates parent dirs | Blocked in `read-only` |
| **EditFile** | Targeted string replacement in existing files | Blocked in `read-only` |
| **FetchRepo** | Clone a public GitHub repo into `.zipcode-remote/` for analysis | Blocked in `read-only` |
| **GlobSearch** | Find files by glob pattern (`**/*.rs`, `src/*.py`) | Always allowed |
| **GrepSearch** | Search file contents with regex, returns `file:line:` | Always allowed |
| **TodoWrite** | Structured todo list persistence (`.zipcode-todos.json`) | Blocked in `read-only` |
| **REPL** | Execute Python/Node.js code snippets | Blocked in `read-only` |
| **Agent** | Spawn sub-agent for delegated tasks | _Stub in v0.1_ |
| **ToolSearch** | Search available tools by keyword | Always allowed |

All tool output is automatically truncated at **8 KB** with a byte-count summary.

`FetchRepo` is the explicit network exception: it only handles repository-root `https://github.com/<owner>/<repo>` URLs, clones them with `git clone --depth 1` into `.zipcode-remote/`, and is unavailable in `read-only` mode. Air-gapped deployments should pre-copy repositories into the workspace instead.

### Permission Modes

| Mode | Bash | Writes | Reads | Use case |
|------|------|--------|-------|----------|
| `read-only` | Denied | Denied | Allowed | Safe exploration |
| `workspace-write` | Approval required | Allowed | Allowed | **Default** |
| `full-access` | Allowed | Allowed | Allowed | Trusted automation |

> In `workspace-write`, approval-gated tools fail closed when zipcode is running non-interactively (for example `prompt` mode with EOF / no TTY). A real interactive terminal is required to approve `bash` or `repl`.

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
|       |Bash |    |FileOps  |  ...11 tools        |
|       +-----+    +---------+                     |
+-------------------------------------------------+
|             Inference Engine                     |
|   candle | llama-cpp | llama-server (SSE)        |
|   GPU offload | KV cache | Flash attention       |
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
 |   |- tools/        11 tool implementations + Tool trait + registry
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
| **inference** | GGUF loading, tokenization, KV cache, SSE streaming, GPU offload. Supports candle, llama-cpp, and llama-server backends. |
| **tools** | 11 tool implementations. `Tool` trait, `ToolRegistry`, `ToolResult` truncation. |
| **runtime** | Agentic conversation loop. Config hierarchy, permission policy, session persistence. |
| **cli** | Binary entry point. REPL (rustyline), ANSI rendering (termimad), clap argument parsing. |

---

## CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `zipcode` | Smart entrypoint: REPL when ready, setup/repair guidance when not |
| `zipcode prompt "..."` | One-shot prompt, then exit |
| `zipcode setup` | Discover prerequisites, write config, optionally run a smoke prompt |
| `zipcode doctor` | Show readiness state and repair hints |

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model <PATH>` | `~/.zipcode/models/` | Path to `.gguf` file |
| `--backend <BACKEND>` | `llama-cpp` | `llama-cpp` / `llama-server` / `candle` |
| `--permission-mode` | `workspace-write` | `read-only` / `workspace-write` / `full-access` |
| `--ui <MODE>` | `fullscreen` | Interactive UI mode: `plain` or `fullscreen` |

### Slash Commands (REPL)

| Command | Action |
|---------|--------|
| `/help` | Show commands |
| `/status` | Session ID, message count, cwd |
| `/clear` | Reset conversation |
| `/quit` | Exit |

### Fullscreen TUI MVP

`zipcode` now launches the fullscreen UI by default when the install is ready. Use
`zipcode --ui fullscreen` (or `zipcode repl --ui fullscreen`) explicitly if you want to force it,
or `zipcode --ui plain` to return to the classic line-oriented REPL.

The fullscreen terminal UI MVP includes:

- transcript area
- bottom composer
- inline status bar
- multiline composer editing
- input history recall
- exact slash-command handling (`/help`, `/status`, `/clear`, `/quit`)
- alternate-screen rendering for a more app-like experience

Key bindings:
- `Enter` — submit
- `Ctrl+J` — newline
- `Arrow keys` — move cursor / recall input history
- `PgUp` / `PgDn` — scroll transcript
- `F1` — help overlay
- `Esc` — clear input or close overlay
- `Ctrl+D` — exit

---

## Configuration

### Project config (`.zipcode.json`)

Loaded from the nearest ancestor project root. Overrides global `~/.zipcode/config.json`.
Paths such as `model_dir`, `model_file`, and `llama_server_bin` support `~/...` home expansion.

```json
{
  "permission_mode": "workspace-write",
  "model_dir": "~/.zipcode/models",
  "gpu_layers": 99,
  "flash_attention": true,
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

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ZIPCODE_LLAMA_SERVER_BIN` | Path to llama-server binary | auto-detect |
| `ZIPCODE_LLAMA_SERVER_CMAKE_ARGS` | Extra cmake flags for `build_llama_server.sh` | none |
| `ZIPCODE_LLAMA_SERVER_CUDA_ARCHITECTURES` | Override detected CUDA arch list (example: `75`) | auto-detect when possible |
| `ZIPCODE_LLAMA_SERVER_CTX` | Context window size | 131072 (Gemma 4 128K native) |
| `ZIPCODE_GPU_LAYERS` | GPU layers to offload (e.g., 99 for all) | none (CPU) |
| `ZIPCODE_FLASH_ATTENTION` | Enable flash attention (`1` or `true`) | false |

---

## Model Setup

### Requirements

| Component | Minimum |
|-----------|---------|
| Model | Gemma 4 E2B IT (GGUF format) |
| VRAM | ~8 GB recommended for Q8_0 quantization |
| CUDA | 12.0+ (optional, CPU fallback available) |
| Disk | ~5 GB for Q8_0 model file |

### Download

```bash
# Option 1: helper script
./scripts/download_model.sh

# Option 2: manual
# Download from: huggingface.co/ggml-org/gemma-4-E2B-it-GGUF
# Also download tokenizer.json from the matching model repo
# Place both files in: ~/.zipcode/models/
```

CPU inference works but is significantly slower. zipcode auto-detects CUDA availability at startup.

### Gemma 4 today

- Native `llama-cpp-2` `0.1.141` still cannot open `general.architecture = gemma4`.
- In practice, Gemma 4 currently works best when zipcode can find a recent `llama-server`.
- Easiest paths:
  1. bundle it during packaging with `./scripts/package.sh --with-llama-server=/path/to/llama-server`
  2. or install it later with `~/.zipcode/install_llama_server.sh /path/to/llama-server ~/.zipcode`
  3. or set `ZIPCODE_LLAMA_SERVER_BIN=/path/to/llama-server`

After that, rerun `zipcode` (default entrypoint) or `zipcode setup` if you want zipcode to refresh its saved config first.

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
