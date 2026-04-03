# zipcode

Local-only AI coding agent powered by Gemma 4 via candle — runs entirely offline, no API keys required.

## Features

- Fully offline inference — no network required after setup
- Designed for air-gapped environments; ships as a single ZIP for USB deployment
- GGUF model loading with CUDA GPU acceleration and transparent CPU fallback
- 10 built-in tools: file operations, shell execution, search, REPL, sub-agents
- Gemma 4 native function calling format with `<tool_call>` tags
- Interactive REPL with line editing, history, and markdown rendering
- One-shot prompt mode for scripted use
- Three permission tiers: read-only, workspace-write, full-access
- Session persistence — save and restore conversation history
- Project-specific instructions via `.zipcode.md`
- Single static binary, Linux x86_64

## Quick Start

### Build

```bash
cargo build --release
# Binary output: target/release/zipcode
```

### Place a model

```bash
mkdir -p ~/.zipcode/models
# Copy your Gemma 4 GGUF file:
cp gemma-4-27b-it-Q8_0.gguf ~/.zipcode/models/
```

### Diagnose your environment

```bash
zipcode doctor
```

Example output:

```
zipcode doctor

Version: 0.1.0

  CUDA available
  Model found: /home/user/.zipcode/models/gemma-4-27b-it-Q8_0.gguf

Done.
```

### Run

```bash
# Interactive REPL
zipcode

# One-shot mode
zipcode prompt "explain src/main.rs"

# Custom model path
zipcode --model ./custom.gguf

# Read-only mode
zipcode --permission-mode read-only
```

## Air-Gapped Deployment

zipcode is packaged as a ZIP archive that can be transferred via USB to networks with no internet access.

### ZIP Archive Structure

```
zipcode-v0.1.0-linux-x86_64-cuda.zip
├── zipcode                       # single static binary (~30 MB)
├── libcudart.so.12               # optional CUDA runtime bundle
├── README.md
├── install.sh
└── models/
    └── PLACE_MODEL_HERE.txt
```

### Deployment Workflow

```
Internet PC                        Air-Gapped Network
──────────────────                 ──────────────────
1. Download zipcode.zip
2. Download gemma-4-27b.gguf
3. Copy both to USB drive  ──USB──> 4. Copy from USB to workstation
                                    5. ./install.sh
                                    6. Place .gguf in ~/.zipcode/models/
                                    7. zipcode doctor
                                    8. zipcode
```

### install.sh

```bash
#!/bin/bash
INSTALL_DIR="${HOME}/.zipcode"
mkdir -p "${INSTALL_DIR}/models" "${INSTALL_DIR}/sessions"
cp zipcode "${INSTALL_DIR}/"
ln -sf "${INSTALL_DIR}/zipcode" /usr/local/bin/zipcode
echo "Done. Place your .gguf model in ${INSTALL_DIR}/models/"
```

Run once after copying from USB. No root required if `/usr/local/bin` is writable; adjust the symlink target as needed.

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────┐
│                 zipcode CLI                  │
│          (REPL + one-shot prompt)            │
├─────────────────────────────────────────────┤
│              Agentic Runtime                 │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Session  │  │Permission│  │  Config   │  │
│  │ Manager  │  │  Policy  │  │  Loader   │  │
│  └─────────┘  └──────────┘  └───────────┘  │
│         ┌──────────────────┐                │
│         │  Conversation    │                │
│         │     Loop         │◄── tool results│
│         └──────┬───────────┘                │
│                │ tool calls                  │
│         ┌──────▼───────────┐                │
│         │   Tool Router    │                │
│         └──────┬───────────┘                │
│    ┌───────────┼──────────────┐             │
│    ▼           ▼              ▼             │
│ ┌──────┐  ┌────────┐  ┌──────────┐         │
│ │ Bash │  │FileOps │  │TodoWrite │  ...     │
│ └──────┘  └────────┘  └──────────┘         │
├─────────────────────────────────────────────┤
│            Inference Engine                  │
│  ┌─────────────────────────────────────┐    │
│  │  candle (GGUF loader + CUDA accel)  │    │
│  │  ┌──────────┐  ┌────────────────┐   │    │
│  │  │Tokenizer │  │ KV Cache Mgmt  │   │    │
│  │  └──────────┘  └────────────────┘   │    │
│  │  ┌──────────────────────────────┐   │    │
│  │  │ Streaming Token Generation   │   │    │
│  │  └──────────────────────────────┘   │    │
│  └─────────────────────────────────────┘    │
├─────────────────────────────────────────────┤
│              Model Store                     │
│  ~/.zipcode/models/gemma-4-27b-Q8.gguf      │
└─────────────────────────────────────────────┘
```

### Crate Dependency Graph

```
cli -> runtime -> inference
           |
           v
         tools
```

### Crates

| Crate | Role |
|-------|------|
| `inference` | GGUF model loading, tokenization, KV cache, streaming token generation via candle. Pure computation — no side effects. |
| `tools` | The 10 tool implementations. Depends only on OS syscalls. Defines the `Tool` trait and `ToolRegistry`. |
| `runtime` | Combines `inference` + `tools` into the agentic conversation loop. Owns session management, config loading, and permission policy. |
| `cli` | Binary entry point. Interactive REPL (rustyline), markdown/ANSI rendering (termimad), slash commands, and the `doctor` subcommand. |

## Tools

| Tool | Description |
|------|-------------|
| `Bash` | Execute shell commands. Requires approval in `workspace-write` mode; blocked in `read-only`. |
| `ReadFile` | Read the contents of a file from disk. |
| `WriteFile` | Create or overwrite a file with new content. |
| `EditFile` | Apply targeted string replacements to an existing file. |
| `GlobSearch` | Find files matching a glob pattern (e.g. `src/**/*.rs`). |
| `GrepSearch` | Search file contents with a regex; returns matching lines and paths. |
| `TodoWrite` | Write a structured todo list to `.zipcode-todos.md` in the working directory. |
| `REPL` | Execute code in a persistent language REPL (Python, Node, etc.). |
| `Agent` | Spawn a sub-agent with its own conversation loop and tool access. |
| `ToolSearch` | Search available tools by name or description. |

Tool output is automatically truncated at 8 KB. Truncated results include a note showing bytes displayed vs. total.

## CLI Reference

### Commands

```
zipcode                          # start interactive REPL
zipcode prompt "<text>"          # one-shot mode — run a single prompt and exit
zipcode doctor                   # check CUDA, model files, and binary version
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model <PATH>` | `~/.zipcode/models/` | Path to a `.gguf` file or directory containing models. |
| `--permission-mode <MODE>` | `workspace-write` | Permission tier: `read-only`, `workspace-write`, `full-access`. |
| `--session <ID>` | — | Resume a saved session by ID. |

### Slash Commands (REPL only)

| Command | Description |
|---------|-------------|
| `/help` | Print available slash commands. |
| `/status` | Show current session ID, model, permission mode, and token count. |
| `/clear` | Clear conversation history and start fresh. |
| `/compact` | Summarize history to reclaim context window space. |
| `/session` | List saved sessions or resume one by ID. |

### Permission Modes

| Mode | Bash | File writes | File reads | Notes |
|------|------|-------------|------------|-------|
| `read-only` | Denied | Denied | Allowed | Safe exploration; no mutations. |
| `workspace-write` | Needs approval | Allowed | Allowed | Default. Bash requires Y/n confirmation. |
| `full-access` | Allowed | Allowed | Allowed | All tools execute without prompts. |

## Configuration

### .zipcode.json (project or global)

Configuration is resolved in layers: global (`~/.zipcode/config.json`) is loaded first, then project-level (`.zipcode.json` in the working directory) overrides specific keys.

```json
{
  "permission_mode": "workspace-write",
  "model_dir": "/path/to/models",
  "model_file": "gemma-4-27b-it-Q8_0.gguf",
  "generation": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 4096
  }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `permission_mode` | `workspace-write` | Default permission tier for this project. |
| `model_dir` | `~/.zipcode/models` | Directory scanned for `.gguf` files on startup. |
| `model_file` | first `.gguf` found | Specific model filename to use. |
| `generation.temperature` | `0.7` | Sampling temperature. |
| `generation.top_p` | `0.9` | Nucleus sampling threshold. |
| `generation.max_tokens` | `4096` | Maximum tokens per generation turn. |

### .zipcode.md (project instructions)

Place a `.zipcode.md` file in any project directory. Its contents are appended to the system prompt when zipcode is run from that directory. Use it to describe project conventions, build commands, and preferred patterns.

```markdown
# My Project

- Language: Rust, edition 2021
- Build: `cargo build --release`
- Test: `cargo test`
- Do not modify files under `vendor/`
```

### Session Storage

```
~/.zipcode/
├── config.json          # global configuration
├── models/              # GGUF model files
└── sessions/            # saved conversation histories
    └── <session-id>.json
```

Sessions are saved automatically after each turn. Use `--session <id>` or the `/session` slash command to restore a previous conversation.

## Model Setup

zipcode requires a Gemma 4 GGUF model. The recommended variant is the Q8_0 quantization of the 27B instruction-tuned model.

### Automated download (requires internet)

```bash
./scripts/download_model.sh
```

The script places the model in `~/.zipcode/models/` by default.

### Manual download

1. Download `gemma-4-27b-it-Q8_0.gguf` from Hugging Face (model card: `google/gemma-4-27b-it-GGUF`).
2. Place the file in `~/.zipcode/models/`.
3. Run `zipcode doctor` to verify detection.

### CUDA requirements

| Component | Minimum |
|-----------|---------|
| CUDA | 12.0 |
| VRAM | 24 GB (Q8_0 27B) |
| Driver | 525+ |

CPU inference works without CUDA but is significantly slower. zipcode detects CUDA via `CUDA_PATH`, `CUDA_HOME`, or `libcuda.so` presence.

### VRAM estimation

zipcode estimates VRAM usage before loading the model. If the estimate exceeds available memory, it shrinks the KV cache. If an out-of-memory error occurs at runtime, it falls back to context compaction before failing.

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `candle-core` | Tensor operations and CUDA backend |
| `candle-nn` | Neural network layers |
| `candle-transformers` | Gemma model architecture |
| `tokenizers` | HuggingFace tokenizer |
| `tokio` | Async runtime |
| `rustyline` | REPL line editing with history |
| `termimad` | Markdown to ANSI terminal rendering |
| `clap` | CLI argument parsing |
| `serde` / `serde_json` | Configuration and session serialization |
| `glob` | File pattern matching |
| `grep-regex` | Content search |

## Platform Support

MVP targets Linux x86_64 only. Windows and macOS are explicitly out of scope. No web UI, no remote API calls, no MCP server integration, no plugin system.

---

Inspired by the architecture patterns of [claw-code-parity](https://github.com/ultraworkers/claw-code-parity).

## License

MIT — see [LICENSE](LICENSE).
