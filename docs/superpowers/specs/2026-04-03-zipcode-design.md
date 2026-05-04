# zipcode — Local-Only Coding Agent Design Spec

Date: 2026-04-03

## Overview

**zipcode** is a Rust-based coding agent that runs offline by default using local LLM inference, with explicit user-requested GitHub repository fetching as the network exception. It provides Claude Code-like functionality (file editing, shell execution, code search) powered by Gemma 4 via the candle ML framework, packaged as a single ZIP for deployment in air-gapped environments.

## Requirements

| Requirement | Decision |
|-------------|----------|
| Inference | Rust-native embedded GGUF via candle |
| Model | Gemma 4 only |
| Platform | Linux x86_64 |
| GPU | CUDA support, models downloaded separately |
| MVP Tools | 10 (Bash, ReadFile, WriteFile, EditFile, GlobSearch, GrepSearch, TodoWrite, REPL, Agent, ToolSearch) |
| Tool Calling | Gemma 4 native function calling format |
| Deployment | ZIP archive — binary + optional CUDA libs; models placed by user |

## Architecture

```
┌─────────────────────────────────────────────┐
│                 zipcode CLI                  │
│          (REPL + one-shot prompt)            │
├─────────────────────────────────────────────┤
│              Agentic Runtime                 │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Session  │  │ Permission│  │  Config   │  │
│  │ Manager  │  │  Policy   │  │  Loader   │  │
│  └─────────┘  └──────────┘  └───────────┘  │
│         ┌──────────────────┐                │
│         │  Conversation    │                │
│         │     Loop         │◄──── tool results
│         └──────┬───────────┘                │
│                │ tool calls                  │
│         ┌──────▼───────────┐                │
│         │   Tool Router    │                │
│         └──────┬───────────┘                │
│    ┌───────────┼───────────────┐            │
│    ▼           ▼               ▼            │
│ ┌──────┐  ┌────────┐  ┌───────────┐        │
│ │ Bash │  │FileOps │  │ TodoWrite │  ...    │
│ └──────┘  └────────┘  └───────────┘        │
├─────────────────────────────────────────────┤
│            Inference Engine                  │
│  ┌─────────────────────────────────────┐    │
│  │  candle (GGUF loader + CUDA accel)  │    │
│  │  ┌──────────┐  ┌────────────────┐   │    │
│  │  │ Tokenizer│  │ KV Cache Mgmt  │   │    │
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

### Data Flow

1. User input → CLI → Runtime Conversation Loop
2. Loop sends input + history to Inference Engine
3. Model streams text or tool calls
4. Tool calls → Tool Router executes → results injected back into Loop
5. Text → streamed to user terminal
6. Repeats until model produces stop token

## Crate Structure

```
zipcode/
├── Cargo.toml                    # workspace root
├── crates/
│   ├── inference/                # Inference engine
│   │   └── src/
│   │       ├── lib.rs            # pub API
│   │       ├── engine.rs         # candle GGUF load + generation
│   │       ├── sampler.rs        # top-k, top-p, temperature
│   │       ├── kv_cache.rs       # KV cache management
│   │       └── chat_template.rs  # Gemma 4 FC format conversion
│   │
│   ├── runtime/                  # Agentic loop
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── conversation.rs   # generate → parse → execute → repeat
│   │       ├── session.rs        # session save/restore
│   │       ├── config.rs         # config hierarchy (.zipcode.json)
│   │       ├── permission.rs     # tool execution permission policy
│   │       └── prompt.rs         # system prompt assembly
│   │
│   ├── tools/                    # 11 tool implementations
│   │   └── src/
│   │       ├── lib.rs            # Tool trait + ToolRegistry
│   │       ├── bash.rs
│   │       ├── read_file.rs
│   │       ├── write_file.rs
│   │       ├── edit_file.rs
│   │       ├── glob_search.rs
│   │       ├── grep_search.rs
│   │       ├── todo_write.rs
│   │       ├── repl.rs
│   │       ├── agent.rs          # sub-agent spawn
│   │       └── tool_search.rs
│   │
│   └── cli/                      # CLI binary
│       └── src/
│           ├── main.rs           # entrypoint
│           ├── repl.rs           # interactive REPL (rustyline)
│           ├── render.rs         # markdown ANSI rendering
│           └── commands.rs       # slash commands
│
├── models/                       # .gitignore'd, model weights location
├── scripts/
│   ├── download_model.sh
│   └── package.sh
└── README.md
```

### Crate Dependency Graph

```
cli → runtime → inference
         ↓
       tools
```

- **inference**: candle only. Pure inference, no side effects.
- **runtime**: combines inference + tools into the agentic loop.
- **tools**: independent. Uses OS syscalls only.
- **cli**: REPL/UI layer on top of runtime.

## Inference Engine

### Core Interface

```rust
pub struct InferenceEngine {
    model: GgufModel,
    tokenizer: Tokenizer,
    device: Device,            // Device::Cuda(0) or Device::Cpu
    kv_cache: KvCache,
    config: GenerationConfig,
}

pub struct GenerationConfig {
    pub temperature: f64,      // default 0.7
    pub top_p: f64,            // default 0.9
    pub max_tokens: usize,     // default 4096
    pub stop_tokens: Vec<u32>,
}

pub enum TokenEvent {
    Token(String),
    ToolCall(ToolCallParsed),
    Done(FinishReason),
    Error(InferenceError),
}
```

### Model Loading

1. Scan `~/.zipcode/models/` for GGUF files
2. Detect CUDA availability → `Device::Cuda(0)` or `Device::Cpu`
3. Load weights via `candle::quantized::gguf::Content::read()`
4. Load tokenizer.json (from GGUF metadata or sidecar file)

### Gemma 4 Function Calling Format

```
<start_of_turn>user
You have access to the following tools:
[{"name": "bash", "parameters": {"command": {"type": "string"}}}]

Read the file src/main.rs<end_of_turn>
<start_of_turn>model
<tool_call>
{"name": "read_file", "arguments": {"file_path": "src/main.rs"}}
</tool_call><end_of_turn>
<start_of_turn>tool
{"content": "fn main() { ... }"}<end_of_turn>
<start_of_turn>model
Here's the content of src/main.rs: ...<end_of_turn>
```

`chat_template.rs` handles this format — hardcoded for Gemma 4 (no generic template engine needed).

### CUDA Memory Management

- Pre-estimate VRAM usage on model load
- KV cache dynamically grows with conversation length
- OOM graceful fallback: shrink cache → context compaction

## Tool System

### Tool Trait

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult>;
}
```

### Permission Flow

```
Tool execution request
  → PermissionPolicy.check(tool_name, args)
    → Allowed → execute
    → NeedsApproval → prompt user Y/N
    → Denied → return denial message
```

### Output Truncation

- Auto-truncate tool results exceeding 8KB
- Append `[truncated: showing first 8192 bytes of N]`

## Agentic Runtime

### Conversation Loop

```rust
impl ConversationLoop {
    pub async fn run_turn(&mut self, user_input: &str) -> Result<()> {
        self.history.push(ChatMessage::user(user_input));

        loop {
            let rx = self.engine.generate_stream(&self.history, &self.tools.specs());
            let response = self.collect_response(rx).await?;
            self.history.push(ChatMessage::assistant(response.clone()));

            match response {
                Response::Text(_) => break,
                Response::ToolCalls(calls) => {
                    for call in calls {
                        let result = self.execute_tool(&call).await?;
                        self.history.push(ChatMessage::tool_result(call.id, result));
                    }
                }
            }
        }

        self.session.save(&self.history)?;
        Ok(())
    }
}
```

### System Prompt Assembly

1. Base role definition (coding agent)
2. Available tools list + JSON Schema
3. Project context (cwd, git status)
4. .zipcode.md contents (project-specific instructions)
5. Permission mode description

### Session Management

```
~/.zipcode/
├── sessions/{session-id}.json
├── models/*.gguf
├── config.json
└── (per-project: .zipcode.md in cwd)
```

- Auto-save after each turn
- Resume via `zipcode --session {id}` or `/session` slash command
- Context compaction when history reaches 80% of model context window

### Error Handling

- Tool execution failure → error message as tool_result (model decides retry)
- Inference OOM → shrink KV cache, retry
- Model file corruption → checksum verification at startup

## CLI

### Usage

```
$ zipcode                          # interactive REPL
$ zipcode prompt "explain main.rs" # one-shot mode
$ zipcode --model ./custom.gguf    # custom model path
$ zipcode doctor                   # environment diagnostics
```

### REPL Features

- `rustyline` line editing (history, tab completion)
- Real-time streaming of tool execution results
- Markdown → ANSI terminal rendering (`termimad`)
- Slash commands: `/help`, `/status`, `/clear`, `/compact`, `/session`

### Doctor Output

```
$ zipcode doctor
  Binary:   zipcode v0.1.0 (linux-x86_64)
  CUDA:     ✅ CUDA 12.4 detected (RTX 4090, 24GB VRAM)
  Models:   ✅ gemma-4-27b-it-Q8_0.gguf (28.3 GB)
  VRAM:     ✅ Estimated usage: 22.1 GB / 24.0 GB
  Ready:    ✅ All checks passed
```

## ZIP Packaging

### Archive Structure

```
zipcode-v0.1.0-linux-x86_64-cuda.zip
├── zipcode                       # single binary (~30MB)
├── libcudart.so.12               # CUDA runtime (optional bundle)
├── README.md
├── install.sh
└── models/
    └── PLACE_MODEL_HERE.txt
```

### Air-Gapped Deployment Scenario

```
Internet PC                      Air-Gapped Network
───────────                      ──────────────────
1. Download zipcode.zip
2. Download gemma-4-27b.gguf
3. Copy to USB              ──→  4. Copy from USB
                                  5. ./install.sh
                                  6. Place .gguf in models/
                                  7. Run zipcode
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

## Key Dependencies (Rust crates)

| Crate | Purpose |
|-------|---------|
| `candle-core` | Tensor ops, CUDA backend |
| `candle-nn` | Neural network layers |
| `candle-transformers` | Gemma model architecture |
| `tokenizers` | HuggingFace tokenizer |
| `serde` / `serde_json` | Serialization |
| `tokio` | Async runtime |
| `rustyline` | REPL line editing |
| `termimad` | Markdown ANSI rendering |
| `glob` | File pattern matching |
| `grep-regex` | Content search |
| `clap` | CLI argument parsing |

## Non-Goals (explicitly out of scope)

- API calling to any remote service
- Multi-model support (Gemma 4 only for MVP)
- Windows / macOS support
- Web UI
- Plugin / skill system
- MCP server integration
