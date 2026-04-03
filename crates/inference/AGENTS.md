# AGENTS.md - zipcode-inference

**Generated:** 2026-04-03  
**Crate Type:** Rust library - GGUF model inference abstraction  
**Parent:** ../AGENTS.md

---

## Crate Purpose

zipcode-inference is a **dual-backend inference abstraction layer** that provides streaming token generation for GGUF models (primarily Gemma 4). It abstracts hardware-accelerated inference via two interchangeable backends (candle, llama.cpp) behind a single `InferenceProvider` trait, enabling flexible deployment without code changes.

**Key Characteristics:**
- Trait-based backend abstraction (candle, llama-cpp, mock)
- Streaming token generation via `mpsc::Receiver<TokenEvent>`
- Gemma 4 chat template with tool-calling support
- Feature-gated compilation (candle default, llama-cpp optional)
- Zero unsafe code (enforced at workspace level)
- CUDA GPU support with automatic fallback to CPU
- Full test coverage with MockInferenceProvider for integration testing

---

## Key Files

| File | Purpose | Key Types |
|------|---------|-----------|
| `lib.rs` | Public API, trait definitions, engine factory | `InferenceProvider`, `Backend`, `create_engine()` |
| `types.rs` | Message and event types | `ChatMessage`, `TokenEvent`, `FinishReason`, `GenerationConfig` |
| `chat_template.rs` | Gemma 4 prompt formatting and tool parsing | `format_conversation()`, `parse_tool_calls()`, `ToolSpec` |
| `sampler.rs` | Token sampling (temperature, top-p, top-k, repeat penalty) | `Sampler` |
| `engine.rs` | Candle backend implementation | `InferenceEngine` |
| `device.rs` | Device detection and selection (CUDA/CPU) | `select_device()` |
| `llama_cpp_backend.rs` | llama-cpp backend implementation | `LlamaCppProvider` |
| `mock.rs` | Mock implementation for testing | `MockInferenceProvider`, `MockResponse` |

---

## Architecture Overview

### Trait-Based Design

```
InferenceProvider (trait)
    ├── InferenceEngine (candle backend)
    ├── LlamaCppProvider (llama-cpp backend)
    └── MockInferenceProvider (testing)
```

All backends implement a single interface:
```rust
pub trait InferenceProvider: Send {
    fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
    ) -> std::sync::mpsc::Receiver<TokenEvent>;
}
```

This allows runtime backend selection without duplicating generation logic across implementations.

### Feature-Gated Backends

| Feature | Enabled By | Backend | Status |
|---------|-----------|---------|--------|
| `candle` (default) | `--features candle` | Pure-Rust, CPU/CUDA via candle-core | Partial (no Gemma GGUF) |
| `llama-cpp` | `--features llama-cpp` | C++ bindings to llama.cpp | Partial (Gemma 4 unsupported) |
| Both | `--features candle,llama-cpp` | Runtime selection via `Backend::from_name()` | Recommended |
| Neither | `--no-default-features` | Compilation error (informative) | Not recommended |

### Generation Pipeline

```
create_engine(backend, model_path, tokenizer_path, config)
    ↓
Backend-specific load (InferenceEngine::load / LlamaCppProvider::load)
    ↓
generate_stream(messages, tools)
    ↓
format_conversation(messages, tools) [Gemma 4 template]
    ↓
tokenize(prompt)
    ↓
autoregressive loop:
    for i in 0..max_tokens {
        sample_next_token(logits, past_tokens)
        decode_token(token_id)
        send TokenEvent::Token(text)
        if stop_token? break
        if tool_call_detected? send TokenEvent::ToolCall + Done(ToolUse)
    }
    ↓
send TokenEvent::Done(FinishReason)
```

---

## Core Components

### 1. InferenceProvider Trait

**Location:** `lib.rs`

Minimal interface that all backends must implement:

```rust
pub trait InferenceProvider: Send {
    fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        tools: &[chat_template::ToolSpec],
    ) -> std::sync::mpsc::Receiver<TokenEvent>;
}
```

**Adding a new backend:**
1. Create a new struct (e.g., `MyBackendProvider`)
2. Implement `InferenceProvider`
3. Add feature flag to `Cargo.toml`
4. Add conditional compilation block to `create_engine_inner()`
5. Update this AGENTS.md

### 2. ChatMessage & Role Types

**Location:** `types.rs`

Model messages with flexible tool-call tracking:

```rust
pub struct ChatMessage {
    pub role: Role,                      // User, Model, Tool, System
    pub content: String,
    pub tool_call_id: Option<String>,    // For tool result messages
    pub tool_calls: Option<Vec<ToolCallParsed>>,
}

pub enum Role { User, Model, Tool, System }
```

**Convenience constructors:**
- `ChatMessage::user(content)` — User message
- `ChatMessage::system(content)` — System prompt
- `ChatMessage::assistant(content)` — Model response
- `ChatMessage::assistant_with_tool_calls(content, calls)` — Model with tool calls
- `ChatMessage::tool_result(call_id, content)` — Tool execution result

### 3. Chat Template (Gemma 4)

**Location:** `chat_template.rs`

Converts message arrays into Gemma 4 prompt format with tool specifications:

```
<start_of_turn>user
You have access to the following tools:
[{"name": "bash", "description": "...", "parameters": {...}}, ...]

user input text here
<end_of_turn>
<start_of_turn>model
model response
<end_of_turn>
<start_of_turn>tool
tool result
<end_of_turn>
<start_of_turn>model
```

**Key Functions:**

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `format_conversation()` | `&[ChatMessage]`, `&[ToolSpec]` | `String` | Full prompt; tools injected into first user turn only |
| `format_message()` | Single `ChatMessage`, `&[ToolSpec]` | `String` | Single formatted turn (used internally) |
| `parse_tool_calls()` | Model output text | `Vec<ToolCallParsed>` | Extract `<tool_call>...</tool_call>` JSON blocks |
| `extract_text_content()` | Model output text | `String` | Strip tool-call tags, keep plain text |

**Tool Call Format (Gemma 4 Standard):**
```json
{
  "name": "read_file",
  "arguments": {
    "file_path": "src/main.rs"
  }
}
```

**Modifying the Chat Template:**

If supporting a new model (Llama, Qwen, etc.), update `format_message()` with new turn markers. Example for Llama 2:
```rust
Role::User => format!("[INST] {content} [/INST]\n"),
```

Remember to test with `parse_tool_calls()` to ensure tool-calling still works.

### 4. Token Sampling

**Location:** `sampler.rs` (candle only)

Applies generation parameters to logits before sampling:

```rust
pub struct Sampler {
    temperature: f64,
    top_p: f64,
    top_k: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
}
```

**Sampling Pipeline:**
1. Repeat penalty (penalize recently-seen tokens)
2. Temperature scaling (higher = more random)
3. Greedy selection if temperature == 0
4. Softmax normalization
5. Top-k filtering (keep k highest probability tokens)
6. Top-p (nucleus) filtering (keep tokens until cumulative prob > p)
7. Random sampling from filtered distribution

**Parameters:**
- `temperature`: 0 = greedy, 0.7 = default, >1 = more random
- `top_k`: 40 (default) — keep 40 highest-probability tokens
- `top_p`: 0.9 (default) — nucleus sampling threshold
- `repeat_penalty`: 1.1 (default) — penalize token repetition
- `repeat_last_n`: 64 (default) — window for repetition tracking

### 5. Candle Backend

**Location:** `engine.rs`

Pure-Rust inference using `candle-transformers` with quantized Llama weights (GGUF):

```rust
pub struct InferenceEngine {
    model: gemma::ModelWeights,  // Uses quantized_llama (Gemma stand-in)
    tokenizer: Tokenizer,
    device: Device,
    config: GenerationConfig,
}
```

**Device Selection (`device.rs`):**
- Attempts CUDA GPU device
- Falls back to CPU if CUDA unavailable
- Logged via `tracing::info!`

**Current Limitation:**
candle 0.8 lacks `quantized_gemma2` module. Currently using `quantized_llama` as a stand-in (same GGUF interface). This compiles but won't load actual Gemma GGUF files. Upgrade candle-transformers when Gemma support is added.

**Generation Loop:**
1. Tokenize prompt using HuggingFace tokenizer
2. Feed full prompt through model to build KV cache
3. Autoregressive loop:
   - Extract logits for last token
   - Sample next token using `Sampler`
   - Check for stop tokens (`<eos>`, `<end_of_turn>`)
   - Decode token to text
   - Stream via `tx.send(TokenEvent::Token(...))`
4. Detect tool calls from generated text
5. Send `TokenEvent::Done(FinishReason)`

### 6. llama.cpp Backend

**Location:** `llama_cpp_backend.rs`

C++ llama.cpp bindings via `llama-cpp-2` crate:

```rust
pub struct LlamaCppProvider {
    model: LlamaModel,
    backend: LlamaBackend,
    config: GenerationConfig,
}
```

**Differences from Candle:**
- Uses llama.cpp's native C++ inference (faster, more optimized)
- Requires cmake + C++ compiler to build
- Handles backend initialization quirks (BackendAlreadyInitialized error)
- Uses `LlamaSampler` chain with top-k → top-p → temp → dist
- Decodes tokens using UTF-8 streaming decoder (`encoding_rs`)
- Checks `is_eog_token()` instead of hardcoded token IDs

**Current Limitation:**
llama-cpp-2 0.1.141 bundles old llama.cpp lacking Gemma 4 support. Tool calls may fail or not be recognized. Upgrade llama-cpp-2 when Gemma support is available.

### 7. Mock Provider (Testing)

**Location:** `mock.rs`

Deterministic test double that queues predetermined responses:

```rust
pub enum MockResponse {
    Text(String),
    ToolCall { name: String, args: serde_json::Value },
    Error(InferenceError),
}

pub struct MockInferenceProvider {
    responses: VecDeque<MockResponse>,
}
```

**Usage in Integration Tests:**
```rust
let mut provider = MockInferenceProvider::new(vec![
    MockResponse::ToolCall {
        name: "read_file".to_string(),
        args: serde_json::json!({"file_path": "src/main.rs"}),
    },
    MockResponse::Text("file contents".to_string()),
]);

let rx = provider.generate_stream(&messages, &tools);
// Receive TokenEvent::ToolCall + TokenEvent::Done(FinishReason::ToolUse)
```

**Advantages:**
- No model dependency
- Instant results
- Fully deterministic
- Easy to test error conditions

---

## Generation Configuration

**Location:** `types.rs`

```rust
pub struct GenerationConfig {
    pub temperature: f64,           // Sampling randomness (0 = greedy)
    pub top_p: f64,                 // Nucleus sampling threshold
    pub top_k: usize,               // Keep top K tokens
    pub max_tokens: usize,          // Generation limit (default 4096)
    pub repeat_penalty: f32,        // Penalize repetition
    pub repeat_last_n: usize,       // Repetition window
}

impl Default for GenerationConfig {
    // temperature: 0.7, top_p: 0.9, top_k: 40, max_tokens: 4096, ...
}
```

**Typical Adjustments:**
- **For creative responses:** `temperature: 1.0, top_p: 0.95`
- **For deterministic code generation:** `temperature: 0.1`
- **For tool calling:** `temperature: 0.3` (reduce randomness)

---

## Token Events

**Location:** `types.rs`

Generation is streamed as an event sequence:

```rust
pub enum TokenEvent {
    Token(String),                  // Partial text decode
    ToolCall(ToolCallParsed),       // Model called a tool
    Done(FinishReason),             // Generation complete
    Error(InferenceError),          // Error occurred
}

pub enum FinishReason {
    Stop,        // Hit stop token
    MaxTokens,   // Exceeded max_tokens limit
    ToolUse,     // Tool call detected
}
```

**Example Event Sequence (text generation):**
```
Token("Hello")
Token(" world")
Done(FinishReason::Stop)
```

**Example Event Sequence (tool calling):**
```
Token("I'll")
Token(" read")
Token(" the")
Token(" file")
ToolCall(ToolCallParsed { name: "read_file", arguments: {...} })
Done(FinishReason::ToolUse)
```

---

## Public API

```rust
// Main factory
pub fn create_engine(
    backend: Backend,
    model_path: &Path,
    tokenizer_path: &Path,
    config: GenerationConfig,
) -> Result<Box<dyn InferenceProvider>>

// Backend selection
pub enum Backend { LlamaCpp, Candle }
impl Backend {
    pub fn from_name(s: &str) -> Self  // "candle" -> Candle, _ -> LlamaCpp
}

// Device utilities (candle only)
#[cfg(feature = "candle")]
pub fn select_device() -> Device

// Chat formatting
pub fn format_conversation(messages: &[ChatMessage], tools: &[ToolSpec]) -> String
pub fn format_message(msg: &ChatMessage, tools: &[ToolSpec]) -> String
pub fn parse_tool_calls(output: &str) -> Vec<ToolCallParsed>
pub fn extract_text_content(output: &str) -> String
```

---

## Testing

**Test Count:** 14 unit tests across all modules

| Module | Tests | Focus |
|--------|-------|-------|
| `types.rs` | 4 | ChatMessage constructors, tool calls, defaults |
| `chat_template.rs` | 8 | Turn formatting, tool parsing, text extraction |
| `sampler.rs` | 1 | Greedy sampling |
| `device.rs` | 1 | Device selection |

**Run Tests:**
```bash
# All tests
cargo test -p zipcode-inference

# Single test
cargo test -p zipcode-inference test_parse_tool_call_from_output

# With output
cargo test -p zipcode-inference -- --nocapture

# Only integration-style tests (mock provider)
# Currently in parent crate (zipcode-runtime) - see ConversationLoop tests
```

**Integration Testing Pattern:**

Use `MockInferenceProvider` to test conversation logic without model dependency:

```rust
#[test]
fn test_tool_calling_flow() {
    let mut provider = MockInferenceProvider::new(vec![
        MockResponse::ToolCall {
            name: "bash".to_string(),
            args: serde_json::json!({"command": "ls"}),
        },
    ]);
    
    let messages = vec![ChatMessage::user("list files")];
    let rx = provider.generate_stream(&messages, &[]);
    
    // Collect events and assert
    let events: Vec<_> = rx.iter().collect();
    assert_eq!(events.len(), 2); // ToolCall + Done
}
```

---

## Feature Flags

| Flag | Default | Dependencies | Use Case |
|------|---------|--------------|----------|
| `candle` | Yes | candle-core, candle-nn, candle-transformers, tokenizers | Pure-Rust GPU/CPU inference |
| `llama-cpp` | No | llama-cpp-2 (0.1) | C++ llama.cpp bindings (faster) |

**Build Combinations:**

```bash
# Candle only (default)
cargo build --release

# llama-cpp only
cargo build --release --no-default-features --features llama-cpp

# Both (recommended)
cargo build --release --features candle,llama-cpp

# Neither (error)
cargo build --release --no-default-features
# Error: No inference backend enabled. Enable at least one of the `candle` or `llama-cpp` features.
```

---

## Dependencies

| Crate | Version | Purpose | Notes |
|-------|---------|---------|-------|
| **candle-core** | 0.8 (optional) | Tensor operations | GPU support via CUDA |
| **candle-nn** | 0.8 (optional) | Neural network layers | Required by candle backend |
| **candle-transformers** | 0.8 (optional) | Pre-built model architectures | Uses quantized_llama (Gemma stand-in) |
| **tokenizers** | 0.21 (optional) | HuggingFace tokenizers | GGUF model vocabulary |
| **llama_cpp** | 0.1 | C++ llama.cpp bindings | Feature-gated |
| **encoding_rs** | 0.8 | UTF-8 streaming | Used by llama-cpp backend |
| **fastrand** | 2 | Random number generation | Token sampling |
| **serde** | workspace | Serialization | ChatMessage JSON |
| **serde_json** | workspace | JSON | Tool arguments parsing |
| **anyhow** | workspace | Error handling | Context wrapping |
| **thiserror** | workspace | Error types | Custom error enums |
| **tracing** | workspace | Logging | Model load, device selection |

---

## Known Limitations & Workarounds

### 1. Gemma 4 GGUF Unsupported (Both Backends)

**Problem:** Neither candle 0.8 nor llama-cpp-2 0.1.141 fully support Gemma 4.

| Backend | Issue | Workaround |
|---------|-------|-----------|
| Candle | No `quantized_gemma` module | Using `quantized_llama` as stand-in (compiles but won't load Gemma files) |
| llama-cpp | Old llama.cpp lacks Gemma 4 | Wait for llama-cpp-2 0.1.142+ |

**Fix Timeline:** Upgrade when available; test with actual Gemma 4 GGUF files.

### 2. Chat Template Gemma-Only

**Problem:** Hardcoded Gemma 4 format won't work for other models (Llama, Qwen, Mistral).

**Workaround:** Add model detection to `format_message()`:
```rust
pub fn format_message(msg: &ChatMessage, model: &str, tools: &[ToolSpec]) -> String {
    match model {
        "gemma" => { /* current code */ },
        "llama" => { /* Llama format */ },
        _ => { /* fallback */ },
    }
}
```

**Current Scope:** Gemma 4 only. Document if supporting other models.

### 3. No True Streaming

**Problem:** `generate_stream()` is synchronous, queues all events, returns receiver. No first-token latency improvement.

**Current Behavior:**
```rust
for i in 0..max_tokens {
    let token = sampler.sample(...)?;  // blocking
    tx.send(TokenEvent::Token(...))?;   // enqueue
}
```

**Future Improvement:** Spawn thread or use async to improve time-to-first-token.

### 4. Sampler Repeat Penalty Not in llama-cpp

**Problem:** llama.cpp backend doesn't apply repeat penalty (uses different sampling chain).

**Impact:** Minimal — most generation use cases don't rely on it.

**Fix:** Update LlamaSampler chain in llama_cpp_backend.rs when llama-cpp-2 adds support.

---

## AI Agent Instructions

### Adding a New Backend

1. **Define the Provider struct:**
   ```rust
   pub struct MyBackendProvider {
       model: /* backend-specific model type */,
       config: GenerationConfig,
   }
   ```

2. **Implement InferenceProvider:**
   ```rust
   impl InferenceProvider for MyBackendProvider {
       fn generate_stream(
           &mut self,
           messages: &[ChatMessage],
           tools: &[ToolSpec],
       ) -> mpsc::Receiver<TokenEvent> {
           // Format prompt using chat_template::format_conversation()
           // Tokenize, generate loop, stream events via tx.send()
       }
   }
   ```

3. **Add feature flag to Cargo.toml:**
   ```toml
   [features]
   my-backend = ["dep:my-backend-crate"]
   
   [dependencies]
   my-backend-crate = { version = "X", optional = true }
   ```

4. **Add conditional compilation to create_engine_inner():**
   ```rust
   #[cfg(all(feature = "my-backend", ...))]
   fn create_engine_inner(...) -> Result<Box<dyn InferenceProvider>> {
       match backend {
           Backend::MyBackend => {
               let provider = MyBackendProvider::load(model_path)?;
               Ok(Box::new(provider))
           },
           // ...
       }
   }
   ```

5. **Add Backend variant:**
   ```rust
   pub enum Backend {
       MyBackend,
       // ...
   }
   ```

6. **Write integration tests:**
   Use MockInferenceProvider to verify token streaming without backend dependency.

7. **Update AGENTS.md** with new backend details.

### Modifying Chat Template

**Before changing format_message():**
1. Verify the new model's exact prompt format
2. Test tool-call parsing with parse_tool_calls()
3. Ensure <tool_call> tags are preserved
4. Update all references in chat_template.rs tests

**Example: Adding Llama 2 Support**
```rust
pub fn format_message(msg: &ChatMessage, model: &str, tools: &[ToolSpec]) -> String {
    match model {
        "gemma4" => { /* current Gemma format */ },
        "llama2" => {
            match msg.role {
                Role::User => format!("[INST] {content} [/INST]\n"),
                Role::Model => format!("{content}\n"),
                // ...
            }
        },
        _ => { /* fallback to Gemma */ },
    }
}
```

Remember to update the public API signature and caller sites.

### Debugging Generation Issues

| Symptom | Debug Steps |
|---------|------------|
| "Model hung during inference" | Check max_tokens limit, verify device selection (CUDA/CPU), inspect logits dims |
| "Tool calls not detected" | Use `extract_text_content()` to inspect raw output, verify `<tool_call>` tags present |
| "Repetitive output" | Increase repeat_penalty, reduce temperature |
| "Out of memory" | Reduce max_tokens, switch to CPU device, try llama-cpp with smaller context |
| "Wrong sampling" | Verify GenerationConfig passed to Sampler, check temperature/top_p values |

Enable logging:
```bash
RUST_LOG=debug cargo test -p zipcode-inference -- --nocapture
```

---

## Common Patterns

### Creating an Engine at Runtime

```rust
use zipcode_inference::{Backend, create_engine, GenerationConfig};

let config = GenerationConfig {
    temperature: 0.3,
    top_p: 0.9,
    ..Default::default()
};

let mut engine = create_engine(
    Backend::from_name("candle"),
    "models/gemma-4-9b.gguf",
    "models/tokenizer.json",
    config,
)?;
```

### Streaming Generation

```rust
let messages = vec![ChatMessage::user("What is 2+2?")];
let tools = vec![];

let rx = engine.generate_stream(&messages, &tools);

for event in rx {
    match event {
        TokenEvent::Token(text) => print!("{text}"),
        TokenEvent::ToolCall(call) => println!("Tool: {}", call.name),
        TokenEvent::Done(reason) => println!("Finished: {:?}", reason),
        TokenEvent::Error(e) => eprintln!("Error: {e}"),
    }
}
```

### Testing with MockInferenceProvider

```rust
#[test]
fn test_my_logic() {
    let mut mock = MockInferenceProvider::new(vec![
        MockResponse::Text("answer".to_string()),
    ]);
    
    // Pass mock as Box<dyn InferenceProvider>
    let rx = mock.generate_stream(&messages, &tools);
    assert_eq!(rx.recv(), Some(TokenEvent::Token("answer".to_string())));
}
```

---

## Development Guidelines

### Code Style

- No unsafe code (workspace-wide enforcement via `unsafe_code = "forbid"`)
- Clippy pedantic (`clippy::all = "warn"`)
- Module names allowed to repeat in child items
- Always use `.with_context()` with anyhow for error propagation

### Before Committing

```bash
# Tests must pass
cargo test -p zipcode-inference

# Zero clippy warnings
cargo clippy -p zipcode-inference --all-targets -- -D warnings

# Format check
cargo fmt -p zipcode-inference -- --check
```

### Documentation

- Public types: doc comment with example
- Public functions: describe inputs, outputs, errors, example
- Trait implementations: note any backend-specific behavior
- Feature flags: document requirements (cmake, C++ compiler, etc.)

---

## Related Documentation

- **Parent:** `/AGENTS.md` — Workspace overview
- **Build Guide:** `/CLAUDE.md` — Build commands, architecture diagram
- **User Guide:** `/README.md` — Features, CLI reference, deployment
- **Runtime:** `../runtime/AGENTS.md` — ConversationLoop, integration tests
- **Tools:** `../tools/AGENTS.md` — Tool execution, output truncation

---

<!-- MANUAL -->

## Manual Maintenance

- **candle-transformers upgrade:** When Gemma 4 support added, replace quantized_llama import
- **llama-cpp-2 upgrade:** When version > 0.1.142, test with actual Gemma GGUF file
- **New model support:** Add format_message() branches, document prompt format
- **Sampler tuning:** Adjust default config based on user feedback
- **Error handling:** Add new InferenceError variants as backends expand

<!-- /MANUAL -->
