# inference — Backends, Sampling, Types

The `zipcode-inference` crate. Home of the `InferenceProvider` god node.

**Crate path:** [`crates/inference/`](../../crates/inference/)
**Depends on nothing from this workspace** — it's the bottom of the dependency graph.

---

## Module layout

| File | Owns |
|------|------|
| `src/lib.rs` | `InferenceProvider` trait, `Backend` enum, `create_engine()` factory |
| `src/types.rs` | `ChatMessage`, `Role`, `ToolCallParsed`, `TokenEvent`, `FinishReason`, `GenerationConfig` |
| `src/chat_template.rs` | Gemma 4 chat format, `<tool_call>` parsing — see [chat-template](chat-template.md) |
| `src/llama_cpp_backend.rs` | `LlamaCppProvider` — `llama-cpp-2` native bindings |
| `src/llama_server_backend.rs` | `LlamaServerProvider` — llama.cpp HTTP subprocess — see [llama-server](llama-server.md) |
| `src/engine.rs` | `InferenceEngine` — pure-Rust candle backend |
| `src/mock.rs` | `MockInferenceProvider` — queue-based test double |
| `src/device.rs` | CUDA/CPU device selection (candle-only) |
| `src/sampler.rs` | Temperature, top-p, top-k, repeat penalty |

---

## `InferenceProvider` trait

**EXTRACTED** `crates/inference/src/lib.rs:28-33`

```rust
pub trait InferenceProvider: Send {
    fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        tools: &[chat_template::ToolSpec],
    ) -> std::sync::mpsc::Receiver<TokenEvent>;
}
```

**Implementors** (all return `Box<dyn InferenceProvider>` from `create_engine()`):

| Impl | File | Feature flag | Status |
|------|------|--------------|--------|
| `LlamaCppProvider` | `llama_cpp_backend.rs:29` | `llama-cpp` | Native bindings; **Gemma 4 blocked upstream** (see [gotchas](gotchas.md#backend-reality-check)) |
| `LlamaServerProvider` | `llama_server_backend.rs:40` | always on | **Fully functional.** Subprocess + SSE. See [llama-server](llama-server.md) |
| `InferenceEngine` (candle) | `engine.rs:19` | `candle` (default) | Pure-Rust; **uses `quantized_llama` as Gemma placeholder** — doesn't actually load real Gemma |
| `MockInferenceProvider` | `mock.rs:20` | test-only | Queue-based; used by integration tests |

**GOTCHA:** `candle` is the default feature, but the only production-ready backend today is `llama-server`. The `--backend llama-server` flag is the path you actually want. See [CLAUDE.md](../../CLAUDE.md) "Current Limitations".

### Adding a backend

1. Create `src/my_backend.rs` with a struct implementing `InferenceProvider`.
2. Add a variant to `Backend` enum (`lib.rs:37-45`).
3. Wire it into `create_engine()` (`lib.rs:71-196`) — dispatch on the new variant.
4. Feature-gate with `#[cfg(feature = "my-backend")]` if heavy deps.
5. Add a test using `MockInferenceProvider` as the contract baseline.

---

## Core types

**EXTRACTED** `crates/inference/src/types.rs`

### `ChatMessage` (lines 13-67)

Wire format for every message across every crate.

```rust
pub struct ChatMessage {
    pub role: Role,                              // User | Model | System | Tool
    pub content: String,
    pub tool_calls: Option<Vec<ToolCallParsed>>, // Set when role == Model and output had <tool_call> blocks
    pub tool_call_id: Option<String>,            // Set when role == Tool
}
```

Constructors (lines 24-66): `system()`, `user()`, `model()`, `model_with_tool_calls()`, `tool_result()`.

### `TokenEvent` (lines 77-82)

Streamed by `generate_stream()` receiver channel.

```rust
pub enum TokenEvent {
    Token(String),
    ToolCall(ToolCallParsed),
    Done(FinishReason),
    Error(InferenceError),
}
```

### `FinishReason` (lines 84-89)

```rust
pub enum FinishReason {
    Stop,      // Hit <eos> or <end_of_turn>
    MaxTokens, // Hit GenerationConfig.max_tokens
    ToolUse,   // Model emitted <tool_call> block(s)
}
```

### `GenerationConfig` (lines 103-124)

Sampling knobs with defaults:

| Field | Default | Notes |
|-------|---------|-------|
| `temperature` | 0.7 | |
| `top_p` | 0.9 | |
| `top_k` | 40 | |
| `max_tokens` | 4096 | |
| `repeat_penalty` | 1.1 | |
| `repeat_last_n` | 64 | |

Overridable via `~/.zipcode/config.json` under `generation.*`. See [config](config.md#fields).

### `ToolCallParsed` (lines 69-74)

```rust
pub struct ToolCallParsed {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}
```

Produced by `chat_template::parse_tool_calls()` — see [chat-template](chat-template.md).

---

## Factory: `create_engine()`

**EXTRACTED** `crates/inference/src/lib.rs:71-196`

Dispatches on `Backend` enum, returns `Box<dyn InferenceProvider>`.

Parameters (from `ServerOptions` struct):
- `gpu_layers: Option<i32>` — `-ngl` for llama-server/llama-cpp
- `flash_attention: bool` — `--flash-attn on`
- `context_size: usize` — default 8192

**INFERRED:** CLI reads env vars (`ZIPCODE_GPU_LAYERS`, `ZIPCODE_FLASH_ATTENTION`) before calling `create_engine()` — env takes precedence over config.

---

## Sampling

**EXTRACTED** `crates/inference/src/sampler.rs` (candle-only)

Chain: `top_k → top_p → temperature → sample`. Used by `InferenceEngine`. `LlamaCppProvider` and `LlamaServerProvider` use their own sampler chains (llama-cpp-2's builder API / llama-server HTTP options).

---

## Tests

**EXTRACTED** — counted during mapping pass.

- `lib.rs` — 5 tests (backend parsing, llama-server e2e marked `#[ignore]` needing `ZIPCODE_TEST_MODEL_PATH`)
- `types.rs` — 5 tests (ChatMessage constructors, GenerationConfig defaults)
- `chat_template.rs` — 7 tests (format, parse, extract) — see [chat-template](chat-template.md)
- Backend-specific tests are minimal because they need real GGUF models.

---

## Related pages

- [chat-template](chat-template.md) — how messages become prompts
- [llama-server](llama-server.md) — the primary working backend
- [conversation-loop](conversation-loop.md) — the consumer of `InferenceProvider`
- [gotchas](gotchas.md#backend-reality-check) — why candle and llama-cpp are half-broken
