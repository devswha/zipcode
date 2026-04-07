---
title: Inference Backends
tags: [modules]
sources: [session-2026-04-08]
updated: 2026-04-08
---

# Inference Backends

The `zipcode-inference` crate provides three pluggable backends behind the `InferenceProvider` trait.

## InferenceProvider Trait

```rust
pub trait InferenceProvider: Send {
    fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
    ) -> mpsc::Receiver<TokenEvent>;
}
```

Returns an `mpsc::Receiver<TokenEvent>` — consumers iterate token-by-token.

## Backends

### llama-server (primary)

HTTP subprocess using OpenAI-compatible API. SSE streaming enabled (`stream: true`). Supports GPU offload (`-ngl`), flash attention (`--flash-attn on`), and KV cache reuse (`--slot-save-path` + `id_slot: 0`).

- Binary resolved from: `ZIPCODE_LLAMA_SERVER_BIN` env → `LLAMA_SERVER_BIN` env → `llama-server` on PATH
- Health check via `/health` endpoint (waits for `"ok"` status, handles 503 during model loading)
- Server process killed on `Drop`

### llama-cpp (native bindings)

Direct `llama-cpp-2` v0.1.141 Rust bindings. Works for Gemma 2 models. **Gemma 4 blocked** — `llama-cpp-rs` doesn't recognize `gemma4` architecture yet. See [[llama-cpp-gemma4]].

When llama-cpp fails on a Gemma 4 model, the CLI automatically falls back to llama-server.

### candle (pure Rust)

Uses `quantized_llama` as placeholder — compiles but cannot actually load Gemma GGUF. See [[candle-limitations]].

## ServerOptions

```rust
pub struct ServerOptions {
    pub gpu_layers: Option<i32>,
    pub flash_attention: bool,
    pub context_size: usize,
}
```

Passed to `LlamaServerProvider::load()`. Populated from `ZipcodeConfig` fields + env var overrides.

## See Also
- [[adr-sse-streaming]]
- [[adr-gpu-offload]]
- [[cuda-compatibility]]
- [[llama-cpp-gemma4]]
- [[candle-limitations]]
