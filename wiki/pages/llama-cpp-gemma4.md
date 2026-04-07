---
title: Gemma 4 Support in llama-cpp-rs
tags: [dependencies]
sources: [session-2026-04-08]
updated: 2026-04-08
---

# Gemma 4 Support in llama-cpp-rs

## Status
**Blocked** as of 2026-04-08. `llama-cpp-rs` v0.1.141 does not recognize the `gemma4` architecture.

## Error
```
llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'gemma4'
```

## Workaround
The CLI auto-detects Gemma 4 models (filename contains "gemma-4" or "gemma4") and falls back to llama-server backend when llama-cpp fails. See `is_probably_gemma4_model()` in `crates/cli/src/repl.rs`.

## Upstream Status
- llama.cpp (C library) supports Gemma 4 fully
- `llama-cpp-rs` Rust bindings lag behind — needs upstream update to expose Gemma 4 architecture
- No known timeline for the fix

## Impact
- Native bindings unusable for Gemma 4 — must use llama-server subprocess
- llama-server subprocess adds startup latency (model loading ~5-10s)
- Once llama-cpp-rs updates, native bindings would eliminate subprocess overhead

## See Also
- [[inference-backends]]
- [[candle-limitations]]
