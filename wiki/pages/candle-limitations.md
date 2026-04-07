---
title: Candle Backend Limitations
tags: [dependencies]
sources: [session-2026-04-08]
updated: 2026-04-08
---

# Candle Backend Limitations

## Status
**Placeholder only** as of 2026-04-08. Compiles but cannot load Gemma GGUF models.

## Problem
Candle v0.8 has no `quantized_gemma` module. The inference engine uses `quantized_llama` as a stand-in, which has incompatible architecture.

## Impact
- `Backend::Candle` compiles and is selectable
- Actually loading a Gemma GGUF file will fail at runtime
- The candle backend is effectively a dead code path for Gemma models

## Why It Exists
- Candle is pure Rust — no C dependencies, no cmake, no libclang
- Ideal for air-gapped environments where building C deps is hard
- If/when candle adds Gemma support, it becomes the simplest deployment option

## See Also
- [[inference-backends]]
- [[llama-cpp-gemma4]]
