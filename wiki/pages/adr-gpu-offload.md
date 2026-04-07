---
title: "ADR: GPU Offload and Flash Attention"
tags: [decisions]
sources: [session-2026-04-08, commit-59bbaea]
updated: 2026-04-08
---

# ADR: GPU Offload and Flash Attention

## Status
Accepted (2026-04-08)

## Context
CPU inference of Gemma 4 E2B Q8_0 (4.6GB) runs at ~0.8 tok/s. Users with NVIDIA GPUs should be able to offload computation for dramatically better performance.

## Decision
Add `gpu_layers` and `flash_attention` to `ZipcodeConfig` and `ServerOptions`. Pass `-ngl` and `--flash-attn on` flags to llama-server when configured.

## Configuration

Config file (`~/.zipcode/config.json` or `.zipcode.json`):
```json
{"gpu_layers": 99, "flash_attention": true}
```

Environment variable overrides (take priority):
- `ZIPCODE_GPU_LAYERS=99`
- `ZIPCODE_FLASH_ATTENTION=1`

## Results
RTX 2070 SUPER: ~150s → ~7.6s (20x speedup)

## Caveats
- `--flash-attn` flag format changed between llama-server versions: old versions use `-fa` (bare), new versions require `--flash-attn on`. Current code uses `--flash-attn on`.
- KV cache reuse enabled via `--slot-save-path ~/.zipcode/cache` + `"id_slot": 0` in request body

## See Also
- [[inference-backends]]
- [[cuda-compatibility]]
- [[cuda-build-errors]]
