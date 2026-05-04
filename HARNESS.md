# Harness

Local-only AI coding agent (Rust + Gemma 4 via candle / llama-cpp / llama-server). Air-gapped target.

## Stack

- **Main**: `omo` (oh-my-openagent) + `ouroboros`
- **Side**: —
- **Final pass**: `patina --lang en` on README / CLI help / setup interview text

## Why

- zipcode's own thesis is "ride every model" (Gemma + multiple backends) — `omo`'s multi-model philosophy is the same idea, clean dogfooding
- Rust + air-gapped verification benefits from `ouroboros` 3-stage gate (Mechanical → Semantic → Multi-Model Consensus)
- Multi-backend compatibility testing benefits from `omo`'s native model diversity

## Lane

| Lane | Owner |
|---|---|
| Rust implementation, candle / llama-cpp integration | omo + ouroboros |
| Backend matrix / compatibility tests | omo (native multi-model) |
| README / CLI help / setup interview | omo, then patina |
| Cross-runtime regression on Gemma family | omo |

## Do NOT

- Do not switch primary harness to `omx`/`omo` — Korean tone regresses
- Do not skip ouroboros seed for pattern changes — anchor verification depends on the spec being explicit
- Do not regress the single-static-binary contract — always verify air-gapped install path
