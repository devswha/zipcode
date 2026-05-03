# Harness

Local-only AI coding agent (Rust + Gemma 4 via candle / llama-cpp / llama-server). Air-gapped target.

## Stack

- **Main**: `omo` (oh-my-openagent) + `ouroboros`
- **Side**: `omc` for `ouroboros` seed authoring (spec-leak mitigation)
- **Final pass**: `patina --lang en` on README / CLI help / setup interview text

## Why

- zipcode's own thesis is "ride every model" (Gemma + multiple backends) — `omo`'s multi-model philosophy is the same idea, clean dogfooding
- Rust + air-gapped verification benefits from `ouroboros` 3-stage gate (Mechanical → Semantic → Multi-Model Consensus)
- Multi-backend compatibility testing benefits from `omo`'s native model diversity

## Spec-leak mitigation (Gemini caveat)

`omo` running on Kimi/GLM under pressure is known to leak out of spec-first loops.
**Compromise**: write the `ouroboros` seed/spec on `omc` (gold-standard ouroboros host), then implement on `omo`.

## Lane

| Lane | Owner |
|---|---|
| Rust implementation, candle / llama-cpp integration | omo + ouroboros (impl) / omc (seed) |
| Backend matrix / compatibility tests | omo (native multi-model) |
| README / CLI help / setup interview | omo, then patina |
| Cross-runtime regression on Gemma family | omo |

## Do NOT

- Do not author `ouroboros` seeds on `omo` directly — risk of spec leak
- Do not commit `omc`-written Rust without running it through omo's verification (ensures air-gapped path actually works)
- Do not regress the single-static-binary contract — always verify air-gapped install path
