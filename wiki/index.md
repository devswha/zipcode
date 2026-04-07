# Wiki Index

## Modules
- [[inference-backends]] — Candle, llama-cpp, and llama-server provider implementations
- [[tool-system]] — Tool trait, registry, execution, and 8KB truncation
- [[runtime-loop]] — ConversationLoop, session persistence, permission policy
- [[cli-entrypoints]] — REPL, one-shot, doctor, setup commands

## Decisions
- [[adr-sse-streaming]] — SSE streaming over blocking HTTP for llama-server
- [[adr-gpu-offload]] — GPU layer offload and flash attention support
- [[adr-health-check]] — /health endpoint over /v1/models for readiness

## Dependencies
- [[cuda-compatibility]] — CUDA version requirements and build configuration
- [[llama-cpp-gemma4]] — Gemma 4 architecture support in llama-cpp-rs
- [[candle-limitations]] — Candle backend gaps and workarounds

## Troubleshooting
- [[cuda-build-errors]] — CUDA build failures: toolkit versions, compiler flags, solutions
- [[test-scenarios]] — Validated test scenarios and results
