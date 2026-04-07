---
title: "ADR: SSE Streaming over Blocking HTTP"
tags: [decisions]
sources: [session-2026-04-08, commit-edbd659]
updated: 2026-04-08
---

# ADR: SSE Streaming over Blocking HTTP

## Status
Accepted (2026-04-08)

## Context
The llama-server backend originally used `"stream": false`, waiting for the complete response before displaying anything. For a 4.6GB model on CPU, this meant ~150 seconds of blank screen.

## Decision
Switch to `"stream": true` with SSE (Server-Sent Events) parsing. Tokens are delivered to the terminal as they're generated via a background thread + mpsc channel.

## Implementation
- `build_chat_request()` sets `"stream": true`
- `stream_sse_events()` opens raw TCP, sends HTTP POST, parses `data:` lines
- `SseEvent` enum: Token, ToolCallDelta, FinishReason, Done
- `ToolCallAccumulator` collects streamed tool call fragments
- Background `std::thread::spawn` feeds tokens through `mpsc::Sender<TokenEvent>`
- Chunked transfer encoding hex lines are skipped via heuristic

## Consequences
- First token appears immediately instead of after full generation
- Tool calls are accumulated from delta chunks (id, name, arguments streamed separately)
- `post_json()` and `http_request()` removed — replaced by `stream_sse_events()` and `health_check()`

## See Also
- [[inference-backends]]
- [[adr-health-check]]
