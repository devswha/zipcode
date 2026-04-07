---
title: "ADR: /health Endpoint for Readiness Check"
tags: [decisions]
sources: [session-2026-04-08, commit-7615b43]
updated: 2026-04-08
---

# ADR: /health Endpoint for Readiness Check

## Status
Accepted (2026-04-08)

## Context
The original health check used `GET /v1/models`. This endpoint returns HTTP 200 even while the model is still loading. When a chat completion request hits during loading, llama-server returns 503 `"Loading model"`.

For large models (4.6GB+), model loading takes 5-10 seconds. The previous code saw 200 from `/v1/models`, waited 1 second, then tried to use the server — and got 503.

## Decision
Replace `/v1/models` with `/health` endpoint. Parse the response body: only consider the server ready when the body contains `"ok"`. A dedicated `health_check()` function with `HealthStatus` enum (Ready, Loading, Unreachable) handles all cases including 503.

## Implementation
- `HealthStatus::Ready` — body contains `"ok"`
- `HealthStatus::Loading` — server responded but not ready (503 or body without "ok")
- `HealthStatus::Unreachable` — connection refused or write failed

The old `http_request()` function was removed entirely — `stream_sse_events()` handles chat requests, and `health_check()` handles readiness.

## See Also
- [[inference-backends]]
- [[adr-sse-streaming]]
