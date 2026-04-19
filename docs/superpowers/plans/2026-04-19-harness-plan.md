# zipcode Harness Plan — Goose-Informed Multi-Agent & Skills Build-Out

**Date:** 2026-04-19
**Scope:** Transform zipcode from single-agent tool-runner into Claude-Code-class harness: sub-agent spawning, skills, tiered context compaction, model-aware tool-call parsing.
**Reference:** `block/goose` (Rust AI coding agent, v1.31.0) — patterns borrowed with attribution.

---

## Motivation

Current zipcode gaps:

- `Agent` tool is a stub (`crates/tools/src/agent.rs:29-33`) — returns "not yet implemented"
- No Skills system — project instructions only via static `.zipcode.md`
- Single-path context compaction (`crates/runtime/src/session.rs:88-138`) — reactive only, no background summarization
- Hardcoded Gemma 3 chat template (`crates/inference/src/chat_template.rs`) — broken for Gemma 4 on native backends
- No per-agent permission scoping — child inherits parent mode without downgrade path
- Latent double-history bug: `ConversationLoop` accumulates messages locally while `llama-server` manages its own KV cache

These block the agentic workflows the project is designed for.

---

## Why Goose

Goose (block/goose, Apache-2.0) is the closest production-grade reference: Rust, local GGUF support via `llama-cpp-2`, MCP-based tools, sub-agent delegation, and Recipe-based skills. We do not fork — we borrow validated patterns.

### Five Goose patterns adopted

1. **`manages_own_context()` provider escape hatch** (`providers/base.rs`) — lets a provider signal that it owns its KV cache so the harness skips local message accumulation. Fixes zipcode's latent double-history bug.

2. **Two-tier context compaction** (`context_mgmt/mod.rs`)
   - Background tool-pair summarization (proactive, async) — batches oldest 10 pairs once count > cutoff
   - Reactive full compaction (blocking) — triggers at ~80% context usage with progressive tool-response eviction if the compaction prompt itself overflows

3. **Sub-agent as independent Agent instance** (`subagent_handler.rs`)
   - New `Agent`, new provider connection, new extension set — zero shared mutable state
   - Communication via `tokio::sync::mpsc::UnboundedSender` for progress events
   - Returns `Conversation` + optional structured output

4. **Recipe schema** (`recipe/mod.rs`)
   - `instructions` → system prompt verbatim
   - `prompt` → first user message
   - `parameters: Vec<RecipeParameter>` with `{{ key }}` Mustache substitution
   - `tool_allowlist` to scope sub-agent capabilities
   - `sub_recipes` auto-spawn sub-agents

5. **Model-aware tool-call parsing** (`native_tool_calling` registry flag)
   - Per-model flag selects native JSON/Jinja path vs text emulator path
   - Supports Gemma, Llama 3.1, ChatML, and text-based emulator mode
   - Covers the "new model family" case without per-addition refactors

---

## Current State Inventory

### What's solid

| Component | Location | Status |
|---|---|---|
| `Tool` trait + `ToolRegistry` | `crates/tools/src/lib.rs:365-398` | HashMap-backed, 10 tools, clean registration |
| `InferenceProvider` trait | `crates/inference/src/lib.rs` | Generic abstraction; feature-gated candle/llama-cpp/llama-server |
| `ConversationLoop` | `crates/runtime/src/conversation.rs` (328 lines) | Generic over provider; 25-iter cap; session auto-save |
| `MockInferenceProvider` | `crates/inference/src/mock.rs` | Queue-based; sufficient for harness tests |
| Session persistence | `crates/runtime/src/session.rs` | JSON at `~/.zipcode/sessions/{id}.json` |
| Permission modes | `crates/runtime/src/permission.rs` | 3 modes: ReadOnly / WorkspaceWrite / FullAccess |

### Gaps (by file)

| Gap | File | Notes |
|---|---|---|
| Agent tool stub | `crates/tools/src/agent.rs:29-33` | Accepts `task: string`, returns error |
| Skills absent | — | No loader, no injection, no `.zipcode/skills/` convention |
| Single-tier compaction | `crates/runtime/src/session.rs:88-138` | Deterministic summary only; no background path |
| Legacy chat template | `crates/inference/src/chat_template.rs:35-80` | Gemma 3 tags hardcoded; wiki doc `wiki/pages/gemma4-format-spec.md` not wired in |
| Per-agent permission scoping | `crates/runtime/src/permission.rs` | No inherit-with-downgrade path |
| `manages_own_context` absent | `crates/inference/src/lib.rs` | Trait lacks the method; double-history risk |
| Tool-call fallback parser | `crates/inference/src/chat_template.rs` | Regex-based, legacy, broken on Gemma 4 |

### Risks if Agent tool built naively

1. **KV cache explosion** — N sub-agents × full context = linear VRAM blowup. Mitigation: hard depth cap + budget inheritance.
2. **Context budget bleed** — parent's remaining tokens not accounted when sub-agent runs.
3. **Permission escalation** — child inheriting `FullAccess` from a user who meant to sandbox the sub-agent.
4. **Session orphans** — crashed child leaves `~/.zipcode/sessions/{child}.json` untracked.
5. **Circular delegation** — agent calls agent calls agent → infinite loop.
6. **Chat template mismatch on backend fallback** — llama-server unavailable → candle/llama-cpp path → legacy template → tool calls don't parse.

---

## Phased Plan

### Phase 0 — Foundation (1–2 days)

Fix latent bugs before stacking new features.

- [ ] Add `fn manages_own_context(&self) -> bool { false }` to `InferenceProvider` trait
- [ ] `LlamaServerProvider` overrides to return `true`
- [ ] In `ConversationLoop::run_turn`, when provider manages context, send only the latest turn (not full history)
- [ ] Extend `ToolContext` (`crates/tools/src/lib.rs:289`) with:
  - `parent_session_id: Option<String>`
  - `depth: u32`
  - `budget_tokens: Option<usize>`
- [ ] Verify `cargo test --workspace` passes and `cargo clippy --workspace --all-targets -- -D warnings` is clean

**Success:** existing tests green; no behavior change user-visible.

### Phase 1 — Agent Tool MVP (5–7 days)

Subagent delegation, one level deep, with strict isolation.

- [ ] Rewrite `crates/tools/src/agent.rs`:
  - Input schema: `{ task: str, skill?: str, tool_allowlist?: string[], max_tokens?: int }`
  - Output: string summary of child's final assistant turn + tool-call count
- [ ] Add `ConversationLoop::spawn_child(...)` helper
  - Independent provider instance (or reuse slot; see risk mitigation below)
  - Filtered `ToolRegistry`
  - Downgraded permissions (see below)
  - Own session file with `parent_id` metadata
- [ ] Constraints:
  - `MAX_AGENT_DEPTH = 2` (no grandchildren)
  - Child context budget = max(parent_remaining / 2, 4096), capped at 32768
  - Child permissions ≤ parent permissions (downgrade-only, no escalate)
  - `agent` tool denied inside child (blocks circular delegation)
- [ ] Session cleanup: on parent exit, delete orphan child session files
- [ ] Extend `crates/runtime/src/permission.rs`:
  - `PermissionPolicy::inherit_for_child(&self, override: Option<PermissionMode>) -> PermissionPolicy`
  - Override may only restrict, never broaden

**Out of scope for MVP:**
- Parallel sub-agents (single llama-server slot contention)
- 3+ level nesting
- Cross-agent structured message passing beyond result summary

**Test plan:**
- `MockInferenceProvider` scenarios: happy path, depth exceeded, permission downgrade, budget exhaustion, allowlist enforcement
- Integration: parent runs a read_file, delegates grep to child, child returns summary, parent acts on it

**Success:** 10 mock-based integration tests green. No real-model dependency.

### Phase 2 — Skills System (3–4 days)

Recipe-pattern skills, file-based, injected via system prompt.

- [ ] Convention: `.zipcode/skills/{name}.md` with YAML frontmatter
  ```yaml
  ---
  name: code-review
  description: Review recently changed code for logic/security/style
  tool_allowlist: [read_file, grep_search, glob_search]
  parameters:
    - name: scope
      default: "HEAD~1..HEAD"
  ---
  You are a code reviewer. Focus on changes in {{ scope }}.
  Report findings as a bullet list...
  ```
- [ ] New module `crates/runtime/src/skills.rs`:
  - `Skill { name, description, tool_allowlist, parameters, body }`
  - `SkillRegistry::load_from(&Path)` walks `.zipcode/skills/*.md`
  - `Skill::render(&self, params: &HashMap<String, String>) -> String` (Mustache-lite: `{{ key }}` only, no conditionals)
- [ ] Prompt integration (`crates/runtime/src/prompt.rs`):
  - Inject skill catalog (name + description only) into system prompt when registry non-empty
  - Keep catalog under 500 tokens
- [ ] New `skill` tool:
  - Input: `{ name: str, params?: object }`
  - Resolves skill → merges params → delegates to Agent tool with rendered `instructions`
- [ ] Optional `--skill {name}` CLI flag for one-shot skill invocation

**Success:** Loading a 3-skill directory and invoking one via the `skill` tool produces a scoped child agent with correct instructions and tool allowlist.

### Phase 3 — Two-Tier Context Compaction (3 days)

Replace single-path summarizer with background + reactive tiers.

- [ ] **Tier 1 — background tool-pair summarization**
  - After each `run_turn` completion, `tokio::spawn` to summarize oldest tool call+response pairs
  - Trigger when pair count > `cutoff + 10` where cutoff scales with context window
  - Replace summarized pairs with one-line summary, mark originals `agent_invisible: true`
  - Non-blocking: main loop proceeds even if summarizer fails
- [ ] **Tier 2 — reactive full compaction**
  - Detect usage via llama-server `prompt_eval_count` response field
  - Threshold: 80% of context window
  - Compaction prompt: "Summarize the conversation so far, preserving goals, decisions, and unresolved items"
  - Progressive fallback if compaction prompt overflows: evict 10% → 20% → 50% → 100% of tool responses
- [ ] Extend `CompactPolicy` to support both tiers; keep existing retain-last-N as a sanity floor

**Success:** 100-turn synthetic agent dialog completes without context overflow at 32K window.

### Phase 4 — Model-Aware Tool-Call Registry (2–3 days)

Decouple chat template + parser from Gemma 3 hardcoding.

- [ ] Registry file: `~/.zipcode/models/registry.json`
  ```jsonc
  {
    "gemma-4-*":   { "tool_format": "gemma_native",  "native_tool_calling": true },
    "qwen2.5-*":   { "tool_format": "chatml",        "native_tool_calling": true },
    "llama-3.1-*": { "tool_format": "llama31_json",  "native_tool_calling": true },
    "*-emulator":  { "tool_format": "emulator",      "native_tool_calling": false }
  }
  ```
- [ ] Refactor `crates/inference/src/chat_template.rs`:
  - Extract format logic behind `trait ChatTemplate { fn render(...) fn parse_tool_calls(...) }`
  - Implementations: `GemmaTemplate`, `ChatMLTemplate`, `Llama31Template`, `EmulatorTemplate`
  - Model file name glob match → registry entry → template selection
- [ ] Emulator mode (fallback for models without native tool calling):
  - Inject system prompt teaching `<<<tool name=... args=...>>>` text syntax
  - Regex-extract in a dedicated parser module
  - Limit emulator tools to a safe subset (bash, read_file, grep_search)

**Success:** `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` loads and runs a tool call via ChatML without code changes.

### Phase 5 — 26B Acceptance Test (½ day)

Now and only now, hit the Windows PC remote llama-server.

- [ ] Point `ZIPCODE_LLAMA_SERVER_URL` at the 26B A4B Opus Distill
- [ ] Run the test suite from Phases 1–4 against it
- [ ] Note any prompt/template friction — fix prompts only, not harness logic
- [ ] Evaluate whether goose's `toolshim.rs` pattern (2nd-pass LLM call to extract tool calls from text) is needed for the Opus Distill's custom `<|turn>`/`<|tool>` template

**Success:** 3 representative skills (code-review, refactor, run-tests) execute end-to-end on the 26B model with correct tool calls and returned summaries.

---

## Timeline

| Week | Days | Work |
|---|---|---|
| W1 | 1–2 | Phase 0 |
| W1 | 3–7 | Phase 1 |
| W2 | 1–4 | Phase 2 |
| W2 | 5–7 | Phase 3 |
| W3 | 1–3 | Phase 4 |
| W3 | 4   | Phase 5 |

**Total:** ~3 weeks. Phases 0–4 depend only on local small models + `MockInferenceProvider`; no remote dependency until Phase 5.

---

## What This Plan Does Not Cover

- Full MCP server integration (goose's biggest feature; deferred — our 10 built-in tools are sufficient for now)
- Parallel sub-agent dispatch (blocked on single llama-server slot; revisit if we add a slot-pool abstraction)
- `on_message` streaming callback for sub-agent progress to UI (Phase 1 returns only the final summary; add later if TUI demands it)
- Security scanning of skill files for prompt-injection payloads (goose's `check_for_security_warnings()` — add in a later hardening phase)

---

## Verification Strategy

Every phase ends with:

1. `cargo test --workspace` green
2. `cargo clippy --workspace --all-targets -- -D warnings` clean
3. `cargo fmt --all -- --check` clean
4. Mock-based integration test for the phase's primary capability
5. Manual smoke test via CLI against the currently configured local model (E2B or Qwen 0.5B)

Phase 5 adds the real-model acceptance gate.

---

## References

- **Goose research brief:** in-session agent output, 2026-04-19
- **zipcode inventory:** in-session agent output, 2026-04-19
- **Goose repo:** https://github.com/aaif-goose/goose
- **Key goose files referenced:**
  - `crates/goose/src/providers/base.rs` (`Provider` trait, `manages_own_context`)
  - `crates/goose/src/providers/local_inference.rs` (local GGUF runtime)
  - `crates/goose/src/agents/agent.rs` + `subagent_handler.rs` (sub-agent pattern)
  - `crates/goose/src/recipe/mod.rs` + `template_recipe.rs` (Recipe schema)
  - `crates/goose/src/context_mgmt/mod.rs` (two-tier compaction)
  - `crates/goose/src/agents/inference_native_tools.rs` / `inference_emulated_tools.rs` / `tool_parsing.rs` (tool-call paths)
  - `crates/goose/src/agents/permission_judge.rs` (LLM-judged permission — future reference)
