# Wiki Regeneration Prompt

Paste the block below into Claude Code (running in the zipcode repo root) to regenerate `wiki/` from scratch. The script `scripts/wiki_sync_check.sh` cannot rewrite content by itself — it only detects drift. Use this prompt when drift is detected and you want a full refresh.

---

## Context you give Claude

This repo keeps a hand-maintained "graphify-style" developer wiki at `wiki/`. The structure is deliberate:

```
wiki/
├── index.md                    # entry point: god nodes, communities, "where do I go to change X"
├── GRAPH_REPORT.md             # ranked god nodes, cross-crate surprises, suggested questions, gotcha summary
└── pages/
    ├── inference.md            # InferenceProvider trait + 4 backends + core types + sampler
    ├── chat-template.md        # hardcoded Gemma 4 format, <tool_call> parsing
    ├── tools.md                # Tool trait + ToolRegistry + 11 tools + path safety + truncation
    ├── conversation-loop.md    # ConversationLoop + 25-iter cap + StreamCallback
    ├── permissions.md          # PermissionMode + policy + 3-tier matrix
    ├── config.md               # ZipcodeConfig + global/project layering + env precedence
    ├── session.md              # ~/.zipcode/sessions/{uuid}.json persistence
    ├── cli.md                  # clap + REPL + TUI + doctor + setup
    ├── llama-server.md         # subprocess lifecycle, /health, SSE, GPU offload
    ├── gotchas.md              # consolidated non-obvious couplings and silent failures
    └── recipes.md              # "How do I..." playbook for common dev tasks
```

**Provenance rule:** every factual claim must carry a `file:line` reference (for example `crates/inference/src/lib.rs:28`). Tag claims with:

- **EXTRACTED** — directly verified against code
- **INFERRED** — reasoned from surrounding code, not a literal quote
- **GOTCHA** — non-obvious coupling, silent failure, invariant worth flagging
- **STUB** — code exists but isn't implemented yet

Cross-page links use plain markdown (`[text](other-page.md)`), not Obsidian `[[wikilinks]]`.

---

## Exact task for Claude

```
Regenerate wiki/ from crates/ following the graphify-style structure documented
in scripts/wiki_prompt.md. Specifically:

1. Delete the existing wiki/ directory entirely.
2. Use the Explore agent (subagent_type=Explore, "very thorough") to map crates/.
   Ask for: per-crate module layout, public traits, implementors, constants,
   feature flags, god nodes (highest-degree concepts), cross-crate wiring,
   non-obvious couplings, and test topology. Demand absolute file:line
   references — precision is required.
3. Recreate these files using the explorer's findings:
     wiki/index.md
     wiki/GRAPH_REPORT.md
     wiki/pages/inference.md
     wiki/pages/chat-template.md
     wiki/pages/tools.md
     wiki/pages/conversation-loop.md
     wiki/pages/permissions.md
     wiki/pages/config.md
     wiki/pages/session.md
     wiki/pages/cli.md
     wiki/pages/llama-server.md
     wiki/pages/gotchas.md
     wiki/pages/recipes.md
4. Keep the five god nodes front and center in index.md and GRAPH_REPORT.md:
     InferenceProvider, Tool+ToolRegistry, ConversationLoop, ChatMessage,
     PermissionMode.
5. Every claim gets a file:line ref with an EXTRACTED/INFERRED/GOTCHA/STUB tag.
6. Every page ends with a "Related pages" section that cross-links the other
   community pages it depends on.
7. Verify: all markdown links resolve, all crates/*.rs:N references point to
   real lines. Run: grep -rhoE 'crates/[A-Za-z0-9_./-]+\.rs:[0-9]+' wiki/
8. Record the new baseline:
     scripts/wiki_sync_check.sh --mark-synced
```

---

## Why this is a paste-only prompt, not a shell script

Walking the code graph, deciding what's a "god node", separating EXTRACTED from INFERRED, and writing dense natural-language explanations all require an LLM. A shell script can't do it. The `scripts/wiki_sync_check.sh` hook therefore only detects drift — actual regeneration is a manual step where you open Claude Code and paste this prompt.

If you want to shortcut things, there's a token-cheap middle ground: run the hook, see which files drifted, then ask Claude Code to update only the affected community pages rather than rewriting the whole wiki.
