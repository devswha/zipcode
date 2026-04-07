# zipcode Wiki — Design Spec

**Goal:** Build a persistent, LLM-maintained knowledge base for the zipcode project using Karpathy's Wiki pattern. Knowledge accumulates naturally during Claude Code development sessions.

**Inspired by:** [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), [wiki-skills](https://github.com/kfchou/wiki-skills)

---

## Directory Structure

```
wiki/
├── SCHEMA.md          # Wiki rules, categories, cross-reference conventions
├── index.md           # Full page catalog (one-line summary per page, by category)
├── log.md             # Append-only operation log
├── overview.md        # Evolving project synthesis
└── pages/             # All wiki pages (flat, slug-named, NO subdirectories)
    ├── inference-backends.md
    ├── cuda-compatibility.md
    ├── sse-streaming.md
    ├── tool-system.md
    └── ...
```

Location: `wiki/` at project root. Git tracked — ships with the codebase, reviewable in PRs, available in air-gapped deployments.

---

## Categories (index.md)

| Category | Content |
|----------|---------|
| **Modules** | Per-crate structure, responsibilities, interfaces, key types |
| **Decisions** | Architecture/design decisions with rationale (ADR-style) |
| **Dependencies** | External dependency knowledge (llama.cpp, candle, CUDA versions) |
| **Troubleshooting** | Problem → diagnosis → solution records |

---

## Page Format

Every page uses this frontmatter:

```markdown
---
title: Human-Readable Title
tags: [category, subcategory]
sources: [session-YYYY-MM-DD, commit-sha, url]
updated: YYYY-MM-DD
---

# Title

Content with [[cross-references]] to other pages.

## See Also
- [[related-page-slug]]
```

### Cross-references

- Format: `[[slug]]` where slug matches filename without `.md`
- Example: `[[cuda-compatibility]]` → `wiki/pages/cuda-compatibility.md`
- Bidirectional: when page A links to page B, page B should link back to A

### Slug convention

- Lowercase, hyphen-separated: `sse-streaming`, `tool-system`, `gemma4-support`
- No nested directories — all pages flat in `wiki/pages/`

---

## SCHEMA.md

Defines wiki conventions so any LLM session can maintain consistency:

- Wiki root path (relative to project root)
- Frontmatter required fields
- Cross-reference format
- Category taxonomy
- Log entry format
- Naming conventions

---

## Core Files

### index.md

Content catalog. Every page listed with one-line summary, organized by category:

```markdown
# Wiki Index

## Modules
- [[inference-backends]] — Candle, llama-cpp, and llama-server provider implementations
- [[tool-system]] — Tool trait, registry, execution, and 8KB truncation

## Decisions
- [[adr-sse-streaming]] — Why SSE over non-streaming for llama-server backend

## Dependencies
- [[cuda-compatibility]] — CUDA version requirements, build flags, known issues
- [[llama-cpp-gemma4]] — Gemma 4 support status in llama-cpp-rs

## Troubleshooting
- [[cuda-build-errors]] — Common CUDA build failures and solutions
```

### log.md

Append-only operation log. Each entry:

```markdown
- 2026-04-08 | ingest | Created [[cuda-compatibility]], [[sse-streaming]] from session
- 2026-04-08 | update | Updated [[inference-backends]] with ServerOptions details
```

### overview.md

Evolving synthesis of the entire project. Updated when significant knowledge changes. Not a README — more like a living architectural overview that includes current state, known limitations, and open questions.

---

## Workflow

### Initial Seed

Generate ~10 core pages from current project knowledge:

1. **Modules**: one page per crate (inference-backends, tool-system, runtime-loop, cli-entrypoints)
2. **Decisions**: key ADRs (sse-streaming, gpu-offload, health-check-endpoint)
3. **Dependencies**: external knowledge (cuda-compatibility, llama-cpp-gemma4, candle-limitations)
4. **Troubleshooting**: recent learnings (cuda-build-errors)

### Accumulation (ongoing)

During Claude Code sessions:
- New design decisions → new or updated decision page
- Debugging sessions → troubleshooting page with problem/solution
- New dependency knowledge → update or create dependency page
- Always update index.md and log.md when pages change

### Lint (periodic)

Check wiki health:
- **Broken links**: `[[slug]]` with no matching page
- **Orphan pages**: pages with zero inbound links
- **Stale pages**: pages not updated in 90+ days with time-sensitive claims
- **Missing cross-refs**: pages discussing the same entity without linking each other
- **Contradictions**: conflicting claims across pages

---

## What This Is NOT

- **Not auto-generated from code** — wiki is human+LLM authored prose, not rustdoc
- **Not a CLI feature** — no `zipcode wiki` subcommand; maintained through Claude Code sessions
- **No raw/ directory** — source documents are already in git history
- **Not a replacement for CLAUDE.md** — CLAUDE.md is build/test instructions; wiki is deep knowledge

---

## Success Criteria

1. Core pages exist covering all 4 crates, key decisions, and major dependencies
2. Cross-references form a connected graph (no isolated pages)
3. Today's CUDA troubleshooting knowledge is captured and findable
4. A new developer (or fresh Claude Code session) can query the wiki to get project context faster than reading raw source
