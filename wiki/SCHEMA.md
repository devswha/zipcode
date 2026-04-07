---
wiki_root: wiki
updated: 2026-04-08
---

# zipcode Wiki Schema

## Location

Wiki root: `wiki/` relative to project root.

## Directory Layout

```
wiki/
├── SCHEMA.md       # This file — conventions and configuration
├── index.md        # Page catalog (every page, one-line summary, by category)
├── log.md          # Append-only operation log
├── overview.md     # Evolving project synthesis
└── pages/          # All wiki pages (flat, slug-named)
```

## Page Frontmatter (required)

```yaml
---
title: Human-Readable Title
tags: [category, ...]
sources: [session-YYYY-MM-DD, commit-SHA, URL]
updated: YYYY-MM-DD
---
```

## Categories

| Category | Slug prefix | Content |
|----------|-------------|---------|
| Modules | (none) | Per-crate structure, responsibilities, interfaces |
| Decisions | `adr-` | Architecture/design decisions with rationale |
| Dependencies | (none) | External dependency knowledge |
| Troubleshooting | (none) | Problem → diagnosis → solution records |

## Cross-References

- Format: `[[slug]]` where slug matches filename without `.md`
- Example: `[[cuda-compatibility]]` → `wiki/pages/cuda-compatibility.md`
- Bidirectional: if A links to B, B should link back to A

## Slug Convention

- Lowercase, hyphen-separated: `sse-streaming`, `tool-system`
- Decision pages prefixed with `adr-`: `adr-sse-streaming`
- No nested directories — all pages flat in `wiki/pages/`

## Log Entry Format

```
- YYYY-MM-DD | operation | Description with [[page-refs]]
```

Operations: `init`, `ingest`, `update`, `lint`, `query`

## Maintenance

- Update `index.md` whenever pages are added or removed
- Append to `log.md` for every operation
- Update `overview.md` when significant knowledge changes
- Run lint periodically: check broken links, orphans, contradictions
