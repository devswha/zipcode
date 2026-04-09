# config — Global + project layering

**File:** [`crates/runtime/src/config.rs`](../../crates/runtime/src/config.rs)

---

## Hierarchy

**EXTRACTED** `config.rs:95-137`

```
defaults (hardcoded)
    ↓ override with
~/.zipcode/config.json               (global)
    ↓ override with
<project-root>/.zipcode.json         (project, selective merge)
```

### Project root detection

**EXTRACTED** `find_project_root()` at `config.rs:151-163`

Walks ancestors from cwd looking for one of:
1. `.zipcode.json`
2. `.zipcode.md`
3. `.git`

First match wins. If none, project root falls back to cwd.

**GOTCHA:** the order matters — a nested `.git` without `.zipcode.*` will anchor project root at the nested repo, not the outer parent.

---

## Fields

**EXTRACTED** `config.rs` — top-level fields.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `model_dir` | `PathBuf` | `~/.zipcode/models` | Where GGUF files are discovered |
| `model_file` | `Option<String>` | `None` | Specific GGUF filename; if absent, CLI scans `model_dir` |
| `llama_server_bin` | `Option<PathBuf>` | `None` | Overridden by `ZIPCODE_LLAMA_SERVER_BIN` env |
| `permission_mode` | `PermissionMode` | `WorkspaceWrite` | Parsed via kebab-case; see [permissions](permissions.md) |
| `generation` | `GenerationConfig` override | defaults | temperature, top_p, top_k, max_tokens, repeat_penalty, repeat_last_n — see [inference › GenerationConfig](inference.md#generationconfig-lines-103-124) |
| `gpu_layers` | `Option<i32>` | `None` | Overridden by `ZIPCODE_GPU_LAYERS` env; passed to llama-server as `-ngl` |
| `flash_attention` | `bool` | `false` | Overridden by `ZIPCODE_FLASH_ATTENTION` env |

### Env var precedence (INFERRED from `crates/cli/src/repl.rs`)

`env var > config.<field> > hardcoded default`

Example: `ZIPCODE_GPU_LAYERS=99 zipcode repl` wins over whatever `~/.zipcode/config.json` sets.

---

## Path expansion

**EXTRACTED** `config.rs:41-67`

| Function | Purpose |
|----------|---------|
| `expand_user_path()` | Handles `~` and `~/...` (lines 41-57) |
| `resolve_project_path()` | Relative paths are resolved against project root (lines 60-67) |

Absolute paths pass through untouched.

---

## `.zipcode.json` override semantics

**INFERRED** from tests in `config.rs`. Override is a **selective merge**: keys present in the project file replace the corresponding global keys, other global keys are preserved. Nested structures like `generation` are merged field-by-field.

Example:

```jsonc
// ~/.zipcode/config.json
{
  "permission_mode": "workspace-write",
  "generation": { "temperature": 0.7, "top_p": 0.9 }
}

// .zipcode.json (project)
{
  "generation": { "temperature": 0.3 }
}

// Effective:
{
  "permission_mode": "workspace-write",
  "generation": { "temperature": 0.3, "top_p": 0.9 }
}
```

---

## Tests

**EXTRACTED** `config.rs` — 13 inline tests:
- `default_config`
- `load_global_only`
- `project_overrides_global` (merge semantics)
- `relative_paths_resolved_against_project_root`
- `tilde_expansion`
- `gpu_layers_field`
- `flash_attention_field`
- Plus ~6 more covering edge cases.

---

## Related pages

- [inference › GenerationConfig](inference.md#generationconfig-lines-103-124) — the sampling overrides source
- [permissions](permissions.md) — where `permission_mode` lands
- [llama-server](llama-server.md) — consumer of `gpu_layers`, `flash_attention`, `llama_server_bin`
- [cli](cli.md) — calls `ZipcodeConfig::load(cwd)` during REPL startup
