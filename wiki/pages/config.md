# config — Global + project layering

**File:** [`crates/runtime/src/config.rs`](../../crates/runtime/src/config.rs)

---

## Hierarchy

**EXTRACTED** `config.rs:6-24`

```
defaults (hardcoded)
    ↓ override with
~/.zipcode/config.json               (global)
    ↓ override with
<project-root>/.zipcode.json         (project, selective merge)
```

### Project root detection

**EXTRACTED** `find_project_root()` at `config.rs:248`

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
| `permission_mode` | `String` | `"workspace-write"` | Stored as a raw string; parsed into `PermissionMode` via `parse_permission_mode()` at use sites (`config.rs:14`, `permission.rs:102`); see [permissions](permissions.md) |
| `generation` | `GenerationOverrides` | defaults | Only `temperature`, `top_p`, `max_tokens` — see [inference › GenerationConfig](inference.md#generationconfig-lines-103-124) (`config.rs:23-28`) |
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

**EXTRACTED** `config.rs` — 20 inline tests:
- `test_default_config`
- `test_load_from_empty_dir`
- `test_load_project_override` (merge semantics)
- `test_load_project_override_model_dir`
- `test_load_project_override_relative_model_dir`
- `test_load_project_override_from_subdirectory`
- `test_find_project_root_uses_ancestor_marker`
- `test_load_project_override_gpu_settings`
- `test_load_project_override_gpu_layers_overflow_rejected`
- `test_load_project_override_relative_llama_server_bin`
- `test_load_project_override_tilde_model_dir`
- `test_load_project_override_tilde_llama_server_bin`
- `test_load_global_tilde_llama_server_bin`
- `test_load_project_rejects_invalid_permission_mode`
- `test_load_global_rejects_invalid_permission_mode`
- `test_load_rejects_negative_temperature`
- `test_load_rejects_zero_top_p`
- `test_load_rejects_top_p_above_one`
- `test_load_rejects_zero_max_tokens`
- `test_load_accepts_valid_full_access`

---

## Related pages

- [inference › GenerationConfig](inference.md#generationconfig-lines-103-124) — the sampling overrides source
- [permissions](permissions.md) — where `permission_mode` lands
- [llama-server](llama-server.md) — consumer of `gpu_layers`, `flash_attention`, `llama_server_bin`
- [cli](cli.md) — calls `ZipcodeConfig::load(cwd)` during REPL startup
