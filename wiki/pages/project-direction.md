# project-direction — Practical backend strategy and product priorities

This page records the current intended direction for zipcode as a local/offline coding agent.

It is not a promise that every long-term item is already implemented. It is the shortest accurate summary of what the codebase is already optimizing for and what it should optimize for next.

---

## Short version

**INFERRED** from the current CLI/backend/install flow:

- **Primary practical engine for Gemma 4:** `llama-server`
- **Primary product goal:** make the local coding-agent UX reliable (`install`, `setup`, `doctor`, `update`, recovery, speed defaults)
- **Longer-term secondary track:** native direct / in-process inference for performance, only when Gemma 4 support is clearly stable

---

## Why `llama-server` is the primary path today

### 1) The CLI now chooses it up front for Gemma 4

**EXTRACTED** `crates/cli/src/repl.rs:266-299,339-383`

- `build_startup_notices()` emits:
  - `Using llama-server directly for Gemma 4 to avoid slow native fallback.`
  - slow-setting notices for missing `gpu_layers` / `flash_attention`
- `prepare_loop()`:
  - discovers the helper path
  - builds `ServerOptions`
  - resolves the **effective backend** before model load
  - bails if Gemma 4 would still go through unsupported native `llama-cpp`

This means the intended Gemma 4 path is no longer “try native first and hope.” The code now treats helper-backed execution as the practical route.

### 2) Setup and install are being shaped around helper-backed execution

**EXTRACTED** `crates/cli/src/commands.rs:159-193,324-360`

- `setup` persists:
  - `model_dir`
  - `model_file`
  - `llama_server_bin`
- It also upgrades stale configs with recommended performance defaults when CUDA is available:
  - `gpu_layers=999`
  - `flash_attention=true`

**EXTRACTED** `scripts/lib/install_common.sh:99-147`

Fresh installs now write those same helper-backed performance defaults into the generated config.

### 3) Reinstall and repair flows are being hardened around the helper

**EXTRACTED** `install.sh:60-93,324-349,474-535,630-650`

The source-checkout installer now prefers to reuse:
- the configured model from saved config
- the already-installed helper in `~/.zipcode/bin/llama-server`

before it falls back to interactive prompts.

**EXTRACTED** `scripts/install_llama_server.sh:45-130`

The helper installer now normalizes an installed wrapper path back to the real helper payload so reinstalls do not corrupt `llama-server-real`.

---

## What this implies for product work

### Near-term / medium-term priority

**INFERRED** from the code above plus recent CLI work:

The main product lane is **not** “invent a new backend first.”
It is:

1. reliable install
2. reliable setup / doctor / repair
3. safe update flow
4. sane speed defaults
5. clear user-facing notices when the machine is running in a slow configuration

That is the difference between “a local demo” and “a local coding agent people can actually live with.”

### Native direct backend is still valuable

**INFERRED** from the current architecture and backend constraints.

A native direct/in-process backend still has obvious upside:
- lower overhead
- fewer moving parts
- better long-term performance potential

But the current codebase already signals that this should be treated as a **secondary long-term performance track**, not the main short-term delivery path for Gemma 4.

The strongest clue is that the CLI explicitly routes Gemma 4 away from native `llama-cpp` today instead of trying to force it.

---

## Recommended decision rule

When there is a tradeoff between:

- polishing helper-backed local execution (`llama-server`), or
- pushing native direct Gemma 4 support further before it is stable,

prefer:

> **the change that makes helper-backed local execution safer, clearer, easier to install, easier to recover, and easier to run fast by default.**

That is the current best path to a dependable local coding agent.

---

## Related pages

- [llama-server](llama-server.md) — primary working backend details
- [cli](cli.md) — startup, setup, and notices
- [config](config.md) — persistent speed settings and helper path
- [gotchas](gotchas.md#backend-reality-check) — backend constraints that shape this direction
- [recipes](recipes.md) — operational playbook
