# zipcode harness benchmark suite

A small, reusable benchmark harness for comparing **zipcode + model**
configurations. The goal is to answer questions like:

- "Does the 31B Claude-Opus-Distill on a hypothetical RTX 3090 do
  better than the 26B A4B Opus-Distill on my current 4080 Super?"
- "How much does swapping `ZIPCODE_LLAMA_SERVER_URL` between local E4B
  and a remote big model change end-to-end task success?"
- "Did the last zipcode harness change (e.g. auto-retry, context
  bump) regress easy tasks?"

Everything is plain bash + python3 (no Rust deps to install) and
produces JSON per task plus a Markdown summary that diffs cleanly
between runs.

## Running a profile

`zipcode` picks its backend from env vars (`ZIPCODE_LLAMA_SERVER_URL`,
`ZIPCODE_LLAMA_SERVER_ALIAS`). The bench just observes whatever
zipcode is configured to do, so switching profiles = switching env:

```bash
# Build the release binary once
cargo build --release -p zipcode

# Profile 1: remote 26B A4B Opus Distill on Windows 4080 Super
export ZIPCODE_LLAMA_SERVER_URL=http://100.108.247.75:8080
export ZIPCODE_LLAMA_SERVER_ALIAS=zipcode-remote
./benches/run.sh --profile remote-26B-A4B-Opus \
    --model-hint "TeichAI gemma-4-26B-A4B-it-Claude-Opus-Distill Q4_K_M"

# Profile 2: local bundled E4B (unset the remote vars)
unset ZIPCODE_LLAMA_SERVER_URL ZIPCODE_LLAMA_SERVER_ALIAS
./benches/run.sh --profile local-E4B \
    --model-hint "unsloth gemma-4-E4B-it Q4_K_M"
```

Each run writes:

```
benches/results/<profile>-<timestamp>/
├── meta.json          # profile, git SHA, server /props snapshot, host info
├── T01/
│   ├── verdict.json   # PASS/PARTIAL/FAIL + metrics
│   ├── session.json   # zipcode's conversation transcript (→ tool_calls, turns)
│   ├── stdout.log     # what zipcode printed
│   ├── stderr.log     # spinner + info traces
│   └── workspace/     # files zipcode actually created (stripped of target/, .git/)
├── T02/ ...
├── T03/ ...
└── summary.md         # human-readable aggregate
```

Run a subset of tasks with `--tasks T01,T02`.

## Comparing two runs

```bash
./benches/compare.py \
    benches/results/remote-26B-A4B-Opus-2026-04-17-153000 \
    benches/results/local-E4B-2026-04-17-154200
```

Output is a side-by-side Markdown table: per-task verdict, wall time
delta, tool call counts.

## Task catalog (V1)

The tasks mirror the three tiers from the original Gemma 4 E4B probe
(see `docs/experiments/2026-04-16-gemma4-e4b-gamedev-probe.md`):

| ID | Tier | What it exercises |
|----|------|-------------------|
| T01 | 1 — single file, no debug | Python number-guessing game. One `write_file`, maybe one `bash`. Measures whether the model can follow a simple spec end-to-end. |
| T02 | 2 — single file, structural | Pong in a self-contained `pong.html`. Verifies canvas size, keydown bindings, JS syntax via `node --check`. Measures whether the model can produce a non-trivial app in one shot. |
| T03 | 3 — multi-file + debug loop | Rust `snake_demo` crate with cargo tests. Exercises zipcode's agentic debug loop: compile error → `edit_file` → re-run. The real separator between model tiers. |

All three are **non-interactive** and **automatically graded** — no
human in the loop. V1 is intentionally modest; more tasks can be added
as `tasks/T04_*.sh` following the contract below.

## Task contract (for adding new tasks)

Every `tasks/TXX_name.sh`:

1. Receives these env vars from `run.sh`:
   - `$ZIPCODE_BIN` — absolute path to the zipcode binary.
   - `$ZIPCODE_MODEL` — path to *any* local GGUF (still required by the
     CLI parser; remote-mode ignores it but you have to pass *something*).
   - `$WORKSPACE_DIR` — a fresh `mktemp -d`. Do all work there.

2. Runs `$ZIPCODE_BIN prompt "..."` with whatever flags make sense
   (`--permission-mode`, `--ui plain`, etc.).

3. Verifies the result itself (grep the output file, run `cargo test`,
   pipe stdin to the script, whatever is appropriate).

4. Writes `$WORKSPACE_DIR/verdict.json` in this shape:

   ```json
   {
     "task": "T04_my_task",
     "tier": 2,
     "verdict": "PASS | PARTIAL | FAIL",
     "failure_reason": "optional human-readable reason, or null",
     "metrics": { "task-specific key": "task-specific value" }
   }
   ```

`run.sh` will add `wall_clock_s` and `task_exit_code` to the verdict
automatically. Keep tasks self-contained — if you need fixtures, put
them under `benches/fixtures/TXX/` and copy them into `$WORKSPACE_DIR`
at the top of the script.

## What "PASS" means

Tasks grade structurally, not stylistically. A `PASS` means the
artifact exists and satisfies the concrete spec (file structure, test
pass rate, stdin handling). A `PARTIAL` means the model produced
something usable but one check failed (e.g. canvas exists but no
keydown listener). A `FAIL` means the artifact is missing, unusable,
or the model gave up mid-task.

Deliberately NOT graded:

- Code style / idiomaticity — too subjective.
- Runtime correctness beyond the checks — the harness isn't a
  test suite for the generated code, it's a probe for the model's
  *agentic reliability*.
- First-token latency / intermediate throughput — the point is
  end-to-end task success, not tok/s microbenchmarks.

## Known caveats

1. **Non-determinism.** Local models at temperature 0.2 are NOT
   deterministic. Running the same profile twice will show some
   variance — especially on T03 where initial code generation can go
   down different error paths (see the gamedev probe doc for evidence
   of this on E4B). Treat single-run results as one data point; run
   3× and look at the distribution for anything important.

2. **Network latency shows up.** Remote profiles include the
   Tailscale/LAN round-trip on every SSE chunk. That's by design — we
   want end-to-end numbers including network cost. To isolate pure
   model throughput, compare two profiles served from the same host.

3. **`cargo test --offline` needs a cached registry.** T03 works
   without network because the generated crate has no dependencies.
   If you add a task that pulls crates, pre-populate `~/.cargo/` or
   drop the `--offline` flag in that task.

4. **T03 runs a real cargo build** inside a tempdir. That's 10-30 s
   extra on first run while `target/` warms up. Subsequent runs reuse
   the sysroot.
