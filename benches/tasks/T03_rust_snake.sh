#!/usr/bin/env bash
# T03 — Tier 3: Rust crate with cargo tests + debug loop.
#
# The hardest standard task. Asks zipcode to write a snake_demo crate
# with 4 unit tests, then RUN cargo test. This exercises zipcode's
# agentic debug loop: if the model's first draft has a compile or
# semantic error it has to diagnose the error, edit the file, and
# re-run.
#
# PASS criteria:
#   - Cargo.toml + src/lib.rs exist
#   - `cargo test --offline` returns 0 and at least 3 of 4 tests pass
#
# PARTIAL:
#   - Crate compiles but some tests fail
# FAIL:
#   - Crate doesn't compile, or model gave up with no cargo test run

set -u
set -o pipefail
cd "$WORKSPACE_DIR"

PROMPT='Create a minimal Rust library crate at ./ called snake_demo that implements the core logic for a terminal Snake game (board, snake segments, food spawning, movement, collision detection). Do NOT use any external dependencies — use only std. Structure: Cargo.toml + src/lib.rs. The Game struct should have: new(width, height) -> Self, tick(&mut self, dir: Direction) -> TickResult where TickResult is an enum { Alive, Ate, Dead }. Direction is an enum { Up, Down, Left, Right }. Include unit tests in src/lib.rs that verify: (1) snake moves forward and tail follows, (2) eating food grows the snake and spawns new food, (3) hitting a wall returns Dead, (4) hitting yourself returns Dead. After writing the files, run "cargo test --offline" to confirm all tests pass. If any test fails, read the error, fix the code, and re-run. You may iterate up to 5 times.'

"$ZIPCODE_BIN" prompt "$PROMPT" \
    --model "$ZIPCODE_MODEL" \
    --permission-mode full-access \
    --backend llama-server \
    --ui plain >/dev/null 2>&1

# ─── Verification ────────────────────────────────────────────────────
verdict="FAIL"
reason=""
compiled=false
tests_passed=0
tests_total=0

if [[ ! -f Cargo.toml || ! -f src/lib.rs ]]; then
    reason="Cargo.toml or src/lib.rs missing"
else
    # Run cargo test fresh to get authoritative results (model may have
    # left the target/ in any state, or skipped the final run).
    test_out="$(cargo test --offline --manifest-path Cargo.toml 2>&1 || true)"

    # Parse: "test result: ok. N passed; M failed"
    #   or  "test result: FAILED. N passed; M failed"
    line="$(echo "$test_out" | grep -E '^test result:' | head -1 || true)"
    if [[ -n "$line" ]]; then
        compiled=true
        tests_passed=$(echo "$line" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | head -1)
        tests_failed=$(echo "$line" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' | head -1)
        tests_passed="${tests_passed:-0}"
        tests_failed="${tests_failed:-0}"
        tests_total=$((tests_passed + tests_failed))

        if [[ $tests_passed -ge 3 && $tests_failed -le 1 ]]; then
            verdict="PASS"
        elif [[ $tests_passed -ge 1 ]]; then
            verdict="PARTIAL"
            reason="compiled but only $tests_passed/$tests_total tests pass"
        else
            verdict="FAIL"
            reason="compiled but 0 tests pass ($tests_total total)"
        fi
    else
        reason="cargo test did not produce a result line (compile error?)"
        # Try to capture the first compile error for diagnosis
        err="$(echo "$test_out" | grep -E '^error(\[E[0-9]+\])?:' | head -1 || true)"
        if [[ -n "$err" ]]; then
            reason="$reason — $err"
        fi
    fi
fi

python3 - <<PY
import json, os
workspace = os.environ["WORKSPACE_DIR"]
lib_size  = os.path.getsize(os.path.join(workspace, "src/lib.rs")) if os.path.exists(os.path.join(workspace, "src/lib.rs")) else 0
out = {
    "task": "T03_rust_snake",
    "tier": 3,
    "verdict": "$verdict",
    "failure_reason": "$reason" or None,
    "metrics": {
        "lib_rs_bytes": lib_size,
        "compiled": "$compiled" == "true",
        "tests_passed": int("$tests_passed" or 0),
        "tests_total":  int("$tests_total" or 0),
    },
}
with open(os.path.join(workspace, "verdict.json"), "w") as f:
    json.dump(out, f, indent=2)
PY
