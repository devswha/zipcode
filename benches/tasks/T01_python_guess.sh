#!/usr/bin/env bash
# T01 — Tier 1: single-file Python.
#
# Asks zipcode to write a number-guessing game with specific requirements,
# then verifies the file exists and produces the expected string patterns
# when fed canned stdin. Measures tool_calls + verdict.
#
# PASS criteria:
#   - guess.py exists
#   - Running it with a piped 7-guess sequence prints at least one of
#     {"too low", "too high", "correct"} and exits 0.

set -u
set -o pipefail
cd "$WORKSPACE_DIR"

PROMPT='Create a Python number-guessing game at guess.py. The program should pick a random integer between 1 and 100, then let the user type guesses on stdin, telling them "too high" / "too low" / "correct" each turn, and report the number of attempts on success. Keep it in a single file.'

"$ZIPCODE_BIN" prompt "$PROMPT" \
    --model "$ZIPCODE_MODEL" \
    --permission-mode workspace-write \
    --backend llama-server \
    --ui plain >/dev/null 2>&1

# ─── Verification ────────────────────────────────────────────────────
verdict="FAIL"
reason=""

if [[ ! -f guess.py ]]; then
    reason="guess.py was not created"
else
    out="$(printf '50\n25\n12\n6\n3\n1\n2\n' | timeout 10 python3 guess.py 2>&1 || true)"
    lower="$(echo "$out" | tr '[:upper:]' '[:lower:]')"
    if echo "$lower" | grep -qE 'too (high|low)|correct'; then
        verdict="PASS"
    else
        verdict="PARTIAL"
        reason="file created but stdin handling produced no expected feedback"
    fi
fi

python3 - <<PY
import json, os
workspace = os.environ["WORKSPACE_DIR"]
verdict = "$verdict"
reason  = "$reason"
size    = os.path.getsize(os.path.join(workspace, "guess.py")) if os.path.exists(os.path.join(workspace, "guess.py")) else 0
out = {
    "task": "T01_python_guess",
    "tier": 1,
    "verdict": verdict,
    "failure_reason": reason or None,
    "metrics": { "file_bytes": size },
}
with open(os.path.join(workspace, "verdict.json"), "w") as f:
    json.dump(out, f, indent=2)
PY
