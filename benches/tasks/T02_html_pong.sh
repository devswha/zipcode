#!/usr/bin/env bash
# T02 — Tier 2: single self-contained HTML + Canvas + JS.
#
# Asks zipcode to write a Pong game in one pong.html file. Verifies
# structural requirements via grep, and that the embedded JavaScript
# passes `node --check` syntax validation.
#
# PASS criteria:
#   - pong.html exists
#   - Contains <canvas width="800" height="500">
#   - Has a keydown event listener and at least one of W/S/ArrowUp/ArrowDown
#   - Has a game loop driver (requestAnimationFrame or setInterval)
#   - Extracted <script> body passes `node --check`
# PARTIAL: file exists but one structural check fails.

set -u
set -o pipefail
cd "$WORKSPACE_DIR"

PROMPT='Create a classic Pong game as a single self-contained HTML file at pong.html. Requirements: HTML5 canvas 800x500, two paddles (left and right), a bouncing ball, left paddle controlled by W/S keys, right paddle controlled by Arrow Up/Down, score display at the top (left vs right). The ball should reset to center when it passes a paddle and award a point to the opponent. Put all CSS and JavaScript inline in the single pong.html file.'

"$ZIPCODE_BIN" prompt "$PROMPT" \
    --model "$ZIPCODE_MODEL" \
    --permission-mode workspace-write \
    --backend llama-server \
    --ui plain >/dev/null 2>&1

# ─── Verification ────────────────────────────────────────────────────
verdict="FAIL"
reason=""
failed_checks=()

if [[ ! -f pong.html ]]; then
    reason="pong.html was not created"
else
    grep -qE '<canvas[^>]+width="800"[^>]+height="500"' pong.html \
        || failed_checks+=("canvas 800x500")
    grep -qE 'addEventListener\([^)]*keydown' pong.html \
        || failed_checks+=("keydown listener")
    grep -qE 'KeyW|"w"|\bw\b|ArrowUp|ArrowDown' pong.html \
        || failed_checks+=("paddle key bindings")
    grep -qE 'requestAnimationFrame|setInterval' pong.html \
        || failed_checks+=("game loop driver")

    # Extract the first <script>...</script> block and syntax-check it
    python3 - <<'PY' > /tmp/pong-extracted.js || true
import re, sys
html = open("pong.html").read()
m = re.search(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
if m:
    sys.stdout.write(m.group(1))
PY
    if [[ -s /tmp/pong-extracted.js ]]; then
        if ! node --check /tmp/pong-extracted.js 2>/dev/null; then
            failed_checks+=("JS syntax")
        fi
    else
        failed_checks+=("script block missing")
    fi

    if [[ ${#failed_checks[@]} -eq 0 ]]; then
        verdict="PASS"
    else
        verdict="PARTIAL"
        reason="failed checks: ${failed_checks[*]}"
    fi
fi

python3 - <<PY
import json, os
workspace = os.environ["WORKSPACE_DIR"]
verdict = "$verdict"
reason  = "$reason"
path = os.path.join(workspace, "pong.html")
size = os.path.getsize(path) if os.path.exists(path) else 0
out = {
    "task": "T02_html_pong",
    "tier": 2,
    "verdict": verdict,
    "failure_reason": reason or None,
    "metrics": { "file_bytes": size },
}
with open(os.path.join(workspace, "verdict.json"), "w") as f:
    json.dump(out, f, indent=2)
PY
