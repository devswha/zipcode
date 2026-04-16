#!/usr/bin/env bash
#
# zipcode harness benchmark orchestrator.
#
# Runs a suite of tasks against the currently-configured zipcode
# (honoring ZIPCODE_LLAMA_SERVER_URL / ZIPCODE_LLAMA_SERVER_ALIAS) and
# produces a per-run results directory with per-task verdicts, session
# transcripts, and a summary.md.
#
# Usage:
#   ./benches/run.sh --profile <name> [--tasks T01,T02] [--model-hint <text>]
#
# The profile name is a free-form label you pick to identify this run
# (e.g. "local-E4B", "remote-26B-A4B-Opus", "rtx3090-31B"). It ends up
# in the results directory name and in the summary.
#
# Task contract (see tasks/README.md for details):
#   - Script receives $ZIPCODE_BIN, $WORKSPACE_DIR, $ZIPCODE_MODEL as env.
#   - Script does its work inside $WORKSPACE_DIR and writes
#     $WORKSPACE_DIR/verdict.json before exiting.
#   - verdict.json shape:
#     { "verdict": "PASS|PARTIAL|FAIL",
#       "tool_calls": N, "wall_clock_s": N,
#       "failure_reason": "..." (optional),
#       "metrics": { ... task-specific ... } }

set -u
set -o pipefail

# ─── Defaults / arg parsing ──────────────────────────────────────────
PROFILE=""
TASKS="all"
MODEL_HINT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)    PROFILE="$2"; shift 2 ;;
        --tasks)      TASKS="$2"; shift 2 ;;
        --model-hint) MODEL_HINT="$2"; shift 2 ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# //; s/^#//'
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$PROFILE" ]]; then
    echo "Error: --profile is required (e.g. 'local-E4B' or 'remote-26B-A4B-Opus')" >&2
    exit 2
fi

# ─── Resolve paths & environment ─────────────────────────────────────
BENCH_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$BENCH_ROOT/.." && pwd)"
ZIPCODE_BIN="${ZIPCODE_BIN:-$REPO_ROOT/target/release/zipcode}"

if [[ ! -x "$ZIPCODE_BIN" ]]; then
    echo "Error: zipcode binary not found at $ZIPCODE_BIN" >&2
    echo "  Build it first: cargo build --release -p zipcode" >&2
    exit 1
fi

# Model path is still required by zipcode's CLI even in remote mode,
# but we point at any existing GGUF; the remote server's model is what
# actually runs.
ZIPCODE_MODEL="${ZIPCODE_MODEL:-$HOME/.zipcode/models/gemma-4-E4B-it-Q4_K_M.gguf}"
if [[ ! -f "$ZIPCODE_MODEL" ]]; then
    # Fall back to any .gguf the user has
    ZIPCODE_MODEL="$(find "$HOME/.zipcode/models" -maxdepth 1 -name '*.gguf' -size +100M 2>/dev/null | head -1)"
fi

export ZIPCODE_BIN
export ZIPCODE_MODEL

TS="$(date +%Y-%m-%d-%H%M%S)"
RUN_DIR="$BENCH_ROOT/results/${PROFILE}-${TS}"
mkdir -p "$RUN_DIR"

echo "=== zipcode harness bench ==="
echo "profile       : $PROFILE"
echo "run_dir       : $RUN_DIR"
echo "zipcode_bin   : $ZIPCODE_BIN"
echo "zipcode_model : $ZIPCODE_MODEL"
echo ""

# ─── Capture run metadata ────────────────────────────────────────────
GIT_SHA="$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(cd "$REPO_ROOT" && [[ -n "$(git status --porcelain 2>/dev/null)" ]] && echo true || echo false)"

# Query the remote llama-server /props if it's configured
SERVER_URL="${ZIPCODE_LLAMA_SERVER_URL:-local-spawn}"
SERVER_ALIAS="${ZIPCODE_LLAMA_SERVER_ALIAS:-zipcode}"
PROPS_JSON="{}"
if [[ "$SERVER_URL" != "local-spawn" ]]; then
    PROPS_JSON="$(curl -s --max-time 3 "$SERVER_URL/props" || echo '{}')"
fi

export PROFILE TS GIT_SHA GIT_DIRTY ZIPCODE_MODEL SERVER_URL SERVER_ALIAS MODEL_HINT
PROPS_JSON_RAW="$PROPS_JSON" python3 - <<'PY' > "$RUN_DIR/meta.json"
import json, os, platform
props = {}
try:
    props = json.loads(os.environ.get('PROPS_JSON_RAW', '{}'))
except Exception:
    props = {}
meta = {
    "profile": os.environ.get("PROFILE", ""),
    "timestamp": os.environ.get("TS", ""),
    "git_sha": os.environ.get("GIT_SHA", ""),
    "git_dirty": os.environ.get("GIT_DIRTY", "false") == "true",
    "zipcode_model_arg": os.environ.get("ZIPCODE_MODEL", ""),
    "server_url": os.environ.get("SERVER_URL", ""),
    "server_alias": os.environ.get("SERVER_ALIAS", ""),
    "model_hint": os.environ.get("MODEL_HINT", ""),
    "server_props": {
        "model_path":         props.get("model_path"),
        "model_alias":        props.get("model_alias"),
        "build_info":         props.get("build_info"),
        "total_slots":        props.get("total_slots"),
        "chat_template_caps": props.get("chat_template_caps"),
        "bos_token":          props.get("bos_token"),
        "eos_token":          props.get("eos_token"),
    },
    "host": {
        "uname":  platform.platform(),
        "python": platform.python_version(),
    },
}
print(json.dumps(meta, indent=2))
PY

# ─── Task selection ──────────────────────────────────────────────────
if [[ "$TASKS" == "all" ]]; then
    TASK_FILES=( "$BENCH_ROOT"/tasks/T*.sh )
else
    TASK_FILES=()
    IFS=',' read -ra SELECTED <<< "$TASKS"
    for t in "${SELECTED[@]}"; do
        TASK_FILES+=( "$BENCH_ROOT/tasks/${t}"*.sh )
    done
fi

# ─── Run each task ───────────────────────────────────────────────────
for task_script in "${TASK_FILES[@]}"; do
    [[ -f "$task_script" ]] || continue
    task_name="$(basename "$task_script" .sh)"
    task_id="${task_name%%_*}"

    echo "── $task_name ──"
    WORKSPACE_DIR="$(mktemp -d -t "zipcode-bench-${task_id}-XXXXXX")"
    export WORKSPACE_DIR

    # Snapshot session files so we can identify the one this run creates
    sessions_before="$(mktemp)"
    ls -1 "$HOME/.zipcode/sessions"/*.json 2>/dev/null | sort > "$sessions_before"

    task_out="$RUN_DIR/$task_id"
    mkdir -p "$task_out"

    # Run the task — capture stdout/stderr separately
    start=$(date +%s.%N)
    bash "$task_script" \
        > "$task_out/stdout.log" \
        2> "$task_out/stderr.log"
    rc=$?
    end=$(date +%s.%N)
    elapsed=$(python3 -c "print(f'{$end - $start:.2f}')")

    # Find the new session file (if zipcode created one)
    sessions_after="$(mktemp)"
    ls -1 "$HOME/.zipcode/sessions"/*.json 2>/dev/null | sort > "$sessions_after"
    new_session="$(comm -13 "$sessions_before" "$sessions_after" | head -1 || true)"
    if [[ -n "$new_session" && -f "$new_session" ]]; then
        cp "$new_session" "$task_out/session.json"
    fi
    rm -f "$sessions_before" "$sessions_after"

    # Copy workspace artifacts (exclude build detritus)
    mkdir -p "$task_out/workspace"
    if [[ -d "$WORKSPACE_DIR" ]]; then
        # Use rsync if available, else find+cp (skip target/ .git/ for sanity)
        if command -v rsync >/dev/null 2>&1; then
            rsync -a --exclude target --exclude .git "$WORKSPACE_DIR/" "$task_out/workspace/"
        else
            (cd "$WORKSPACE_DIR" && find . -type f \
                -not -path './target/*' -not -path './.git/*' \
                -exec cp --parents {} "$task_out/workspace/" \;)
        fi
        # Verdict must live at workspace root — copy it to task_out top-level
        [[ -f "$WORKSPACE_DIR/verdict.json" ]] && cp "$WORKSPACE_DIR/verdict.json" "$task_out/verdict.json"
    fi

    # Augment verdict with orchestrator-measured wall clock and exit code
    python3 - <<PY
import json, os, sys
vpath = "$task_out/verdict.json"
try:
    with open(vpath) as f:
        v = json.load(f)
except Exception:
    v = {"verdict": "FAIL", "failure_reason": f"task wrote no verdict.json (rc=$rc)"}
v.setdefault("metrics", {})
v["wall_clock_s"] = float("$elapsed")
v["task_exit_code"] = int($rc)
with open(vpath, "w") as f:
    json.dump(v, f, indent=2)
PY

    # Short status line
    verdict="$(python3 -c "import json; print(json.load(open('$task_out/verdict.json')).get('verdict','?'))")"
    echo "  verdict=$verdict  wall=${elapsed}s  rc=$rc"

    # Clean up the tempdir
    rm -rf "$WORKSPACE_DIR"
    unset WORKSPACE_DIR
done

echo ""

# ─── Aggregate analysis ──────────────────────────────────────────────
python3 "$BENCH_ROOT/analyze.py" "$RUN_DIR"

echo ""
echo "=== Done ==="
echo "Results: $RUN_DIR"
echo "Summary: $RUN_DIR/summary.md"
