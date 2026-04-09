#!/bin/bash
# scripts/wiki_sync_check.sh — Warn when wiki/ drifts from crates/
#
# Safe to call from a post-commit hook. NEVER modifies the wiki itself —
# it only detects staleness and prints a report.
#
# Usage:
#   scripts/wiki_sync_check.sh                 Run the check
#   scripts/wiki_sync_check.sh --mark-synced   Record current HEAD as sync baseline
#   scripts/wiki_sync_check.sh --install-hook  Install as .git/hooks/post-commit
#   scripts/wiki_sync_check.sh --help          Show this message

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SYNC_STATE_FILE="wiki/.sync-state"
WIKI_DIR="wiki"
WATCH_PATH="crates"

# ANSI colors only when stdout is a tty
if [ -t 1 ]; then
    BOLD=$'\033[1m'
    DIM=$'\033[2m'
    RED=$'\033[31m'
    YELLOW=$'\033[33m'
    GREEN=$'\033[32m'
    RESET=$'\033[0m'
else
    BOLD=""
    DIM=""
    RED=""
    YELLOW=""
    GREEN=""
    RESET=""
fi

warn() { printf "%s%s%s\n" "$YELLOW" "$*" "$RESET" >&2; }
err()  { printf "%s%s%s\n" "$RED" "$*" "$RESET" >&2; }
ok()   { printf "%s%s%s\n" "$GREEN" "$*" "$RESET"; }

# ---------- subcommands ----------

cmd_help() {
    cat <<EOF
${BOLD}scripts/wiki_sync_check.sh${RESET} — Warn when wiki/ drifts from crates/

Usage:
    scripts/wiki_sync_check.sh                 Run the check
    scripts/wiki_sync_check.sh --mark-synced   Record current HEAD as sync baseline
    scripts/wiki_sync_check.sh --install-hook  Install as .git/hooks/post-commit
    scripts/wiki_sync_check.sh --help          Show this message

The check:
  - compares HEAD against the commit recorded in $SYNC_STATE_FILE
  - lists files under $WATCH_PATH/ that changed since that commit
  - validates every 'crates/...:N' file:line reference in $WIKI_DIR/
  - prints a warning report if anything is stale

The script never modifies $WIKI_DIR/. To rewrite the wiki, use the prompt
template at scripts/wiki_prompt.md with Claude Code, then run:
    scripts/wiki_sync_check.sh --mark-synced
EOF
}

cmd_mark_synced() {
    local head
    head="$(git rev-parse HEAD)"
    printf "%s\n" "$head" > "$SYNC_STATE_FILE"
    ok "wiki sync baseline recorded: $head"
}

cmd_install_hook() {
    local hook_dir hook_path
    hook_dir="$(git rev-parse --git-path hooks)"
    hook_path="$hook_dir/post-commit"

    if [ -e "$hook_path" ] && ! grep -q "wiki_sync_check.sh" "$hook_path" 2>/dev/null; then
        err "post-commit hook already exists and does not reference wiki_sync_check.sh:"
        err "    $hook_path"
        err "Inspect it first, then remove it or merge the call manually."
        return 1
    fi

    cat > "$hook_path" <<'HOOK'
#!/bin/bash
# Auto-installed by scripts/wiki_sync_check.sh --install-hook
# Runs the zipcode wiki staleness check after every commit.
# Never blocks, never modifies files — it just prints a warning on drift.
repo_root="$(git rev-parse --show-toplevel)"
exec "$repo_root/scripts/wiki_sync_check.sh"
HOOK
    chmod +x "$hook_path"
    ok "installed post-commit hook at $hook_path"
}

# ---------- main check ----------

check() {
    if [ ! -d "$WIKI_DIR" ]; then
        warn "wiki/ missing — nothing to check"
        return 0
    fi

    local head baseline=""
    head="$(git rev-parse HEAD 2>/dev/null || echo "")"
    [ -z "$head" ] && return 0  # not in a git repo yet

    if [ -f "$SYNC_STATE_FILE" ]; then
        baseline="$(tr -d '[:space:]' < "$SYNC_STATE_FILE")"
    fi

    # 1. Diff changed files under $WATCH_PATH/ since baseline
    local changed=""
    if [ -n "$baseline" ] && git cat-file -e "$baseline" 2>/dev/null; then
        changed="$(git diff --name-only "$baseline" "$head" -- "$WATCH_PATH" 2>/dev/null || true)"
    fi

    # 2. Validate crates/...:N references in the wiki
    local broken_refs=""
    local refs
    refs="$(grep -rhoE 'crates/[A-Za-z0-9_./-]+\.rs:[0-9]+' "$WIKI_DIR" 2>/dev/null | sort -u || true)"

    if [ -n "$refs" ]; then
        while IFS= read -r ref; do
            [ -z "$ref" ] && continue
            local path line len
            path="${ref%:*}"
            line="${ref##*:}"
            if [ ! -f "$path" ]; then
                broken_refs+="  missing file: $ref"$'\n'
                continue
            fi
            len=$(awk 'END { print NR }' "$path" 2>/dev/null || echo 0)
            if [ "$line" -gt "$len" ]; then
                broken_refs+="  out of range: $ref (file has $len lines)"$'\n'
            fi
        done <<< "$refs"
    fi

    # 3. Report
    local have_changes=0 have_broken=0
    [ -n "$changed" ]     && have_changes=1
    [ -n "$broken_refs" ] && have_broken=1

    if [ "$have_changes" -eq 0 ] && [ "$have_broken" -eq 0 ]; then
        # Clean state — stay quiet so the hook doesn't spam every commit
        return 0
    fi

    printf "\n%s[wiki-sync]%s drift detected\n" "$BOLD$YELLOW" "$RESET"
    printf "  baseline: %s\n" "${baseline:-<none recorded>}"
    printf "  head:     %s\n\n" "$head"

    if [ "$have_changes" -eq 1 ]; then
        printf "  %schanged under %s/ since baseline:%s\n" "$BOLD" "$WATCH_PATH" "$RESET"
        printf "%s\n" "$changed" | sed 's/^/    /'
        printf "\n"
    fi

    if [ "$have_broken" -eq 1 ]; then
        printf "  %sbroken file:line references in %s/:%s\n" "$BOLD" "$WIKI_DIR" "$RESET"
        printf "%s" "$broken_refs" | sed 's/^/  /'
        printf "\n"
    fi

    printf "  %sto regenerate:%s paste %sscripts/wiki_prompt.md%s into Claude Code\n" "$DIM" "$RESET" "$BOLD" "$RESET"
    printf "  %sto accept current state:%s scripts/wiki_sync_check.sh --mark-synced\n\n" "$DIM" "$RESET"
}

# ---------- dispatch ----------

case "${1:-}" in
    ""|check)       check ;;
    --mark-synced)  cmd_mark_synced ;;
    --install-hook) cmd_install_hook ;;
    -h|--help)      cmd_help ;;
    *)
        err "unknown argument: $1"
        cmd_help
        exit 2
        ;;
esac
