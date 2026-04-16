#!/usr/bin/env python3
"""
Aggregate per-task verdicts + session JSON into a single summary.md
for a bench run directory.

Usage:
    python3 analyze.py <run_dir>

The session JSON carries the richer metrics we can't measure from
outside zipcode: tool call count, reasoning vs content bytes (when the
bench is run with verbose mode enabled — off by default, so these will
usually be 0 for reasoning bytes).
"""

import json
import os
import sys
from pathlib import Path


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def analyze_session(session_path):
    """Return (tool_calls, model_turns, total_content_bytes, had_auto_retry)."""
    session = load_json(session_path)
    if not session:
        return (0, 0, 0, False)

    tool_calls = 0
    model_turns = 0
    total_content = 0
    had_auto_retry = False

    for msg in session.get("messages", []):
        if msg.get("role") != "model":
            continue
        model_turns += 1
        tc = msg.get("tool_calls") or []
        tool_calls += len(tc)
        content = msg.get("content", "") or ""
        total_content += len(content.encode("utf-8"))
        if "re-read the error" in content:
            had_auto_retry = True

    return (tool_calls, model_turns, total_content, had_auto_retry)


def render_summary(run_dir: Path) -> str:
    meta = load_json(run_dir / "meta.json") or {}
    task_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("T"))

    # Per-task table
    rows = []
    totals = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "time": 0.0, "tools": 0}

    for t in task_dirs:
        v = load_json(t / "verdict.json") or {"verdict": "FAIL", "failure_reason": "no verdict.json"}
        tool_calls, model_turns, content_bytes, retried = analyze_session(t / "session.json")

        verdict = v.get("verdict", "?")
        if verdict in totals:
            totals[verdict] += 1
        wall = v.get("wall_clock_s", 0.0) or 0.0
        totals["time"] += wall
        totals["tools"] += tool_calls

        metrics = v.get("metrics", {}) or {}
        metric_bits = []
        if "tests_passed" in metrics and "tests_total" in metrics:
            metric_bits.append(f"tests {metrics['tests_passed']}/{metrics['tests_total']}")
        if "file_bytes" in metrics and metrics["file_bytes"]:
            metric_bits.append(f"{metrics['file_bytes']} B out")
        if "lib_rs_bytes" in metrics:
            metric_bits.append(f"lib.rs {metrics['lib_rs_bytes']} B")

        rows.append({
            "task": t.name,
            "verdict": verdict,
            "wall": wall,
            "tools": tool_calls,
            "turns": model_turns,
            "content_b": content_bytes,
            "retried": retried,
            "metric_bits": " · ".join(metric_bits),
            "reason": v.get("failure_reason") or "",
        })

    # Emoji-free verdict marker (deliberate — some terminals fonts
    # butcher emoji alignment in fixed-width tables).
    def badge(v):
        return {"PASS": "✓ PASS", "PARTIAL": "~ PARTIAL", "FAIL": "✗ FAIL"}.get(v, v)

    lines = []
    lines.append(f"# zipcode bench — {meta.get('profile', 'unknown')}")
    lines.append("")
    lines.append(f"**Run at:** {meta.get('timestamp', '?')}")
    lines.append(f"**zipcode SHA:** `{meta.get('git_sha', 'unknown')[:12]}`" +
                 (" (dirty)" if meta.get("git_dirty") else ""))
    lines.append(f"**Server URL:** `{meta.get('server_url', '?')}`")

    props = meta.get("server_props") or {}
    if props.get("model_path"):
        lines.append(f"**Server model:** `{props['model_path']}`")
    if props.get("build_info"):
        lines.append(f"**llama.cpp build:** `{props['build_info']}`")
    if meta.get("model_hint"):
        lines.append(f"**Model hint:** {meta['model_hint']}")

    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Task | Verdict | Wall | Tools | Model turns | Content bytes | Auto-retry | Metrics |")
    lines.append("|---|---|---:|---:|---:|---:|:---:|---|")
    for r in rows:
        lines.append("| {task} | {verdict} | {wall:.1f}s | {tools} | {turns} | {content_b} | {retry} | {metrics} |".format(
            task=r["task"],
            verdict=badge(r["verdict"]),
            wall=r["wall"],
            tools=r["tools"],
            turns=r["turns"],
            content_b=r["content_b"],
            retry="yes" if r["retried"] else "—",
            metrics=r["metric_bits"] or "—",
        ))

    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    n_tasks = len(rows) or 1
    lines.append(f"- **Pass rate:** {totals['PASS']}/{n_tasks}")
    lines.append(f"- **Partial:** {totals['PARTIAL']}/{n_tasks}")
    lines.append(f"- **Fail:** {totals['FAIL']}/{n_tasks}")
    lines.append(f"- **Total wall time:** {totals['time']:.1f} s")
    lines.append(f"- **Total tool calls:** {totals['tools']}")

    # Failure reasons (quick-glance debugging aid)
    failures = [r for r in rows if r["verdict"] != "PASS" and r["reason"]]
    if failures:
        lines.append("")
        lines.append("## Failure reasons")
        lines.append("")
        for f in failures:
            lines.append(f"- **{f['task']}** ({f['verdict']}): {f['reason']}")

    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) != 2:
        print("Usage: analyze.py <run_dir>", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    summary = render_summary(run_dir)
    (run_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
