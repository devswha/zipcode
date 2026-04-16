#!/usr/bin/env python3
"""
Side-by-side comparison of two bench runs. Useful for asking questions
like "does the 31B Opus-distilled on an RTX 3090 do better than 26B
A4B Opus-distilled on a 4080 Super?"

Usage:
    python3 compare.py <run_dir_A> <run_dir_B>

Both arguments are paths under benches/results/ (use their full or
trailing-slash names, either works).
"""

import json
import sys
from pathlib import Path


def load_verdict(run_dir: Path, task: str):
    path = run_dir / task / "verdict.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_session(run_dir: Path, task: str):
    path = run_dir / task / "session.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def session_stats(session):
    if not session:
        return (0, 0)
    tc, turns = 0, 0
    for m in session.get("messages", []):
        if m.get("role") != "model":
            continue
        turns += 1
        tc += len(m.get("tool_calls") or [])
    return (tc, turns)


def main():
    if len(sys.argv) != 3:
        print("Usage: compare.py <run_dir_A> <run_dir_B>", file=sys.stderr)
        sys.exit(2)

    a = Path(sys.argv[1]).resolve()
    b = Path(sys.argv[2]).resolve()

    meta_a = json.loads((a / "meta.json").read_text()) if (a / "meta.json").exists() else {}
    meta_b = json.loads((b / "meta.json").read_text()) if (b / "meta.json").exists() else {}

    tasks = sorted({p.name for run in (a, b) for p in run.iterdir()
                    if p.is_dir() and p.name.startswith("T")})

    # Header
    print(f"# Comparison: {meta_a.get('profile','A')} vs {meta_b.get('profile','B')}")
    print()
    print(f"- **A** `{a.name}` — server `{meta_a.get('server_url','?')}`")
    if (meta_a.get("server_props") or {}).get("model_path"):
        print(f"        model: `{meta_a['server_props']['model_path']}`")
    print(f"- **B** `{b.name}` — server `{meta_b.get('server_url','?')}`")
    if (meta_b.get("server_props") or {}).get("model_path"):
        print(f"        model: `{meta_b['server_props']['model_path']}`")
    print()

    # Table
    print("| Task | A verdict | B verdict | A wall | B wall | Δ wall | A tools | B tools |")
    print("|---|---|---|---:|---:|---:|---:|---:|")

    tot = {"a_pass": 0, "b_pass": 0, "a_wall": 0.0, "b_wall": 0.0}

    for t in tasks:
        va = load_verdict(a, t) or {}
        vb = load_verdict(b, t) or {}
        tc_a, _ = session_stats(load_session(a, t))
        tc_b, _ = session_stats(load_session(b, t))

        va_v = va.get("verdict", "—")
        vb_v = vb.get("verdict", "—")
        wa = va.get("wall_clock_s") or 0.0
        wb = vb.get("wall_clock_s") or 0.0

        if va_v == "PASS":
            tot["a_pass"] += 1
        if vb_v == "PASS":
            tot["b_pass"] += 1
        tot["a_wall"] += wa
        tot["b_wall"] += wb

        dw = (wb - wa)
        dw_s = f"{dw:+.1f}s"

        print(f"| {t} | {va_v} | {vb_v} | {wa:.1f}s | {wb:.1f}s | {dw_s} | {tc_a} | {tc_b} |")

    print()
    print("## Aggregate")
    print()
    n = len(tasks) or 1
    print(f"- Pass rate: **A** {tot['a_pass']}/{n}  vs  **B** {tot['b_pass']}/{n}")
    print(f"- Total wall: **A** {tot['a_wall']:.1f} s  vs  **B** {tot['b_wall']:.1f} s  (Δ {tot['b_wall']-tot['a_wall']:+.1f} s)")


if __name__ == "__main__":
    main()
