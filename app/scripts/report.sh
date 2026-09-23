#!/usr/bin/env bash
# Summarise state/runs.jsonl into the matrix tables in BENCHMARKS.md.
#
# Rewrites only the section between the "## Game matrix" marker and the next
# "## " heading, so hand-written analysis elsewhere in the file is preserved.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
. scripts/_common.sh

python3 - "$STATE_DIR/runs.jsonl" BENCHMARKS.md <<'PY'
import json, re, sys
from pathlib import Path

runs_path, bench_path = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
if runs_path.exists():
    for line in runs_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

# Only benchmark runs (those carrying a task id) belong in the matrix.
rows = [r for r in rows if r.get("task_id")]

def fmt_secs(v):
    if v in (None, ""):
        return "–"
    v = float(v)
    if v >= 3600:
        return f"{v/3600:.2f} h"
    if v >= 60:
        return f"{v/60:.1f} m"
    return f"{v:.1f} s"

lines = [
    "## Game matrix", "",
    "Produced by `scripts/report.sh` from `state/runs.jsonl`. "
    "Export the full data as CSV from the History tab.", "",
]
if not rows:
    lines += ["_No benchmark runs recorded yet._", ""]
else:
    lines += [
        "| Task | Model | Harness | Outcome | Wall | TTFT | 1st tool | Dec tok/s | Checks |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r.get("task_id",""), r.get("model_id",""))):
        m = r.get("metrics", {}) or {}
        checks = r.get("checks", {}) or {}
        passed = sum(1 for v in checks.values() if v)
        ext = r.get("time_limit_extensions", 0)
        outcome = r.get("outcome", "?")
        if ext:
            outcome += f" (+{ext})"
        lines.append(
            f"| {r.get('task_id','')} | {r.get('model_id','')} | "
            f"{r.get('harness_id','')} | {outcome} | "
            f"{fmt_secs(r.get('wall_seconds'))} | {fmt_secs(m.get('ttft_seconds'))} | "
            f"{fmt_secs(m.get('time_to_first_tool_call'))} | "
            f"{m.get('decode_tps') or '–'} | "
            f"{passed}/{len(checks) if checks else '–'} |"
        )
    lines.append("")

text = bench_path.read_text()
start = text.find("## Game matrix")
if start == -1:
    text = text.rstrip() + "\n\n" + "\n".join(lines) + "\n"
else:
    # Stop at the next heading of ANY level. Searching only for "## " would
    # swallow a "###" subsection that follows the generated table.
    m = re.search(r"\n#{2,6} ", text[start + 1:])
    nxt = start + 1 + m.start() if m else -1
    tail = text[nxt:] if nxt != -1 else "\n"
    text = text[:start] + "\n".join(lines) + tail
bench_path.write_text(text)
print(f"BENCHMARKS.md updated from {len(rows)} benchmark run(s)")
PY
