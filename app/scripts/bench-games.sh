#!/usr/bin/env bash
# Run a model x harness x task matrix through the running app.
#
#   scripts/bench-games.sh                       # everything runnable
#   scripts/bench-games.sh --tasks T1 --models deepseek-v4-flash
#
# Uses the current time limit setting. Each cell starts from a fresh copy of
# the template project. Results land in state/runs.jsonl and are summarised
# into BENCHMARKS.md by scripts/report.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
. scripts/_common.sh

MODELS=""; HARNESSES=""; TASKS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --models)    MODELS="$2"; shift 2 ;;
    --harnesses) HARNESSES="$2"; shift 2 ;;
    --tasks)     TASKS="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

curl -fsS -m 5 "$UI_URL/api/health" >/dev/null || {
  echo "The app is not running. Start it with scripts/up.sh" >&2; exit 1; }

# Default to everything the registries define.
json_list() { printf '%s' "$1" | tr ' ,' '\n\n' | grep -v '^$' \
  | python3 -c 'import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))'; }

if [ -z "$MODELS" ]; then
  MODELS_JSON=$(curl -fsS "$UI_URL/api/models" | python3 -c \
    'import sys,json; print(json.dumps([m["id"] for m in json.load(sys.stdin)["models"]]))')
else MODELS_JSON=$(json_list "$MODELS"); fi

if [ -z "$HARNESSES" ]; then
  HARNESSES_JSON=$(curl -fsS "$UI_URL/api/harnesses" | python3 -c \
    'import sys,json; print(json.dumps([h["id"] for h in json.load(sys.stdin)["harnesses"]]))')
else HARNESSES_JSON=$(json_list "$HARNESSES"); fi

if [ -z "$TASKS" ]; then
  TASKS_JSON=$(curl -fsS "$UI_URL/api/bench/tasks" | python3 -c \
    'import sys,json; print(json.dumps([t["id"] for t in json.load(sys.stdin)["tasks"]]))')
else TASKS_JSON=$(json_list "$TASKS"); fi

BODY=$(printf '{"model_ids":%s,"harness_ids":%s,"task_ids":%s}' \
  "$MODELS_JSON" "$HARNESSES_JSON" "$TASKS_JSON")

echo "Matrix preview:"
curl -fsS -X POST -H 'Content-Type: application/json' -d "$BODY" \
  "$UI_URL/api/bench/matrix" | python3 -c '
import sys, json
d = json.load(sys.stdin)
print(f"  runnable: {d[\"runnable\"]}   skipped: {d[\"skipped\"]}")
for c in d["cells"]:
    if c["skipped"]:
        print(f"  SKIP  {c[\"task_id\"]:4} {c[\"model_id\"]:20} {c[\"harness_id\"]:9} {c[\"skipped\"]}")
'

echo
echo "Starting. This can run for many hours; follow along in the UI."
curl -fsS -X POST -H 'Content-Type: application/json' -d "$BODY" \
  "$UI_URL/api/bench/run" | python3 -m json.tool

echo
echo "Watching events (Ctrl-C to detach; the matrix keeps running):"
curl -sN "$UI_URL/api/stream" | while IFS= read -r line; do
  case "$line" in
    data:*) printf '%s\n' "${line#data: }" | python3 -c '
import sys, json
try: e = json.load(sys.stdin)
except Exception: raise SystemExit
t = e.get("type","")
if t == "bench_cell_start":
    c = e["cell"]
    print(f"[{e[\"index\"]}/{e[\"total\"]}] {c[\"task_id\"]} {c[\"model_id\"]} x {c[\"harness_id\"]}")
elif t == "bench_cell_end":
    passed = sum(1 for v in e.get("checks",{}).values() if v)
    total = len(e.get("checks",{}))
    print(f"    -> {e[\"outcome\"]}  ({passed}/{total} checks)")
elif t == "bench_cell_error":
    print(f"    -> error: {e[\"error\"]}")
elif t == "bench_done":
    print("Matrix complete.")
' ;;
  esac
done
