#!/usr/bin/env bash
# Build the three game tasks one after another, on one model and harness.
#
#   scripts/run-games.sh                                  # defaults below
#   scripts/run-games.sh --model glm53-flash --hours 6
#   scripts/run-games.sh --harness pi --tasks G1
#
# Cells run strictly sequentially: llama-server runs --parallel 1, so
# overlapping them would only make each slower and the numbers meaningless.
# Each game starts from a fresh copy of the template project.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
. scripts/_common.sh

MODEL="deepseek-v4-flash"
HARNESS="opencode"
TASKS="G1,G2,G3"
HOURS="6"

while [ $# -gt 0 ]; do
  case "$1" in
    --model)   MODEL="$2"; shift 2 ;;
    --harness) HARNESS="$2"; shift 2 ;;
    --tasks)   TASKS="$2"; shift 2 ;;
    --hours)   HOURS="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

curl -fsS -m 5 "$UI_URL/api/health" >/dev/null || {
  echo "The app is not running. Start it with scripts/up.sh" >&2; exit 1; }

SECS=$(( HOURS * 3600 ))
echo "Time limit: ${HOURS}h per game (${SECS}s)"
curl -fsS -X POST -H 'Content-Type: application/json' \
  -d "{\"time_limit_enabled\":true,\"time_limit_seconds\":$SECS}" \
  "$UI_URL/api/settings" >/dev/null

# The MCP bridge is a singleton on 9080; a stray server would stop the
# harness's own from binding it, and the editor would talk to the wrong one.
if ss -lntp 2>/dev/null | grep -q ':9080'; then
  echo "WARNING: something already holds port 9080 (the Godot MCP bridge)." >&2
  echo "         Stop it, or the harness's MCP server cannot start." >&2
fi

TASKS_JSON=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -v '^$' \
  | python3 -c 'import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))')
BODY=$(printf '{"model_ids":["%s"],"harness_ids":["%s"],"task_ids":%s}' \
  "$MODEL" "$HARNESS" "$TASKS_JSON")

echo "Matrix:"
curl -fsS -X POST -H 'Content-Type: application/json' -d "$BODY" \
  "$UI_URL/api/bench/matrix" | python3 -c '
import sys, json
d = json.load(sys.stdin)
print(f"  runnable: {d[\"runnable\"]}  skipped: {d[\"skipped\"]}")
for c in d["cells"]:
    mark = "SKIP" if c["skipped"] else "run "
    print(f"  {mark} {c[\"task_id\"]} {c[\"model_id\"]} x {c[\"harness_id\"]} "
          f"{c[\"skipped\"] or \"\"}")'

echo
read -r -p "Start? This can run for $(( HOURS * 3 )) hours. [y/N] " ans
case "$ans" in [yY]*) ;; *) echo "aborted"; exit 0 ;; esac

curl -fsS -X POST -H 'Content-Type: application/json' -d "$BODY" \
  "$UI_URL/api/bench/run" | python3 -m json.tool

cat <<NOTE

Started. Watch progress in the UI (Benchmark tab), or follow events with:
  curl -sN $UI_URL/api/stream

Finished games appear in the Games tab with a Play button. For that button to
open a window, run this on the host in another terminal:
  scripts/play-watcher.sh &
NOTE
