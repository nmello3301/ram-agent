#!/usr/bin/env bash
# Bridges the web UI and the benchmark runner to your desktop session.
#
#   scripts/play-watcher.sh &
#
# The backend runs in a container and cannot open a window, so it drops request
# files in state/ and this launches them on the host. Two kinds:
#
#   *.request  -> play a finished game in its own window
#   *.editor   -> open a project in the Godot editor, which is what connects the
#                 MCP addon to the bridge. Scene-editing tools drive the LIVE
#                 editor, so a benchmark run cannot build anything without this.
#
# Only one editor at a time: the MCP bridge is a single WebSocket and a second
# editor would fight the first for it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
. scripts/_common.sh

REQ_DIR="$STATE_DIR/play-requests"
mkdir -p "$REQ_DIR"
GODOT_BIN="${GODOT_BIN:-$HOME/.local/bin/godot}"
EDITOR_PID=""
EDITOR_PROJECT=""

open_editor() {
  local name="$1" dir="$RAM_AGENT_ROOT/projects/$1"
  [ -f "$dir/project.godot" ] || { echo "  unknown project: $name" >&2; return; }
  if [ "$name" = "$EDITOR_PROJECT" ] && kill -0 "$EDITOR_PID" 2>/dev/null; then
    echo "$(date +%H:%M:%S) editor already on $name"; return
  fi
  if [ -n "$EDITOR_PID" ] && kill -0 "$EDITOR_PID" 2>/dev/null; then
    echo "$(date +%H:%M:%S) closing editor on $EDITOR_PROJECT"
    kill "$EDITOR_PID" 2>/dev/null || true
    sleep 3
  fi
  # Seed the addon if this project predates it.
  if [ ! -d "$dir/addons/godot_mcp" ] && [ -d "templates/addons/godot_mcp" ]; then
    mkdir -p "$dir/addons"; cp -r "templates/addons/godot_mcp" "$dir/addons/"
  fi
  echo "$(date +%H:%M:%S) opening editor on $name"
  GODOT_MCP_BRIDGE_URL="ws://127.0.0.1:9080" \
    setsid "$GODOT_BIN" --editor --path "$dir" >/dev/null 2>&1 &
  EDITOR_PID=$!
  EDITOR_PROJECT="$name"
  # Publish what the editor is on. The backend is in a container and cannot
  # see host processes, and it cannot use port 9080 as the signal either --
  # that is bound by the MCP server, which the harness only starts afterwards.
  printf '%s\n' "$name" > "$STATE_DIR/editor-project"
}

rm -f "$STATE_DIR/editor-project"
echo "Watching $REQ_DIR (Ctrl-C to stop)"
while true; do
  for req in "$REQ_DIR"/*.editor; do
    [ -e "$req" ] || continue
    name="$(head -1 "$req" | tr -d '\r\n')"; rm -f "$req"
    open_editor "$name"
  done
  for req in "$REQ_DIR"/*.request; do
    [ -e "$req" ] || continue
    name="$(head -1 "$req" | tr -d '\r\n')"; rm -f "$req"
    dir="$RAM_AGENT_ROOT/projects/$name"
    if [ -f "$dir/project.godot" ]; then
      echo "$(date +%H:%M:%S) playing: $name"
      setsid "$GODOT_BIN" --path "$dir" >/dev/null 2>&1 &
    else
      echo "$(date +%H:%M:%S) unknown project: $name" >&2
    fi
  done
  sleep 1
done
