#!/usr/bin/env bash
# Play a finished game in a real window.
#
#   scripts/play.sh "bench_G1_glm53-flash_opencode_1789..."
#
# This runs the project directly rather than opening the editor, so it starts
# in the game instead of in the IDE.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
. scripts/_common.sh

NAME="${1:-}"
if [ -z "$NAME" ]; then
  echo "usage: $0 <project-name>" >&2
  echo "available:" >&2
  ls -1 "$RAM_AGENT_ROOT/projects" 2>/dev/null | sed 's/^/  /' >&2 || true
  exit 1
fi

DIR="$RAM_AGENT_ROOT/projects/$NAME"
[ -f "$DIR/project.godot" ] || { echo "no Godot project at: $DIR" >&2; exit 1; }

GODOT_BIN="${GODOT_BIN:-$HOME/.local/bin/godot}"
[ -x "$GODOT_BIN" ] || { echo "Godot not installed; run scripts/install-godot.sh" >&2; exit 1; }

echo "Playing $NAME  (close the window to quit)"
exec "$GODOT_BIN" --path "$DIR"
