#!/usr/bin/env bash
# Open a project in the host Godot editor with the MCP addon enabled.
#
#   scripts/godot-editor.sh "Pong Clone"
#
# The editor runs on the host (Wayland/Hyprland, real GPU); the MCP server runs
# in the container and binds the bridge the addon dials out to.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
. scripts/_common.sh

PROJECT_NAME="${1:-}"
if [ -z "$PROJECT_NAME" ]; then
  echo "usage: $0 <project-name>" >&2
  echo "available:" >&2
  ls -1 "$RAM_AGENT_ROOT/projects" 2>/dev/null | sed 's/^/  /' >&2 || true
  exit 1
fi

PROJECT_DIR="$RAM_AGENT_ROOT/projects/$PROJECT_NAME"
[ -f "$PROJECT_DIR/project.godot" ] || {
  echo "no Godot project at: $PROJECT_DIR" >&2; exit 1; }

GODOT_BIN="${GODOT_BIN:-$HOME/.local/bin/godot}"
[ -x "$GODOT_BIN" ] || { echo "Godot not installed; run scripts/install-godot.sh" >&2; exit 1; }

# Seed the addon if this project predates it.
if [ ! -d "$PROJECT_DIR/addons/godot_mcp" ] && [ -d "templates/addons/godot_mcp" ]; then
  echo "Installing the MCP addon into $PROJECT_NAME..."
  mkdir -p "$PROJECT_DIR/addons"
  cp -r "templates/addons/godot_mcp" "$PROJECT_DIR/addons/"
fi

export GODOT_MCP_BRIDGE_URL="${GODOT_MCP_BRIDGE_URL:-ws://127.0.0.1:9080}"
echo "Opening $PROJECT_NAME (bridge: $GODOT_MCP_BRIDGE_URL)"
echo "If the MCP panel is not visible: Project > Project Settings > Plugins > Godot MCP > Enable."
exec "$GODOT_BIN" --editor --path "$PROJECT_DIR"
