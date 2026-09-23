#!/usr/bin/env bash
# Ask an MCP server what it actually exposes, and price the catalogue.
#
# Every tool-cost number in mcp.yaml should come from here rather than from a
# project's release notes. The Blender entry was originally sized from a blog
# post about a different server and was wrong by an order of magnitude; this
# script is the correction mechanism.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
. scripts/_common.sh

VENV="$STATE_DIR/venv"
PY="${PYTHON:-$VENV/bin/python}"
[ -x "$PY" ] || PY=python3

case "${1:-}" in
  blender)
    exec "$PY" bench/probe_mcp.py "$VENV/bin/blender-mcp-server"
    ;;
  godot)
    exec env \
      GODOT_MCP_DEFAULT_TOOLSETS="${2:-scene_edit,scripts,runtime}" \
      GODOT_MCP_BRIDGE_URL="ws://127.0.0.1:9080" \
      "$PY" bench/probe_mcp.py godot-editor-mcp
    ;;
  *)
    echo "usage: $0 {blender|godot [toolsets]}" >&2
    echo "  godot toolsets default to: scene_edit,scripts,runtime" >&2
    exit 2
    ;;
esac
