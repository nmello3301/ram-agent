#!/usr/bin/env bash
# Blender + the Blender MCP bridge, without root.
#
# Mirrors scripts/install-godot.sh deliberately: a portable tarball under
# ~/.local rather than a distro package, so nothing here needs sudo and the
# version is pinned to what the benchmark was run against.
#
# The architecture is the same shape as Godot MCP, which is why the app can
# treat them alike:
#
#   MCP server (stdio)  ──TCP──▶  add-on inside Blender  ──▶  bpy
#                    127.0.0.1:9876
#
# The add-on listens; the server dials out to it. Blender must be running with
# the add-on enabled before the agent can do anything.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
. scripts/_common.sh

BLENDER_VERSION="${BLENDER_VERSION:-5.2.2}"
BLENDER_SERIES="${BLENDER_VERSION%.*}"          # 5.2
MCP_REF="${BLENDER_MCP_REF:-main}"

OPT="$HOME/.local/share"
BIN="$HOME/.local/bin/blender"
TARBALL="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
URL="https://download.blender.org/release/Blender${BLENDER_SERIES}/${TARBALL}"
DEST="$OPT/blender-${BLENDER_VERSION}"
CACHE="${BLENDER_CACHE:-$STATE_DIR/downloads}"
VENV="$STATE_DIR/venv"
SRC="$STATE_DIR/blender-mcp-server"

mkdir -p "$OPT" "$(dirname "$BIN")" "$CACHE" "$STATE_DIR"

# --- 1. Blender itself ----------------------------------------------------
if [ -x "$DEST/blender" ]; then
  echo "Blender ${BLENDER_VERSION} already installed at $DEST"
else
  if [ ! -f "$CACHE/$TARBALL" ]; then
    echo "Downloading $TARBALL (~400 MB)"
    curl -fL --progress-bar -o "$CACHE/$TARBALL.part" "$URL"
    mv "$CACHE/$TARBALL.part" "$CACHE/$TARBALL"
  else
    echo "Using cached $CACHE/$TARBALL"
  fi

  # Verify against upstream's published checksum rather than trusting the
  # transfer. Non-fatal if the file is unavailable, but say so out loud.
  if curl -fsSL -o "$CACHE/blender.sha256" \
       "https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}.sha256" 2>/dev/null; then
    want="$(grep -E "${TARBALL}\$" "$CACHE/blender.sha256" | awk '{print $1}' | head -1)"
    if [ -n "$want" ]; then
      got="$(sha256sum "$CACHE/$TARBALL" | awk '{print $1}')"
      if [ "$want" != "$got" ]; then
        echo "FATAL: checksum mismatch for $TARBALL" >&2
        echo "  expected $want" >&2
        echo "  got      $got" >&2
        exit 1
      fi
      echo "checksum OK"
    else
      echo "WARNING: no checksum line for $TARBALL; continuing unverified" >&2
    fi
  else
    echo "WARNING: upstream checksum file unavailable; continuing unverified" >&2
  fi

  echo "Extracting to $DEST"
  tmp="$(mktemp -d)"
  tar -xJf "$CACHE/$TARBALL" -C "$tmp"
  rm -rf "$DEST"
  mv "$tmp/blender-${BLENDER_VERSION}-linux-x64" "$DEST"
  rmdir "$tmp" 2>/dev/null || true
fi

ln -sfn "$DEST/blender" "$BIN"
echo "blender -> $("$BIN" --version 2>/dev/null | head -1)"

# --- 2. The MCP server ----------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
fi

# Cloned rather than installed from PyPI, because the add-on zip is built from
# the repo and the two halves must come from the same revision -- a server
# talking to an add-on from a different version fails at the protocol, not at
# import, which is a miserable thing to debug mid-run.
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$MCP_REF" \
    https://github.com/djeada/blender-mcp-server.git "$SRC"
else
  git -C "$SRC" fetch --depth 1 origin "$MCP_REF" && git -C "$SRC" checkout -q FETCH_HEAD
fi
"$VENV/bin/pip" install -q -e "$SRC"
git -C "$SRC" rev-parse HEAD > "$STATE_DIR/BLENDER_MCP_COMMIT"

if [ ! -x "$VENV/bin/blender-mcp-server" ]; then
  echo "FATAL: blender-mcp-server console script not found after install" >&2
  exit 1
fi
echo "blender-mcp-server -> $VENV/bin/blender-mcp-server"
echo "  revision $(cut -c1-8 < "$STATE_DIR/BLENDER_MCP_COMMIT")"

# --- 3. The add-on --------------------------------------------------------
ADDON_ZIP="$STATE_DIR/blender_mcp_addon.zip"
if [ -x "$SRC/scripts/build_addon_zip.sh" ]; then
  ( cd "$SRC" && ./scripts/build_addon_zip.sh >/dev/null 2>&1 ) || true
  found="$(find "$SRC" -maxdepth 2 -name '*addon*.zip' -newermt '-5 minutes' | head -1)"
  [ -n "$found" ] && cp "$found" "$ADDON_ZIP"
fi
if [ ! -f "$ADDON_ZIP" ] && [ -d "$SRC/addon" ]; then
  # Fall back to zipping the addon directory ourselves.
  ( cd "$SRC" && zip -qr "$ADDON_ZIP" addon ) 2>/dev/null || true
fi

cat <<EOF

--------------------------------------------------------------------
Blender ${BLENDER_VERSION} and the MCP server are installed.

ONE MANUAL STEP REMAINS, because Blender has no headless add-on install:

  1. blender
  2. Edit -> Preferences -> Add-ons -> Install from Disk
       $ADDON_ZIP
  3. Enable "Blender MCP Bridge"
  4. In the 3D viewport press N -> MCP tab
     It must say:  Listening on 127.0.0.1:9876

The add-on LISTENS and the server DIALS OUT, so Blender has to be running
with the bridge enabled before an agent can reach it. Same as Godot.

VRAM: the model holds the NVIDIA card. Set Blender's viewport to EEVEE and
render Cycles on CPU, or unload the model first. See docs/vram-budget.md.
--------------------------------------------------------------------
EOF
