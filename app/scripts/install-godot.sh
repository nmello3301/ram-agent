#!/usr/bin/env bash
# Install the Godot 4.7.2 editor for the current user, plus the MCP addon that
# new projects are seeded from. No root needed: everything lands under ~/.local.
set -euo pipefail

VERSION="4.7.2-stable"
# Pinned from the release's own SHA512-SUMS.txt.
SHA512="9aa00f7a605200940bce3027a567b782f49bd8e940dd06ae9e987bd65aee1b1467edd56ed84fcdcbdd44354bf613bdbb4e5d2913e925850368e150c59ed54c65"
ZIP="Godot_v${VERSION}_linux.x86_64.zip"
URL="https://github.com/godotengine/godot-builds/releases/download/${VERSION}/${ZIP}"

ROOT="${RAM_AGENT_ROOT:-$HOME/Desktop/RAM_Agent}"
PREFIX="$HOME/.local"
OPT="$PREFIX/share/godot"
BIN="$PREFIX/bin/godot"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$OPT" "$PREFIX/bin" "$PREFIX/share/applications"

if [ -x "$BIN" ] && "$BIN" --version 2>/dev/null | grep -q "4.7.2"; then
  echo "Godot 4.7.2 already installed at $BIN"
else
  echo "Downloading Godot $VERSION..."
  curl -fsSL -o "$TMP/$ZIP" "$URL"
  echo "Verifying checksum..."
  echo "${SHA512}  $TMP/$ZIP" | sha512sum -c -
  unzip -qo "$TMP/$ZIP" -d "$OPT"
  mv -f "$OPT/Godot_v${VERSION}_linux.x86_64" "$OPT/godot-${VERSION}"
  chmod +x "$OPT/godot-${VERSION}"
  ln -sf "$OPT/godot-${VERSION}" "$BIN"
  echo "Installed: $("$BIN" --version | head -1)"
fi

cat > "$PREFIX/share/applications/godot-4.7.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Godot Engine 4.7.2
Comment=Godot editor used by RAM_Agent
Exec=$BIN %f
Icon=godot
Terminal=false
Categories=Development;IDE;
MimeType=application/x-godot-project;
DESKTOP
update-desktop-database "$PREFIX/share/applications" 2>/dev/null || true

# The MCP addon, cached in templates/ so new projects can be seeded offline.
ADDON_DEST="$ROOT/app/templates/addons"
if [ -d "$ADDON_DEST/godot_mcp" ]; then
  echo "MCP addon already present in templates/"
else
  echo "Fetching the godot-mcp editor addon..."
  # hybridindie/godot-mcp. NOT Coding-Solo/godot-mcp, which is affected by
  # CVE-2026-25546 (command injection -> RCE) below 0.1.1.
  if curl -fsSL -o "$TMP/addon.zip" \
       "https://github.com/hybridindie/godot-mcp/releases/latest/download/godot_mcp_addon.zip"; then
    mkdir -p "$ADDON_DEST" "$TMP/addon"
    unzip -qo "$TMP/addon.zip" -d "$TMP/addon"
    if [ -d "$TMP/addon/addons/godot_mcp" ]; then
      cp -r "$TMP/addon/addons/godot_mcp" "$ADDON_DEST/"
    elif [ -d "$TMP/addon/godot_mcp" ]; then
      cp -r "$TMP/addon/godot_mcp" "$ADDON_DEST/"
    else
      echo "  unexpected addon layout:"; ls "$TMP/addon"
    fi
    [ -d "$ADDON_DEST/godot_mcp" ] && echo "  addon cached in templates/addons/"
  else
    echo "  WARNING: addon download failed. Install from the Godot Asset"
    echo "  Library instead (Editor > AssetLib > 'Godot MCP')."
  fi
fi

echo
echo "Done. Open a project with: scripts/godot-editor.sh <project-name>"
