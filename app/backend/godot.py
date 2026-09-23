"""Godot projects, the MCP addon, and headless validation.

The interactive editor runs on the host under Wayland; only the headless binary
lives in the container, and it is used purely to check that a project loads and
runs. The MCP server (in the container) binds a WebSocket bridge that the
addon (in the host editor) dials out to.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import GODOT_BIN, PROJECTS_DIR

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 9080
ADDON_DIR_NAME = "godot_mcp"

GODOT_VERSION = "4.7.2"

# Minimal, valid Godot 4.7 project. Written rather than copied so a new project
# never depends on a template having been downloaded first.
PROJECT_GODOT = """\
config_version=5

[application]

config/name="{name}"
run/main_scene="res://main.tscn"
config/features=PackedStringArray("4.7", "GL Compatibility")

[editor_plugins]

enabled=PackedStringArray("res://addons/{addon}/plugin.cfg")

[rendering]

renderer/rendering_method="gl_compatibility"
"""

MAIN_SCENE = """\
[gd_scene format=3 uid="uid://b0ramagent0001"]

[node name="Main" type="Node2D"]
"""


@dataclass
class GodotStatus:
    bridge_listening: bool = False
    editor_connected: bool = False
    mcp_installed: bool = False
    godot_version: str = ""
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def bridge_listening(host: str = BRIDGE_HOST, port: int = BRIDGE_PORT) -> bool:
    """True when something is accepting connections on the MCP bridge port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def godot_version() -> str:
    """Version string of the headless binary, or '' when it is missing."""
    if not Path(GODOT_BIN).exists():
        return ""
    try:
        out = subprocess.run(
            [str(GODOT_BIN), "--version"], capture_output=True, text=True, timeout=30
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    except (OSError, subprocess.SubprocessError, IndexError):
        return ""


def status(project: Path | None = None) -> GodotStatus:
    st = GodotStatus(
        bridge_listening=bridge_listening(),
        godot_version=godot_version(),
    )
    if project is not None:
        st.mcp_installed = (project / "addons" / ADDON_DIR_NAME).is_dir()
    if not st.bridge_listening:
        st.message = (
            "The Godot editor is not connected. Start it with "
            "scripts/godot-editor.sh <project>, then enable "
            "Project > Project Settings > Plugins > Godot MCP."
        )
    else:
        st.editor_connected = True
        st.message = "Editor bridge is up."
    return st


def list_projects(projects_dir: Path = PROJECTS_DIR) -> list[dict[str, Any]]:
    projects_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for child in sorted(projects_dir.iterdir()):
        if child.is_dir() and (child / "project.godot").exists():
            out.append({
                "name": child.name,
                "path": str(child),
                "mcp_installed": (child / "addons" / ADDON_DIR_NAME).is_dir(),
            })
    return out


def _safe_name(name: str) -> str:
    """Keep project names to something that is safe as a directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9 ._-]", "", name).strip()
    return cleaned or "Untitled"


def create_project(name: str, projects_dir: Path = PROJECTS_DIR,
                   addon_source: Path | None = None) -> Path:
    """Create a minimal Godot 4.7 project with the MCP addon installed."""
    safe = _safe_name(name)
    dest = projects_dir / safe
    if dest.exists():
        raise FileExistsError(f"project already exists: {safe}")
    dest.mkdir(parents=True)
    (dest / "project.godot").write_text(
        PROJECT_GODOT.format(name=safe, addon=ADDON_DIR_NAME)
    )
    (dest / "main.tscn").write_text(MAIN_SCENE)
    install_addon(dest, addon_source)
    return dest


def install_addon(project: Path, addon_source: Path | None = None) -> bool:
    """Copy the MCP addon into a project and enable it in project.godot."""
    source = addon_source or Path("/app/templates/addons") / ADDON_DIR_NAME
    target = project / "addons" / ADDON_DIR_NAME
    if not source.is_dir():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)

    # Make sure the plugin is listed as enabled.
    cfg = project / "project.godot"
    text = cfg.read_text() if cfg.exists() else ""
    line = f'enabled=PackedStringArray("res://addons/{ADDON_DIR_NAME}/plugin.cfg")'
    if "[editor_plugins]" not in text:
        text += f"\n\n[editor_plugins]\n\n{line}\n"
    elif ADDON_DIR_NAME not in text:
        text = text.replace("[editor_plugins]", f"[editor_plugins]\n\n{line}", 1)
    cfg.write_text(text)
    return True


@dataclass
class ValidationResult:
    loads: bool = False
    no_script_errors: bool = False
    ran_clean: bool = False
    errors: list[str] = None
    output: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "loads": self.loads,
            "no_script_errors": self.no_script_errors,
            "ran_clean": self.ran_clean,
            "errors": self.errors or [],
            "output": self.output[-4000:],
        }


_ERROR_RE = re.compile(
    r"(SCRIPT ERROR|ERROR:|Parse Error|Failed to load|Cannot open|"
    r"Invalid call|Attempt to call)", re.I
)

# Godot prints these on a clean shutdown of the editor/import pass. They are
# teardown bookkeeping, not defects in the project, and matching them would
# fail every task that is actually fine. Verified against a freshly created,
# empty 4.7.2 project, which emits both.
_BENIGN_RE = re.compile(
    r"(resources still in use at exit"
    r"|ObjectDB instances were leaked"
    r"|were leaked at exit"
    r"|--verbose for details)", re.I
)


def _real_errors(output: str) -> list[str]:
    """Lines that indicate a genuine problem with the project."""
    return [
        line for line in output.splitlines()
        if _ERROR_RE.search(line) and not _BENIGN_RE.search(line)
    ]


def validate(project: Path, run_seconds: int = 10) -> ValidationResult:
    """Headless checks: the project imports, and the main scene runs clean.

    Godot writes most diagnostics to stdout, so both streams are merged and
    scanned together.
    """
    result = ValidationResult(errors=[])
    if not Path(GODOT_BIN).exists():
        result.errors.append("headless Godot binary not found")
        return result
    if not (project / "project.godot").exists():
        result.errors.append("project.godot missing")
        return result

    env = os.environ.copy()
    env["DISPLAY"] = ""      # force headless even if a display leaks in

    # 1. Import/open the project once so resources are built.
    imported = subprocess.run(
        [str(GODOT_BIN), "--headless", "--path", str(project), "--import"],
        capture_output=True, text=True, timeout=600, env=env,
    )
    import_out = imported.stdout + imported.stderr
    result.loads = imported.returncode == 0
    result.output += import_out

    errors = _real_errors(import_out)
    result.no_script_errors = not errors
    result.errors.extend(errors[:50])

    if not result.loads:
        return result

    # 2. Run the main scene for a fixed window and see if it stays clean.
    try:
        ran = subprocess.run(
            [str(GODOT_BIN), "--headless", "--path", str(project),
             "--quit-after", str(run_seconds * 60)],   # quit-after counts frames
            capture_output=True, text=True, timeout=run_seconds + 120, env=env,
        )
        run_out = ran.stdout + ran.stderr
        result.output += "\n" + run_out
        run_errors = _real_errors(run_out)
        result.errors.extend(run_errors[:50])
        result.ran_clean = ran.returncode == 0 and not run_errors
    except subprocess.TimeoutExpired:
        result.errors.append(f"main scene did not exit within {run_seconds}s")
        result.ran_clean = False

    result.no_script_errors = not result.errors
    return result


def scene_has_nodes(project: Path, scene: str, node_types: list[str]) -> dict[str, bool]:
    """Check a .tscn for required node types, by reading the scene file.

    Text-based and cheap, which matters because the benchmark checks run after
    every task and must not themselves be slow.
    """
    path = project / scene
    if not path.exists():
        return {t: False for t in node_types}
    text = path.read_text(errors="replace")
    return {t: bool(re.search(rf'type="{re.escape(t)}"', text)) for t in node_types}
