"""Finished games: what was built, whether it works, and how to play it.

A "game" here is a project produced by a run of a task whose `kind` is `game`,
joined with that run's record and a fresh look at what is on disk. The point is
to answer, per game: did it end up playable, and how do I start it.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import godot
from .config import PROJECTS_DIR, STATE_DIR

PLAY_REQUESTS = STATE_DIR / "play-requests"


def request_play(project: str) -> dict[str, Any]:
    """Ask the host to open this game in a window.

    The backend runs in a container with no access to the desktop session, so
    it leaves a request for `scripts/play-watcher.sh` to pick up. The UI always
    shows the equivalent command too, so this degrades to a copy-paste rather
    than to nothing when the watcher is not running.
    """
    target = PROJECTS_DIR / project
    if not (target / "project.godot").exists():
        raise FileNotFoundError(f"no Godot project named {project!r}")
    PLAY_REQUESTS.mkdir(parents=True, exist_ok=True)
    req = PLAY_REQUESTS / f"{int(time.time() * 1000)}-{project}.request"
    req.write_text(project + "\n")
    return {
        "requested": project,
        "watcher_running": watcher_running(),
        "command": f'scripts/play.sh "{project}"',
    }


def request_editor(project: str) -> Path:
    """Ask the host to open this project in the Godot editor.

    Scene-editing MCP tools drive the *live* editor, so a run against a freshly
    created project does nothing until the editor is open on that project and
    its addon has dialled the bridge.
    """
    target = PROJECTS_DIR / project
    if not (target / "project.godot").exists():
        raise FileNotFoundError(f"no Godot project named {project!r}")
    PLAY_REQUESTS.mkdir(parents=True, exist_ok=True)
    req = PLAY_REQUESTS / f"{int(time.time() * 1000)}-{project}.editor"
    req.write_text(project + "\n")
    return req


def editor_project() -> str | None:
    """Which project the host editor is currently open on, per the watcher."""
    marker = STATE_DIR / "editor-project"
    try:
        name = marker.read_text().strip()
        return name or None
    except OSError:
        return None


def watcher_running() -> bool:
    """Whether requests are being consumed.

    Inferred from the queue rather than from process tables, which the
    container cannot see: a watcher drains requests within a second, so a
    backlog older than a few seconds means nothing is listening.
    """
    if not PLAY_REQUESTS.is_dir():
        return False
    now = time.time()
    stale = [
        p for p in PLAY_REQUESTS.glob("*.request")
        if now - p.stat().st_mtime > 5
    ]
    return not stale


def _main_scene_nodes(project: Path) -> int:
    """Nodes declared in main.tscn. The starting template has exactly one."""
    path = project / "main.tscn"
    if not path.exists():
        return 0
    try:
        return path.read_text(errors="replace").count("[node ")
    except OSError:
        return 0


def _script_stats(project: Path) -> dict[str, int]:
    scripts = [p for p in project.rglob("*.gd") if "addons" not in p.parts]
    lines = 0
    for p in scripts:
        try:
            lines += len(p.read_text(errors="replace").splitlines())
        except OSError:
            continue
    scenes = [p for p in project.rglob("*.tscn") if "addons" not in p.parts]
    return {"scripts": len(scripts), "script_lines": lines, "scenes": len(scenes)}


def list_games(runs: list[dict[str, Any]], tasks: dict[str, Any]) -> list[dict[str, Any]]:
    """Every game project on disk, joined with the run that produced it."""
    game_task_ids = {tid for tid, t in tasks.items()
                     if getattr(t, "kind", None) == "game"}

    # Latest run per project wins: a project may be re-run.
    by_project: dict[str, dict[str, Any]] = {}
    for r in runs:
        proj = r.get("project")
        if not proj:
            continue
        prev = by_project.get(proj)
        if prev is None or r.get("started_at", 0) >= prev.get("started_at", 0):
            by_project[proj] = r

    out = []
    for name, run in sorted(by_project.items(), key=lambda kv: -kv[1].get("started_at", 0)):
        if run.get("task_id") not in game_task_ids:
            continue
        path = PROJECTS_DIR / name
        if not (path / "project.godot").exists():
            continue
        checks = run.get("checks") or {}
        task = tasks.get(run["task_id"])
        out.append({
            "project": name,
            "task_id": run.get("task_id"),
            "task_name": getattr(task, "name", run.get("task_id")),
            "model_id": run.get("model_id"),
            "harness_id": run.get("harness_id"),
            "outcome": run.get("outcome"),
            "wall_seconds": run.get("wall_seconds"),
            "started_at": run.get("started_at"),
            "checks_passed": sum(1 for v in checks.values() if v),
            "checks_total": len(checks),
            "checks": checks,
            "metrics": run.get("metrics", {}),
            # An empty project passes "loads" and "no script errors" trivially,
            # so those alone marked untouched templates as playable. A game is
            # only playable if the agent actually produced something: at least
            # one script of its own, and a main scene with more than the single
            # root node the template ships.
            "playable": (
                bool(checks.get("project loads"))
                and bool(checks.get("no script errors"))
                and _script_stats(path)["scripts"] > 0
                and _main_scene_nodes(path) > 1
            ),
            "command": f'scripts/play.sh "{name}"',
            **_script_stats(path),
        })
    return out


def validate_now(project: str, seconds: int = 10) -> dict[str, Any]:
    """Re-check a game on demand, without re-running the agent."""
    path = PROJECTS_DIR / project
    if not (path / "project.godot").exists():
        raise FileNotFoundError(f"no Godot project named {project!r}")
    return godot.validate(path, run_seconds=seconds).as_dict()
