"""Benchmark: fixed tasks, automatic checks, and the model x harness matrix.

Checks are run by headless Godot plus cheap text inspection of the project.
They are deliberately shallow -- they establish that a task produced something
that loads, runs and contains the required pieces, not that the game is fun.
A deep check would be slower and more subjective than the thing it measures.
"""
from __future__ import annotations

import fnmatch
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import godot
from .config import APP_DIR, PROJECTS_DIR
from .runs import Outcome

# /app is where the image puts it; APP_DIR is the checkout, which is what runs
# when the backend is started directly for a test.
TASKS_DIR = (
    Path("/app/bench/tasks") if Path("/app/bench/tasks").is_dir()
    else APP_DIR / "bench" / "tasks"
)
TEMPLATE_ADDON = (
    Path("/app/templates/addons/godot_mcp")
    if Path("/app/templates/addons/godot_mcp").is_dir()
    else APP_DIR / "templates" / "addons" / "godot_mcp"
)


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    name: str
    version: int
    prompt: str
    checks: list[dict[str, Any]]
    requires_vision: bool = False
    depends_on: str | None = None
    fixture: str | None = None
    # "game" marks a task that should produce a playable project, which the
    # Games tab lists and offers to launch.
    kind: str = "task"

    @classmethod
    def load(cls, path: Path) -> "Task":
        d = yaml.safe_load(path.read_text())
        return cls(
            id=d["id"], name=d["name"], version=d.get("version", 1),
            prompt=d["prompt"], checks=d.get("checks", []),
            requires_vision=d.get("requires_vision", False),
            depends_on=d.get("depends_on"), fixture=d.get("fixture"),
            kind=d.get("kind", "task"),
        )


def load_tasks(tasks_dir: Path = TASKS_DIR) -> dict[str, Task]:
    if not tasks_dir.is_dir():
        return {}
    tasks = {}
    for path in sorted(tasks_dir.glob("*.yaml")):
        task = Task.load(path)
        tasks[task.id] = task
    return tasks


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _all_scripts(project: Path) -> str:
    """Every .gd script in the project, concatenated and lowercased."""
    out = []
    for p in project.rglob("*.gd"):
        # Skip the MCP addon: it is our code, not the model's.
        if "addons" in p.parts:
            continue
        try:
            out.append(p.read_text(errors="replace"))
        except OSError:
            continue
    return "\n".join(out).lower()


def _all_text(project: Path) -> str:
    """Scripts and scenes together, for checks that may match either."""
    out = [_all_scripts(project)]
    for pattern in ("*.tscn", "*.tres"):
        for p in project.rglob(pattern):
            if "addons" in p.parts:
                continue
            try:
                out.append(p.read_text(errors="replace"))
            except OSError:
                continue
    return "\n".join(out).lower()


def run_checks(task: Task, project: Path) -> dict[str, bool]:
    """Run every check for a task. Returns {check_label: passed}."""
    results: dict[str, bool] = {}
    validation: godot.ValidationResult | None = None

    def _validate(seconds: int) -> godot.ValidationResult:
        nonlocal validation
        if validation is None:
            validation = godot.validate(project, run_seconds=seconds)
        return validation

    for check in task.checks:
        kind = check["kind"]
        if kind == "project_loads":
            results["project loads"] = _validate(10).loads
        elif kind == "no_script_errors":
            results["no script errors"] = _validate(10).no_script_errors
        elif kind == "runs_clean":
            seconds = int(check.get("seconds", 10))
            results[f"runs {seconds}s clean"] = _validate(seconds).ran_clean
        elif kind == "nodes_present":
            scene = check.get("scene", "main.tscn")
            ok = True
            for node in check.get("all_of", []):
                present = godot.scene_has_nodes(project, scene, [node])[node]
                results[f"has {node}"] = present
                ok = ok and present
            any_of = check.get("any_of", [])
            if any_of:
                found = godot.scene_has_nodes(project, scene, any_of)
                results[f"has one of {'/'.join(any_of[:3])}…"] = any(found.values())
        elif kind == "files_present":
            for pattern in check.get("patterns", []):
                hit = any(
                    fnmatch.fnmatch(p.name, pattern) and "addons" not in p.parts
                    for p in project.rglob("*")
                )
                results[f"file {pattern}"] = hit
        elif kind == "text_in_scripts":
            text = _all_scripts(project)
            needles = [n.lower() for n in check.get("any_of", [])]
            label = f"script mentions {needles[0]}" if needles else "script text"
            results[label] = any(n in text for n in needles)
        elif kind == "text_anywhere":
            text = _all_text(project)
            needles = [n.lower() for n in check.get("any_of", [])]
            label = f"mentions {needles[0]}" if needles else "text"
            results[label] = any(n in text for n in needles)
    return results


def outcome_from_checks(checks: dict[str, bool]) -> Outcome:
    if not checks:
        return Outcome.ERROR
    return Outcome.PASS if all(checks.values()) else Outcome.FAIL


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    model_id: str
    harness_id: str
    task_id: str
    skipped: str | None = None      # reason, when the cell cannot be run

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def build_matrix(models: list[dict[str, Any]], harnesses: list[dict[str, Any]],
                 tasks: dict[str, Task], model_ids: list[str],
                 harness_ids: list[str], task_ids: list[str]) -> list[Cell]:
    """Expand the selection into cells, marking the impossible ones.

    A cell is skipped rather than silently dropped, so the matrix shows *why*
    a combination has no number instead of leaving a blank.
    """
    by_model = {m["id"]: m for m in models}
    by_harness = {h["id"]: h for h in harnesses}
    cells: list[Cell] = []
    for task_id in task_ids:
        task = tasks.get(task_id)
        if task is None:
            continue
        for model_id in model_ids:
            model = by_model.get(model_id)
            if model is None:
                continue
            for harness_id in harness_ids:
                harness = by_harness.get(harness_id)
                if harness is None:
                    continue
                skip = None
                if task.requires_vision and not model["vision"]:
                    skip = "N/A - engine is text-only"
                elif harness_id != "direct" and not model["tools"]:
                    skip = "N/A - engine has no tool calling"
                elif harness_id == "direct":
                    # Direct chat has no tools and no file access, so it cannot
                    # produce a project. It is a speed baseline only.
                    skip = "baseline only - cannot build a game"
                cells.append(Cell(model_id, harness_id, task_id, skip))
    return cells


def fresh_project(task: Task, model_id: str, harness_id: str,
                  projects_dir: Path = PROJECTS_DIR,
                  template_addon: Path | None = None,
                  previous: Path | None = None) -> Path:
    """A clean project for one cell.

    Edit tasks (those with depends_on) start from a copy of the dependency's
    result, so what is measured is editing existing code rather than writing
    from scratch.
    """
    name = f"bench_{task.id}_{model_id}_{harness_id}_{int(time.time())}"
    dest = projects_dir / name
    if task.depends_on and previous and previous.exists():
        shutil.copytree(previous, dest)
    else:
        godot.create_project(name, projects_dir, template_addon)
        dest = projects_dir / name
    return dest
