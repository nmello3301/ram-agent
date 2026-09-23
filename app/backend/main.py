"""FastAPI app: REST endpoints, SSE streaming, static UI."""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from . import bench, games, godot, tools_catalog
from .config import (
    APP_DIR, MODELS_DIR, PROJECTS_DIR, STATE_DIR, Settings,
    hf_token, load_harnesses, load_models,
)
from .models_mgr import ModelManager
from .orchestrator import Orchestrator
from .runs import RunLog

UI_DIR = Path("/app/ui") if Path("/app/ui").is_dir() else APP_DIR / "ui"

# Left for the OS, the container, Firefox and the Godot editor. The VRAM
# half of the budget is a user setting; this half is not, because it is a
# property of the machine rather than a preference.
RAM_HEADROOM_GB = 12

app = FastAPI(title="RAM_Agent", docs_url=None, redoc_url=None)

settings = Settings.load()
models_registry = load_models()
harnesses_registry = load_harnesses()
manager = ModelManager(models_registry, MODELS_DIR)
orch = Orchestrator(manager, harnesses_registry, settings)
runlog = RunLog(STATE_DIR / "runs.jsonl")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "time": time.time(), **orch.state()}


def _hardware() -> dict[str, Any]:
    """RAM, VRAM and the budget a model is sized against."""
    from .engine import fast_memory
    return fast_memory(
        ram_headroom_gb=RAM_HEADROOM_GB,
        vram_reserve_mb=Settings.load().vram_reserve_mb,
    )


def _model_rows() -> list[dict[str, Any]]:
    """Every model, annotated with whether it is here and whether it fits.

    The UI needs three things this adds on top of the registry row: which
    declared files are actually on disk, where to put them if they are not,
    and whether the machine could hold the model at all. All three are computed
    from the filesystem and /proc at request time rather than cached, because
    the honest answer changes when someone drops a file in or starts Blender.
    """
    hw = _hardware()
    usable = hw["usable_gb"]
    fast_total = hw["fast_total_gb"]

    rows = []
    for entry in manager.entries.values():
        row = entry.as_dict()
        missing = entry.missing_files(MODELS_DIR)
        row["missing_files"] = missing
        row["present"] = not missing
        row["target_dir"] = str(entry.dir(MODELS_DIR))
        row["fits"] = entry.size_gb <= usable
        row["fits_at_all"] = entry.size_gb <= fast_total
        if not row["fits_at_all"]:
            row["fit_note"] = (
                f"{entry.size_gb:.1f} GB exceeds this machine's "
                f"{fast_total:.0f} GB of fast memory"
            )
        elif not row["fits"]:
            row["fit_note"] = (
                f"{entry.size_gb:.1f} GB against {usable:.1f} GB usable — "
                "it would fit only by squeezing the editors"
            )
        else:
            row["fit_note"] = (
                f"{entry.size_gb:.1f} GB of {usable:.1f} GB usable"
            )
        rows.append(row)
    return rows


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    manager.refresh_states()
    return {
        "llamacpp_ref": manager.llamacpp_ref,
        "default": manager.default_id,
        "loaded": orch.engine.loaded_model_id,
        "hardware": _hardware(),
        "models": _model_rows(),
    }


@app.post("/api/models/refresh")
def refresh_models() -> dict[str, Any]:
    """Re-read the disk and report what changed.

    This is the button behind the model list. It exists because the supported
    way to get a model onto this machine is not only the download button: the
    UI tells you the exact files and the exact directory, and dropping them
    there by hand is a first-class path. Nothing notices that until something
    looks, so something has to look on demand.
    """
    before = {e.id: e.state.value for e in manager.entries.values()}
    manager.refresh_states()
    rows = _model_rows()
    after = {e.id: e.state.value for e in manager.entries.values()}
    changed = [mid for mid in after if before.get(mid) != after[mid]]
    return {
        "models": rows,
        "hardware": _hardware(),
        "changed": changed,
        "present": sum(1 for r in rows if r["present"]),
        "total": len(rows),
    }


@app.get("/api/hardware")
def hardware() -> dict[str, Any]:
    return _hardware()


@app.get("/api/harnesses")
def list_harnesses() -> dict[str, Any]:
    return harnesses_registry


@app.get("/api/projects")
def list_projects() -> dict[str, Any]:
    return {"projects": godot.list_projects(PROJECTS_DIR)}


@app.post("/api/projects")
async def create_project(request: Request) -> dict[str, Any]:
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "a project name is required")
    try:
        path = godot.create_project(name, PROJECTS_DIR)
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"created": str(path), "projects": godot.list_projects(PROJECTS_DIR)}


@app.get("/api/godot")
def godot_status(project: str | None = None) -> dict[str, Any]:
    target = PROJECTS_DIR / project if project else None
    return godot.status(target).as_dict()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    data = settings.as_dict()
    # Never return the token itself, only whether one is configured.
    data["hf_token_set"] = hf_token() is not None
    return data


@app.post("/api/settings")
async def update_settings(request: Request) -> dict[str, Any]:
    body = await request.json()
    for key in ("time_limit_enabled", "inhibit_sleep", "mcp_enabled"):
        if key in body:
            setattr(settings, key, bool(body[key]))
    if "ram_headroom_gb" in body:
        settings.ram_headroom_gb = max(0, int(body["ram_headroom_gb"]))
    if "time_limit_seconds" in body:
        value = int(body["time_limit_seconds"])
        if value <= 0:
            raise HTTPException(400, "time limit must be positive")
        # Deliberately no upper bound: an overnight run is a valid choice.
        settings.time_limit_seconds = value
    if "mcp_toolsets" in body:
        settings.godot_toolsets = [str(t) for t in body["mcp_toolsets"]]
    if "model_flags" in body:
        settings.model_flags = dict(body["model_flags"])
    settings.save()
    return get_settings()


# ---------------------------------------------------------------------------
# Models: download / load
# ---------------------------------------------------------------------------

@app.post("/api/models/{model_id}/download")
def download_model(model_id: str) -> dict[str, Any]:
    try:
        entry = manager.get(model_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    threading.Thread(
        target=manager.download,
        args=(model_id, hf_token(), orch.emit),
        daemon=True,
    ).start()
    return {"started": True, "model": entry.as_dict()}


@app.post("/api/models/{model_id}/cancel")
def cancel_download(model_id: str) -> dict[str, Any]:
    manager.cancel_download(model_id)
    return {"cancelled": True}


@app.post("/api/models/{model_id}/load")
def load_model(model_id: str) -> dict[str, Any]:
    entry = manager.get(model_id)
    if orch.busy:
        raise HTTPException(409, "a run is in progress; stop it first")

    def _load() -> None:
        try:
            orch.engine.start(
                entry, manager.defaults,
                flag_overrides=settings.model_flags.get(model_id),
                on_log=lambda l: orch.emit({"type": "engine_log", "line": l}),
                ram_headroom_gb=settings.ram_headroom_gb,
            )
        except Exception as exc:                       # noqa: BLE001
            orch.emit({"type": "error", "error": str(exc)})

    threading.Thread(target=_load, daemon=True).start()
    return {"loading": True}


@app.get("/api/engine/logs")
def engine_logs(n: int = 200) -> dict[str, Any]:
    """The engine's own output. The first place to look when a load fails."""
    return {"lines": orch.engine.logs(n), "status": orch.engine.status.as_dict()}


@app.post("/api/models/unload")
def unload_model() -> dict[str, Any]:
    orch.engine.stop()
    return {"unloaded": True}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

@app.post("/api/run")
async def start_run(request: Request) -> dict[str, Any]:
    body = await request.json()
    try:
        record = orch.start_run(
            model_id=body["model_id"],
            harness_id=body["harness_id"],
            project=body.get("project", ""),
            prompt=body.get("prompt", ""),
            task_id=body.get("task_id"),
            images=body.get("images") or [],
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return record.as_dict()


@app.post("/api/run/stop")
def stop_run() -> dict[str, Any]:
    orch.stop_run()
    return {"stopping": True}


@app.post("/api/run/continue")
async def continue_run(request: Request) -> dict[str, Any]:
    """Resume a run that hit its time limit, with another full window."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        body = {}
    extra = body.get("extra_seconds")
    try:
        orch.continue_run(int(extra) if extra else None)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"continued": True, **orch.state()}


@app.get("/api/runs")
def list_runs(limit: int = 200) -> dict[str, Any]:
    rows = runlog.read_all()
    return {"runs": rows[-limit:][::-1]}


@app.get("/api/runs.csv")
def runs_csv() -> PlainTextResponse:
    return PlainTextResponse(
        runlog.to_csv(),
        headers={"Content-Disposition": 'attachment; filename="runs.csv"'},
        media_type="text/csv",
    )


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

@app.get("/api/games")
def list_games() -> dict[str, Any]:
    """Playable projects produced by game tasks, newest first."""
    return {
        "games": games.list_games(runlog.read_all(), bench.load_tasks()),
        "watcher_running": games.watcher_running(),
    }


@app.post("/api/games/{project}/play")
def play_game(project: str) -> dict[str, Any]:
    try:
        return games.request_play(project)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/games/{project}/validate")
def validate_game(project: str) -> dict[str, Any]:
    """Re-run the headless checks on a finished game."""
    try:
        return games.validate_now(project)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@app.get("/api/tools")
def tools() -> dict[str, Any]:
    """The whole Godot MCP catalogue, with what is enabled and what it costs."""
    return tools_catalog.catalog(settings.godot_toolsets)


@app.post("/api/tools")
async def set_tools(request: Request) -> dict[str, Any]:
    """Change which toolsets are enabled. Takes effect on the next run."""
    body = await request.json()
    wanted = body.get("toolsets")
    if not isinstance(wanted, list):
        raise HTTPException(400, "`toolsets` must be an array")
    known = {t["name"] for t in tools_catalog.load_snapshot().get("toolsets", [])}
    unknown = [t for t in wanted if t not in known]
    if unknown:
        raise HTTPException(400, f"unknown toolset(s): {', '.join(unknown)}")
    settings.godot_toolsets = [str(t) for t in wanted]
    settings.save()
    return tools_catalog.catalog(settings.godot_toolsets)


@app.get("/api/lineup")
def lineup() -> dict[str, Any]:
    """Models that exist upstream but are not in this build, and why.

    Kept visible rather than implied. The interesting cases are not the ones
    that are obviously too big -- they are Qwen3.8-Flash-Next, which "fits" in
    46 GB only by paging a 51B n-gram table off the SSD, and GLM-5.3-Flash,
    which is the model Colibri ran at 0.1 tok/s. Both are excluded on purpose.
    """
    return {"excluded": models_registry.get("excluded", [])}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@app.get("/api/bench/tasks")
def bench_tasks() -> dict[str, Any]:
    tasks = bench.load_tasks()
    return {
        "tasks": [
            {"id": t.id, "name": t.name, "version": t.version,
             "requires_vision": t.requires_vision, "depends_on": t.depends_on}
            for t in tasks.values()
        ]
    }


@app.post("/api/bench/matrix")
async def bench_matrix(request: Request) -> dict[str, Any]:
    """Expand a selection into cells without running anything.

    Lets the UI show which combinations are impossible, and why, before
    committing to a matrix that may take all night.
    """
    body = await request.json()
    tasks = bench.load_tasks()
    manager.refresh_states()
    cells = bench.build_matrix(
        [e.as_dict() for e in manager.entries.values()],
        harnesses_registry["harnesses"], tasks,
        body.get("model_ids", []), body.get("harness_ids", []),
        body.get("task_ids", []),
    )
    return {
        "cells": [c.as_dict() for c in cells],
        "runnable": sum(1 for c in cells if not c.skipped),
        "skipped": sum(1 for c in cells if c.skipped),
    }


@app.post("/api/bench/run")
async def bench_run(request: Request) -> dict[str, Any]:
    """Queue a matrix. Cells run one at a time; progress arrives over SSE."""
    body = await request.json()
    if orch.busy:
        raise HTTPException(409, "a run is already in progress")
    tasks = bench.load_tasks()
    manager.refresh_states()
    cells = bench.build_matrix(
        [e.as_dict() for e in manager.entries.values()],
        harnesses_registry["harnesses"], tasks,
        body.get("model_ids", []), body.get("harness_ids", []),
        body.get("task_ids", []),
    )
    runnable = [c for c in cells if not c.skipped]
    threading.Thread(
        target=_run_matrix, args=(runnable, tasks), daemon=True
    ).start()
    return {"queued": len(runnable),
            "skipped": [c.as_dict() for c in cells if c.skipped]}


def _run_matrix(cells: list, tasks: dict) -> None:
    """Run each cell to completion, in order, from a fresh project."""
    previous: dict[str, Any] = {}
    for i, cell in enumerate(cells, 1):
        task = tasks[cell.task_id]
        orch.emit({"type": "bench_cell_start", "index": i, "total": len(cells),
                   "cell": cell.as_dict()})
        try:
            project = bench.fresh_project(
                task, cell.model_id, cell.harness_id, PROJECTS_DIR,
                bench.TEMPLATE_ADDON,
                previous.get(task.depends_on or ""),
            )
            record = orch.start_run(
                model_id=cell.model_id, harness_id=cell.harness_id,
                project=project.name, prompt=task.prompt, task_id=task.id,
            )
            # One cell at a time: llama-server runs --parallel 1 anyway, and a
            # shared machine would make every number meaningless.
            while orch.busy:
                time.sleep(2.0)
            record.checks = bench.run_checks(task, project)
            if record.outcome.value == "pass":
                record.outcome = bench.outcome_from_checks(record.checks)
            orch.runlog.update(record)
            previous[task.id] = project
            orch.emit({"type": "bench_cell_end", "cell": cell.as_dict(),
                       "outcome": record.outcome.value, "checks": record.checks})
        except Exception as exc:                        # noqa: BLE001
            orch.emit({"type": "bench_cell_error", "cell": cell.as_dict(),
                       "error": str(exc)})
    orch.emit({"type": "bench_done"})


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------

@app.get("/api/stream")
async def stream(request: Request) -> StreamingResponse:
    """Server-sent events for the whole app.

    A heartbeat goes out every 15 s. Prefill on these engines can produce
    nothing for an hour, and without a heartbeat Firefox and any intermediary
    would give up on an idle connection long before the model speaks.
    """
    q = orch.bus.subscribe()
    loop = asyncio.get_running_loop()

    async def gen():
        try:
            yield _sse({"type": "hello", **orch.state()})
            last_beat = time.monotonic()
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await loop.run_in_executor(None, q.get, True, 1.0)
                    yield _sse(event)
                except queue.Empty:
                    pass
                if time.monotonic() - last_beat >= 15:
                    last_beat = time.monotonic()
                    yield ": heartbeat\n\n"
        finally:
            orch.bus.unsubscribe(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


if UI_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")
