"""Model manager: registry, disk-space checks, downloads, state machine.

The state machine is deliberately small and explicit, because the UI shows it
and the benchmark depends on it:

    absent -> downloading -> verifying -> ready -> loading -> loaded
                  |             |          |        |
                  +-------------+----------+--------+--> error

`error` is reachable from anywhere and always carries a human-readable reason.

Carried over from Colibri unchanged in shape. One thing changed substantively:
a model is no longer a *repository*, it is **one or two files inside one**.
The GGUF repos here publish every quantisation side by side -- pulling
`unsloth/Qwen3-Coder-Next-GGUF` wholesale is over 900 GB for a 46 GB model --
so downloads are restricted to the exact files the registry names.
"""
from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class State(str, Enum):
    ABSENT = "absent"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    READY = "ready"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


# Which transitions are legal. Anything not listed is a bug, not a user error.
ALLOWED: dict[State, set[State]] = {
    State.ABSENT: {State.DOWNLOADING, State.VERIFYING, State.ERROR},
    State.DOWNLOADING: {State.VERIFYING, State.ABSENT, State.ERROR},
    State.VERIFYING: {State.READY, State.ERROR},
    State.READY: {State.LOADING, State.ABSENT, State.ERROR},
    State.LOADING: {State.LOADED, State.READY, State.ERROR},
    State.LOADED: {State.READY, State.ERROR},
    # Recoverable: retrying a failed download goes back to the start.
    State.ERROR: {State.ABSENT, State.DOWNLOADING, State.VERIFYING, State.READY},
}


class TransitionError(RuntimeError):
    """Raised on an illegal state transition."""


@dataclass
class Progress:
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bps: float = 0.0
    eta_seconds: float | None = None

    @property
    def percent(self) -> float:
        if not self.total_bytes:
            return 0.0
        return min(100.0, 100.0 * self.downloaded_bytes / self.total_bytes)


@dataclass
class ModelEntry:
    """One row of models.yaml plus its live state."""

    id: str
    name: str
    repo: str
    revision: str
    gguf: str
    size_gb: float
    tools: bool
    vision: bool
    gpu: bool
    context_ceiling: int
    # Vision projector. None for text-only models; llama-server is simply not
    # given --mmproj and the screenshot toolsets are dropped for the run.
    mmproj: str | None = None
    # Speculative decoding, e.g. {"type": "draft-mtp", "draft_n_max": 2}.
    spec: dict[str, Any] | None = None
    # Passed to llama-server verbatim. The engine validates that this is JSON
    # and does not look inside it -- which is what keeps the engine ignorant of
    # any particular model family's prompt conventions.
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    # Per-model sampling defaults. A harness may override them per request.
    sampling: dict[str, Any] = field(default_factory=dict)

    # Descriptive only: shown in the UI, never branched on.
    family: str = ""
    model_page: str = ""
    license: str = ""
    released: str = ""
    params_total: str = ""
    params_active: str = ""
    reasoning: str = ""
    benchmarks: dict[str, Any] = field(default_factory=dict)
    # Per-model overrides of the registry `defaults` block (ctx, threads,
    # n_cpu_moe, ...). Merged under any live override from the sweep.
    engine_opts: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    default: bool = False
    deferred: bool = False
    notes: str = ""

    # Every model in this registry runs on the same engine. Kept as a field so
    # run records stay self-describing and a second backend can be added later
    # without reshaping them.
    engine: str = "llamacpp"

    state: State = State.ABSENT
    error: str | None = None
    progress: Progress = field(default_factory=Progress)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def size_bytes(self) -> int:
        return int(self.size_gb * 1_000_000_000)

    @property
    def files(self) -> list[str]:
        """Repo-relative paths this model needs on disk."""
        return [f for f in (self.gguf, self.mmproj) if f]

    @property
    def hf_url(self) -> str:
        """The repo the weights come from -- where a manual download starts."""
        return f"https://huggingface.co/{self.repo}"

    @property
    def hf_tree_url(self) -> str:
        """The file listing, pinned to the revision this registry expects.

        Deliberately the pinned tree rather than `main`: a GGUF repo can be
        re-quantised in place, and a file downloaded from `main` months later
        is not necessarily the file this registry was measured against.
        """
        return f"https://huggingface.co/{self.repo}/tree/{self.revision}"

    def file_urls(self) -> list[dict[str, str]]:
        """Direct download links for each file, pinned to the revision."""
        return [
            {"name": f,
             "url": f"https://huggingface.co/{self.repo}/resolve/"
                    f"{self.revision}/{f}?download=true"}
            for f in self.files
        ]

    def missing_files(self, models_dir: Path) -> list[str]:
        """Which declared files are not yet on disk."""
        d = self.dir(models_dir)
        return [f for f in self.files if not (d / f).exists()]

    def dir(self, models_dir: Path) -> Path:
        return models_dir / self.id

    def gguf_path(self, models_dir: Path) -> Path:
        return self.dir(models_dir) / self.gguf

    def mmproj_path(self, models_dir: Path) -> Path | None:
        return self.dir(models_dir) / self.mmproj if self.mmproj else None

    def transition(self, new: State, error: str | None = None) -> None:
        if new is not self.state and new not in ALLOWED[self.state]:
            raise TransitionError(f"{self.id}: {self.state.value} -> {new.value}")
        self.state = new
        # Clear a stale message whenever we leave the error state.
        self.error = error if new is State.ERROR else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "engine": self.engine,
            "repo": self.repo,
            "gguf": self.gguf,
            "mmproj": self.mmproj,
            "size_gb": self.size_gb,
            "tools": self.tools,
            "vision": self.vision,
            "gpu": self.gpu,
            "context_ceiling": self.context_ceiling,
            "spec": self.spec,
            "default": self.default,
            "family": self.family,
            "model_page": self.model_page,
            "license": self.license,
            "released": self.released,
            "params_total": self.params_total,
            "params_active": self.params_active,
            "reasoning": self.reasoning,
            "benchmarks": self.benchmarks,
            "sampling": self.sampling,
            "hf_url": self.hf_url,
            "hf_tree_url": self.hf_tree_url,
            "files": self.files,
            "file_urls": self.file_urls(),
            "notes": self.notes.strip(),
            "deferred": self.deferred,
            "state": self.state.value,
            "error": self.error,
            "progress": {
                "percent": round(self.progress.percent, 2),
                "downloaded_bytes": self.progress.downloaded_bytes,
                "total_bytes": self.progress.total_bytes,
                "speed_bps": round(self.progress.speed_bps, 1),
                "eta_seconds": self.progress.eta_seconds,
            },
        }


def free_bytes(path: Path) -> int:
    """Free space on the filesystem holding `path`, creating it if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


class InsufficientSpace(RuntimeError):
    """Not enough disk for a download. Message is shown verbatim in the UI."""

    def __init__(self, need_bytes: int, have_bytes: int) -> None:
        self.need_bytes = need_bytes
        self.have_bytes = have_bytes
        super().__init__(
            f"Not enough disk space: need {need_bytes / 1e9:.1f} GB, "
            f"have {have_bytes / 1e9:.1f} GB"
        )


def check_space(entry: ModelEntry, models_dir: Path) -> None:
    """Fail before starting a download that cannot finish.

    Only the *remaining* bytes are required, so resuming a part-finished
    download does not demand room for what is already on disk.
    """
    already = dir_size(entry.dir(models_dir))
    remaining = max(0, entry.size_bytes - already)
    have = free_bytes(models_dir)
    if remaining > have:
        raise InsufficientSpace(remaining, have)


def dir_size(path: Path) -> int:
    """Bytes currently on disk under `path`. Incomplete downloads included."""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            # A file being written can vanish between rglob and stat.
            continue
    return total


class ModelManager:
    """Owns every ModelEntry and the download threads."""

    def __init__(self, registry: dict[str, Any], models_dir: Path) -> None:
        self.models_dir = models_dir
        self.defaults = registry.get("defaults", {})
        self.llamacpp_ref = registry.get("llamacpp_ref", "unknown")
        self.excluded = registry.get("excluded", [])
        self.entries: dict[str, ModelEntry] = {}
        for row in registry["models"]:
            entry = ModelEntry(
                id=row["id"], name=row["name"],
                repo=row["repo"], revision=row["revision"],
                gguf=row["gguf"], mmproj=row.get("mmproj"),
                size_gb=row["size_gb"],
                tools=row["tools"], vision=row["vision"],
                gpu=row.get("gpu", False),
                context_ceiling=row.get("context_ceiling", 65536),
                spec=row.get("spec"),
                chat_template_kwargs=row.get("chat_template_kwargs") or {},
                sampling=row.get("sampling") or {},
                family=row.get("family", ""),
                model_page=row.get("model_page", ""),
                license=row.get("license", ""),
                released=str(row.get("released", "")),
                params_total=str(row.get("params_total", "")),
                params_active=str(row.get("params_active", "")),
                reasoning=row.get("reasoning", ""),
                benchmarks=row.get("benchmarks") or {},
                engine_opts=row.get("engine_opts") or {},
                env={k: str(v) for k, v in (row.get("env") or {}).items()},
                default=bool(row.get("default", False)),
                deferred=bool(row.get("deferred", False)),
                notes=row.get("notes", ""),
            )
            self.entries[entry.id] = entry
        self.refresh_states()

    @property
    def default_id(self) -> str | None:
        """The model the UI should preselect."""
        for entry in self.entries.values():
            if entry.default:
                return entry.id
        return next(iter(self.entries), None)

    def refresh_states(self) -> None:
        """Set each model's state from what is actually on disk."""
        for entry in self.entries.values():
            if entry.state in (State.DOWNLOADING, State.VERIFYING,
                               State.LOADING, State.LOADED):
                continue  # something is actively working on it
            marker = entry.dir(self.models_dir) / ".complete"
            entry.state = State.READY if marker.exists() else State.ABSENT
            entry.progress.downloaded_bytes = dir_size(entry.dir(self.models_dir))
            entry.progress.total_bytes = entry.size_bytes

    def get(self, model_id: str) -> ModelEntry:
        if model_id not in self.entries:
            raise KeyError(f"unknown model: {model_id}")
        return self.entries[model_id]

    def cancel_download(self, model_id: str) -> None:
        self.get(model_id)._cancel.set()

    def download(
        self,
        model_id: str,
        token: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Download one model's files. Blocking; callers run it on a thread.

        Resumable: huggingface_hub skips files already present and complete, so
        an interrupted download continues rather than restarting.
        """
        entry = self.get(model_id)
        emit = on_event or (lambda _e: None)

        try:
            check_space(entry, self.models_dir)
        except InsufficientSpace as exc:
            entry.transition(State.ERROR, str(exc))
            emit({"type": "model_error", "id": entry.id, "error": str(exc)})
            return

        entry._cancel.clear()
        entry.transition(State.DOWNLOADING)
        dest = entry.dir(self.models_dir)
        dest.mkdir(parents=True, exist_ok=True)
        entry.progress = Progress(
            downloaded_bytes=dir_size(dest), total_bytes=entry.size_bytes
        )
        emit({"type": "model_state", "id": entry.id, "state": entry.state.value})

        stop = threading.Event()
        watcher = threading.Thread(
            target=self._watch_progress, args=(entry, dest, stop, emit), daemon=True
        )
        watcher.start()
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=entry.repo,
                revision=entry.revision,
                local_dir=str(dest),
                token=token,
                max_workers=8,
                # THE important argument. These repos carry every quantisation
                # of the model; without this a 27 GB model is a 500 GB pull.
                allow_patterns=entry.files,
            )
            if entry._cancel.is_set():
                entry.transition(State.ABSENT)
                emit({"type": "model_state", "id": entry.id, "state": "absent"})
                return

            entry.transition(State.VERIFYING)
            emit({"type": "model_state", "id": entry.id, "state": "verifying"})
            ok, why = self.verify(entry)
            if not ok:
                entry.transition(State.ERROR, why)
                emit({"type": "model_error", "id": entry.id, "error": why})
                return

            (dest / ".complete").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            entry.transition(State.READY)
            emit({"type": "model_state", "id": entry.id, "state": "ready"})
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI verbatim
            entry.transition(State.ERROR, f"{type(exc).__name__}: {exc}")
            emit({"type": "model_error", "id": entry.id, "error": str(exc)})
        finally:
            stop.set()

    def _watch_progress(
        self, entry: ModelEntry, dest: Path, stop: threading.Event,
        emit: Callable[[dict[str, Any]], None],
    ) -> None:
        """Poll the directory size for progress, speed and ETA."""
        last_bytes = dir_size(dest)
        last_t = time.monotonic()
        while not stop.wait(2.0):
            now = time.monotonic()
            cur = dir_size(dest)
            dt = now - last_t
            if dt > 0:
                # Smoothed so the UI does not jitter between polls.
                inst = max(0.0, (cur - last_bytes) / dt)
                entry.progress.speed_bps = (
                    inst if entry.progress.speed_bps == 0
                    else 0.7 * entry.progress.speed_bps + 0.3 * inst
                )
            entry.progress.downloaded_bytes = cur
            remaining = max(0, entry.size_bytes - cur)
            entry.progress.eta_seconds = (
                remaining / entry.progress.speed_bps
                if entry.progress.speed_bps > 1e3 else None
            )
            last_bytes, last_t = cur, now
            emit({
                "type": "model_progress",
                "id": entry.id,
                "percent": round(entry.progress.percent, 2),
                "speed_bps": entry.progress.speed_bps,
                "eta_seconds": entry.progress.eta_seconds,
            })

    def verify(self, entry: ModelEntry) -> tuple[bool, str]:
        """Integrity check after a download.

        Checks that each declared file is present and that the GGUF really is
        one -- a truncated or HTML-error-page download is the common failure and
        it otherwise surfaces much later as an unreadable engine crash.
        Full hashing is left to huggingface_hub, which already verifies ETags.
        """
        dest = entry.dir(self.models_dir)
        for rel in entry.files:
            path = dest / rel
            if not path.exists():
                return False, f"{rel} missing after download"
            if path.suffix == ".gguf":
                try:
                    with path.open("rb") as fh:
                        if fh.read(4) != b"GGUF":
                            return False, f"{rel} is not a GGUF file (truncated?)"
                except OSError as exc:
                    return False, f"{rel} is unreadable: {exc}"

        actual = dir_size(dest)
        # 2% tolerance: .complete markers, .cache sidecars and Hub rounding.
        if actual < entry.size_bytes * 0.98:
            return (
                False,
                f"incomplete: {actual / 1e9:.1f} GB on disk, "
                f"expected {entry.size_gb:.1f} GB",
            )
        return True, ""
