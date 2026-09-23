"""Run orchestration: load a model, run a harness, stream events, honour the clock.

One run at a time. llama-server is started with --parallel 1, and the whole
point of this app is to measure a single model/harness pair cleanly.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import godot
from . import mcp as mcp_mod
from .config import (
    ASSETS_DIR, MODELS_DIR, PROJECTS_DIR, STATE_DIR, Settings,
)
from .engine import EngineManager, base_url
from .harness import (
    HarnessRun, HarnessSpec, write_goose_config, write_opencode_config,
    write_pi_config,
)
from .models_mgr import ModelManager, State
from .runs import Metrics, Outcome, Phase, RunLog, RunRecord
from .timelimit import ClockState, LimitWatcher, RunClock, SleepInhibitor


class EventBus:
    """Fan-out of run events to any number of SSE subscribers."""

    def __init__(self) -> None:
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._history: list[dict[str, Any]] = []

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=10_000)
        with self._lock:
            self._subs.append(q)
            # Replay so a browser that connects mid-run is not left blank.
            for event in self._history[-500:]:
                q.put_nowait(event)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict[str, Any]) -> None:
        event.setdefault("t", time.time())
        with self._lock:
            self._history.append(event)
            del self._history[:-2000]
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass        # a stalled browser must not block the run

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()


class Orchestrator:
    def __init__(self, models: ModelManager, harnesses: dict[str, Any],
                 settings: Settings) -> None:
        self.models = models
        self.harnesses = harnesses
        self.settings = settings
        self.engine = EngineManager()
        self.bus = EventBus()
        self.runlog = RunLog(STATE_DIR / "runs.jsonl")

        self.record: RunRecord | None = None
        self.clock: RunClock | None = None
        self.harness_run: HarnessRun | None = None
        self.inhibitor: SleepInhibitor | None = None
        self._watcher: LimitWatcher | None = None
        self._thread: threading.Thread | None = None
        self._resume = threading.Event()
        self._stop_requested = threading.Event()
        self._editor_project: str | None = None
        self._lock = threading.Lock()

    # -- helpers -----------------------------------------------------------

    def emit(self, event: dict[str, Any]) -> None:
        self.bus.publish(event)

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def state(self) -> dict[str, Any]:
        return {
            "busy": self.busy,
            "engine": self.engine.status.as_dict(),
            "run": self.record.as_dict() if self.record else None,
            "clock": self.clock.as_dict() if self.clock else None,
            "awaiting_decision": (
                self.clock is not None and self.clock.state is ClockState.PAUSED
            ),
        }

    # -- run ---------------------------------------------------------------

    def start_run(self, *, model_id: str, harness_id: str, project: str,
                  prompt: str, task_id: str | None = None,
                  images: list[str] | None = None) -> RunRecord:
        with self._lock:
            if self.busy:
                raise RuntimeError("a run is already in progress")
            entry = self.models.get(model_id)
            harness = self._harness(harness_id)

            if harness["id"] != "direct" and not entry.tools:
                raise ValueError(
                    f"{entry.name} does not support tool calling, so it cannot "
                    f"drive {harness['name']}. Use Direct chat."
                )
            if images and not entry.vision:
                raise ValueError(
                    f"{entry.name} has no vision path, so screenshots "
                    "cannot reach it. Pick a model with vision, or drop the "
                    "images and rely on text checks."
                )

            limit = (self.settings.time_limit_seconds
                     if self.settings.time_limit_enabled else None)
            self.record = RunRecord(
                prompt=prompt, model_id=model_id, harness_id=harness_id,
                project=project, task_id=task_id,
                llamacpp_ref=self.models.llamacpp_ref,
                harness_version=str(harness.get("version", "")),
                engine_flags=dict(entry.env),
                phase=self.settings.phase,
                godot_toolsets=list(self.settings.godot_toolsets),
                blender_bin=self.settings.blender_bin or None,
                time_limit_seconds=limit,
            )
            self.clock = RunClock.start(limit)
            self._stop_requested.clear()
            self._resume.clear()
            self.bus.clear_history()
            self.runlog.append(self.record)

            self._thread = threading.Thread(
                target=self._run_thread,
                args=(entry, harness, project, prompt, images or []),
                daemon=True,
            )
            self._thread.start()
            return self.record

    def _harness(self, harness_id: str) -> dict[str, Any]:
        for h in self.harnesses["harnesses"]:
            if h["id"] == harness_id:
                return h
        raise KeyError(f"unknown harness: {harness_id}")

    def _run_thread(self, entry, harness, project, prompt, images) -> None:
        record, clock = self.record, self.clock
        assert record is not None and clock is not None
        # Hold the machine awake: these runs are long and unattended, and a
        # suspend mid-prefill throws away everything done so far.
        self.inhibitor = SleepInhibitor(enabled=self.settings.inhibit_sleep)
        self.inhibitor.__enter__()
        self.emit({"type": "run_start", "run": record.as_dict()})
        try:
            # 1. Download if needed. The prompt is queued behind it.
            if entry.state is State.ABSENT:
                self.emit({"type": "phase", "phase": "downloading"})
                self.models.download(
                    entry.id, token=_hf_token(), on_event=self.emit
                )
                if entry.state is not State.READY:
                    raise RuntimeError(entry.error or "download failed")

            # 2. Load the engine if it is not already the loaded one.
            if self.engine.loaded_model_id != entry.id:
                record.metrics.phase = Phase.LOADING
                self.emit({"type": "phase", "phase": "loading",
                           "model": entry.name})
                overrides = dict(self.settings.model_flags.get(entry.id) or {})
                overrides.setdefault("vram_reserve_mb",
                                     self.settings.vram_reserve_mb)
                self.engine.start(
                    entry, self.models.defaults,
                    flag_overrides=overrides,
                    on_log=lambda l: self.emit({"type": "engine_log", "line": l}),
                )
            # Record how the model actually landed, not how it was asked to.
            record.placement = dict(self.engine.status.placement)
            record.llamacpp_build = _llamacpp_build()
            self.emit({"type": "placement", "placement": record.placement})

            # 3. Watch the clock.
            self._watcher = LimitWatcher(clock, self._on_limit_expired)
            self._watcher.start()

            # 4. Run, possibly across several windows if the user continues.
            while True:
                self._execute(entry, harness, project, prompt, images, record, clock)
                if self._stop_requested.is_set():
                    record.outcome = Outcome.STOPPED
                    break
                if clock.state is ClockState.PAUSED:
                    # A benchmark matrix runs unattended, often overnight. If a
                    # cell waited for a human here, one slow cell would block
                    # every cell behind it, so a benchmark run finalises itself
                    # instead of asking. Only manual runs prompt.
                    if record.task_id is not None:
                        self.emit({
                            "type": "time_limit_reached",
                            "run_id": record.id,
                            "elapsed": round(clock.elapsed),
                            "unattended": True,
                            "message": ("Time limit reached on a benchmark cell; "
                                        "output saved and the cell recorded as "
                                        "time_limit."),
                        })
                        record.outcome = Outcome.TIME_LIMIT
                        break

                    # Everything is already flushed; wait for the decision.
                    self.emit({
                        "type": "time_limit_reached",
                        "run_id": record.id,
                        "elapsed": round(clock.elapsed),
                        "unattended": False,
                        "message": ("Time limit reached. Output has been saved. "
                                    "Continue or stop?"),
                    })
                    record.outcome = Outcome.TIME_LIMIT_PAUSED
                    self.runlog.update(record)
                    self._resume.wait()          # no timeout: waits indefinitely
                    self._resume.clear()
                    if self._stop_requested.is_set():
                        record.outcome = Outcome.TIME_LIMIT
                        break
                    record.time_limit_extensions = clock.extensions
                    self._watcher = LimitWatcher(clock, self._on_limit_expired)
                    self._watcher.start()
                    prompt = ("Continue where you left off. The previous "
                              "session was paused at a time limit.")
                    continue
                # A run that emitted nothing has not succeeded, whatever the
                # transport did. Without this, an engine that accepts the
                # request and returns an empty stream is recorded as a pass.
                if (record.metrics.decode_tokens == 0
                        and record.metrics.tool_calls == 0):
                    record.outcome = Outcome.FAIL
                    record.error = record.error or (
                        "the model produced no output and made no tool calls")
                else:
                    record.outcome = Outcome.PASS
                break
        except Exception as exc:                 # noqa: BLE001
            record.outcome = Outcome.ERROR
            record.error = f"{type(exc).__name__}: {exc}"
            self.emit({"type": "error", "error": record.error})
        finally:
            if self._watcher:
                self._watcher.stop()
            clock.finish()
            record.ended_at = time.time()
            record.metrics.phase = Phase.DONE
            self.runlog.update(record)
            if self.inhibitor:
                self.inhibitor.release()
            self.emit({"type": "run_end", "run": record.as_dict()})

    def _execute(self, entry, harness, project, prompt, images,
                 record: RunRecord, clock: RunClock) -> None:
        """One window of work: configure, launch, stream until done or paused."""
        project_dir = PROJECTS_DIR / project if project else PROJECTS_DIR
        if harness["id"] == "direct":
            self._run_direct(entry, prompt, images, record, clock)
            return

        # Scene tools act on the live editor, so it has to be open on THIS
        # project before the harness starts. Only needed when MCP is attached;
        # without it the agent never touches the editor.
        if self.settings.mcp_enabled:
            self._ensure_editor(project_dir.name)

        spec = HarnessSpec(
            id=harness["id"], name=harness["name"], bin=harness["bin"],
            run=list(harness["run"]), protocol=harness.get("protocol", "openai"),
        )
        env = self._configure(harness, entry, project_dir)
        self.harness_run = HarnessRun(spec, project_dir, prompt, env=env)
        record.metrics.phase = Phase.PREFILL
        self.emit({"type": "phase", "phase": "prefill"})

        started = time.monotonic()
        first_token_seen = False
        self.harness_run.start()
        for event in self.harness_run.events():
            if event["type"] == "text" and event.get("text"):
                if not first_token_seen:
                    first_token_seen = True
                    record.metrics.ttft_seconds = round(time.monotonic() - started, 2)
                    record.metrics.phase = Phase.DECODE
                    self.emit({"type": "phase", "phase": "decode",
                               "ttft": record.metrics.ttft_seconds})
                record.metrics.decode_tokens += max(1, len(event["text"]) // 4)
            elif event["type"] == "tool_call":
                record.metrics.tool_calls += 1
                if record.metrics.time_to_first_tool_call is None:
                    record.metrics.time_to_first_tool_call = round(
                        time.monotonic() - started, 2)
                record.metrics.phase = Phase.TOOL
            elif event["type"] == "tool_result" and event.get("is_error"):
                record.metrics.tool_errors += 1
            elif event["type"] == "error":
                record.error = event.get("error")

            self.emit({"type": "event", "run_id": record.id, "event": event})
            self.emit({"type": "metrics", "metrics": record.metrics.as_dict(),
                       "clock": clock.as_dict()})
            if clock.state is ClockState.PAUSED or self._stop_requested.is_set():
                break

        for line in self.harness_run.stderr_lines[-200:]:
            self.emit({"type": "stderr", "line": line})
        self.harness_run.stop()
        self.harness_run = None

    def _run_direct(self, entry, prompt, images, record, clock) -> None:
        """Baseline: talk to llama-server with no harness, no tools."""
        import httpx

        content: Any = prompt
        if images and entry.vision:
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": img}} for img in images
            ]
        body = {
            "model": entry.id,
            "messages": [{"role": "user", "content": content}],
            "stream": True,
        }
        record.metrics.phase = Phase.PREFILL
        self.emit({"type": "phase", "phase": "prefill"})
        started = time.monotonic()
        first = False
        # No read timeout: prefill legitimately produces nothing for a long time.
        timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=None)
        with httpx.stream("POST", f"{base_url()}/v1/chat/completions",
                          json=body, timeout=timeout) as resp:
            # A non-200 body is not SSE, so the loop below would skip every
            # line and the run would be recorded as a clean pass that produced
            # nothing. Fail on the status instead.
            if resp.status_code != 200:
                detail = resp.read().decode(errors="replace")[:500]
                raise RuntimeError(
                    f"llama-server returned HTTP {resp.status_code}: {detail}")
            for line in resp.iter_lines():
                if clock.state is ClockState.PAUSED or self._stop_requested.is_set():
                    break
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                text = delta.get("content") or ""
                if text:
                    if not first:
                        first = True
                        record.metrics.ttft_seconds = round(
                            time.monotonic() - started, 2)
                        record.metrics.phase = Phase.DECODE
                        self.emit({"type": "phase", "phase": "decode",
                                   "ttft": record.metrics.ttft_seconds})
                    record.metrics.decode_tokens += 1
                    self.emit({"type": "event", "run_id": record.id,
                               "event": {"type": "text", "text": text}})
                usage = chunk.get("usage")
                if usage:
                    record.metrics.tokens_in = usage.get("prompt_tokens", 0)
                    record.metrics.tokens_out = usage.get("completion_tokens", 0)
                    record.metrics.prefill_tokens = record.metrics.tokens_in
                    # Prefill rate is the prompt divided by the time spent
                    # before the first token: on these engines that is the
                    # number that explains a long silence.
                    if record.metrics.ttft_seconds:
                        record.metrics.prefill_tps = round(
                            record.metrics.tokens_in / record.metrics.ttft_seconds, 3)
        elapsed = time.monotonic() - started
        # A rate needs at least one interval between tokens. With a single
        # token the divisor is the sub-second tail after TTFT, which produced
        # nonsense like "7.2 tok/s" on a one-word reply.
        if record.metrics.decode_tokens >= 2 and record.metrics.ttft_seconds is not None:
            decode_time = max(1e-6, elapsed - record.metrics.ttft_seconds)
            record.metrics.decode_tps = round(
                (record.metrics.decode_tokens - 1) / decode_time, 3)
        self.emit({"type": "metrics", "metrics": record.metrics.as_dict(),
                   "clock": clock.as_dict()})

    def _ensure_editor(self, project: str, wait_seconds: int = 180) -> bool:
        """Open the editor on `project` and wait for its addon to connect."""
        from . import games
        if games.editor_project() == project:
            self._editor_project = project
            return True
        try:
            games.request_editor(project)
        except FileNotFoundError as exc:
            self.emit({"type": "warning", "warning": str(exc)})
            return False
        self.emit({"type": "phase", "phase": "opening editor",
                   "project": project})
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if games.editor_project() == project:
                self._editor_project = project
                self.emit({"type": "editor_ready", "project": project})
                return True
            time.sleep(2.0)
        # Not fatal: script and runtime tools still work on the filesystem, and
        # saying so is more useful than refusing to start.
        self.emit({"type": "warning", "warning": (
            f"the Godot editor did not connect for {project} within "
            f"{wait_seconds}s. Scene-editing tools will fail. Is "
            f"scripts/play-watcher.sh running on the host?")})
        return False

    def _configure(self, harness, entry, project_dir: Path) -> dict[str, str]:
        """Write the harness's config and return any env it needs.

        This is where the phase split is enforced. The agent is given Godot's
        tools or Blender's, not both, unless the operator explicitly asked for
        `both` -- because together they are 40-60k tokens of schema and the
        context window is 64k. See RAM_AGENT.md.
        """
        url = base_url()
        env: dict[str, str] = {"OPENAI_API_KEY": "local"}

        servers: dict[str, mcp_mod.ServerSpec] = {}
        if self.settings.mcp_enabled:
            # A model with no vision cannot use the screenshot tools, so they
            # are dropped rather than offered and then failing mid-run.
            servers = mcp_mod.servers_for(
                self.settings.phase,
                project_dir=project_dir,
                assets_dir=ASSETS_DIR,
                godot_toolsets=list(self.settings.godot_toolsets),
                blender_bin=self.settings.blender_bin or None,
                vision=entry.vision,
            )

        record = self.record
        if record is not None:
            ctx = int(self.models.defaults.get("ctx", 65536))
            if entry.context_ceiling:
                ctx = min(ctx, entry.context_ceiling)
            record.mcp_servers = sorted(servers)
            record.tool_budget = mcp_mod.budget(servers, ctx)
            self.emit({"type": "tool_budget", "budget": record.tool_budget})
            if record.tool_budget.get("warning"):
                self.emit({"type": "warning",
                           "message": record.tool_budget["warning"]})
            for sid, spec in servers.items():
                if not spec.verified:
                    self.emit({
                        "type": "warning",
                        "message": (
                            f"MCP server {sid!r} has not been verified against "
                            "an installed binary; its invocation in mcp.yaml is "
                            "the documented shape, not a measured one."
                        ),
                    })

        if harness["id"] == "opencode":
            write_opencode_config(project_dir, entry.id, url, servers,
                                  mcp=self.settings.mcp_enabled)
        elif harness["id"] == "goose":
            cfg_home = Path.home() / ".config" / "goose"
            write_goose_config(cfg_home, entry.id, url, servers,
                               mcp=self.settings.mcp_enabled)
            env.update({"GOOSE_MODE": "auto", "OPENAI_HOST": url,
                        "OPENAI_API_KEY": "local"})
        elif harness["id"] == "pi":
            write_pi_config(project_dir, entry.id, url, servers,
                            mcp=self.settings.mcp_enabled)
            env.update({"PI_BASE_URL": f"{url}/v1", "PI_API_KEY": "local",
                        "PI_MODEL": entry.id})
        return env

    # -- control -----------------------------------------------------------

    def _on_limit_expired(self) -> None:
        """Stop generating, but keep everything produced so far."""
        if self.clock is None:
            return
        self.clock.pause()
        self.emit({"type": "phase", "phase": "paused"})
        if self.harness_run is not None:
            self.harness_run.stop()
        if self.record is not None:
            self.record.time_limit_extensions = self.clock.extensions
            self.runlog.update(self.record)

    def continue_run(self, extra_seconds: int | None = None) -> None:
        if self.clock is None or self.clock.state is not ClockState.PAUSED:
            raise RuntimeError("no paused run to continue")
        self.clock.extend(extra_seconds)
        self._resume.set()

    def stop_run(self) -> None:
        self._stop_requested.set()
        if self.harness_run is not None:
            self.harness_run.stop()
        self._resume.set()      # release a paused run so the thread can finish

    def switch_model(self, model_id: str) -> None:
        """Stop whatever is running and load a different model."""
        self.stop_run()
        self.engine.stop()


def _hf_token() -> str | None:
    from .config import hf_token
    return hf_token()


def _llamacpp_build() -> str:
    """The engine build actually in use, for the run record.

    A benchmark whose engine moved under it is not a benchmark, and llama.cpp
    moves fast. This reads what is installed rather than trusting the pin in
    models.yaml, so a rebuilt image cannot silently invalidate a table.
    """
    import subprocess

    from .engine import LLAMA_SERVER

    try:
        out = subprocess.run([LLAMA_SERVER, "--version"], capture_output=True,
                             text=True, timeout=30)
        text = (out.stdout + out.stderr).strip()
        for line in text.splitlines():
            if "version" in line.lower() or "build" in line.lower():
                return line.strip()
        return text.splitlines()[0] if text else "unknown"
    except (OSError, IndexError, subprocess.SubprocessError):
        return "unknown"
