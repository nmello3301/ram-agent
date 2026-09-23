"""Run clock and sleep inhibition.

Two jobs:

1. Enforce a per-run time limit that has no hard ceiling and that, when it
   expires, *pauses* the run rather than discarding it. Everything produced is
   flushed first, then the UI is asked whether to continue or stop. Continuing
   extends the window and resumes the same harness session, so llama.cpp's prefix
   cache and the harness's own context are reused instead of re-prefilled.

2. Keep the machine awake for as long as a run is in flight. Runs are long and
   unattended; a suspend mid-run loses hours of prefill.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class ClockState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"       # limit hit, waiting for continue/stop
    FINISHED = "finished"


@dataclass
class RunClock:
    """Tracks elapsed time against a limit that can be extended or removed.

    `limit_seconds` of None means unlimited. There is no maximum: a limit of
    30 hours is as valid as one of 30 minutes.
    """

    limit_seconds: int | None
    started_at: float
    # Time already banked from earlier windows, when a run has been resumed.
    banked: float = 0.0
    state: ClockState = ClockState.RUNNING
    extensions: int = 0
    # The size of one window, remembered so that "Continue" can grant another
    # window of the same length rather than whatever is left of the first.
    window_seconds: int | None = None

    @classmethod
    def start(cls, limit_seconds: int | None) -> "RunClock":
        return cls(limit_seconds=limit_seconds, started_at=time.monotonic(),
                   window_seconds=limit_seconds)

    @property
    def elapsed(self) -> float:
        if self.state is ClockState.RUNNING:
            return self.banked + (time.monotonic() - self.started_at)
        return self.banked

    @property
    def remaining(self) -> float | None:
        if self.limit_seconds is None:
            return None
        return self.limit_seconds - self.elapsed

    def expired(self) -> bool:
        rem = self.remaining
        return rem is not None and rem <= 0 and self.state is ClockState.RUNNING

    def pause(self) -> None:
        """Bank the elapsed time and stop the clock."""
        if self.state is ClockState.RUNNING:
            self.banked = self.elapsed
            self.state = ClockState.PAUSED

    def extend(self, extra_seconds: int | None = None) -> None:
        """Resume, giving the run another full window.

        `extra_seconds` of None reuses the original window size, so "Continue"
        grants the same duration again.

        The new limit is measured from the time already banked, not added to
        the old limit. Expiry is noticed on a poll, so a run can overshoot its
        limit slightly before it is paused; adding to the old limit would
        silently deduct that overshoot from the next window.
        """
        if self.limit_seconds is not None:
            extra = extra_seconds or self.window_seconds or self.limit_seconds
            self.limit_seconds = int(self.elapsed + extra)
        self.started_at = time.monotonic()
        self.state = ClockState.RUNNING
        self.extensions += 1

    def finish(self) -> None:
        if self.state is ClockState.RUNNING:
            self.banked = self.elapsed
        self.state = ClockState.FINISHED

    def as_dict(self) -> dict:
        return {
            "state": self.state.value,
            "elapsed": round(self.elapsed, 1),
            "remaining": round(self.remaining, 1) if self.remaining is not None else None,
            "limit_seconds": self.limit_seconds,
            "extensions": self.extensions,
        }


class SleepInhibitor:
    """Holds the machine awake for the lifetime of a run.

    Two mechanisms, because they cover different things:
      * systemd-inhibit blocks idle, suspend and the lid switch;
      * hypridle is Hyprland's own idle daemon and does not consult logind
        inhibitors for its own timers, so it is paused separately.

    Both are released on exit, including on a crash. A benchmark run should not
    leave a laptop that never sleeps again.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._proc: subprocess.Popen | None = None
        self._paused_hypridle = False

    def __enter__(self) -> "SleepInhibitor":
        if not self.enabled:
            return self
        if shutil.which("systemd-inhibit"):
            try:
                # Sleeps forever; killed on release. The inhibitor lives as long
                # as this child process does.
                self._proc = subprocess.Popen(
                    ["systemd-inhibit", "--what=idle:sleep:handle-lid-switch",
                     "--who=ram-agent", "--why=long model run in progress",
                     "--mode=block", "sleep", "infinity"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except OSError:
                self._proc = None
        self._pause_hypridle()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()

    def _pause_hypridle(self) -> None:
        if not shutil.which("systemctl"):
            return
        try:
            active = subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet", "hypridle"],
                timeout=10,
            ).returncode == 0
            if active:
                subprocess.run(["systemctl", "--user", "stop", "hypridle"],
                               timeout=10, check=False)
                self._paused_hypridle = True
        except (OSError, subprocess.SubprocessError):
            pass

    def release(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._paused_hypridle and shutil.which("systemctl"):
            subprocess.run(["systemctl", "--user", "start", "hypridle"],
                           timeout=10, check=False)
            self._paused_hypridle = False

    @property
    def active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


class LimitWatcher(threading.Thread):
    """Calls `on_expire` once, the moment the clock runs out."""

    def __init__(self, clock: RunClock, on_expire: Callable[[], None],
                 poll_seconds: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.clock = clock
        self.on_expire = on_expire
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            if self.clock.state is ClockState.FINISHED:
                return
            if self.clock.expired():
                self.on_expire()
                return

    def stop(self) -> None:
        self._stop.set()
