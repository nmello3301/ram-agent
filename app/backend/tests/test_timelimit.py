"""Run clock: parametric limit, expiry, pause and resume."""
from __future__ import annotations

import time

import pytest

from backend.timelimit import ClockState, RunClock


@pytest.fixture
def fake_clock(monkeypatch):
    """A settable monotonic clock.

    Patching `time.monotonic` with a lambda that reads `clock.started_at` is
    self-referential once `extend()` reassigns it, so the tests drive an
    explicit counter instead.
    """
    now = {"t": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: now["t"])
    return now


class TestRunClock:
    def test_unlimited_never_expires(self):
        c = RunClock.start(None)
        assert c.remaining is None
        assert not c.expired()

    def test_expires_once_the_limit_passes(self, fake_clock):
        c = RunClock.start(10)
        fake_clock["t"] += 11
        assert c.expired()

    def test_not_expired_before_the_limit(self, fake_clock):
        c = RunClock.start(10)
        fake_clock["t"] += 9
        assert not c.expired()

    def test_accepts_a_limit_far_beyond_three_hours(self):
        """There is no ceiling: a 30 h run is as valid as a 30 min one."""
        c = RunClock.start(30 * 3600)
        assert c.limit_seconds == 108_000
        assert not c.expired()

    def test_pause_banks_elapsed_and_stops_the_clock(self, fake_clock):
        c = RunClock.start(100)
        fake_clock["t"] += 40
        c.pause()
        assert c.state is ClockState.PAUSED
        assert round(c.elapsed) == 40
        # Time passing while paused must not count against the run.
        fake_clock["t"] += 360
        assert round(c.elapsed) == 40

    def test_continue_grants_another_full_window(self, fake_clock):
        c = RunClock.start(100)
        fake_clock["t"] += 101          # overshot the limit before the poll
        assert c.expired()
        c.pause()
        c.extend()
        assert c.state is ClockState.RUNNING
        assert c.extensions == 1
        assert not c.expired()
        # A full 100 s more, not 99: the overshoot is not deducted.
        assert round(c.remaining) == 100

    def test_extend_by_an_explicit_amount(self, fake_clock):
        c = RunClock.start(100)
        fake_clock["t"] += 100
        c.pause()
        c.extend(3600)
        assert round(c.remaining) == 3600

    def test_repeated_extensions_each_give_a_full_window(self, fake_clock):
        c = RunClock.start(60)
        for i in range(3):
            fake_clock["t"] += 61
            c.pause()
            c.extend()
            assert round(c.remaining) == 60, f"extension {i}"
        assert c.extensions == 3

    def test_finish_freezes_elapsed(self, fake_clock):
        c = RunClock.start(100)
        fake_clock["t"] += 20
        c.finish()
        assert c.state is ClockState.FINISHED
        fake_clock["t"] += 880
        assert round(c.elapsed) == 20

    def test_expired_is_false_once_finished(self, fake_clock):
        c = RunClock.start(1)
        fake_clock["t"] += 100
        c.finish()
        assert not c.expired()

    def test_serialises_for_the_ui(self):
        d = RunClock.start(7200).as_dict()
        assert d["limit_seconds"] == 7200
        assert d["state"] == "running"
        assert d["remaining"] is not None


class TestDecodeRateGuard:
    """A decode rate needs at least one interval between tokens.

    Measured directly: a one-word reply reported "7.196 tok/s" on an engine
    whose real rate is around 0.6, because the divisor was the sub-second tail
    after TTFT.
    """

    @staticmethod
    def rate(decode_tokens: int, elapsed: float, ttft: float | None):
        if decode_tokens >= 2 and ttft is not None:
            return round((decode_tokens - 1) / max(1e-6, elapsed - ttft), 3)
        return None

    def test_single_token_reports_no_rate(self):
        assert self.rate(1, 72.9, 72.8) is None

    def test_zero_tokens_reports_no_rate(self):
        assert self.rate(0, 10.0, None) is None

    def test_many_tokens_uses_intervals_not_count(self):
        # 61 tokens over 100 s after TTFT is 60 intervals, not 61.
        assert self.rate(61, 110.0, 10.0) == 0.6
