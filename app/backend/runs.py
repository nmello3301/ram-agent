"""Run records, live metrics and history.

Every run is appended to state/runs.jsonl as one JSON object per line, so the
history survives container restarts and can be read with anything.

The metrics exist because of what these engines cost. Prefill on a CPU engine
runs at a few tokens per second, so a harness's baseline prompt can be an hour
of silence before the first token. Reporting only wall time and pass/fail would
make every slow combination look identical; TTFT, the baseline prompt size and
the time to the first tool call are what actually distinguish them.
"""
from __future__ import annotations

import csv
import io
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class Outcome(str, Enum):
    RUNNING = "running"
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    STOPPED = "stopped"              # stopped by the user
    TIME_LIMIT = "time_limit"        # limit hit and the user chose to stop
    TIME_LIMIT_PAUSED = "time_limit_paused"   # limit hit, awaiting a decision


class Phase(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    PREFILL = "prefill"
    DECODE = "decode"
    TOOL = "tool"
    DONE = "done"


@dataclass
class Metrics:
    """Live counters for one run."""

    phase: Phase = Phase.IDLE
    ttft_seconds: float | None = None          # to first assistant token
    time_to_first_tool_call: float | None = None
    prefill_tokens: int = 0
    prefill_tps: float | None = None
    decode_tokens: int = 0
    decode_tps: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    retries: int = 0
    rss_bytes: int = 0
    # Size of the harness's system prompt + tool schemas, measured once.
    baseline_prompt_tokens: int | None = None
    # Prefill cost of a single screenshot, when one is sent.
    screenshot_tokens: int | None = None
    screenshot_prefill_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["phase"] = self.phase.value
        return d


@dataclass
class RunRecord:
    """Everything worth knowing about one run."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    prompt: str = ""
    model_id: str = ""
    harness_id: str = ""
    project: str = ""
    task_id: str | None = None          # set for benchmark runs

    # Provenance, so a number in BENCHMARKS.md can be traced back.
    llamacpp_ref: str = ""
    llamacpp_build: str = ""
    harness_version: str = ""
    engine_flags: dict[str, Any] = field(default_factory=dict)

    # Which editor the agent drove, and the tool surface it was given.
    phase: str = "godot"
    mcp_servers: list[str] = field(default_factory=list)
    godot_toolsets: list[str] = field(default_factory=list)
    # What the tool schemas cost as a share of the context window. Colibri's
    # most expensive mistake was that this number existed and was never
    # recorded: three 6 h runs produced nothing because a 15,074-token
    # preamble was never priced until afterwards.
    tool_budget: dict[str, Any] = field(default_factory=dict)
    # How the model was actually split across GPU and RAM. The requested
    # configuration and the resolved one are routinely different on a 6 GB
    # card, and only the resolved one explains a slow run.
    placement: dict[str, Any] = field(default_factory=dict)

    time_limit_seconds: int | None = None
    time_limit_extensions: int = 0

    outcome: Outcome = Outcome.RUNNING
    error: str | None = None
    metrics: Metrics = field(default_factory=Metrics)
    checks: dict[str, bool] = field(default_factory=dict)   # benchmark checks

    @property
    def wall_seconds(self) -> float:
        return (self.ended_at or time.time()) - self.started_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "wall_seconds": round(self.wall_seconds, 2),
            "prompt": self.prompt,
            "model_id": self.model_id,
            "harness_id": self.harness_id,
            "project": self.project,
            "task_id": self.task_id,
            "llamacpp_ref": self.llamacpp_ref,
            "llamacpp_build": self.llamacpp_build,
            "harness_version": self.harness_version,
            "engine_flags": self.engine_flags,
            "phase": self.phase,
            "mcp_servers": self.mcp_servers,
            "godot_toolsets": self.godot_toolsets,
            "tool_budget": self.tool_budget,
            "placement": self.placement,
            "time_limit_seconds": self.time_limit_seconds,
            "time_limit_extensions": self.time_limit_extensions,
            "outcome": self.outcome.value,
            "error": self.error,
            "metrics": self.metrics.as_dict(),
            "checks": self.checks,
        }


class RunLog:
    """Append-only JSONL history, with a CSV export for the benchmark."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: RunRecord) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(record.as_dict()) + "\n")

    def update(self, record: RunRecord) -> None:
        """Rewrite the record in place, matched by id.

        Runs are long and can be paused and resumed, so a record is written
        when it starts and refreshed as it changes; a crash still leaves the
        last written version on disk.
        """
        rows = [r for r in self.read_all() if r.get("id") != record.id]
        rows.append(record.as_dict())
        rows.sort(key=lambda r: r.get("started_at", 0))
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        tmp.replace(self.path)

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue      # a torn line from a hard kill must not break history
        return out

    def to_csv(self) -> str:
        rows = self.read_all()
        columns = [
            "id", "started_at", "model_id", "harness_id", "task_id", "project",
            "outcome", "wall_seconds", "time_limit_seconds",
            "time_limit_extensions",
            "ttft_seconds", "time_to_first_tool_call",
            "baseline_prompt_tokens", "prefill_tokens", "prefill_tps",
            "decode_tokens", "decode_tps", "tokens_in", "tokens_out",
            "tool_calls", "tool_errors", "retries",
            "screenshot_tokens", "screenshot_prefill_seconds",
            "llamacpp_ref", "harness_version", "checks_passed", "checks_total",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            m = r.get("metrics", {})
            checks = r.get("checks", {}) or {}
            writer.writerow({
                **{k: r.get(k) for k in columns if k in r},
                **{k: m.get(k) for k in m},
                "checks_passed": sum(1 for v in checks.values() if v),
                "checks_total": len(checks),
            })
        return buf.getvalue()


def parse_v41_stats(line: str) -> dict[str, Any] | None:
    """Read a DeepSeek V4.1 `V41_STATS` line from the engine's stderr.

    The engine prints per-turn accounting when V41_STATS is set: expert bytes
    and their rate, the n-gram cache, and how many drafts were accepted. It is
    the only direct view of what the disk is doing during a turn.
    """
    if "expert" not in line.lower() and "tok/s" not in line.lower():
        return None
    out: dict[str, Any] = {}
    for key, pattern in (
        ("tok_s", r"([0-9.]+)\s*tok/s"),
        ("expert_mb", r"([0-9.]+)\s*MB.*expert"),
        ("accepted", r"accepted[:= ]+([0-9]+)"),
    ):
        import re
        m = re.search(pattern, line, re.I)
        if m:
            out[key] = float(m.group(1))
    return out or None
