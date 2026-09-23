"""Harness integration: config generation, non-interactive runs, output parsing.

Every harness is pointed at the local llama-server endpoint over the
**OpenAI** protocol, which is the one llama.cpp serves natively and the one
every harness here speaks.

The requirement that a screenshot must reach the model is unchanged from
Colibri -- it is now more important, not less, because the default model can
actually see. Images travel as content blocks on /v1/chat/completions.

Each harness prints something different. The parsers below normalise all of it
into one event shape so the UI and the benchmark do not care which harness ran:

    {"type": "text"       , "text": ...}
    {"type": "tool_call"  , "name": ..., "args": ..., "id": ...}
    {"type": "tool_result", "id": ..., "result": ..., "is_error": bool}
    {"type": "stderr"     , "line": ...}
    {"type": "raw"        , "line": ...}   # unparseable, shown not swallowed
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:            # avoids a cycle: mcp imports config, not harness
    from .mcp import ServerSpec

# A dummy key satisfies clients that refuse to start without one; llama-server
# only enforces a key when --api-key is passed.
DUMMY_KEY = "local"


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def parse_opencode_line(line: str) -> dict[str, Any] | None:
    """OpenCode prints JSON objects per event when run with --print-logs."""
    line = line.strip()
    if not line:
        return None
    if not line.startswith("{"):
        return {"type": "raw", "line": line}
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        # Malformed output must be shown, never crash the run.
        return {"type": "raw", "line": line}
    kind = obj.get("type") or obj.get("event")
    if kind in ("text", "message", "assistant"):
        return {"type": "text", "text": obj.get("text") or obj.get("content") or ""}
    if kind in ("tool_call", "tool-call", "tool_use"):
        return {
            "type": "tool_call",
            "name": obj.get("name") or obj.get("tool") or "?",
            "args": obj.get("args") or obj.get("input") or {},
            "id": obj.get("id") or obj.get("callId"),
        }
    if kind in ("tool_result", "tool-result"):
        return {
            "type": "tool_result",
            "id": obj.get("id") or obj.get("callId"),
            "result": obj.get("result") or obj.get("output") or "",
            "is_error": bool(obj.get("isError") or obj.get("is_error")),
        }
    if kind == "error":
        return {"type": "error", "error": obj.get("message") or str(obj)}
    return {"type": "raw", "line": line}


def parse_pi_line(line: str) -> dict[str, Any] | None:
    """Pi's `--json` print mode emits one JSON event per line."""
    line = line.strip()
    if not line:
        return None
    if not line.startswith("{"):
        return {"type": "raw", "line": line}
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return {"type": "raw", "line": line}
    kind = obj.get("type")
    if kind in ("assistant", "text", "content"):
        text = obj.get("text") or obj.get("content") or ""
        if isinstance(text, list):     # content blocks
            text = "".join(
                b.get("text", "") for b in text if isinstance(b, dict)
            )
        return {"type": "text", "text": text}
    if kind in ("tool", "tool_call", "toolCall"):
        return {
            "type": "tool_call",
            "name": obj.get("name") or "?",
            "args": obj.get("arguments") or obj.get("args") or {},
            "id": obj.get("id"),
        }
    if kind in ("tool_result", "toolResult"):
        return {
            "type": "tool_result",
            "id": obj.get("id"),
            "result": obj.get("result") or obj.get("output") or "",
            "is_error": bool(obj.get("isError") or obj.get("is_error")),
        }
    if kind == "error":
        return {"type": "error", "error": obj.get("message") or str(obj)}
    return {"type": "raw", "line": line}


_GOOSE_TOOL_RE = re.compile(r"^\s*(?:─+\s*)?(\w[\w.-]*)\s*\|\s*(.*)$")


def parse_goose_line(line: str) -> dict[str, Any] | None:
    """Goose prints human-readable text with tool banners, not JSON.

    It has no machine-readable run output, so this is a best-effort reader:
    anything not recognised is passed through as `raw` rather than dropped.
    """
    stripped = line.rstrip("\n")
    if not stripped.strip():
        return None
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if obj.get("type") == "error":
                return {"type": "error", "error": obj.get("message", "")}
        except json.JSONDecodeError:
            pass
    low = stripped.lower()
    if low.startswith(("◓", "─── ", "--- ")) or " | " in stripped[:60]:
        m = _GOOSE_TOOL_RE.match(stripped.lstrip("◓─- "))
        if m:
            return {"type": "tool_call", "name": m.group(1),
                    "args": m.group(2), "id": None}
    if low.startswith(("error", "failed", "panic")):
        return {"type": "error", "error": stripped}
    return {"type": "text", "text": stripped}


PARSERS: dict[str, Callable[[str], dict[str, Any] | None]] = {
    "opencode": parse_opencode_line,
    "pi": parse_pi_line,
    "goose": parse_goose_line,
}


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _mcp_entries(servers: dict[str, "ServerSpec"]) -> dict[str, Any]:
    """Normalise ServerSpecs into the shape each harness config wants."""
    return {
        sid: {"command": spec.command, "env": dict(spec.env),
              "timeout_ms": spec.timeout_ms}
        for sid, spec in servers.items()
    }


def write_opencode_config(project_dir: Path, model_id: str, base_url: str,
                          servers: dict[str, "ServerSpec"] | None = None,
                          mcp: bool = True) -> Path:
    entries = _mcp_entries(servers or {})
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "llamacpp": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "llama.cpp (local)",
                "options": {
                    "baseURL": f"{base_url}/v1",
                    "apiKey": DUMMY_KEY,
                    # All three default to 5 minutes and all three abort the
                    # request when it goes quiet.
                    #
                    # Under Colibri this was fatal and non-obvious: a cold
                    # prefill ran for an hour emitting nothing, so every turn
                    # was aborted mid-flight and three 6 h runs died at ~31
                    # minutes with zero tool calls.
                    #
                    # With a resident model a turn is seconds and these would
                    # very likely never fire. They stay disabled anyway: a
                    # single Blender render or a headless Godot import can
                    # still outlast five minutes of silence, and the backend
                    # owns the clock. This costs nothing to keep.
                    #   headerTimeout - waiting for response headers
                    #   chunkTimeout  - gap between streamed SSE chunks
                    #   timeout       - the whole request
                    "timeout": False,
                    "headerTimeout": False,
                    "chunkTimeout": False,
                },
                "models": {model_id: {"name": model_id}},
            }
        },
        "model": f"llamacpp/{model_id}",
        # Full auto-approve: no confirmation prompts of any kind.
        "permission": {"edit": "allow", "bash": "allow", "webfetch": "allow"},
    }
    # Omitted entirely rather than set to enabled:false -- OpenCode still
    # loads a disabled entry's tools, so the key has to be absent.
    if mcp and entries:
        cfg["mcp"] = {
            sid: {
                "type": "local",
                "enabled": True,
                "command": e["command"],
                "environment": e["env"],
                "timeout": e["timeout_ms"],
            }
            for sid, e in entries.items()
        }
    path = project_dir / "opencode.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def write_goose_config(config_home: Path, model_id: str, base_url: str,
                       servers: dict[str, "ServerSpec"] | None = None,
                       mcp: bool = True) -> Path:
    import yaml

    entries = _mcp_entries(servers or {})
    cfg: dict[str, Any] = {
        "GOOSE_PROVIDER": "openai",
        "GOOSE_MODEL": model_id,
        # Goose reads the OpenAI host from config; the /v1 suffix is added by
        # the client, so the host is given without it.
        "OPENAI_HOST": base_url,
        "OPENAI_BASE_PATH": "v1/chat/completions",
        "GOOSE_MODE": "auto",          # no approval prompts
    }
    if mcp and entries:
        cfg["extensions"] = {
            sid: {
                "enabled": True,
                "type": "stdio",
                "cmd": e["command"][0],
                "args": e["command"][1:],
                "envs": e["env"],
                "timeout": int(e["timeout_ms"] / 1000),
            }
            for sid, e in entries.items()
        }
    path = config_home / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def write_pi_config(project_dir: Path, model_id: str, base_url: str,
                    servers: dict[str, "ServerSpec"] | None = None,
                    mcp: bool = True) -> Path:
    entries = _mcp_entries(servers or {})
    cfg: dict[str, Any] = {
        "provider": {
            "type": "openai-compatible",
            "baseUrl": f"{base_url}/v1",
            "apiKey": DUMMY_KEY,
            "model": model_id,
        },
    }
    if mcp and entries:
        # Supplied by pi-mcp-adapter; Pi itself ships no MCP support.
        cfg["mcp"] = {
            sid: {"command": e["command"], "env": e["env"]}
            for sid, e in entries.items()
        }
    path = project_dir / ".pi" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    return path


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class HarnessSpec:
    id: str
    name: str
    bin: str
    run: list[str]
    protocol: str = "openai"


class HarnessRun:
    """One non-interactive harness invocation, streamed line by line.

    The time limit is enforced here by killing the process *tree*: harnesses
    spawn MCP servers and shells, and killing only the parent leaves those
    running. No timeout is passed to the harness itself -- the backend owns the
    clock so that a paused run can be resumed rather than lost.
    """

    def __init__(self, spec: HarnessSpec, project_dir: Path, prompt: str,
                 env: dict[str, str] | None = None) -> None:
        self.spec = spec
        self.project_dir = project_dir
        self.prompt = prompt
        self.env = env or {}
        self.proc: subprocess.Popen | None = None
        self.stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> None:
        if not shutil.which(self.spec.bin):
            raise FileNotFoundError(f"{self.spec.bin} is not installed in the image")
        env = os.environ.copy()
        env.update(self.env)
        cmd = [*self.spec.run, self.prompt]
        self.proc = subprocess.Popen(
            cmd, cwd=str(self.project_dir), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
            start_new_session=True,      # own process group, so we can kill the tree
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        if self.proc is None or self.proc.stderr is None:
            return
        for line in self.proc.stderr:
            self.stderr_lines.append(line.rstrip("\n"))
            del self.stderr_lines[:-2000]

    def events(self) -> Iterator[dict[str, Any]]:
        """Yield normalised events until the harness exits."""
        if self.proc is None or self.proc.stdout is None:
            return
        parser = PARSERS.get(self.spec.id, lambda line: {"type": "raw", "line": line})
        for line in self.proc.stdout:
            event = parser(line)
            if event is not None:
                yield event
        self.proc.wait()
        if self.proc.returncode not in (0, None, -signal.SIGTERM, -signal.SIGKILL):
            yield {
                "type": "error",
                "error": f"{self.spec.name} exited {self.proc.returncode}",
                "stderr": "\n".join(self.stderr_lines[-40:]),
            }

    def stop(self) -> None:
        """Kill the whole process tree, escalating if it does not go."""
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self.proc.wait(timeout=10)
        except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
            self.proc.kill()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None
