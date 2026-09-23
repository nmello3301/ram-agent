"""MCP server selection: which tools the agent gets, and what they cost.

Under Colibri this was four lines inside harness.py, because there was one
server and the only question was which of its toolsets to enable. RAM_Agent has
two servers whose catalogues differ by an order of magnitude, so the decision
earns its own module.

The governing constraint has changed shape. It used to be prefill time: 7,000
tokens of schema was an hour before the first output token. It is now context
budget: the same 7,000 tokens are ~45 seconds of prefill, but they sit in the
window permanently, for every turn, competing with the transcript the agent
needs in order to finish the job.

The two servers are asymmetric, and not the way this module originally assumed.
Godot MCP is 181 tools across 27 gated toolsets and genuinely needs selecting
down. Blender MCP is 27 tools and ~3,183 tokens total, with no gating offered
and none needed. The selection logic is therefore real for one and a pass-
through for the other, which is why they have separate builders rather than a
shared abstraction that would flatter neither.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import load_mcp_servers


@dataclass
class ServerSpec:
    """One MCP server, resolved for a particular run."""

    id: str
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    # Milliseconds. A tool call can legitimately take a while -- a Blender
    # render or a headless Godot import -- so this is generous, but it is no
    # longer the hours Colibri needed to survive a cold prefill.
    timeout_ms: int = 600_000
    tools: int = 0
    tokens: int = 0
    estimated: bool = False
    verified: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "command": self.command,
            "env": self.env, "tools": self.tools, "tokens": self.tokens,
            "estimated": self.estimated, "verified": self.verified,
        }


class MCPError(RuntimeError):
    pass


def _godot_spec(row: dict[str, Any], toolsets: list[str], project_dir: Path,
                vision: bool) -> ServerSpec:
    """Godot MCP with only the requested toolsets exposed.

    Toolsets are seeded through GODOT_MCP_DEFAULT_TOOLSETS; without it the
    server starts with only `inspection` exposed. `core` and `inspection` are
    always on and must not be listed.
    """
    catalogue = row.get("toolsets", {})
    always_on = set(row.get("always_on", []))

    wanted: list[str] = []
    for name in toolsets:
        if name in always_on:
            continue           # listing these is a no-op at best
        if name not in catalogue:
            raise MCPError(
                f"unknown Godot toolset {name!r}; "
                f"known: {', '.join(sorted(catalogue))}"
            )
        # A screenshot tool the model cannot see through is pure context cost.
        # Colibri dropped the toolset in the orchestrator; doing it here means
        # every harness gets the same treatment for free.
        if catalogue[name].get("vision") and not vision:
            continue
        wanted.append(name)

    tools = row.get("always_on_tools", 0) + sum(
        catalogue[n].get("tools", 0) for n in wanted)
    tokens = row.get("always_on_tokens", 0) + sum(
        catalogue[n].get("tokens", 0) for n in wanted)

    env_map = row.get("env_map", {})
    env = {
        env_map.get("toolsets", "GODOT_MCP_DEFAULT_TOOLSETS"): ",".join(wanted),
        env_map.get("bridge", "GODOT_MCP_BRIDGE_URL"): row.get("bridge_url", ""),
        env_map.get("project", "GODOT_MCP_PROJECT"): str(project_dir),
    }
    return ServerSpec(
        id="godot", name=row.get("name", "Godot MCP"),
        command=_resolve(list(row["command"])), env=env,
        tools=tools, tokens=tokens,
        verified=bool(row.get("verified", False)),
    )


def _resolve(command: list[str]) -> list[str]:
    """Find the server binary on PATH, else in the project venv.

    Both servers are installed into the container's PATH, but a native run puts
    them in state/venv/bin. Resolving here means mcp.yaml can name the command
    plainly instead of carrying two absolute paths that are wrong half the time.
    """
    if not command:
        return command
    exe = command[0]
    found = shutil.which(exe)
    if found:
        return [found, *command[1:]]
    from .config import STATE_DIR
    candidate = STATE_DIR / "venv" / "bin" / exe
    if candidate.exists():
        return [str(candidate), *command[1:]]
    return command


def _blender_spec(server_id: str, row: dict[str, Any],
                  blender_bin: str | None) -> ServerSpec:
    """Blender MCP: the whole catalogue, because there is nothing to gate.

    27 tools and ~3,183 tokens -- 5% of a 64k window. The server exposes no
    profiles, no namespaces switch and no CLI flags; its only environment
    variable is BLENDER_BIN. Verified against the source, and the counts here
    were probed from the running server by scripts/probe-mcp.sh.
    """
    env_map = row.get("env_map", {})
    env: dict[str, str] = {}
    if blender_bin and "blender_bin" in env_map:
        env[env_map["blender_bin"]] = blender_bin

    return ServerSpec(
        id=server_id, name=row.get("name", "Blender MCP"),
        command=_resolve(list(row["command"])), env=env,
        tools=row.get("always_on_tools", 0),
        tokens=row.get("always_on_tokens", 0),
        estimated=False,
        verified=bool(row.get("verified", False)),
    )



def servers_for(
    phase: str,
    *,
    project_dir: Path,
    assets_dir: Path,
    godot_toolsets: list[str],
    blender_bin: str | None = None,
    blender_server: str = "blender",
    vision: bool = False,
    registry: dict[str, Any] | None = None,
) -> dict[str, ServerSpec]:
    """Resolve the MCP servers for one run.

    Returns a mapping of server id -> ServerSpec, empty when MCP is off.
    """
    reg = registry or load_mcp_servers()
    rows = reg["servers"]
    out: dict[str, ServerSpec] = {}

    if phase in ("godot", "both"):
        out["godot"] = _godot_spec(
            rows["godot"], godot_toolsets, project_dir, vision)

    if phase in ("blender", "both"):
        row = rows.get(blender_server)
        if row is None:
            raise MCPError(f"unknown Blender server {blender_server!r}")
        if phase not in row.get("phases", []):
            raise MCPError(
                f"{blender_server!r} does not support phase {phase!r}")
        out[blender_server] = _blender_spec(blender_server, row, blender_bin)

    return out


def budget(specs: dict[str, ServerSpec], ctx: int) -> dict[str, Any]:
    """What this tool selection costs, as a share of the context window.

    Surfaced in the UI and written into every run record. Colibri's single most
    expensive mistake was invisible: a 15,074-token preamble that nobody priced
    until three 6 h runs had produced nothing.
    """
    tokens = sum(s.tokens for s in specs.values())
    tools = sum(s.tools for s in specs.values())
    estimated = any(s.estimated for s in specs.values())
    unverified = [s.id for s in specs.values() if not s.verified]
    share = (tokens / ctx) if ctx else 0.0
    return {
        "tools": tools,
        "tokens": tokens,
        "context": ctx,
        "share_of_context": round(share, 4),
        "percent_of_context": f"{share * 100:.0f}%",
        "estimated": estimated,
        "unverified_servers": unverified,
        # A third of the window spent before the agent reads a file is the
        # configuration that looks reasonable and is not.
        "warning": (
            "tool schemas occupy more than a third of the context window"
            if share > 0.33 else None
        ),
    }
