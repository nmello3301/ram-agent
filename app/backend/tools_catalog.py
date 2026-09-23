"""The Godot MCP tool catalogue, priced in tokens.

Every tool schema is prefill, and prefill is the dominant cost on these
engines, so which toolsets are on is a performance decision rather than a
convenience one. This module exists so the UI can show the whole catalogue --
what is enabled, what is excluded, and what each one costs -- instead of
asking the user to trust a list of names in a config file.

A snapshot measured from godot-editor-mcp is shipped with the app, so the
catalogue renders without a running MCP server. When one is running its live
`godot_list_toolsets` output is preferred, because the truth can change with
the Godot version (some toolsets are gated on 4.4+).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import APP_DIR

SNAPSHOT = (
    Path("/app/bench/godot-tool-catalog.json")
    if Path("/app/bench/godot-tool-catalog.json").exists()
    else APP_DIR / "bench" / "godot-tool-catalog.json"
)


def load_snapshot() -> dict[str, Any]:
    if not SNAPSHOT.exists():
        return {"always_on": [], "baseline_tools": [], "toolsets": [],
                "source": "unavailable"}
    return json.loads(SNAPSHOT.read_text())


def catalog(enabled: list[str]) -> dict[str, Any]:
    """The full catalogue, annotated with what the current settings enable."""
    snap = load_snapshot()
    baseline_tools = snap.get("baseline_tools", [])
    baseline_tokens = sum(t.get("tokens", 0) for t in baseline_tools)

    toolsets = []
    for ts in snap.get("toolsets", []):
        on = ts["name"] in enabled
        toolsets.append({
            "name": ts["name"],
            "description": (ts.get("description") or "").strip(),
            "min_godot": ts.get("min_godot"),
            "tool_count": ts.get("tool_count", len(ts.get("tools", []))),
            "tokens": ts.get("tokens", 0),
            "enabled": on,
            "tools": ts.get("tools", []),
        })

    enabled_tokens = baseline_tokens + sum(t["tokens"] for t in toolsets if t["enabled"])
    enabled_count = len(baseline_tools) + sum(
        t["tool_count"] for t in toolsets if t["enabled"])
    all_tokens = baseline_tokens + sum(t["tokens"] for t in toolsets)
    all_count = len(baseline_tools) + sum(t["tool_count"] for t in toolsets)
    excluded = [t for t in toolsets if not t["enabled"]]

    return {
        "source": snap.get("source", "snapshot"),
        "always_on": snap.get("always_on", []),
        "baseline": {
            "tools": baseline_tools,
            "tool_count": len(baseline_tools),
            "tokens": baseline_tokens,
        },
        "toolsets": toolsets,
        "enabled_toolsets": list(enabled),
        "totals": {
            "enabled_tool_count": enabled_count,
            "enabled_tokens": enabled_tokens,
            "all_tool_count": all_count,
            "all_tokens": all_tokens,
            "excluded_toolset_count": len(excluded),
            "excluded_tool_count": sum(t["tool_count"] for t in excluded),
            "excluded_tokens": sum(t["tokens"] for t in excluded),
        },
        # What the enabled catalogue costs in wall time, at the prefill rates
        # actually measured on this machine. This is the number that decides
        # whether a combination is usable, so it is computed rather than left
        # for the reader to work out.
        "prefill_estimate_seconds": {
            "cpu_2_tok_s": round(enabled_tokens / 2.0),
            "gpu_3_tok_s": round(enabled_tokens / 3.0),
        },
    }
