"""Paths, registry loading and the settings that persist between runs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

APP_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
PROJECTS_DIR = Path(os.environ.get("PROJECTS_DIR", "/projects"))
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", "/assets"))
STATE_DIR = Path(os.environ.get("STATE_DIR", "/state"))
LLAMA_HOME = Path(os.environ.get("LLAMA_HOME", "/opt/llama.cpp"))
GODOT_BIN = Path(os.environ.get("GODOT_BIN", "/opt/godot/godot"))
BLENDER_BIN = Path(os.environ.get("BLENDER_BIN", "/usr/bin/blender"))

SETTINGS_FILE = STATE_DIR / "settings.json"
RUNS_FILE = STATE_DIR / "runs.jsonl"

# Colibri defaulted to 10 hours because DeepSeek V4.1 Flash decoded at roughly
# 1 tok/s and a single turn could take three hours. With the weights resident
# a turn is seconds, so an hour is a generous budget for a whole game rather
# than a floor for one exchange. Still parametric, still no upper bound: a long
# autonomous run is a legitimate thing to ask for.
DEFAULT_TIME_LIMIT_SECONDS = 3600


# The two phases an agent can run in. They are separate on purpose: Godot MCP
# and Blender MCP together are 40-60k tokens of tool schema, which crowds out
# the context the agent needs to do the work. See RAM_AGENT.md.
PHASES = ("godot", "blender", "both")


@dataclass
class Settings:
    """User-editable settings. Persisted to state/settings.json."""

    time_limit_enabled: bool = True
    time_limit_seconds: int = DEFAULT_TIME_LIMIT_SECONDS

    # Which editor the agent is driving. `both` is offered because it is
    # occasionally the right answer for a small task, not because it is
    # recommended -- it roughly triples the preamble.
    phase: str = "godot"

    # Godot MCP toolsets, by their real names in godot-editor-mcp. `core` and
    # `inspection` are always on and are not listed here.
    #
    # Measured with godot-editor-mcp 2026.9.10:
    #   core + inspection (always on)   18 tools   ~2,557 tokens
    #   scene_edit                      21 tools   ~2,498 tokens
    #   runtime                          9 tools   ~1,566 tokens
    #   scripts                          6 tools     ~690 tokens
    #   debugger                        11 tools   ~1,171 tokens   <- dropped
    #   this default (54 tools)                    ~7,312 tokens
    #
    # Under Colibri this list was a *performance* decision -- 7,000 tokens was
    # an hour of prefill. It is now a *context* decision: ~45 seconds of
    # prefill, but still 7k tokens permanently resident in a 64k window.
    godot_toolsets: list[str] = field(
        default_factory=lambda: ["scene_edit", "scripts", "runtime"]
    )

    # Blender MCP has no profiles to choose between: 27 tools, ~3,183 tokens,
    # the whole catalogue or nothing. This points the server at a Blender
    # binary; empty means "whatever is on PATH".
    blender_bin: str = ""

    # Attach MCP to the harness at all. Colibri's measurement stands as the
    # reason this switch exists: with MCP the OpenCode preamble was 15,074
    # tokens, without it 7,723, and the 24 h run that actually produced a game
    # was the one with MCP off. That trade is no longer forced -- but a Godot
    # project is still text, and writing .tscn/.gd directly and validating with
    # headless Godot remains a legitimate strategy worth benchmarking against.
    mcp_enabled: bool = True

    # Per-model engine flag overrides, keyed by model id. This is what the
    # -ncmoe and MTP sweeps write into.
    model_flags: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Hold the machine awake for the duration of a run.
    inhibit_sleep: bool = True

    # VRAM held back from the expert-offload calculation, for activations,
    # scratch, the CUDA context and whatever else has the card.
    #
    # This replaces Colibri's ram_headroom_gb, and the reason is the inversion
    # at the heart of this project: RAM stopped being the scarce resource and
    # VRAM became it. colibri had to be told not to eat 53 of 62 GB. llama.cpp
    # needs to be told how much of a 6 GB card it may not touch -- because
    # Blender, ComfyUI and the vision projector all want the same card.
    vram_reserve_mb: int = 1024

    @classmethod
    def load(cls) -> "Settings":
        if SETTINGS_FILE.exists():
            try:
                raw = json.loads(SETTINGS_FILE.read_text())
                known = {f for f in cls.__dataclass_fields__}
                return cls(**{k: v for k, v in raw.items() if k in known})
            except (json.JSONDecodeError, TypeError, ValueError):
                # A corrupt settings file must not stop the app from starting.
                pass
        return cls()

    def save(self) -> None:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(self.__dict__, indent=2))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def validate(self) -> None:
        """Reject settings that would fail much later, and less legibly."""
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {self.phase!r}")
        if self.time_limit_enabled and self.time_limit_seconds < 60:
            raise ValueError("time limit must be at least 60 seconds")
        if self.vram_reserve_mb < 0:
            raise ValueError("vram_reserve_mb cannot be negative")


def _load_yaml(name: str) -> dict[str, Any]:
    path = APP_DIR / name
    with path.open() as fh:
        return yaml.safe_load(fh)


def load_models() -> dict[str, Any]:
    return _load_yaml("models.yaml")


def load_harnesses() -> dict[str, Any]:
    return _load_yaml("harnesses.yaml")


def load_mcp_servers() -> dict[str, Any]:
    return _load_yaml("mcp.yaml")


def hf_token() -> str | None:
    """Read HF_TOKEN from the environment or app/.env. Never logged."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip() or None
    env_file = APP_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip() or None
    return None
