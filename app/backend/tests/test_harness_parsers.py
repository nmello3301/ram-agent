"""Harness output parsers: normalise three formats, never crash on junk."""
from __future__ import annotations

from pathlib import Path

import json

from backend.harness import parse_goose_line, parse_opencode_line, parse_pi_line


class TestOpenCodeParser:
    def test_text_event(self):
        e = parse_opencode_line(json.dumps({"type": "text", "text": "hello"}))
        assert e == {"type": "text", "text": "hello"}

    def test_tool_call(self):
        e = parse_opencode_line(json.dumps(
            {"type": "tool_call", "name": "add_node", "args": {"t": "Label"}, "id": "c1"}))
        assert e["type"] == "tool_call"
        assert e["name"] == "add_node"
        assert e["args"] == {"t": "Label"}

    def test_tool_result_error_flag(self):
        e = parse_opencode_line(json.dumps(
            {"type": "tool_result", "id": "c1", "result": "boom", "isError": True}))
        assert e["is_error"] is True

    def test_malformed_json_is_surfaced_not_swallowed(self):
        e = parse_opencode_line('{"type": "text", "text": ')
        assert e["type"] == "raw"

    def test_blank_line_ignored(self):
        assert parse_opencode_line("   ") is None


class TestPiParser:
    def test_content_blocks_are_joined(self):
        e = parse_pi_line(json.dumps(
            {"type": "assistant", "content": [{"text": "a"}, {"text": "b"}]}))
        assert e == {"type": "text", "text": "ab"}

    def test_tool_call_arguments_key(self):
        e = parse_pi_line(json.dumps(
            {"type": "tool", "name": "run_scene", "arguments": {"p": 1}, "id": "x"}))
        assert e["type"] == "tool_call" and e["args"] == {"p": 1}

    def test_non_json_line_is_raw(self):
        assert parse_pi_line("Loading skills...")["type"] == "raw"


class TestGooseParser:
    def test_plain_text(self):
        assert parse_goose_line("Creating the scene")["type"] == "text"

    def test_error_line(self):
        e = parse_goose_line("Error: could not reach the editor")
        assert e["type"] == "error"

    def test_blank_ignored(self):
        assert parse_goose_line("\n") is None

    def test_never_raises_on_arbitrary_bytes(self):
        for junk in ("\x00\x01", "─── ", "{not json", "| | |", "◓"):
            parse_goose_line(junk)      # must not raise


class TestOpenCodeMcpConfig:
    """OpenCode's MCP schema is exact, and getting it wrong fails silently.

    With `command` as a string or the env under `env`, OpenCode logs
    "Ignoring MCP config entry without type" and runs with only its own ten
    built-in tools. The agent then looks like it is working while never
    touching Godot at all. Verified against https://opencode.ai/config.json:
    `command` is an array of strings and the key is `environment`.
    """

    def _cfg(self, tmp_path, toolsets=("scene_edit", "scripts"), mcp=True):
        import json

        import yaml

        from backend.harness import write_opencode_config
        from backend.mcp import servers_for

        registry = yaml.safe_load(
            (Path(__file__).resolve().parents[2] / "mcp.yaml").read_text())
        servers = servers_for(
            "godot", project_dir=tmp_path, assets_dir=tmp_path / "assets",
            godot_toolsets=list(toolsets), registry=registry,
        ) if mcp else {}
        path = write_opencode_config(
            tmp_path, "m", "http://127.0.0.1:8077", servers, mcp=mcp)
        return json.loads(path.read_text())

    def test_command_is_a_list(self, tmp_path):
        godot = self._cfg(tmp_path)["mcp"]["godot"]
        assert isinstance(godot["command"], list)
        assert godot["command"][0] == "godot-editor-mcp"

    def test_environment_key_not_env(self, tmp_path):
        godot = self._cfg(tmp_path)["mcp"]["godot"]
        assert "environment" in godot
        assert "env" not in godot

    def test_type_and_enabled_present(self, tmp_path):
        godot = self._cfg(tmp_path)["mcp"]["godot"]
        assert godot["type"] == "local"
        assert godot["enabled"] is True

    def test_toolsets_are_passed_through(self, tmp_path):
        env = self._cfg(tmp_path)["mcp"]["godot"]["environment"]
        assert env["GODOT_MCP_DEFAULT_TOOLSETS"] == "scene_edit,scripts"

    def test_mcp_timeout_outlives_a_slow_tool_call(self, tmp_path):
        # Colibri needed hours here because a cold turn took hours. What still
        # needs headroom is the tool call itself: a Blender render or a
        # headless Godot import can run for minutes.
        assert self._cfg(tmp_path)["mcp"]["godot"]["timeout"] >= 60_000

    def test_approvals_are_all_disabled(self, tmp_path):
        perm = self._cfg(tmp_path)["permission"]
        assert set(perm.values()) == {"allow"}


class TestOpenCodeTimeouts:
    """OpenCode aborts a request that goes quiet for five minutes.

    `headerTimeout` and `chunkTimeout` both default to 300000 ms. Under Colibri
    this was fatal: a cold prefill produced nothing for far longer, every turn
    aborted, OpenCode retried ~6 times and exited 1 after ~31 minutes with zero
    tool calls, three times in a row, while the run's own 6 h limit went unused.

    A resident model would very likely never trip these. They stay disabled
    because a single Blender render can still outlast five minutes of silence,
    and because the backend owns the clock -- and this test stays because
    re-enabling them by accident would reintroduce a failure that took three
    wasted 6 h runs to diagnose.
    """

    def _options(self, tmp_path):
        import json
        from backend.harness import write_opencode_config
        path = write_opencode_config(tmp_path, "m", "http://127.0.0.1:8077",
                                     {}, mcp=False)
        cfg = json.loads(path.read_text())
        return cfg["provider"]["llamacpp"]["options"]

    def test_all_three_timeouts_disabled(self, tmp_path):
        opts = self._options(tmp_path)
        for key in ("timeout", "headerTimeout", "chunkTimeout"):
            assert opts[key] is False, f"{key} must be disabled, got {opts[key]!r}"

    def test_base_url_and_key_still_set(self, tmp_path):
        opts = self._options(tmp_path)
        assert opts["baseURL"].endswith("/v1")
        assert opts["apiKey"]


class TestMcpToggle:
    """With MCP off the key must be absent, not merely disabled.

    Measured: setting `enabled: false` still had OpenCode load all 58 tools
    (11,591 tokens of the 15,074-token preamble). Only removing the key
    entirely dropped it to 7,723.
    """

    def _cfg(self, tmp_path, mcp):
        import json

        import yaml

        from backend.harness import write_opencode_config
        from backend.mcp import servers_for

        registry = yaml.safe_load(
            (Path(__file__).resolve().parents[2] / "mcp.yaml").read_text())
        servers = servers_for(
            "godot", project_dir=tmp_path, assets_dir=tmp_path / "assets",
            godot_toolsets=["scene_edit"], registry=registry,
        )
        path = write_opencode_config(tmp_path, "m", "http://127.0.0.1:8077",
                                     servers, mcp=mcp)
        return json.loads(path.read_text())

    def test_mcp_key_absent_when_disabled(self, tmp_path):
        assert "mcp" not in self._cfg(tmp_path, mcp=False)

    def test_mcp_present_when_enabled(self, tmp_path):
        cfg = self._cfg(tmp_path, mcp=True)
        assert cfg["mcp"]["godot"]["type"] == "local"

    def test_provider_survives_either_way(self, tmp_path):
        for mcp in (True, False):
            cfg = self._cfg(tmp_path, mcp=mcp)
            assert cfg["provider"]["llamacpp"]["options"]["chunkTimeout"] is False
