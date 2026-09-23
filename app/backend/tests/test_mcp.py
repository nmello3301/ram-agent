"""MCP server selection and the context budget it costs."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.mcp import MCPError, budget, servers_for


def call(phase="godot", *, registry, vision=False, toolsets=None,
         blender_bin=None, tmp=Path("/tmp/p")):
    return servers_for(
        phase, project_dir=tmp, assets_dir=tmp / "assets",
        godot_toolsets=toolsets if toolsets is not None
        else ["scene_edit", "scripts", "runtime"],
        blender_bin=blender_bin, vision=vision, registry=registry,
    )


class TestPhaseSplit:
    def test_godot_phase_loads_only_godot(self, mcp_registry):
        assert set(call("godot", registry=mcp_registry)) == {"godot"}

    def test_blender_phase_loads_only_blender(self, mcp_registry):
        assert set(call("blender", registry=mcp_registry)) == {"blender"}

    def test_both_loads_both(self, mcp_registry):
        assert set(call("both", registry=mcp_registry)) == {"godot", "blender"}

    def test_both_costs_more_than_either_alone(self, mcp_registry):
        """The arithmetic behind the phase default.

        It is a smaller effect than originally assumed -- both servers together
        are 16% of a 64k window, not the 36% estimated from the wrong Blender
        server -- so the split is now a default rather than a hard constraint.
        """
        one = budget(call("godot", registry=mcp_registry), 65536)["tokens"]
        two = budget(call("both", registry=mcp_registry), 65536)["tokens"]
        assert two > one


class TestGodotToolsets:
    def test_unknown_toolset_is_refused(self, mcp_registry):
        with pytest.raises(MCPError, match="unknown Godot toolset"):
            call(registry=mcp_registry, toolsets=["nonsense"])

    def test_always_on_toolsets_are_not_listed(self, mcp_registry):
        """Listing core/inspection is a no-op at best; they must be filtered."""
        spec = call(registry=mcp_registry,
                    toolsets=["core", "inspection", "scripts"])["godot"]
        assert spec.env["GODOT_MCP_DEFAULT_TOOLSETS"] == "scripts"

    def test_screenshots_are_dropped_without_vision(self, mcp_registry):
        with_vision = call(registry=mcp_registry, vision=True,
                           toolsets=["scene_edit", "screenshots"])["godot"]
        without = call(registry=mcp_registry, vision=False,
                       toolsets=["scene_edit", "screenshots"])["godot"]
        assert "screenshots" in with_vision.env["GODOT_MCP_DEFAULT_TOOLSETS"]
        assert "screenshots" not in without.env["GODOT_MCP_DEFAULT_TOOLSETS"]
        assert without.tokens < with_vision.tokens

    def test_more_toolsets_cost_more_tokens(self, mcp_registry):
        small = call(registry=mcp_registry, toolsets=["scripts"])["godot"]
        large = call(registry=mcp_registry,
                     toolsets=["scripts", "scene_edit", "runtime",
                               "debugger"])["godot"]
        assert large.tokens > small.tokens
        assert large.tools > small.tools

    def test_always_on_cost_is_never_free(self, mcp_registry):
        """core + inspection are 18 tools we pay for whatever we select."""
        spec = call(registry=mcp_registry, toolsets=[])["godot"]
        assert spec.tools == 18
        assert spec.tokens > 0


class TestBlenderServer:
    """Blender MCP: 27 tools, no gating, and the numbers are measured.

    An earlier version of this file tested a `profile` argument with values
    like `core` and `full`, sized from the release notes of a different and
    much larger community server. That server is not the one
    `pip install blender-mcp-server` installs. These tests assert against what
    was actually probed from the running server.
    """

    def test_exposes_the_measured_catalogue(self, mcp_registry):
        spec = call("blender", registry=mcp_registry)["blender"]
        assert spec.tools == 27
        assert spec.tokens == 3183

    def test_costs_are_measured_not_estimated(self, mcp_registry):
        assert call("blender", registry=mcp_registry)["blender"].estimated is False

    def test_server_is_verified(self, mcp_registry):
        """It was probed on this machine; the flag must say so."""
        assert call("blender", registry=mcp_registry)["blender"].verified is True

    def test_blender_bin_is_passed_through(self, mcp_registry):
        spec = call("blender", registry=mcp_registry,
                    blender_bin="/home/u/.local/bin/blender")["blender"]
        assert spec.env["BLENDER_BIN"] == "/home/u/.local/bin/blender"

    def test_blender_bin_is_omitted_when_unset(self, mcp_registry):
        """Empty means 'whatever is on PATH', not an empty BLENDER_BIN."""
        assert "BLENDER_BIN" not in call("blender", registry=mcp_registry)["blender"].env

    def test_whole_catalogue_is_a_small_share_of_context(self, mcp_registry):
        """5% of 64k. There is nothing here worth gating."""
        b = budget(call("blender", registry=mcp_registry), 65536)
        assert b["share_of_context"] < 0.10


class TestBudget:
    def test_reports_share_of_context(self, mcp_registry):
        b = budget(call("godot", registry=mcp_registry), 65536)
        assert 0 < b["share_of_context"] < 1
        assert b["context"] == 65536

    def test_warns_when_tools_eat_a_third_of_the_window(self, mcp_registry):
        """The guard still has to fire, even though no current config trips it.

        Both real servers together are 16% of 64k. The warning exists for the
        configuration someone reaches for next -- a smaller context, or a
        Godot toolset list with everything switched on.
        """
        servers = call("both", registry=mcp_registry)
        assert budget(servers, 65536)["warning"] is None
        assert budget(servers, 16384)["warning"] is not None

    def test_no_warning_for_a_sane_configuration(self, mcp_registry):
        assert budget(call("godot", registry=mcp_registry), 65536)["warning"] is None

    def test_a_small_window_makes_the_same_tools_expensive(self, mcp_registry):
        """Colibri's CTX default was 4096 -- smaller than one preamble."""
        servers = call("godot", registry=mcp_registry)
        assert budget(servers, 8192)["share_of_context"] > \
               budget(servers, 65536)["share_of_context"]

    def test_empty_selection_costs_nothing(self, mcp_registry):
        b = budget({}, 65536)
        assert b["tokens"] == 0 and b["warning"] is None

    def test_verified_servers_are_not_flagged(self, mcp_registry):
        """Both have been probed on this machine, so nothing should be named."""
        assert budget(call("both", registry=mcp_registry), 65536)[
            "unverified_servers"] == []
