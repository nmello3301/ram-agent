"""Godot helpers: error classification and project scaffolding."""
from __future__ import annotations

from backend.godot import _real_errors, _safe_name, scene_has_nodes


class TestErrorClassification:
    def test_script_error_is_real(self):
        out = "SCRIPT ERROR: Parse Error: Identifier 'foo' not declared"
        assert len(_real_errors(out)) == 1

    def test_clean_shutdown_noise_is_not_an_error(self):
        """A freshly created, valid 4.7.2 project emits both of these on exit.

        Treating them as failures would fail every benchmark task that is
        actually fine.
        """
        out = (
            "WARNING: 68 ObjectDB instances were leaked at exit "
            "(run with `--verbose` for details).\n"
            "ERROR: 33 resources still in use at exit "
            "(run with --verbose for details).\n"
        )
        assert _real_errors(out) == []

    def test_real_error_alongside_noise_is_still_caught(self):
        out = (
            "ERROR: 33 resources still in use at exit\n"
            "SCRIPT ERROR: Invalid call to function 'start'\n"
        )
        errors = _real_errors(out)
        assert len(errors) == 1
        assert "Invalid call" in errors[0]

    def test_empty_output_is_clean(self):
        assert _real_errors("") == []


class TestSafeName:
    def test_strips_path_separators(self):
        assert "/" not in _safe_name("../../etc/passwd")

    def test_keeps_readable_names(self):
        assert _safe_name("Pong Clone 2") == "Pong Clone 2"

    def test_empty_falls_back(self):
        assert _safe_name("!!!") == "Untitled"


class TestSceneHasNodes:
    def test_finds_declared_types(self, tmp_path):
        (tmp_path / "main.tscn").write_text(
            '[node name="P1" type="CharacterBody2D"]\n'
            '[node name="Score" type="Label"]\n'
        )
        got = scene_has_nodes(tmp_path, "main.tscn", ["CharacterBody2D", "Label", "Timer"])
        assert got == {"CharacterBody2D": True, "Label": True, "Timer": False}

    def test_missing_scene_reports_all_false(self, tmp_path):
        assert scene_has_nodes(tmp_path, "nope.tscn", ["Label"]) == {"Label": False}


class TestPlayableFlag:
    """An untouched template must not be reported as a playable game.

    The first flag counted "loads" + "no script errors", which an empty project
    passes trivially, so three failed runs that produced nothing were listed as
    playable in the Games tab.
    """

    TEMPLATE = '[gd_scene format=3 uid="uid://x"]\n\n[node name="Main" type="Node2D"]\n'

    def _make(self, tmp_path, scene, scripts=()):
        (tmp_path / "project.godot").write_text("")
        (tmp_path / "main.tscn").write_text(scene)
        for i, body in enumerate(scripts):
            (tmp_path / f"s{i}.gd").write_text(body)
        return tmp_path

    def test_untouched_template_is_not_playable(self, tmp_path):
        from backend.games import _main_scene_nodes, _script_stats
        p = self._make(tmp_path, self.TEMPLATE)
        assert _script_stats(p)["scripts"] == 0
        assert _main_scene_nodes(p) == 1

    def test_real_game_counts_as_playable(self, tmp_path):
        from backend.games import _main_scene_nodes, _script_stats
        scene = self.TEMPLATE + '\n[node name="Paddle" type="CharacterBody2D" parent="."]\n'
        p = self._make(tmp_path, scene, scripts=("extends Node2D\n",))
        assert _script_stats(p)["scripts"] == 1
        assert _main_scene_nodes(p) == 2

    def test_addon_scripts_do_not_count(self, tmp_path):
        """The MCP addon is our code, not the model's."""
        from backend.games import _script_stats
        (tmp_path / "addons" / "godot_mcp").mkdir(parents=True)
        (tmp_path / "addons" / "godot_mcp" / "x.gd").write_text("extends Node\n")
        assert _script_stats(tmp_path)["scripts"] == 0
