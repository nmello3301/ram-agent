"""Model manager: state machine, free-space check, verification."""
from __future__ import annotations

import pytest

from backend.models_mgr import (
    ALLOWED, InsufficientSpace, ModelEntry, State, TransitionError,
    check_space, dir_size,
)


def make_entry(**kw) -> ModelEntry:
    base = dict(
        id="m1", name="M1", repo="org/repo", revision="abc",
        gguf="m1-Q5_K_XL.gguf", size_gb=10.0,
        tools=True, vision=False, gpu=False, context_ceiling=65536,
    )
    base.update(kw)
    return ModelEntry(**base)


class TestStateMachine:
    def test_starts_absent(self):
        assert make_entry().state is State.ABSENT

    def test_happy_path(self):
        e = make_entry()
        for s in (State.DOWNLOADING, State.VERIFYING, State.READY,
                  State.LOADING, State.LOADED):
            e.transition(s)
        assert e.state is State.LOADED

    def test_illegal_transition_raises(self):
        e = make_entry()
        # Cannot load a model that was never downloaded.
        with pytest.raises(TransitionError):
            e.transition(State.LOADED)

    def test_error_is_reachable_from_every_state(self):
        for state in State:
            if state is State.ERROR:
                continue
            assert State.ERROR in ALLOWED[state], state

    def test_error_carries_message_and_clears_on_recovery(self):
        e = make_entry()
        e.transition(State.ERROR, "disk fell over")
        assert e.error == "disk fell over"
        e.transition(State.DOWNLOADING)          # retry
        assert e.error is None

    def test_self_transition_is_allowed(self):
        e = make_entry()
        e.transition(State.DOWNLOADING)
        e.transition(State.DOWNLOADING)          # progress updates re-assert it
        assert e.state is State.DOWNLOADING

    def test_cancel_returns_to_absent(self):
        e = make_entry()
        e.transition(State.DOWNLOADING)
        e.transition(State.ABSENT)
        assert e.state is State.ABSENT

    def test_unload_returns_to_ready(self):
        e = make_entry()
        for s in (State.DOWNLOADING, State.VERIFYING, State.READY,
                  State.LOADING, State.LOADED, State.READY):
            e.transition(s)
        assert e.state is State.READY


class TestFreeSpace:
    def test_passes_when_space_is_sufficient(self, tmp_path, monkeypatch):
        e = make_entry(size_gb=1.0)
        monkeypatch.setattr("backend.models_mgr.free_bytes", lambda _p: 10_000_000_000)
        check_space(e, tmp_path)        # must not raise

    def test_raises_with_need_and_have_in_the_message(self, tmp_path, monkeypatch):
        e = make_entry(size_gb=500.0)
        monkeypatch.setattr("backend.models_mgr.free_bytes", lambda _p: 1_000_000_000)
        with pytest.raises(InsufficientSpace) as exc:
            check_space(e, tmp_path)
        msg = str(exc.value)
        assert "need 500.0 GB" in msg and "have 1.0 GB" in msg

    def test_only_the_remainder_is_required_when_resuming(self, tmp_path, monkeypatch):
        """A part-finished download must not demand room for what it already has."""
        e = make_entry(size_gb=10.0)
        # 9 GB already present, so only ~1 GB more is needed.
        monkeypatch.setattr("backend.models_mgr.dir_size", lambda _p: 9_000_000_000)
        monkeypatch.setattr("backend.models_mgr.free_bytes", lambda _p: 2_000_000_000)
        check_space(e, tmp_path)        # 1 GB needed, 2 GB free -> fine

    def test_exact_fit_is_allowed(self, tmp_path, monkeypatch):
        e = make_entry(size_gb=1.0)
        monkeypatch.setattr("backend.models_mgr.dir_size", lambda _p: 0)
        monkeypatch.setattr("backend.models_mgr.free_bytes", lambda _p: 1_000_000_000)
        check_space(e, tmp_path)


class TestDirSize:
    def test_counts_nested_files(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "f1").write_bytes(b"x" * 100)
        (tmp_path / "f2").write_bytes(b"y" * 50)
        assert dir_size(tmp_path) == 150

    def test_missing_dir_is_zero(self, tmp_path):
        assert dir_size(tmp_path / "nope") == 0


class TestRegistryIntegrity:
    """The registry is the source of truth for what the UI claims.

    These assertions encode the decisions in RAM_AGENT.md, so a careless edit
    to models.yaml cannot quietly reintroduce the problem this project was
    built to escape -- a model that does not fit in RAM.
    """

    # Fast memory on this machine: 62 GB of RAM + 6 GB of VRAM. The budget is
    # the sum, because weights placed on the GPU come out of the VRAM column
    # and the rest sit in RAM. The registry is allowed to assume this machine;
    # RAM_Agent is not a portable product.
    HOST_RAM_GB = 62
    HOST_VRAM_GB = 6
    HOST_FAST_GB = HOST_RAM_GB + HOST_VRAM_GB

    def _registry(self):
        import yaml
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        return yaml.safe_load((root / "models.yaml").read_text())

    def test_every_model_declares_the_capability_fields(self):
        for m in self._registry()["models"]:
            for field in ("gguf", "size_gb", "tools", "vision", "gpu",
                          "context_ceiling", "revision"):
                assert field in m, f"{m['id']} is missing {field}"

    def test_every_model_fits_in_fast_memory(self):
        """The entire premise. A model that does not fit is a Colibri model.

        The budget is RAM + VRAM, minus what must be left for everything that
        is not the model: ~12 GB of RAM for the OS, the container, the Godot
        editor and Firefox, and ~1 GB of VRAM for the CUDA context and
        activations.
        """
        usable = (self.HOST_RAM_GB - 12) + (self.HOST_VRAM_GB - 1)
        for m in self._registry()["models"]:
            assert m["size_gb"] <= usable, (
                f"{m['id']} is {m['size_gb']} GB against a {usable} GB budget "
                f"of usable fast memory"
            )

    def test_no_model_relies_on_the_disk(self):
        """The line that separates this project from Colibri.

        If a model only fits by paging weights off the NVMe, it does not
        belong here whatever its benchmark scores -- that is precisely what
        Qwen3.8-Flash-Next does, and why it is in `excluded`.
        """
        for m in self._registry()["models"]:
            assert m["size_gb"] <= self.HOST_FAST_GB, (
                f"{m['id']} exceeds total fast memory and would page to disk")

    def test_the_default_model_leaves_room_for_blender(self):
        """The default has to be the one you can actually work in.

        Blender with a real scene is several GB on top of the editor. A model
        that cannot coexist with it is a legitimate registry row -- 
        Qwen3-Coder-Next at 46 GB is exactly that -- but it must not be the
        default.
        """
        reg = self._registry()
        default = next(m for m in reg["models"] if m.get("default"))
        assert default["size_gb"] <= self.HOST_FAST_GB - 30, (
            f"the default model {default['id']} is {default['size_gb']} GB, "
            "which does not leave room for Blender and Godot together"
        )

    def test_exactly_one_model_is_the_default(self):
        defaults = [m["id"] for m in self._registry()["models"]
                    if m.get("default")]
        assert len(defaults) == 1, f"expected one default, got {defaults}"

    def test_vision_models_ship_a_projector(self):
        """--mmproj is what makes vision work; claiming vision without one is a lie."""
        for m in self._registry()["models"]:
            if m["vision"]:
                assert m.get("mmproj"), (
                    f"{m['id']} claims vision but declares no mmproj")
            else:
                assert not m.get("mmproj"), (
                    f"{m['id']} has no vision but declares an mmproj")

    def test_every_model_can_call_tools(self):
        """A model that cannot call tools cannot drive Godot or Blender MCP.

        Colibri carried four such engines and had to special-case them
        everywhere. The rule here is simpler: it does not go in the registry.
        """
        for m in self._registry()["models"]:
            assert m["tools"] is True, f"{m['id']} cannot drive MCP"

    def test_context_ceiling_holds_a_preamble_and_a_transcript(self):
        for m in self._registry()["models"]:
            assert m["context_ceiling"] >= 32768, m["id"]

    def test_default_context_leaves_room_after_the_tool_catalogue(self):
        """A 64k window minus a 7.3k Godot catalogue is still a working agent."""
        reg = self._registry()
        assert reg["defaults"]["ctx"] >= 32768

    def test_pinned_revisions_are_full_commit_shas(self):
        for m in self._registry()["models"]:
            rev = m["revision"]
            assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), m["id"]

    def test_gguf_filenames_look_like_gguf(self):
        for m in self._registry()["models"]:
            assert m["gguf"].endswith(".gguf"), m["id"]
            if m.get("mmproj"):
                assert m["mmproj"].endswith(".gguf"), m["id"]

    def test_every_model_declares_its_stats_for_the_ui(self):
        """The UI shows these, so a missing one is a blank cell, not a crash."""
        for m in self._registry()["models"]:
            for field in ("family", "license", "params_total", "params_active",
                          "reasoning", "model_page"):
                assert m.get(field), f"{m['id']} is missing {field}"

    def test_no_code_is_needed_to_add_a_model(self):
        """Model-agnosticism, asserted rather than claimed.

        Every model-specific behaviour must be expressible as registry data.
        If a row needed a code change to work, it would show up here as a key
        the dataclass cannot carry -- which `test_registry_keys_are_all_consumed`
        covers -- or as a family name appearing in the backend, which this
        checks directly.
        """
        from pathlib import Path
        backend = Path(__file__).resolve().parents[1]
        families = {m["family"].lower() for m in self._registry()["models"]}
        offenders = []
        for src in backend.glob("*.py"):
            text = src.read_text().lower()
            for fam in families:
                # Comments and docstrings may name a model; code must not
                # branch on one. Look for it being compared or indexed.
                for pattern in (f'== "{fam}"', f"== '{fam}'", f'["{fam}"]'):
                    if pattern in text:
                        offenders.append(f"{src.name}: {pattern}")
        assert not offenders, f"backend branches on a model family: {offenders}"

    def test_excluded_models_carry_a_reason(self):
        """'Ruled out, not merely absent' -- the reason is the point."""
        for m in self._registry().get("excluded", []):
            assert m.get("reason", "").strip(), f"{m['name']} excluded without a reason"

    def test_the_colibri_models_are_not_back(self):
        """Regression guard with a specific history.

        These three are 818 GB between them and decoded at 0.1-1 tok/s. If one
        reappears in this registry, something has gone badly wrong.
        """
        ids = " ".join(m["id"] + m["repo"] for m in self._registry()["models"]).lower()
        for banned in ("glm-5.3-flash", "deepseek-v4-flash", "deepseek-v4.1"):
            assert banned not in ids, f"{banned} is a Colibri-era model"


class TestRegistryConstructs:
    """The registry and the dataclass must agree.

    A field was once added to models.yaml and to the ModelManager constructor
    call but not to ModelEntry, and the mismatch only surfaced as the container
    crash-looping on startup. This builds every registry row the way
    ModelManager does, so that failure is caught here instead.
    """

    def _registry(self):
        import yaml
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        return yaml.safe_load((root / "models.yaml").read_text())

    def test_model_manager_builds_every_row(self, tmp_path):
        from backend.models_mgr import ModelManager
        mgr = ModelManager(self._registry(), tmp_path)
        assert len(mgr.entries) == len(self._registry()["models"])

    def test_default_id_resolves(self, tmp_path):
        from backend.models_mgr import ModelManager
        mgr = ModelManager(self._registry(), tmp_path)
        assert mgr.default_id in mgr.entries

    def test_download_is_restricted_to_the_declared_files(self, tmp_path):
        """The guard against a 27 GB model becoming a 500 GB pull.

        These repos publish every quantisation side by side. `files` is what is
        handed to snapshot_download as allow_patterns, so it must never be
        empty and must never be a wildcard.
        """
        from backend.models_mgr import ModelManager
        mgr = ModelManager(self._registry(), tmp_path)
        for entry in mgr.entries.values():
            assert entry.files, f"{entry.id} would download the whole repo"
            assert all("*" not in f for f in entry.files), entry.id
            assert entry.gguf in entry.files

    def test_registry_keys_are_all_consumed(self, tmp_path):
        """Every key in a models.yaml row maps to a ModelEntry field."""
        import dataclasses
        from backend.models_mgr import ModelEntry
        fields = {f.name for f in dataclasses.fields(ModelEntry)}
        for row in self._registry()["models"]:
            unknown = set(row) - fields
            assert not unknown, f"{row['id']}: unmapped keys {unknown}"
