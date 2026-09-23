"""Engine adapter: offload sizing and command construction.

The offload calculation is the one piece of arithmetic in this project that
silently decides whether a run is fast, slow, or an OOM four hours in. It gets
tested directly rather than inferred from a benchmark.
"""
from __future__ import annotations

import pytest

from backend.engine import EngineError, auto_n_cpu_moe, build_command
from backend.gguf import read_info
from backend.models_mgr import ModelEntry

GB = 1_073_741_824


def entry(**kw) -> ModelEntry:
    base = dict(
        id="test-model", name="Test", repo="org/repo", revision="a" * 40,
        gguf="model.gguf", size_gb=27.0, tools=True, vision=False, gpu=True,
        context_ceiling=262144,
    )
    base.update(kw)
    return ModelEntry(**base)


class TestAutoNCpuMoe:
    def test_no_gpu_puts_every_expert_in_ram(self, make_gguf):
        info = read_info(make_gguf(n_layers=48))
        n, why = auto_n_cpu_moe(info, 27 * GB, 65536, free_mb=0)
        assert n == 48
        assert why["mode"] == "cpu-only"

    def test_a_small_card_puts_every_expert_in_ram(self, make_gguf):
        """The expected outcome on this machine, and it must not be an error.

        6 GB, minus reserve, minus KV, minus the non-expert trunk, leaves
        nothing for experts. That is the GTX-1650-class configuration that
        measured 20.2 tok/s -- a fine place to start, not a failure.
        """
        info = read_info(make_gguf(n_layers=48))
        n, why = auto_n_cpu_moe(info, 27 * GB, 65536, free_mb=6144,
                                reserve_mb=1024, mmproj_mb=900)
        assert n == 48
        assert "all experts in system RAM" in why["reason"]

    def test_a_large_card_keeps_experts_on_gpu(self, make_gguf):
        info = read_info(make_gguf(n_layers=48))
        n, why = auto_n_cpu_moe(info, 27 * GB, 8192, free_mb=48000,
                                reserve_mb=1024)
        assert n < 48
        assert why["expert_layers_on_gpu"] > 0

    def test_more_vram_never_means_more_cpu_offload(self, make_gguf):
        """Monotonicity. A bigger card must never make the split worse."""
        info = read_info(make_gguf(n_layers=48))
        previous = 49
        for free in (6144, 12288, 24576, 49152):
            n, _ = auto_n_cpu_moe(info, 27 * GB, 16384, free_mb=free)
            assert n <= previous, f"{free} MB produced a worse split"
            previous = n

    def test_a_longer_context_pushes_experts_off_the_gpu(self, make_gguf):
        """KV cache competes with experts for the same VRAM."""
        info = read_info(make_gguf(n_layers=48))
        short, _ = auto_n_cpu_moe(info, 27 * GB, 8192, free_mb=32000)
        long, _ = auto_n_cpu_moe(info, 27 * GB, 131072, free_mb=32000)
        assert long >= short

    def test_reserve_is_honoured(self, make_gguf):
        """A neighbour on the card -- ComfyUI, Blender -- must be respected."""
        info = read_info(make_gguf(n_layers=48))
        small, _ = auto_n_cpu_moe(info, 27 * GB, 8192, free_mb=32000,
                                  reserve_mb=512)
        large, _ = auto_n_cpu_moe(info, 27 * GB, 8192, free_mb=32000,
                                  reserve_mb=16000)
        assert large >= small

    def test_unknown_geometry_falls_back_safely(self, make_gguf):
        from backend.gguf import GGUFInfo
        info = GGUFInfo(path=make_gguf(), n_layers=0)
        n, why = auto_n_cpu_moe(info, 27 * GB, 65536, free_mb=6144)
        assert why["mode"] == "fallback"

    def test_never_returns_more_layers_than_the_model_has(self, make_gguf):
        info = read_info(make_gguf(n_layers=48))
        for free in (0, 1024, 6144, 100000):
            n, _ = auto_n_cpu_moe(info, 27 * GB, 65536, free_mb=free)
            assert 0 <= n <= 48


class TestBuildCommand:
    def _defaults(self, **kw):
        base = dict(ctx=65536, n_cpu_moe="auto", cache_type_k="q8_0",
                    cache_type_v="q8_0", flash_attn="on", threads=8,
                    batch=2048, ubatch=512, vram_reserve_mb=1024)
        base.update(kw)
        return base

    def test_jinja_is_always_passed(self, make_gguf, tmp_path):
        """Without --jinja tool calls come back as text. Non-negotiable."""
        d = tmp_path / "test-model"
        d.mkdir()
        make_gguf().rename(d / "model.gguf")
        cmd, _ = build_command(entry(), self._defaults(), models_dir=tmp_path)
        assert "--jinja" in cmd

    def test_alias_matches_the_model_id(self, make_gguf, tmp_path):
        """/v1/models must advertise our id, or _serves() rejects the server."""
        d = tmp_path / "test-model"
        d.mkdir()
        make_gguf().rename(d / "model.gguf")
        cmd, _ = build_command(entry(), self._defaults(), models_dir=tmp_path)
        assert cmd[cmd.index("--alias") + 1] == "test-model"

    def test_missing_chat_template_is_refused(self, make_gguf, tmp_path):
        d = tmp_path / "test-model"
        d.mkdir()
        make_gguf(chat_template=False).rename(d / "model.gguf")
        with pytest.raises(EngineError, match="no chat template"):
            build_command(entry(), self._defaults(), models_dir=tmp_path)

    def test_missing_model_file_is_refused(self, tmp_path):
        with pytest.raises(EngineError, match="not on disk"):
            build_command(entry(), self._defaults(), models_dir=tmp_path)

    def test_context_is_clamped_to_the_model_ceiling(self, make_gguf, tmp_path):
        d = tmp_path / "test-model"
        d.mkdir()
        make_gguf().rename(d / "model.gguf")
        cmd, _ = build_command(entry(context_ceiling=32768),
                               self._defaults(ctx=131072), models_dir=tmp_path)
        assert cmd[cmd.index("--ctx-size") + 1] == "32768"

    def test_pinned_n_cpu_moe_bypasses_the_calculation(self, make_gguf, tmp_path):
        """What the sweep writes must be what the engine gets."""
        d = tmp_path / "test-model"
        d.mkdir()
        make_gguf().rename(d / "model.gguf")
        cmd, placement = build_command(
            entry(), self._defaults(n_cpu_moe=24), models_dir=tmp_path)
        assert placement["mode"] == "pinned"
        if "--n-cpu-moe" in cmd:
            assert cmd[cmd.index("--n-cpu-moe") + 1] == "24"

    def test_mmproj_is_passed_when_present(self, make_gguf, tmp_path):
        d = tmp_path / "test-model"
        d.mkdir()
        make_gguf().rename(d / "model.gguf")
        make_gguf(name="mmproj.gguf").rename(d / "mmproj.gguf")
        cmd, placement = build_command(
            entry(vision=True, mmproj="mmproj.gguf"),
            self._defaults(), models_dir=tmp_path)
        assert "--mmproj" in cmd
        assert placement["mmproj"] == "mmproj.gguf"

    def test_template_kwargs_are_passed_through_verbatim(
            self, make_gguf, tmp_path):
        """The model-agnostic seam.

        The engine must not know what `preserve_thinking` means -- only that
        the registry asked for it. Anything JSON-serialisable goes through
        untouched.
        """
        import json
        d = tmp_path / "test-model"
        d.mkdir()
        make_gguf().rename(d / "model.gguf")
        e = entry(chat_template_kwargs={"preserve_thinking": True,
                                        "some_future_flag": "x"})
        cmd, _ = build_command(e, self._defaults(), models_dir=tmp_path)
        sent = json.loads(cmd[cmd.index("--chat-template-kwargs") + 1])
        assert sent == {"preserve_thinking": True, "some_future_flag": "x"}

    def test_unserialisable_template_kwargs_fail_at_build_not_at_load(
            self, make_gguf, tmp_path):
        d = tmp_path / "test-model"
        d.mkdir()
        make_gguf().rename(d / "model.gguf")
        e = entry(chat_template_kwargs={"bad": object()})
        with pytest.raises(EngineError, match="not JSON-serialisable"):
            build_command(e, self._defaults(), models_dir=tmp_path)

    def test_no_template_kwargs_means_no_flag(self, make_gguf, tmp_path):
        d = tmp_path / "test-model"
        d.mkdir()
        make_gguf().rename(d / "model.gguf")
        cmd, _ = build_command(entry(), self._defaults(), models_dir=tmp_path)
        assert "--chat-template-kwargs" not in cmd

    def test_per_model_sampling_reaches_the_command(self, make_gguf, tmp_path):
        """Two models in this registry want different sampling. Neither is
        hardcoded anywhere in the engine."""
        d = tmp_path / "test-model"
        d.mkdir()
        make_gguf().rename(d / "model.gguf")
        e = entry(sampling={"temp": 1.0, "top_k": 40, "min_p": 0.01})
        cmd, _ = build_command(e, self._defaults(), models_dir=tmp_path)
        assert cmd[cmd.index("--temp") + 1] == "1.0"
        assert cmd[cmd.index("--top-k") + 1] == "40"
        assert cmd[cmd.index("--min-p") + 1] == "0.01"
        assert "--top-p" not in cmd          # unset keys are not invented


class TestFastMemory:
    """The budget is RAM + VRAM, not either alone."""

    def test_reports_both_kinds_and_their_sum(self, monkeypatch):
        from backend import engine
        monkeypatch.setattr(engine, "total_ram_gb", lambda: 62)
        monkeypatch.setattr(engine, "available_ram_gb", lambda: 58)
        monkeypatch.setattr(engine, "total_vram_mb", lambda: 6144)
        monkeypatch.setattr(engine, "free_vram_mb", lambda: 5685)
        fm = engine.fast_memory(ram_headroom_gb=12, vram_reserve_mb=1024)
        assert fm["fast_total_gb"] == 68.0
        # 50 GB of RAM + ~4.55 GB of VRAM
        assert 54 < fm["usable_gb"] < 55

    def test_a_machine_with_no_gpu_still_reports_ram(self, monkeypatch):
        from backend import engine
        monkeypatch.setattr(engine, "total_ram_gb", lambda: 62)
        monkeypatch.setattr(engine, "available_ram_gb", lambda: 58)
        monkeypatch.setattr(engine, "total_vram_mb", lambda: 0)
        monkeypatch.setattr(engine, "free_vram_mb", lambda: 0)
        fm = engine.fast_memory()
        assert fm["fast_total_gb"] == 62.0
        assert fm["usable_gb"] == 50.0

    def test_headroom_is_never_negative(self, monkeypatch):
        """A tiny machine must report 0 usable, not a negative budget."""
        from backend import engine
        monkeypatch.setattr(engine, "total_ram_gb", lambda: 8)
        monkeypatch.setattr(engine, "available_ram_gb", lambda: 6)
        monkeypatch.setattr(engine, "total_vram_mb", lambda: 512)
        monkeypatch.setattr(engine, "free_vram_mb", lambda: 256)
        fm = engine.fast_memory(ram_headroom_gb=12, vram_reserve_mb=1024)
        assert fm["usable_gb"] == 0.0
