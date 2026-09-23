"""GGUF header reader."""
from __future__ import annotations

import pytest

from backend.gguf import GGUFError, read_info


class TestReadInfo:
    def test_reads_architecture_and_geometry(self, make_gguf):
        info = read_info(make_gguf(n_layers=48, n_experts=128, n_experts_used=8))
        assert info.architecture == "qwen3moe"
        assert info.n_layers == 48
        assert info.n_experts == 128
        assert info.n_experts_used == 8
        assert info.is_moe

    def test_detects_a_chat_template(self, make_gguf):
        assert read_info(make_gguf(chat_template=True)).chat_template
        assert not read_info(make_gguf(chat_template=False)).chat_template

    def test_rejects_a_non_gguf_file(self, make_gguf):
        bad = make_gguf(magic=b"\x89PNG")
        with pytest.raises(GGUFError, match="not a GGUF file"):
            read_info(bad)

    def test_rejects_a_truncated_file(self, tmp_path):
        """The common failure: an HTML error page saved as .gguf."""
        p = tmp_path / "truncated.gguf"
        p.write_bytes(b"GGUF\x03\x00\x00\x00")
        with pytest.raises(GGUFError, match="unexpected end of header"):
            read_info(p)

    def test_unknown_architecture_reports_zero_not_a_crash(self, make_gguf):
        """A new architecture must degrade to the safe path, not break a run."""
        info = read_info(make_gguf(architecture="somethingnew", n_layers=12))
        assert info.architecture == "somethingnew"
        # The namespaced lookup misses, the suffix scan finds it.
        assert info.n_layers == 12

    def test_dense_model_is_not_moe(self, make_gguf):
        info = read_info(make_gguf(n_experts=0, n_experts_used=0))
        assert not info.is_moe

    def test_reads_only_the_header(self, make_gguf):
        """A 27 GB model must not be read to answer 'how many layers'."""
        big = make_gguf(pad_to_bytes=8_000_000)
        info = read_info(big)
        assert info.n_layers == 48
        assert big.stat().st_size == 8_000_000
