"""Shared fixtures.

The important one builds a real GGUF header on disk. Several modules --
the metadata reader, the offload calculator and the command builder -- all
branch on what the header says, and mocking that out would test the mocks
rather than the parser. A header is a few hundred bytes, so building a genuine
one is cheap and considerably more honest.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

_TYPE_STRING = 8
_TYPE_UINT32 = 4


def _kv(key: str, vtype: int, payload: bytes) -> bytes:
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", vtype) + payload


def _string_value(text: str) -> bytes:
    tb = text.encode()
    return struct.pack("<Q", len(tb)) + tb


@pytest.fixture
def make_gguf(tmp_path: Path):
    """Write a minimal but genuine GGUF file and return its path."""

    def _make(
        name: str = "model.gguf",
        *,
        architecture: str = "qwen3moe",
        n_layers: int = 48,
        n_experts: int = 128,
        n_experts_used: int = 8,
        context_length: int = 262144,
        chat_template: bool = True,
        pad_to_bytes: int = 0,
        magic: bytes = b"GGUF",
    ) -> Path:
        entries: list[bytes] = [
            _kv("general.architecture", _TYPE_STRING, _string_value(architecture)),
            _kv(f"{architecture}.block_count", _TYPE_UINT32,
                struct.pack("<I", n_layers)),
            _kv(f"{architecture}.expert_count", _TYPE_UINT32,
                struct.pack("<I", n_experts)),
            _kv(f"{architecture}.expert_used_count", _TYPE_UINT32,
                struct.pack("<I", n_experts_used)),
            _kv(f"{architecture}.context_length", _TYPE_UINT32,
                struct.pack("<I", context_length)),
        ]
        if chat_template:
            entries.append(_kv("tokenizer.chat_template", _TYPE_STRING,
                               _string_value("{{ messages }}")))

        body = (magic + struct.pack("<I", 3)
                + struct.pack("<Q", 0)                  # tensor_count
                + struct.pack("<Q", len(entries))
                + b"".join(entries))
        if pad_to_bytes > len(body):
            # Stand in for the weights, so size-based arithmetic is exercised
            # against a file of a realistic length.
            body += b"\0" * (pad_to_bytes - len(body))

        path = tmp_path / name
        path.write_bytes(body)
        return path

    return _make


@pytest.fixture
def registry() -> dict[str, Any]:
    import yaml
    root = Path(__file__).resolve().parents[2]
    return yaml.safe_load((root / "models.yaml").read_text())


@pytest.fixture
def mcp_registry() -> dict[str, Any]:
    import yaml
    root = Path(__file__).resolve().parents[2]
    return yaml.safe_load((root / "mcp.yaml").read_text())
