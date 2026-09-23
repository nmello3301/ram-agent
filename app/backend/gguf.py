"""Minimal GGUF metadata reader.

Only the header is read -- the key/value block at the front of the file -- so
this is a few hundred bytes of I/O against a 27 GB model, not a load.

It exists for one reason: `-ncmoe` cannot be sized honestly without knowing how
many layers the model actually has. Guessing that number is how you end up
either wasting VRAM or OOM-ing halfway through a benchmark run. The registry
could carry it as a hand-typed field, but then it is one more thing that goes
stale silently when a quant is re-uploaded.

Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"

# GGUF value type enum -> (struct format, size). Strings and arrays are handled
# separately because they are variable length.
_SCALARS: dict[int, tuple[str, int]] = {
    0: ("<B", 1),   # uint8
    1: ("<b", 1),   # int8
    2: ("<H", 2),   # uint16
    3: ("<h", 2),   # int16
    4: ("<I", 4),   # uint32
    5: ("<i", 4),   # int32
    6: ("<f", 4),   # float32
    7: ("<?", 1),   # bool
    10: ("<Q", 8),  # uint64
    11: ("<q", 8),  # int64
    12: ("<d", 8),  # float64
}
_STRING = 8
_ARRAY = 9


class GGUFError(RuntimeError):
    """The file is not GGUF, or its header is unreadable."""


def _read(fh: BinaryIO, n: int) -> bytes:
    buf = fh.read(n)
    if len(buf) != n:
        raise GGUFError(f"unexpected end of header (wanted {n} bytes, got {len(buf)})")
    return buf


def _u32(fh: BinaryIO) -> int:
    return struct.unpack("<I", _read(fh, 4))[0]


def _u64(fh: BinaryIO) -> int:
    return struct.unpack("<Q", _read(fh, 8))[0]


def _string(fh: BinaryIO) -> str:
    return _read(fh, _u64(fh)).decode("utf-8", errors="replace")


def _value(fh: BinaryIO, vtype: int) -> Any:
    if vtype in _SCALARS:
        fmt, size = _SCALARS[vtype]
        return struct.unpack(fmt, _read(fh, size))[0]
    if vtype == _STRING:
        return _string(fh)
    if vtype == _ARRAY:
        item_type = _u32(fh)
        count = _u64(fh)
        # Arrays here are things like tokenizer vocabularies: hundreds of
        # thousands of strings we have no use for. Skip them rather than
        # building the list, but stay byte-exact so the cursor stays aligned.
        if item_type in _SCALARS and count > 64:
            fh.seek(_SCALARS[item_type][1] * count, 1)
            return f"<{count} items>"
        return [_value(fh, item_type) for _ in range(count)]
    raise GGUFError(f"unknown GGUF value type {vtype}")


@dataclass
class GGUFInfo:
    """The handful of header fields this project actually uses."""

    path: Path
    architecture: str = "unknown"
    n_layers: int = 0
    n_experts: int = 0
    n_experts_used: int = 0
    context_length: int = 0
    chat_template: bool = False
    raw: dict[str, Any] | None = None

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "n_experts_used": self.n_experts_used,
            "context_length": self.context_length,
            "chat_template": self.chat_template,
            "is_moe": self.is_moe,
        }


def read_info(path: Path, max_kv: int = 4096) -> GGUFInfo:
    """Read architecture, layer count and expert counts out of a GGUF header.

    Raises GGUFError if the file is not GGUF. Never raises for a *missing*
    field -- an architecture this reader has not seen simply reports zeros, and
    the caller falls back to a safe default rather than crashing a run.
    """
    with path.open("rb") as fh:
        if _read(fh, 4) != MAGIC:
            raise GGUFError(f"{path.name}: not a GGUF file")
        version = _u32(fh)
        if version < 2:
            raise GGUFError(f"{path.name}: GGUF version {version} is too old")
        _u64(fh)                      # tensor_count, unused here
        kv_count = _u64(fh)
        if kv_count > max_kv:
            raise GGUFError(f"{path.name}: implausible metadata count {kv_count}")

        kv: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _string(fh)
            kv[key] = _value(fh, _u32(fh))

    arch = kv.get("general.architecture", "unknown")
    return GGUFInfo(
        path=path,
        architecture=arch,
        # Per the spec these are namespaced by architecture, e.g.
        # "qwen3moe.block_count". Look the arch up first, then fall back to a
        # scan so a new architecture name does not silently report zero.
        n_layers=_lookup(kv, arch, "block_count"),
        n_experts=_lookup(kv, arch, "expert_count"),
        n_experts_used=_lookup(kv, arch, "expert_used_count"),
        context_length=_lookup(kv, arch, "context_length"),
        chat_template="tokenizer.chat_template" in kv,
        raw=kv,
    )


def _lookup(kv: dict[str, Any], arch: str, suffix: str) -> int:
    """Read `<arch>.<suffix>`, else any key ending in `.<suffix>`."""
    exact = kv.get(f"{arch}.{suffix}")
    if isinstance(exact, int):
        return exact
    for key, value in kv.items():
        if key.endswith(f".{suffix}") and isinstance(value, int):
            return value
    return 0
