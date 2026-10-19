"""The ``.kt2n`` engine file: constants, header codec, layout rules, reader.

Container layout:

    [0, 64)                 fixed header (little-endian)
    [64, 64+meta_len)       UTF-8 JSON metadata, sorted keys
    zero padding to data_off (4096-aligned)
    [data_off, file_len)    data segments, each 4096-aligned relative to data_off

Header: magic "KT2NENG1" | u32 version=1 | u32 flags=0 | u64 meta_off |
u64 meta_len | u64 data_off | u64 data_len | u64 file_len | 8 zero bytes.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

MAGIC = b"KT2NENG1"
VERSION = 1
FLAGS = 0
HEADER_SIZE = 64
ALIGN = 4096
GROUP = 128
HEADER_STRUCT = struct.Struct("<8sIIQQQQQ8s")
assert HEADER_STRUCT.size == HEADER_SIZE

KIND_TERNARY = "ternary2"
KIND_DENSE = "dense"

QUANT = {
    "group": GROUP,
    "codes": {"0": 0, "1": "+1", "2": "-1"},
    "word": "int32 LSB-first, column 16j+o at bits 2o..2o+1",
}

# Segment dtype name -> (numpy dtype, item size). bf16 is carried as raw uint16 bits.
SEG_DTYPES: dict[str, tuple[np.dtype, int]] = {
    "bf16": (np.dtype("<u2"), 2),
    "f16": (np.dtype("<f2"), 2),
    "f32": (np.dtype("<f4"), 4),
    "f64": (np.dtype("<f8"), 8),
    "i8": (np.dtype("i1"), 1),
    "u8": (np.dtype("u1"), 1),
    "i16": (np.dtype("<i2"), 2),
    "u16": (np.dtype("<u2"), 2),
    "i32": (np.dtype("<i4"), 4),
    "u32": (np.dtype("<u4"), 4),
    "i64": (np.dtype("<i8"), 8),
    "u64": (np.dtype("<u8"), 8),
    "bool": (np.dtype("u1"), 1),
}


class FormatError(ValueError):
    pass


def align_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


def _is_count(x: Any) -> bool:
    """A non-negative JSON integer (bool is not one)."""
    return type(x) is int and x >= 0


def numel(shape: list[int] | tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def dumps_meta(meta: dict[str, Any]) -> bytes:
    """Canonical metadata bytes: sorted keys, compact, UTF-8."""
    return json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class Header:
    version: int
    flags: int
    meta_off: int
    meta_len: int
    data_off: int
    data_len: int
    file_len: int

    def pack(self) -> bytes:
        return HEADER_STRUCT.pack(
            MAGIC, self.version, self.flags, self.meta_off, self.meta_len,
            self.data_off, self.data_len, self.file_len, b"\0" * 8,
        )

    @classmethod
    def unpack(cls, raw: bytes) -> "Header":
        if len(raw) < HEADER_SIZE:
            raise FormatError("file shorter than the 64-byte header")
        magic, version, flags, meta_off, meta_len, data_off, data_len, file_len, reserved = HEADER_STRUCT.unpack(
            raw[:HEADER_SIZE]
        )
        if magic != MAGIC:
            raise FormatError(f"bad magic {magic!r}")
        if version != VERSION:
            raise FormatError(f"unsupported version {version}")
        if flags != FLAGS:
            raise FormatError(f"unsupported flags {flags:#x}")
        if reserved != b"\0" * 8:
            raise FormatError("reserved header bytes are not zero")
        return cls(version, flags, meta_off, meta_len, data_off, data_len, file_len)

    @classmethod
    def for_meta(cls, meta_len: int, data_len: int) -> "Header":
        data_off = align_up(HEADER_SIZE + meta_len)
        return cls(VERSION, FLAGS, HEADER_SIZE, meta_len, data_off, data_len, data_off + data_len)


def segment(off: int, length: int, shape: list[int], dtype: str, sha256: str) -> dict[str, Any]:
    return {"off": off, "len": length, "shape": list(shape), "dtype": dtype, "sha256": sha256}


def tensor_segments(t: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(role, segment) pairs of one tensor entry, in file order."""
    if t["kind"] == KIND_TERNARY:
        roles = ["codes", "scales"] + (["zero_signs"] if "zero_signs" in t else [])
    elif t["kind"] == KIND_DENSE:
        roles = ["data"]
    else:
        raise FormatError(f"unknown kind {t['kind']!r} for {t.get('name')!r}")
    return [(r, t[r]) for r in roles]


class Kt2nFile:
    """Read-only, memory-mapped ``.kt2n`` file with structural validation."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = os.fspath(path)
        self.size = os.path.getsize(self.path)
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.header = Header.unpack(self._mm[:HEADER_SIZE])
        h = self.header
        if h.meta_off != HEADER_SIZE:
            raise FormatError(f"meta_off {h.meta_off} != {HEADER_SIZE}")
        if h.data_off != align_up(h.meta_off + h.meta_len):
            raise FormatError("data_off is not the first 4096 boundary after the metadata")
        if h.file_len != self.size or h.data_off + h.data_len != h.file_len:
            raise FormatError(f"length mismatch: header says {h.file_len}, file is {self.size}")
        if h.data_len % ALIGN:
            raise FormatError("data_len is not a multiple of 4096")
        self.meta_bytes = bytes(self._mm[h.meta_off : h.meta_off + h.meta_len])
        try:
            self.meta: dict[str, Any] = json.loads(self.meta_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormatError(f"metadata is not UTF-8 JSON: {exc}") from exc
        if dumps_meta(self.meta) != self.meta_bytes:
            raise FormatError("metadata is not in canonical form (sorted keys, compact)")
        if self.meta.get("format") != "kt2n" or self.meta.get("version") != VERSION:
            raise FormatError("metadata format/version mismatch")
        if any(self._mm[h.meta_off + h.meta_len : h.data_off]):
            raise FormatError("padding between metadata and data is not zero")
        self.tensors: list[dict[str, Any]] = self.meta["tensors"]
        self.by_name = {t["name"]: t for t in self.tensors}
        if len(self.by_name) != len(self.tensors):
            raise FormatError("duplicate tensor names")
        self._check_layout()

    def _check_layout(self) -> None:
        spans = []
        for t in self.tensors:
            for role, seg in tensor_segments(t):
                if seg["dtype"] not in SEG_DTYPES:
                    raise FormatError(f"{t['name']}.{role}: unknown dtype {seg['dtype']!r}")
                # Offsets, lengths and dims must be non-negative integers, or the checks below
                # (upper bounds only) would let a segment reach before the data region.
                if not (_is_count(seg["off"]) and _is_count(seg["len"]) and isinstance(seg["shape"], list)
                        and all(_is_count(d) for d in seg["shape"])):
                    raise FormatError(f"{t['name']}.{role}: off, len and shape must be non-negative integers")
                if seg["off"] % ALIGN:
                    raise FormatError(f"{t['name']}.{role}: offset not 4096-aligned")
                if seg["len"] != numel(seg["shape"]) * SEG_DTYPES[seg["dtype"]][1]:
                    raise FormatError(f"{t['name']}.{role}: len does not match shape/dtype")
                if seg["off"] + seg["len"] > self.header.data_len:
                    raise FormatError(f"{t['name']}.{role}: segment past the end of data")
                spans.append((seg["off"], seg["off"] + seg["len"], f"{t['name']}.{role}"))
        spans.sort()
        for (a0, a1, an), (b0, _, bn) in zip(spans, spans[1:]):
            if b0 < a1:
                raise FormatError(f"segments overlap: {an} / {bn}")

    # -- access ------------------------------------------------------------
    def seg_bytes(self, seg: dict[str, Any]) -> memoryview:
        start = self.header.data_off + seg["off"]
        return memoryview(self._mm)[start : start + seg["len"]]

    def seg_array(self, seg: dict[str, Any]) -> np.ndarray:
        dt = SEG_DTYPES[seg["dtype"]][0]
        arr = np.frombuffer(self._mm, dtype=dt, count=numel(seg["shape"]), offset=self.header.data_off + seg["off"])
        return arr.reshape(seg["shape"])

    def padding_is_zero(self) -> bool:
        """All bytes of the data region not covered by a segment are zero."""
        covered = sorted(
            (seg["off"], seg["off"] + seg["len"]) for t in self.tensors for _, seg in tensor_segments(t)
        )
        cursor = 0
        mv = memoryview(self._mm)
        base = self.header.data_off
        for a, b in covered + [(self.header.data_len, self.header.data_len)]:
            if a > cursor and any(mv[base + cursor : base + a]):
                return False
            cursor = max(cursor, b)
        return True

    def close(self) -> None:
        try:
            self._mm.close()
        except BufferError:
            pass
        self._fh.close()

    def __enter__(self) -> "Kt2nFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def sha256_hex(data: bytes | memoryview) -> str:
    h = hashlib.sha256()
    mv = memoryview(data)
    step = 64 << 20
    for i in range(0, len(mv), step):
        h.update(mv[i : i + step])
    return h.hexdigest()
