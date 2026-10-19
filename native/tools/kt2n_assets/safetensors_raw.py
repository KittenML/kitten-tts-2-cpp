"""Minimal, dependency-light safetensors reader (numpy only).

We parse the header ourselves instead of using the ``safetensors`` package so
that the converter (a) keeps the exact header bytes, including the trailing
space padding, which ``reconstruct`` needs to rebuild the file byte for byte,
and (b) can memory-map tensors as raw ``uint16`` words without any float
conversion.

File layout (https://github.com/huggingface/safetensors):
    u64 little-endian N | N bytes of UTF-8 JSON header | data buffer
Each header entry is ``name -> {"dtype", "shape", "data_offsets": [a, b]}``
with offsets relative to the start of the data buffer. ``__metadata__`` is an
optional string -> string map.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

# safetensors dtype name -> (kt2n dtype name, numpy dtype, item size)
DTYPES: dict[str, tuple[str, Any, int]] = {
    "BF16": ("bf16", np.dtype("<u2"), 2),  # raw bits; never converted to float here
    "F16": ("f16", np.dtype("<f2"), 2),
    "F32": ("f32", np.dtype("<f4"), 4),
    "F64": ("f64", np.dtype("<f8"), 8),
    "I8": ("i8", np.dtype("i1"), 1),
    "U8": ("u8", np.dtype("u1"), 1),
    "I16": ("i16", np.dtype("<i2"), 2),
    "U16": ("u16", np.dtype("<u2"), 2),
    "I32": ("i32", np.dtype("<i4"), 4),
    "U32": ("u32", np.dtype("<u4"), 4),
    "I64": ("i64", np.dtype("<i8"), 8),
    "U64": ("u64", np.dtype("<u8"), 8),
    "BOOL": ("bool", np.dtype("u1"), 1),
}

MAX_HEADER = 100 * 1024 * 1024  # same limit as the reference implementation


class SafetensorsError(ValueError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str  # safetensors name, e.g. "BF16"
    shape: tuple[int, ...]
    begin: int  # relative to the data buffer
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


class SafetensorsFile:
    """Read-only view of one ``.safetensors`` file, memory-mapped."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = os.fspath(path)
        self.size = os.path.getsize(self.path)
        with open(self.path, "rb") as fh:
            prefix = fh.read(8)
            if len(prefix) != 8:
                raise SafetensorsError("file shorter than 8 bytes")
            (n,) = struct.unpack("<Q", prefix)
            if n > MAX_HEADER or 8 + n > self.size:
                raise SafetensorsError(f"bad header length {n}")
            self.header_bytes: bytes = fh.read(n)
        if len(self.header_bytes) != n:
            raise SafetensorsError("truncated header")
        self.header_len = n
        self.data_start = 8 + n
        try:
            header = json.loads(self.header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetensorsError(f"header is not valid UTF-8 JSON: {exc}") from exc
        if not isinstance(header, dict):
            raise SafetensorsError("header is not a JSON object")
        self.metadata = header.pop("__metadata__", None)
        tensors: list[TensorInfo] = []
        for name, entry in header.items():
            try:
                dtype = entry["dtype"]
                shape = tuple(int(d) for d in entry["shape"])
                begin, end = (int(x) for x in entry["data_offsets"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SafetensorsError(f"bad header entry for {name!r}") from exc
            if dtype not in DTYPES:
                raise SafetensorsError(f"unsupported dtype {dtype!r} for {name!r}")
            info = TensorInfo(name, dtype, shape, begin, end)
            if any(d < 0 for d in shape) or end < begin or info.nbytes != info.numel * DTYPES[dtype][2]:
                raise SafetensorsError(f"inconsistent shape/offsets for {name!r}")
            tensors.append(info)
        # Header (JSON) order, which for files written by safetensors is also
        # the data order; ``by_offset`` is what reconstruction relies on.
        self.tensors: list[TensorInfo] = tensors
        self.by_name: dict[str, TensorInfo] = {t.name: t for t in tensors}
        if len(self.by_name) != len(tensors):
            raise SafetensorsError("duplicate tensor names")
        self.by_offset: list[TensorInfo] = sorted(tensors, key=lambda t: (t.begin, t.end))
        # The data buffer must be fully covered with no holes or overlaps, and
        # nothing may follow it (the safetensors format requires both). This
        # is what makes byte-exact reconstruction from per-tensor data possible.
        cursor = 0
        for t in self.by_offset:
            if t.begin != cursor:
                raise SafetensorsError(f"data buffer hole/overlap at {t.name!r} ({t.begin} != {cursor})")
            cursor = t.end
        self.data_len = cursor
        if self.data_start + self.data_len != self.size:
            raise SafetensorsError(
                f"file size {self.size} != 8 + header {n} + data {self.data_len}"
            )
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ) if self.size else None

    # -- access ------------------------------------------------------------
    def raw(self, name: str) -> memoryview:
        """The tensor's bytes, zero-copy."""
        t = self.by_name[name]
        assert self._mm is not None
        return memoryview(self._mm)[self.data_start + t.begin : self.data_start + t.end]

    def array(self, name: str) -> np.ndarray:
        """Zero-copy read-only numpy view; BF16 comes back as raw ``uint16`` bits."""
        t = self.by_name[name]
        np_dtype = DTYPES[t.dtype][1]
        assert self._mm is not None
        arr = np.frombuffer(self._mm, dtype=np_dtype, count=t.numel, offset=self.data_start + t.begin)
        return arr.reshape(t.shape)

    def tensor_sha256(self, name: str) -> str:
        return sha256_bytes(self.raw(name))

    def close(self) -> None:
        if self._mm is not None:
            try:
                self._mm.close()
            except BufferError:
                pass  # numpy views still alive; the mapping is freed with them
            self._mm = None
        self._fh.close()

    def __enter__(self) -> "SafetensorsFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def sha256_bytes(data: bytes | memoryview, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    mv = memoryview(data)
    for i in range(0, len(mv), chunk):
        h.update(mv[i : i + chunk])
    return h.hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def iter_file_chunks(path: str, chunk: int = 16 << 20) -> Iterator[bytes]:
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                return
            yield block


def write_safetensors(
    path: str | os.PathLike[str],
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
    metadata: dict[str, str] | None = None,
    *,
    pad_to: int = 8,
) -> None:
    """Tiny writer used by tests: ``name -> (dtype, shape, raw bytes)``.

    Mirrors the reference writer's conventions: tensors laid out in the given
    order, header padded with spaces to a multiple of ``pad_to`` bytes.
    """
    header: dict[str, Any] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    for name, (dtype, shape, data) in tensors.items():
        size = DTYPES[dtype][2]
        n = 1
        for d in shape:
            n *= d
        if len(data) != n * size:
            raise ValueError(f"{name}: {len(data)} bytes for shape {shape} {dtype}")
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-(8 + len(blob))) % pad_to
    blob += b" " * pad
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for _, (_, _, data) in tensors.items():
            fh.write(data)
