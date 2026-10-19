"""The minimal safetensors reader/writer."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from kt2n_assets.safetensors_raw import SafetensorsError, SafetensorsFile, write_safetensors


def test_roundtrip_and_verbatim_header(tmp_path):
    a = np.arange(12, dtype="<u2").reshape(3, 4)
    b = np.arange(5, dtype="<f4")
    p = tmp_path / "x.safetensors"
    write_safetensors(p, {"a": ("BF16", (3, 4), a.tobytes()), "b": ("F32", (5,), b.tobytes())},
                      metadata={"format": "pt"})
    raw = p.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    assert (8 + n) % 8 == 0 and raw[8 + n - 1 : 8 + n] in (b" ", b"}")
    with SafetensorsFile(p) as st:
        assert st.header_bytes == raw[8 : 8 + n]
        assert st.metadata == {"format": "pt"}
        assert np.array_equal(st.array("a"), a) and st.array("a").dtype == np.uint16
        assert np.array_equal(st.array("b"), b)
        assert [t.name for t in st.by_offset] == ["a", "b"]
        assert st.raw("b").tobytes() == b.tobytes()


def _write_raw(path, header: dict, data: bytes, pad: int = 0) -> None:
    blob = json.dumps(header).encode() + b" " * pad
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


def test_rejects_hole(tmp_path):
    p = tmp_path / "hole.safetensors"
    _write_raw(p, {"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                   "b": {"dtype": "U8", "shape": [2], "data_offsets": [3, 5]}}, b"\0" * 5)
    with pytest.raises(SafetensorsError, match="hole"):
        SafetensorsFile(p)


def test_rejects_trailing_bytes(tmp_path):
    p = tmp_path / "tail.safetensors"
    _write_raw(p, {"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}, b"\0" * 3)
    with pytest.raises(SafetensorsError, match="file size"):
        SafetensorsFile(p)


def test_rejects_shape_offset_mismatch(tmp_path):
    p = tmp_path / "bad.safetensors"
    _write_raw(p, {"a": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 4]}}, b"\0" * 4)
    with pytest.raises(SafetensorsError, match="inconsistent"):
        SafetensorsFile(p)


def test_rejects_unknown_dtype(tmp_path):
    p = tmp_path / "dt.safetensors"
    _write_raw(p, {"a": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]}}, b"\0")
    with pytest.raises(SafetensorsError, match="unsupported dtype"):
        SafetensorsFile(p)
