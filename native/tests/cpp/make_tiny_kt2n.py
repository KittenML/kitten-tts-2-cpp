#!/usr/bin/env python3
"""Write tiny KT2N v1 engine files for the C++ / binding tests (numpy only).

Follows the engine-file FORMAT exactly (include/kt2n/engine_file.h and the
converter lane's the container format in tools/kt2n_assets/format.py): 64-byte little-endian header, canonical
metadata (sorted keys, compact separators, UTF-8), data_off = first 4096
boundary after the metadata, segments at 4096-byte multiples relative to
data_off, data_len a multiple of 4096, zero padding. Ternary tensors carry
"zeros" and "neg_zero"; the up_proj test matrix contains -0.0 weights, so it
also has a "zero_signs" segment (sign bit of every zero weight, row-major,
LSB first).

Outputs (in --out-dir):
  tiny.kt2n            valid file: 2 ternary2 matrices (one with all-zero
                       groups), dense bf16 / f32 tensors
  tiny_badseg.kt2n     one byte of a codes segment flipped after hashing
                       (segment sha + reconstruction both mismatch)
  tiny_badsrc.kt2n     a ternary2 src_sha256 that does not match the weights
  tiny_badcode.kt2n    a code word containing code 3 (hashes recorded over the
                       bad codes, so only invalid_code / src_sha256 fail)
  tiny_badzero.kt2n    up_proj records one -0.0 more than its zero_signs hold
  tiny.expected.json   inputs + expected outputs for the reference kernels,
                       computed here independently of the C++ code

The packing mirrors kitten_tts_09/fast_lm/packed_ternary.py pack_exact on raw
bf16 bits: codes 0 -> 0, +1 -> 1, -1 -> 2, 16 codes per int32 LSB-first
(column 16j+o at bits 2o..2o+1), scales = max|w| per 128-group.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct

import numpy as np

GROUP = 128
ALIGN = 4096
MAGIC = b"KT2NENG1"
VERSION = 1


def f32_to_bf16_bits(a) -> np.ndarray:
    """Round-to-nearest-even float32 -> bf16 bits (inputs here are finite)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    bias = ((u >> 16) & 1) + 0x7FFF
    return ((u + bias) >> 16).astype(np.uint16)


def bf16_bits_to_f32(b) -> np.ndarray:
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def make_ternary(rng: np.random.Generator, n: int, k: int, zero_groups: int = 0,
                 neg_zero_frac: float = 0.0) -> np.ndarray:
    """Random exactly-ternary bf16 matrix [n, k] as raw bits (canonical form)."""
    groups = k // GROUP
    codes = rng.integers(0, 3, size=(n, k), dtype=np.int64)  # 0, 1 (+1), 2 (-1)
    scale_f = rng.uniform(0.004, 0.09, size=(n, groups)).astype(np.float32)
    scales = f32_to_bf16_bits(scale_f)
    # Canonical: a group with a non-zero scale has at least one non-zero code.
    grouped = codes.reshape(n, groups, GROUP)
    empty = (grouped != 0).sum(-1) == 0
    grouped[..., 0] = np.where(empty, 1, grouped[..., 0])
    # Some all-zero groups (scale 0, codes 0), like pruned groups.
    for i in range(zero_groups):
        r, g = (7 * i + 1) % n, (3 * i + 1) % groups
        grouped[r, g, :] = 0
        scales[r, g] = 0
    codes = grouped.reshape(n, k)
    s = np.repeat(scales, GROUP, axis=1)
    w = np.where(codes == 1, s, np.where(codes == 2, s | np.uint16(0x8000), np.uint16(0))).astype(np.uint16)
    if neg_zero_frac > 0:  # some zeros stored as -0.0, like the real checkpoint
        flip = (w == 0) & (rng.random(size=w.shape) < neg_zero_frac)
        w[flip] = np.uint16(0x8000)
    return w


def pack_exact(w_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """numpy mirror of pack_exact: returns (codes int32 [n, k/16], scales bf16 bits [n, k/128])."""
    n, k = w_bits.shape
    if k % 256:
        raise ValueError("K must be divisible by 256")
    f = bf16_bits_to_f32(w_bits)
    grouped = f.reshape(n, k // GROUP, GROUP)
    scales_f = np.abs(grouped).max(-1)
    rebuilt = np.sign(grouped) * scales_f[..., None]
    if not np.array_equal(rebuilt, grouped):  # value equality, like torch.equal
        raise ValueError("not exactly ternary per 128-group")
    codes = np.where(f > 0, 1, np.where(f < 0, 2, 0)).astype(np.uint32)
    packed = np.zeros((n, k // 16), dtype=np.uint32)
    for o in range(16):
        packed |= codes[:, o::16] << np.uint32(2 * o)
    return packed.view(np.int32), f32_to_bf16_bits(scales_f)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def align(x: int) -> int:
    return (x + ALIGN - 1) // ALIGN * ALIGN


def build(tensors: list[dict], *, mutate=None) -> bytes:
    """tensors: [{name, kind, dtype, shape, src: bytes, segs: {seg_name: (bytes, shape, dtype)}, extra: {}}]."""
    blobs: list[tuple[int, bytes]] = []
    entries = []
    cursor = 0
    for t in tensors:
        e = {"name": t["name"], "kind": t["kind"], "dtype": t["dtype"], "shape": list(t["shape"]),
             "src_sha256": t.get("src_sha256_override") or sha(t["src"])}
        e.update(t.get("extra", {}))
        for seg_name, (raw, shape, dtype) in t["segs"].items():
            off = align(cursor)
            e[seg_name] = {"off": off, "len": len(raw), "shape": list(shape), "dtype": dtype,
                           "sha256": sha(raw)}
            blobs.append((off, raw))
            cursor = off + len(raw)
        entries.append(e)
    data_len = align(cursor)
    meta = {
        "format": "kt2n",
        "version": VERSION,
        "source": {"kind": "synthetic-test", "generator": "tests/cpp/make_tiny_kt2n.py", "seed": 20261006},
        "config": {"hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 1,
                   "rms_norm_eps": 1e-06, "vocab_size": 32},
        "runtime_config": {"audio_token_start": 16, "audio_vocab": 8, "text_vocab": 16},
        "quant": {"group": GROUP, "codes": {"0": 0, "1": "+1", "2": "-1"},
                  "word": "int32 LSB-first, column 16j+o at bits 2o..2o+1"},
        "tensors": entries,
    }
    meta_bytes = json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    meta_off = 64
    data_off = align(meta_off + len(meta_bytes))
    file_len = data_off + data_len
    out = bytearray(file_len)
    out[0:64] = MAGIC + struct.pack("<IIQQQQQ", VERSION, 0, meta_off, len(meta_bytes), data_off, data_len,
                                    file_len) + bytes(8)
    out[meta_off:meta_off + len(meta_bytes)] = meta_bytes
    for off, raw in blobs:
        out[data_off + off:data_off + off + len(raw)] = raw
    if mutate is not None:
        mutate(out, data_off, entries)
    return bytes(out)


def ternary_tensor(name: str, w_bits: np.ndarray, codes=None, scales=None, neg_zero_delta: int = 0) -> dict:
    if codes is None:
        codes, scales = pack_exact(w_bits)
    n, k = w_bits.shape
    zero = (w_bits & np.uint16(0x7FFF)) == 0
    zeros = int(np.count_nonzero(zero))
    neg_zero = int(np.count_nonzero(w_bits == np.uint16(0x8000)))
    segs = {"codes": (codes.astype("<i4").tobytes(), [n, k // 16], "i32"),
            "scales": (scales.astype("<u2").tobytes(), [n, k // GROUP], "bf16")}
    if neg_zero > 0:
        signs = np.packbits(((w_bits[zero] >> 15) & 1).astype(np.uint8), bitorder="little")
        segs["zero_signs"] = (signs.tobytes(), [len(signs)], "u8")
    return {"name": name, "kind": "ternary2", "dtype": "bf16", "shape": [n, k],
            "src": w_bits.astype("<u2").tobytes(), "segs": segs,
            "extra": {"zeros": zeros, "neg_zero": neg_zero + neg_zero_delta}}


def dense_tensor(name: str, arr: np.ndarray, dtype: str) -> dict:
    raw = arr.tobytes()
    return {"name": name, "kind": "dense", "dtype": dtype, "shape": list(arr.shape), "src": raw,
            "segs": {"data": (raw, list(arr.shape), dtype)}}


def seq_gemv_f32(w_bits: np.ndarray, x_bits: np.ndarray) -> np.ndarray:
    """Reference order: acc = 0f; for k ascending: acc = f32(acc + f32(w*x))."""
    w = bf16_bits_to_f32(w_bits)
    x = bf16_bits_to_f32(x_bits)
    m_rows, n_rows = x.shape[0], w.shape[0]
    y = np.zeros((m_rows, n_rows), dtype=np.float32)
    for m in range(m_rows):
        prods = (w * x[m][None, :]).astype(np.float32)  # exact products
        acc = np.zeros(n_rows, dtype=np.float32)
        for k in range(w.shape[1]):
            acc = (acc + prods[:, k]).astype(np.float32)
        y[m] = acc
    return y


def rmsnorm_hf(x_bits: np.ndarray, w_bits: np.ndarray, eps: float) -> np.ndarray:
    """HF Qwen3RMSNorm on bf16 with a sequential fp32 sum of squares."""
    x = bf16_bits_to_f32(x_bits)
    w = bf16_bits_to_f32(w_bits)
    out = np.zeros(x.shape, dtype=np.uint16)
    eps32 = np.float32(eps)
    for m in range(x.shape[0]):
        ss = np.float32(0.0)
        for v in x[m]:
            ss = np.float32(ss + np.float32(v * v))
        var = np.float32(ss / np.float32(x.shape[1]))
        r = np.float32(np.float32(1.0) / np.sqrt(np.float32(var + eps32)))
        h = bf16_bits_to_f32(f32_to_bf16_bits((x[m] * r).astype(np.float32)))
        out[m] = f32_to_bf16_bits((w * h).astype(np.float32))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(20261006)

    up = make_ternary(rng, 24, 512, neg_zero_frac=0.5)
    q = make_ternary(rng, 16, 256, zero_groups=3)
    embed = f32_to_bf16_bits(rng.normal(0, 0.02, size=(32, 64)))
    norm_w = f32_to_bf16_bits(rng.uniform(0.5, 1.5, size=(256,)))
    bias = rng.normal(0, 0.1, size=(16,)).astype("<f4")

    def tensors(up_codes=None, up_scales=None, src_override=None, neg_zero_delta=0):
        t_up = ternary_tensor("model.layers.0.mlp.up_proj.weight", up, up_codes, up_scales, neg_zero_delta)
        if src_override:
            t_up["src_sha256_override"] = src_override
        return [
            dense_tensor("model.embed_tokens.weight", embed.astype("<u2"), "bf16"),
            t_up,
            ternary_tensor("model.layers.0.self_attn.q_proj.weight", q),
            dense_tensor("model.norm.weight", norm_w.astype("<u2"), "bf16"),
            dense_tensor("spk_proj.0.bias", bias, "f32"),
        ]

    def write(name: str, blob: bytes) -> None:
        with open(os.path.join(args.out_dir, name), "wb") as fh:
            fh.write(blob)

    write("tiny.kt2n", build(tensors()))

    def flip_codes(out: bytearray, data_off: int, entries: list) -> None:
        e = next(x for x in entries if x["name"] == "model.layers.0.mlp.up_proj.weight")
        out[data_off + e["codes"]["off"] + 5] ^= 0x01  # code 0 <-> 1 or 2 <-> 3 in word 1

    write("tiny_badseg.kt2n", build(tensors(), mutate=flip_codes))
    write("tiny_badsrc.kt2n", build(tensors(src_override=sha(b"not the weights"))))

    up_codes, up_scales = pack_exact(up)
    bad_codes = up_codes.view(np.uint32).copy()
    bad_codes[2, 3] |= np.uint32(3 << 6)  # column 16*3+3 of row 2 -> code 3
    write("tiny_badcode.kt2n", build(tensors(up_codes=bad_codes.view(np.int32), up_scales=up_scales)))
    write("tiny_badzero.kt2n", build(tensors(neg_zero_delta=1)))

    # Reference-kernel expectations (computed here, independently of C++).
    x = f32_to_bf16_bits(rng.normal(0, 1.0, size=(3, 512)))
    # Kernels see production's unpack (every zero is +0.0), so the expected
    # GEMV uses the weights with zero signs dropped.
    up_plus_zero = np.where((up & np.uint16(0x7FFF)) == 0, np.uint16(0), up)
    y = seq_gemv_f32(up_plus_zero, x)
    x_norm = f32_to_bf16_bits(rng.normal(0, 2.0, size=(2, 256)))
    expected = {
        "gemv": {"tensor": "model.layers.0.mlp.up_proj.weight", "M": 3, "N": 24, "K": 512,
                 "x_bf16": x.astype(int).ravel().tolist(),
                 "y_f32_bits": y.view(np.uint32).astype(int).ravel().tolist()},
        "rmsnorm": {"weight": "model.norm.weight", "M": 2, "D": 256, "eps": 1e-6,
                    "x_bf16": x_norm.astype(int).ravel().tolist(),
                    "y_bf16": rmsnorm_hf(x_norm, norm_w, 1e-6).astype(int).ravel().tolist()},
        "unpack": {"tensor": "model.layers.0.self_attn.q_proj.weight",
                   "sha256_bf16": sha(q.astype("<u2").tobytes())},
        "zeros": {"tensor": "model.layers.0.mlp.up_proj.weight",
                  "zeros": int(np.count_nonzero((up & np.uint16(0x7FFF)) == 0)),
                  "neg_zero": int(np.count_nonzero(up == np.uint16(0x8000))),
                  "sha256_plus_zero": sha(up_plus_zero.astype("<u2").tobytes())},
    }
    with open(os.path.join(args.out_dir, "tiny.expected.json"), "w", encoding="utf-8") as fh:
        json.dump(expected, fh, sort_keys=True)
    print(f"wrote tiny engine files to {args.out_dir}")


if __name__ == "__main__":
    main()
