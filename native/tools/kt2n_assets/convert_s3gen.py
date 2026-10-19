"""Convert the pinned acoustic checkpoint, preserving tensor order and arithmetic.

The adjacent s3gen_tensors.tsv defines the expected tensors. Weight-normalization
folding runs on CUDA by default to match the inference checkpoint preparation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import struct
import sys
import time
from typing import Any

import numpy as np

from kt2n_assets.format import ALIGN, HEADER_SIZE, align_up, dumps_meta
from kt2n_assets.safetensors_raw import SafetensorsFile

MAGIC = b"KT2NS3G1"
VERSION = 1
HEADER_STRUCT = struct.Struct("<8sIIQQQQQ8s")
CONVERTER = {"name": "kt2n.convert_s3gen", "version": "1.0.0"}
PINNED_SHA256 = "d65cb687a2ed581ee6cc297e919ffefa63386944f42364ae13b78a594945514f"
PINNED_SIZE = 1064875036
PINNED_REVISION = "1e4698ca7cbb41ff030c4185f0927a3b42d76924"
MANIFEST = pathlib.Path(__file__).with_name("s3gen_tensors.tsv")

# Frozen geometry; the C++ loader compares it with render/ops.h.
CONFIG: dict[str, Any] = {
    "sample_rate": 24000, "token_mel_ratio": 2, "pre_lookahead_len": 3, "vocab_size": 6561, "silence_token": 4299,
    "static_chunk_tokens": 25, "static_chunk_frames": 50,
    "encoder": {"d_model": 512, "heads": 8, "linear_units": 2048, "num_blocks": 6, "num_up_blocks": 4,
                "pos_enc": "rel_pos_espnet", "embed_ln_eps": 1e-5, "layer_ln_eps": 1e-12, "after_norm_eps": 1e-5,
                "up_stride": 2, "up_kernel": 5, "pre_lookahead_kernels": [4, 3]},
    "flow": {"spk_embed_dim": 192, "output_size": 80},
    "estimator": {"in_channels": 320, "out_channels": 80, "channels": [256], "n_blocks": 4, "num_mid_blocks": 12,
                  "num_heads": 8, "attention_head_dim": 64, "act_fn": "gelu", "ln_eps": 1e-5, "time_embed_dim": 1024,
                  "time_scale": 1000, "meanflow": True, "mask_bias": -1e10, "flow_steps": 2, "t_mid": 0.5},
    "hift": {"base_channels": 512, "nb_harmonics": 8, "upsample_rates": [8, 5, 3], "upsample_kernel_sizes": [16, 11, 7],
             "resblock_kernel_sizes": [3, 7, 11], "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
             "source_resblock_kernel_sizes": [7, 7, 11],
             "source_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
             "istft_n_fft": 16, "istft_hop": 4, "lrelu_slope": 0.1, "audio_limit": 0.99, "nsf_alpha": 0.1,
             "nsf_sigma": 0.003, "nsf_voiced_threshold": 10, "f0_upsample_scale": 480},
}
SIZES = {"f32": 4, "bf16": 2}


class ConvertError(ValueError):
    pass


def _sha(data: bytes | memoryview) -> str:
    h = hashlib.sha256()
    mv = memoryview(data)
    for i in range(0, len(mv), 64 << 20):
        h.update(mv[i : i + (64 << 20)])
    return h.hexdigest()


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: pathlib.Path) -> list[dict[str, Any]]:
    lines = path.read_text().splitlines()
    if not lines or lines[0].split("\t") != ["native_name", "group", "dtype", "shape", "transform", "sources"]:
        raise ConvertError(f"{path}: unexpected manifest header")
    rows = []
    for line in lines[1:]:
        if not line:
            continue
        name, group, dtype, shape, transform, sources = line.split("\t")
        dims = [] if shape == "scalar" else [int(d) for d in shape.split("x")]
        rows.append({"native_name": name, "group": group, "dtype": dtype, "shape": dims, "transform": transform,
                     "sources": sources.split(",")})
    return rows


def manifest_from_header(src: SafetensorsFile) -> list[dict[str, Any]]:
    from . import s3gen_manifest
    header = json.loads(src.header_bytes.decode("utf-8"))
    header.pop("__metadata__", None)
    return s3gen_manifest.manifest(header)


def bf16_rne_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even f32 -> bf16 bits (NaN refused by the caller)."""
    u = x.astype("<f4", copy=False).view("<u4").astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype("<u2")


def _torch():
    import torch  # noqa: PLC0415 - only the weight-norm fold and the verifier need torch

    return torch


def weight_norm_fold(v: np.ndarray, g: np.ndarray, device: str) -> np.ndarray:
    torch = _torch()
    tv = torch.from_numpy(np.array(v, order="C", copy=True)).to(device)
    tg = torch.from_numpy(np.array(g, order="C", copy=True)).to(device)
    with torch.no_grad():
        w = torch._weight_norm(tv, tg, 0)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return w.cpu().numpy().astype("<f4", copy=False)


def transform_bytes(src: SafetensorsFile, row: dict[str, Any], device: str) -> np.ndarray:
    t = row["transform"]
    if t == "copy":
        (name,) = row["sources"]
        a = src.array(name)
        if a.dtype != np.dtype("<f4"):
            raise ConvertError(f"{name}: copy expects f32 source, got {a.dtype}")
        return np.ascontiguousarray(a)
    if t == "bf16_rne":
        (name,) = row["sources"]
        a = src.array(name)
        if not np.isfinite(a).all():
            raise ConvertError(f"{name}: non-finite value in an estimator weight")
        return bf16_rne_bits(np.ascontiguousarray(a))
    if t == "weight_norm_dim0":
        v_name, g_name = row["sources"]
        v, g = src.array(v_name), src.array(g_name)
        if v.shape[0] != g.shape[0] or math.prod(g.shape) != g.shape[0]:
            raise ConvertError(f"{row['native_name']}: g/v do not fold on dim 0")
        w = weight_norm_fold(v, g, device)
        if not np.isfinite(w).all():
            raise ConvertError(f"{row['native_name']}: non-finite folded weight")
        return w
    raise ConvertError(f"unknown transform {t!r}")


def pack_header(meta_len: int, data_len: int) -> bytes:
    data_off = align_up(HEADER_SIZE + meta_len)
    return HEADER_STRUCT.pack(MAGIC, VERSION, 0, HEADER_SIZE, meta_len, data_off, data_len, data_off + data_len,
                              b"\0" * 8)


def convert(src_path: str, out_path: str, *, manifest_path: pathlib.Path, expect_sha256: str | None,
            revision: str, device: str, extra_groups: list[tuple[str, str]]) -> dict[str, Any]:
    started = time.time()
    src_path = os.path.realpath(src_path)
    size = os.path.getsize(src_path)
    sha = _file_sha(src_path)
    if expect_sha256 and sha != expect_sha256:
        raise ConvertError(f"source sha256 {sha} != expected {expect_sha256}")
    src = SafetensorsFile(src_path)
    rows = read_manifest(manifest_path)
    built = manifest_from_header(src)
    if len(built) != len(rows):
        raise ConvertError(f"manifest has {len(rows)} rows, the source header gives {len(built)}")
    for a, b in zip(rows, built):
        if (a["native_name"], a["group"], a["dtype"], a["shape"], a["transform"], a["sources"]) != (
                b["native_name"], b["group"], b["dtype"], list(b["shape"]), b["transform"], b["sources"]):
            raise ConvertError(f"manifest row differs from the source header: {a['native_name']} / {b['native_name']}")
    extra_files = []
    for group, path in extra_groups:
        ef = SafetensorsFile(path)
        extra_files.append((group, ef))
        for info in ef.tensors:
            if info.dtype != "F32":
                raise ConvertError(f"extra group {group}: {info.name} is {info.dtype}, only F32 copy is supported")
            rows.append({"native_name": f"{group}.{info.name}", "group": group, "dtype": "f32",
                         "shape": list(info.shape), "transform": "copy", "sources": [info.name], "_file": ef})

    # Pass 1: transform + hash (kept in memory: 918 MB for the pinned file).
    blobs: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    off = 0
    groups: dict[str, dict[str, int]] = {}
    for row in rows:
        sfile = row.pop("_file", src)
        data = transform_bytes(sfile, row, device)
        if list(data.shape) != row["shape"] or data.dtype.itemsize != SIZES[row["dtype"]]:
            raise ConvertError(f"{row['native_name']}: produced {data.dtype} {list(data.shape)}")
        raw = data.tobytes()
        seg_sha = _sha(raw)
        entry = {
            "name": row["native_name"], "kind": "dense", "dtype": row["dtype"], "shape": row["shape"],
            "group": row["group"], "transform": row["transform"],
            "sources": [{"name": n, "sha256": _sha(sfile.raw(n))} for n in row["sources"]],
            "data": {"off": off, "len": len(raw), "shape": row["shape"], "dtype": row["dtype"], "sha256": seg_sha},
        }
        if row["transform"] == "copy":
            entry["src_sha256"] = seg_sha
            if entry["sources"][0]["sha256"] != seg_sha:
                raise ConvertError(f"{row['native_name']}: copy changed the bytes")
        entries.append(entry)
        blobs.append(raw)
        g = groups.setdefault(row["group"], {"tensors": 0, "params": 0, "bytes": 0})
        g["tensors"] += 1
        g["params"] += math.prod(row["shape"])
        g["bytes"] += len(raw)
        off = align_up(off + len(raw))
    data_len = off
    header_obj = json.loads(src.header_bytes.decode("utf-8"))
    meta = {
        "format": "kt2n",
        "version": VERSION,
        "config": CONFIG,
        "source": {
            "safetensors": {
                "file": os.path.basename(src_path), "revision": revision, "size": size, "sha256": sha,
                "header_len": src.header_len, "header_sha256": _sha(src.header_bytes),
                "tensor_count": len([k for k in header_obj if k != "__metadata__"]),
            },
            "converter": dict(CONVERTER, weight_norm_device=_device_name(device)),
            "manifest_sha256": _sha(manifest_path.read_bytes()),
            "extra_groups": [{"group": g, "file": os.path.basename(f.path), "sha256": _file_sha(f.path)}
                             for g, f in extra_files],
        },
        "groups": groups,
        "tensors": entries,
    }
    meta_bytes = dumps_meta(meta)
    head = pack_header(len(meta_bytes), data_len)
    data_off = align_up(HEADER_SIZE + len(meta_bytes))
    tmp = out_path + ".partial"
    with open(tmp, "wb") as f:
        f.write(head)
        f.write(meta_bytes)
        f.write(b"\0" * (data_off - HEADER_SIZE - len(meta_bytes)))
        pos = 0
        for entry, raw in zip(entries, blobs):
            seg_off = entry["data"]["off"]
            if seg_off > pos:
                f.write(b"\0" * (seg_off - pos))
            f.write(raw)
            pos = seg_off + len(raw)
        if data_len > pos:
            f.write(b"\0" * (data_len - pos))
    os.replace(tmp, out_path)
    return {"out": out_path, "file_len": os.path.getsize(out_path), "sha256": _file_sha(out_path),
            "tensors": len(entries), "groups": groups, "data_off": data_off, "meta_len": len(meta_bytes),
            "seconds": round(time.time() - started, 1), "weight_norm_device": _device_name(device)}


def _device_name(device: str) -> str:
    if device.startswith("cuda"):
        torch = _torch()
        return f"{device}:{torch.cuda.get_device_name(0)} torch {torch.__version__}"
    return device


def verify(path: str, src_path: str) -> dict[str, Any]:
    """Container rules + segment sha256 + every transform recomputed from the source."""
    import mmap

    torch = _torch()
    problems: list[str] = []
    size = os.path.getsize(path)
    fh = open(path, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    magic, version, flags, meta_off, meta_len, data_off, data_len, file_len, reserved = HEADER_STRUCT.unpack(
        mm[:HEADER_SIZE])
    if magic != MAGIC or version != VERSION or flags != 0 or reserved != b"\0" * 8:
        problems.append("header magic/version/flags/reserved")
    if meta_off != HEADER_SIZE or data_off != align_up(HEADER_SIZE + meta_len) or file_len != size \
            or data_off + data_len != file_len or data_len % ALIGN:
        problems.append("header offsets")
    meta_bytes = bytes(mm[meta_off : meta_off + meta_len])
    meta = json.loads(meta_bytes)
    if dumps_meta(meta) != meta_bytes:
        problems.append("metadata not canonical")
    if any(mm[meta_off + meta_len : data_off]):
        problems.append("metadata padding not zero")
    if meta.get("config") != CONFIG:
        problems.append("config differs from the frozen geometry")
    src = SafetensorsFile(src_path)
    if meta["source"]["safetensors"]["sha256"] != _file_sha(src_path):
        problems.append("source sha256")
    rows = read_manifest(MANIFEST)
    if [t["name"] for t in meta["tensors"][: len(rows)]] != [r["native_name"] for r in rows]:
        problems.append("tensor order differs from s3gen_tensors.tsv")
    device = "cuda" if "cuda" in meta["source"]["converter"]["weight_norm_device"] else "cpu"
    counts = {"copy": 0, "bf16_rne": 0, "weight_norm_dim0": 0}
    wn_cpu_maxdiff = 0.0
    wn_cpu_ulps = 0
    covered = []
    for t, row in zip(meta["tensors"], rows):
        seg = t["data"]
        if seg["off"] % ALIGN or seg["len"] != math.prod(seg["shape"]) * SIZES[seg["dtype"]] \
                or seg["off"] + seg["len"] > data_len or seg["shape"] != row["shape"] or seg["dtype"] != row["dtype"]:
            problems.append(f"{t['name']}: segment geometry")
            continue
        covered.append((seg["off"], seg["off"] + seg["len"]))
        raw = memoryview(mm)[data_off + seg["off"] : data_off + seg["off"] + seg["len"]]
        if _sha(raw) != seg["sha256"]:
            problems.append(f"{t['name']}: segment sha256")
        for s in t["sources"]:
            if _sha(src.raw(s["name"])) != s["sha256"]:
                problems.append(f"{t['name']}: source sha256 {s['name']}")
        kind = t["transform"]
        counts[kind] += 1
        if kind == "copy":
            if bytes(raw) != bytes(src.raw(t["sources"][0]["name"])) or t.get("src_sha256") != seg["sha256"]:
                problems.append(f"{t['name']}: copy bytes")
        elif kind == "bf16_rne":
            a = torch.from_numpy(np.array(src.array(t["sources"][0]["name"])))
            ref = a.to(torch.bfloat16).view(torch.int16).numpy().view("<u2").reshape(-1)
            got = np.frombuffer(raw, dtype="<u2")
            if not np.array_equal(ref, got):
                problems.append(f"{t['name']}: bf16 differs from torch .to(bfloat16)")
        else:
            v = src.array(t["sources"][0]["name"])
            g = src.array(t["sources"][1]["name"])
            ref = weight_norm_fold(v, g, device).reshape(-1)
            got = np.frombuffer(raw, dtype="<f4")
            if not np.array_equal(ref.view("<u4"), got.view("<u4")):
                problems.append(f"{t['name']}: weight-norm fold differs from {device} recompute")
            cpu = weight_norm_fold(v, g, "cpu").reshape(-1)
            wn_cpu_maxdiff = max(wn_cpu_maxdiff, float(np.abs(cpu - got).max()))
            wn_cpu_ulps += int((cpu.view("<u4") != got.view("<u4")).sum())
    covered.sort()
    cursor = 0
    for a, b in covered + [(data_len, data_len)]:
        if a > cursor and any(mm[data_off + cursor : data_off + a]):
            problems.append(f"non-zero padding at data offset {cursor}")
            break
        cursor = max(cursor, b)
    return {"ok": not problems, "problems": problems[:20], "tensors": len(meta["tensors"]), "transforms": counts,
            "groups": meta.get("groups"), "file_sha256": _file_sha(path), "file_len": size,
            "weight_norm_device": meta["source"]["converter"]["weight_norm_device"],
            "weight_norm_vs_cpu": {"max_abs_diff": wn_cpu_maxdiff, "values_differing": wn_cpu_ulps}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--expect-sha256", default=PINNED_SHA256)
    ap.add_argument("--revision", default=PINNED_REVISION)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extra-group", action="append", default=[], help="NAME=FILE.safetensors (reserved slot)")
    ap.add_argument("--verify", metavar="KT2N")
    ap.add_argument("--report")
    a = ap.parse_args(argv)
    if a.verify:
        if not a.src:
            ap.error("--verify needs the source: --verify OUT.kt2n SRC.safetensors")
        result = verify(a.verify, a.src)
    else:
        if not (a.src and a.out):
            ap.error("SRC and OUT are required")
        extras = [tuple(x.split("=", 1)) for x in a.extra_group]
        result = convert(a.src, a.out, manifest_path=pathlib.Path(a.manifest), expect_sha256=a.expect_sha256,
                         revision=a.revision, device=a.device, extra_groups=extras)  # type: ignore[arg-type]
    text = json.dumps(result, indent=1, sort_keys=True)
    print(text)
    if a.report:
        pathlib.Path(a.report).write_text(text + "\n")
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
