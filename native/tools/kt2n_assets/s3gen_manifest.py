#!/usr/bin/env python3
"""The s3gen.kt2n tensor manifest, from the source safetensors header only.

Reads the 8-byte length + JSON header of production's S3Gen artifact (no tensor bytes, no torch) and
writes one row per tensor the native file holds:

    native_name  group  dtype  shape  transform  sources

Rules (the converter, L5a, must produce exactly this list; the C++ loader checks it):
  * `*.num_batches_tracked` (int64 BatchNorm training counters) are dropped;
  * a weight-norm pair `<p>.parametrizations.weight.original0` (g) + `.original1` (v) becomes ONE
    tensor `<p>.weight` = torch._weight_norm(v, g, dim=0) in fp32 (what production evaluates on
    every forward), shape of v;
  * group / dtype by prefix (precision policy):
      flow.decoder.estimator.*                 estimator  bf16 (round-to-nearest-even of the fp32
                                               values = production's estimator.to(torch.bfloat16))
      flow.*  (embedding, encoder, encoder_proj, spk_embed_affine_layer)
                                               encoder    f32
      mel2wav.*                                hift       f32
      tokenizer.*                              clone_s3tokenizer  f32
      speaker_encoder.*                        clone_campplus     f32
  * native names are the source names (weight-norm suffix removed) so every tensor maps 1:1 onto
    the PyTorch module tree used by the oracle dumps.

    python -m kt2n_assets.s3gen_manifest SRC.safetensors --out s3gen_tensors.tsv [--expect-sha256 HEX]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct
import sys

WN_G = ".parametrizations.weight.original0"
WN_V = ".parametrizations.weight.original1"
DTYPE_NAMES = {"F32": "f32", "BF16": "bf16", "F16": "f16", "I64": "i64", "I32": "i32"}
SIZES = {"f32": 4, "bf16": 2, "f16": 2, "i64": 8, "i32": 4}
GROUPS = (
    ("flow.decoder.estimator.", "estimator", "bf16"),
    ("flow.", "encoder", "f32"),
    ("mel2wav.", "hift", "f32"),
    ("tokenizer.", "clone_s3tokenizer", "f32"),
    ("speaker_encoder.", "clone_campplus", "f32"),
)


def read_header(path: pathlib.Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def group_of(name: str) -> tuple[str, str]:
    for prefix, group, dtype in GROUPS:
        if name.startswith(prefix):
            return group, dtype
    raise SystemExit(f"tensor outside every group: {name}")


def manifest(header: dict) -> list[dict]:
    rows = []
    names = list(header)  # file order
    seen = set()
    for name in names:
        if name in seen:
            continue
        info = header[name]
        if name.endswith(".num_batches_tracked"):
            seen.add(name)
            continue
        if name.endswith(WN_G) or name.endswith(WN_V):
            base = name[: -len(WN_G)]
            g, v = base + WN_G, base + WN_V
            if g not in header or v not in header:
                raise SystemExit(f"incomplete weight-norm pair at {base}")
            seen.update((g, v))
            group, dtype = group_of(base)
            if group == "estimator":
                raise SystemExit(f"weight norm inside the estimator is not expected: {base}")
            gshape, vshape = header[g]["shape"], header[v]["shape"]
            if gshape[0] != vshape[0] or math.prod(gshape) != gshape[0]:
                raise SystemExit(f"weight-norm g/v shapes do not fold on dim 0: {base} {gshape} {vshape}")
            rows.append({"native_name": base + ".weight", "group": group, "dtype": dtype, "shape": vshape,
                         "transform": "weight_norm_dim0", "sources": [v, g]})
            continue
        seen.add(name)
        group, dtype = group_of(name)
        src_dtype = DTYPE_NAMES.get(info["dtype"], info["dtype"])
        if src_dtype != "f32":
            raise SystemExit(f"unexpected source dtype {info['dtype']} for {name}")
        transform = "bf16_rne" if dtype == "bf16" else "copy"
        rows.append({"native_name": name, "group": group, "dtype": dtype, "shape": info["shape"],
                     "transform": transform, "sources": [name]})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-sha256", default=None, help="check the whole source file first (reads 1 GB)")
    a = ap.parse_args(argv)
    src = pathlib.Path(a.src)
    if a.expect_sha256:
        h = hashlib.sha256()
        with open(src, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        if h.hexdigest() != a.expect_sha256:
            raise SystemExit(f"source sha256 {h.hexdigest()} != expected {a.expect_sha256}")
    header = read_header(src)
    rows = manifest(header)
    lines = ["native_name\tgroup\tdtype\tshape\ttransform\tsources"]
    totals: dict[str, list[int]] = {}
    for r in rows:
        lines.append("\t".join([r["native_name"], r["group"], r["dtype"], "x".join(map(str, r["shape"])) or "scalar",
                                r["transform"], ",".join(r["sources"])]))
        t = totals.setdefault(r["group"], [0, 0, 0])
        n = math.prod(r["shape"])
        t[0] += 1
        t[1] += n
        t[2] += n * SIZES[r["dtype"]]
    pathlib.Path(a.out).write_text("\n".join(lines) + "\n")
    summary = {"source_tensors": len(header), "native_tensors": len(rows),
               "groups": {g: {"tensors": t[0], "params": t[1], "bytes": t[2]} for g, t in totals.items()}}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
