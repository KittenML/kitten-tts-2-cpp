"""Convert a dense BF16 safetensors checkpoint into a ``.kt2n`` engine file.

    python -m kt2n_assets.convert SRC_DIR OUT.kt2n [--runtime-config PATH]

* All projection matrices (q/k/v/o/gate/up/down of every layer) become
  ``ternary2``: production's ``pack_exact`` layout (2-bit codes, 16 per int32,
  LSB first) plus one BF16 scale per 128 weights. Every 128-group must be
  exactly ``{-s, 0, +s}``; anything else (or NaN/Inf, or K % 256 != 0) is
  refused, never approximated.
* Every other tensor is copied byte for byte (``dense``).
* Metadata keeps what is needed to rebuild the source safetensors exactly
  (``python -m kt2n_assets.reconstruct``) and to tie the file to its source.

numpy only; torch is used solely by ``kt2n.verify``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from . import format as fmt
from .safetensors_raw import DTYPES, SafetensorsFile, sha256_bytes, sha256_file

CONVERTER_NAME = "kt2n.convert"
CONVERTER_VERSION = "1.0.0"

# Full safetensors names of the projections. Same set as production's
# ``PROJECTION_RE`` (which matches names under ``lm.model``), with the
# ``model.`` prefix and ``.weight`` suffix of the checkpoint.
PROJECTION_RE = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)
PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
PROJECTIONS_PER_LAYER = len(PROJECTIONS)

SAFETENSORS_NAME = "model.safetensors"
RUNTIME_CONFIG = "runtime_config.json"
VERBATIM_FILES = ("config.json", RUNTIME_CONFIG)

PRODUCTION_PACKING = {
    "scope": "k2048",
    "max_rows": 1,
    "note": (
        "The decode fast path uses packed projections with in_features == 2048 "
        "(q/k/v/o/gate/up) at batch 1. This file packs all projections with the "
        "same exact rule, retaining the source values and signed-zero metadata "
        "needed for bitwise reconstruction."
    ),
}


class ConvertError(ValueError):
    pass


class NotTernaryError(ConvertError):
    pass


# ---------------------------------------------------------------------------
# Packing (numpy, on raw BF16 bits)
# ---------------------------------------------------------------------------
def pack_ternary_bits(w: np.ndarray, name: str = "?") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack a BF16 matrix given as raw ``uint16`` bits ``[N, K]``.

    Returns ``(codes int32 [N, K/16], scales uint16-bits [N, K/128], zero_signs uint8, neg_zero)``.
    Bit-identical to production's ``pack_exact``:

    * scale = max over the group of ``bits & 0x7FFF`` (equals ``amax|w|``: for
      non-negative finite BF16 the bit order is the numeric order);
    * exact iff every element's magnitude bits are 0 or the scale;
    * code 1 for positive non-zero, 2 for negative non-zero, else 0
      (so ``-0.0`` gets code 0, like ``w < 0`` is False in torch);
    * ``packed |= codes[:, o::16] << 2*o`` for o in 0..15, viewed as int32.

    ``zero_signs`` is the sign bit of every zero weight (code 0), in row-major
    order, packed 8 per byte LSB-first (``np.packbits(..., bitorder="little")``);
    ``neg_zero`` counts the ``-0.0`` values (bits 0x8000). ``pack_exact`` drops
    that sign (both zeros get code 0 and unpack to +0.0, numerically equal), so
    the bitmap exists only to let ``reconstruct`` restore the source bytes.
    """
    if w.dtype != np.uint16 or w.ndim != 2:
        raise ConvertError(f"{name}: expected a 2-D uint16 (BF16 bits) array, got {w.dtype} {w.shape}")
    n, k = w.shape
    if k % 256:
        raise ConvertError(f"{name}: K={k} is not divisible by 256 (pack_exact requirement)")
    mag = w & np.uint16(0x7FFF)
    if (mag >= 0x7F80).any():  # exponent all ones: Inf or NaN
        bad = int(np.flatnonzero((mag >= 0x7F80).reshape(-1))[0])
        raise NotTernaryError(f"{name}: non-finite value at flat index {bad}")
    groups = mag.reshape(n, k // fmt.GROUP, fmt.GROUP)
    scales = groups.max(axis=2)
    exact = (groups == 0) | (groups == scales[..., None])
    if not exact.all():
        bad_rows, bad_groups = np.nonzero(~exact.all(axis=2))
        raise NotTernaryError(
            f"{name}: {len(bad_rows)} group(s) are not exactly {{-s, 0, +s}} "
            f"(first: row {int(bad_rows[0])}, group {int(bad_groups[0])})"
        )
    sign = (w >> np.uint16(15)).astype(np.uint32)
    codes = (mag != 0).astype(np.uint32) * (np.uint32(1) + sign)  # 0, 1 (+), 2 (-)
    packed = np.zeros((n, k // 16), dtype=np.uint32)
    for o in range(16):
        packed |= codes[:, o::16] << np.uint32(2 * o)
    zero_signs = np.packbits(sign[mag == 0].astype(np.uint8), bitorder="little")
    neg_zero = int(np.count_nonzero(w == 0x8000))
    return packed.view("<i4"), np.ascontiguousarray(scales, dtype="<u2"), zero_signs, neg_zero


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------
def find_runtime_config(src_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise ConvertError(f"--runtime-config {p} not found")
        return p
    for cand in (src_dir / RUNTIME_CONFIG, src_dir.parent / RUNTIME_CONFIG):
        if cand.is_file():
            return cand
    raise ConvertError(f"{RUNTIME_CONFIG} not found in {src_dir} or its parent (use --runtime-config)")


def source_files(src_dir: Path, st_name: str, runtime_config: Path) -> dict[str, dict[str, Any]]:
    """sha256 + size of every side file (tokenizer etc.); config files also verbatim (base64)."""
    out: dict[str, dict[str, Any]] = {}
    paths = {p.name: p for p in sorted(src_dir.iterdir())
             if p.is_file() and p.name != st_name and p.suffix in (".json", ".txt")}
    paths[RUNTIME_CONFIG] = runtime_config
    for name, path in sorted(paths.items()):
        data = path.read_bytes()
        entry: dict[str, Any] = {"sha256": sha256_bytes(data), "size": len(data)}
        if name in VERBATIM_FILES:
            entry["b64"] = base64.b64encode(data).decode("ascii")
        if path.parent.resolve() != src_dir.resolve():
            entry["location"] = "parent of the source directory"
        out[name] = entry
    return out


def ternary_names(st: SafetensorsFile) -> list[str]:
    return [t.name for t in st.tensors if PROJECTION_RE.match(t.name)]


def expected_projection_names(layers: int) -> set[str]:
    """The exact projection set of a ``layers``-layer model: seven per layer, layers 0..layers-1."""
    return {f"model.layers.{i}.{p}.weight" for i in range(layers) for p in PROJECTIONS}


def check_projection_set(found: set[str], config: dict[str, Any]) -> None:
    """Refuse unless the projections are exactly the seven of every configured layer."""
    layers = config.get("num_hidden_layers")
    if type(layers) is not int or layers < 1:
        raise ConvertError(f"config.json num_hidden_layers must be a positive integer, got {layers!r}")
    want = expected_projection_names(layers)
    if found != want:
        missing, extra = sorted(want - found), sorted(found - want)
        raise ConvertError(
            f"found {len(found)} projection tensors, expected {layers} layers x {PROJECTIONS_PER_LAYER} "
            f"= {len(want)}; missing {missing[:5]}{' ...' if len(missing) > 5 else ''}, "
            f"unexpected {extra[:5]}{' ...' if len(extra) > 5 else ''}"
        )


def _path_keys(p: Path) -> set[Any]:
    """Every way two paths can name the same file: the directory entry, the final target, the inode."""
    keys: set[Any] = {str(p.absolute().parent.resolve() / p.name), str(p.resolve())}
    try:
        st = p.stat()
        keys.add((st.st_dev, st.st_ino))
    except OSError:
        pass
    return keys


def check_output_path(out: Path, inputs: list[Path]) -> None:
    """Refuse an output (or its ``.partial`` temporary) that is one of the source files.

    ``convert`` truncates ``OUT.partial`` and then renames it over ``OUT``; either would destroy a
    source file (the safetensors is still memory-mapped when the rename happens).
    """
    source_keys: set[Any] = set()
    for p in inputs:
        source_keys |= _path_keys(p)
    for cand in (out, out.with_name(out.name + ".partial")):
        if _path_keys(cand) & source_keys:
            raise ConvertError(f"output {out} would overwrite a source file ({cand}); choose another path")


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def _write_at(fh: BinaryIO, pos: int, data: bytes | memoryview) -> None:
    fh.seek(pos)
    fh.write(data)


def convert(
    src_dir: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    runtime_config: str | None = None,
    safetensors_name: str = SAFETENSORS_NAME,
    log: Any = None,
) -> dict[str, Any]:
    """Convert and return a summary dict. Writes ``out_path`` atomically."""
    t0 = time.time()
    src = Path(src_dir)
    out = Path(out_path)
    say = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    st_path = src / safetensors_name
    if not st_path.is_file():
        raise ConvertError(f"{st_path} not found")
    rc_path = find_runtime_config(src, runtime_config)
    check_output_path(out, [st_path, rc_path] + [p for p in src.iterdir() if p.is_file()])
    files = source_files(src, safetensors_name, rc_path)
    config = json.loads(base64.b64decode(files["config.json"]["b64"])) if "config.json" in files else None
    if config is None:
        raise ConvertError(f"config.json not found in {src}")
    runtime_cfg = json.loads(base64.b64decode(files[RUNTIME_CONFIG]["b64"]))

    say(f"hashing {st_path.name} ...")
    st_sha = sha256_file(st_path)
    st = SafetensorsFile(st_path)
    try:
        tern = set(ternary_names(st))
        check_projection_set(tern, config)
        for name in sorted(tern):
            info = st.by_name[name]
            if info.dtype != "BF16" or len(info.shape) != 2:
                raise ConvertError(f"{name}: projection must be a 2-D BF16 matrix, got {info.dtype} {info.shape}")

        # Pass 1: layout (sizes are known from shapes), metadata with placeholder hashes.
        placeholder = "0" * 64
        entries: list[dict[str, Any]] = []
        cursor = 0

        def take(nbytes: int) -> int:
            nonlocal cursor
            off = cursor
            cursor = fmt.align_up(cursor + nbytes)
            return off

        neg_zero_count: dict[str, int] = {}
        for info in st.by_offset:
            kt2n_dtype = DTYPES[info.dtype][0]
            entry: dict[str, Any] = {
                "name": info.name,
                "dtype": kt2n_dtype,
                "shape": list(info.shape),
                "src_sha256": placeholder,
            }
            if info.name in tern:
                n, k = info.shape
                if k % 256:
                    raise ConvertError(f"{info.name}: K={k} is not divisible by 256 (pack_exact requirement)")
                # The zero / -0.0 counts change the layout, so find them before laying out.
                w = st.array(info.name)
                zeros = int(np.count_nonzero((w & np.uint16(0x7FFF)) == 0))
                neg = int(np.count_nonzero(w == 0x8000))
                neg_zero_count[info.name] = neg
                entry["kind"] = fmt.KIND_TERNARY
                codes_len = n * (k // 16) * 4
                scales_len = n * (k // fmt.GROUP) * 2
                entry["codes"] = fmt.segment(take(codes_len), codes_len, [n, k // 16], "i32", placeholder)
                entry["scales"] = fmt.segment(take(scales_len), scales_len, [n, k // fmt.GROUP], "bf16", placeholder)
                entry["zeros"] = zeros
                entry["neg_zero"] = neg
                if neg:
                    zs_len = (zeros + 7) // 8
                    entry["zero_signs"] = fmt.segment(take(zs_len), zs_len, [zs_len], "u8", placeholder)
            else:
                entry["kind"] = fmt.KIND_DENSE
                entry["data"] = fmt.segment(take(info.nbytes), info.nbytes, list(info.shape), kt2n_dtype, placeholder)
            entries.append(entry)
        data_len = cursor  # already a multiple of ALIGN (every take() rounds up)

        meta: dict[str, Any] = {
            "format": "kt2n",
            "version": fmt.VERSION,
            "source": {
                "safetensors": {
                    "file": safetensors_name,
                    "sha256": st_sha,
                    "size": st.size,
                    "header_len": st.header_len,
                    "header_b64": base64.b64encode(st.header_bytes).decode("ascii"),
                    "header_sha256": sha256_bytes(st.header_bytes),
                    "tensor_count": len(st.tensors),
                },
                "files": files,
                "converter": {"name": CONVERTER_NAME, "version": CONVERTER_VERSION},
                "production_packing": PRODUCTION_PACKING,
                "ternary_rule": (
                    "per 128-group along K: scale = amax|w| (bf16 bits), every |w| in {0, scale}; "
                    "code 1 = +scale, 2 = -scale, 0 = zero; NaN/Inf and K % 256 != 0 refused"
                ),
            },
            "config": config,
            "runtime_config": runtime_cfg,
            "quant": fmt.QUANT,
            "tensors": entries,
        }
        meta_len = len(fmt.dumps_meta(meta))
        header = fmt.Header.for_meta(meta_len, data_len)

        # Pass 2: write data, fill in hashes.
        tmp = out.with_name(out.name + ".partial")
        stats = {"ternary": 0, "dense": 0, "groups": 0, "zero_groups": 0, "neg_zero": 0}
        try:
            with open(tmp, "wb") as fh:
                fh.truncate(header.file_len)
                for i, entry in enumerate(entries):
                    name = entry["name"]
                    entry["src_sha256"] = st.tensor_sha256(name)
                    if entry["kind"] == fmt.KIND_TERNARY:
                        w = st.array(name)
                        packed, scales, zero_signs, neg = pack_ternary_bits(w, name)
                        assert neg == neg_zero_count[name]
                        for role, arr in (("codes", packed), ("scales", scales), ("zero_signs", zero_signs)):
                            if role not in entry:
                                continue
                            raw = memoryview(np.ascontiguousarray(arr)).cast("B")
                            assert len(raw) == entry[role]["len"]
                            entry[role]["sha256"] = sha256_bytes(raw)
                            _write_at(fh, header.data_off + entry[role]["off"], raw)
                        stats["ternary"] += 1
                        stats["groups"] += int(scales.size)
                        stats["zero_groups"] += int((scales == 0).sum())
                        stats["neg_zero"] += neg
                    else:
                        raw = st.raw(name)
                        entry["data"]["sha256"] = entry["src_sha256"]
                        _write_at(fh, header.data_off + entry["data"]["off"], raw)
                        stats["dense"] += 1
                    if (i + 1) % 50 == 0:
                        say(f"  {i + 1}/{len(entries)} tensors written ({time.time() - t0:.1f}s)")
                meta_bytes = fmt.dumps_meta(meta)
                if len(meta_bytes) != meta_len:
                    raise ConvertError("internal: metadata length changed after filling hashes")
                _write_at(fh, 0, header.pack())
                _write_at(fh, fmt.HEADER_SIZE, meta_bytes)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, out)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    finally:
        st.close()
    summary = {
        "out": str(out),
        "file_len": header.file_len,
        "meta_len": meta_len,
        "data_off": header.data_off,
        "tensors": len(entries),
        "ternary": stats["ternary"],
        "dense": stats["dense"],
        "groups": stats["groups"],
        "zero_groups": stats["zero_groups"],
        "neg_zero_values": stats["neg_zero"],
        "source_sha256": st_sha,
        "seconds": round(time.time() - t0, 2),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kt2n_assets.convert", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src_dir", help="directory with model.safetensors + config.json (+ tokenizer files)")
    ap.add_argument("out", help="output .kt2n path")
    ap.add_argument("--runtime-config", default=None,
                    help="runtime_config.json (default: SRC_DIR/runtime_config.json, then SRC_DIR/../)")
    ap.add_argument("--safetensors", default=SAFETENSORS_NAME, help="file name inside SRC_DIR")
    ap.add_argument("--expect-sha256", default=None, help="refuse unless the source safetensors has this sha256")
    args = ap.parse_args(argv)
    try:
        if args.expect_sha256:
            got = sha256_file(Path(args.src_dir) / args.safetensors)
            if got != args.expect_sha256.lower():
                raise ConvertError(f"source sha256 {got} != expected {args.expect_sha256}")
        summary = convert(args.src_dir, args.out, runtime_config=args.runtime_config,
                          safetensors_name=args.safetensors)
    except (ConvertError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
