"""Rebuild the original safetensors file, byte for byte, from a ``.kt2n`` file.

    python -m kt2n_assets.reconstruct OUT.kt2n OUT.safetensors
    python -m kt2n_assets.reconstruct OUT.kt2n - | sha256sum      # stream, no disk

The stored header bytes are written verbatim, then every tensor in the
header's data order: dense tensors are copied, ternary tensors are unpacked
from codes + scales (with recorded ``-0.0`` positions restored). The result's
sha256 must equal ``source.safetensors.sha256``; the tool checks this and every
tensor's ``src_sha256`` on the way and exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import sys
from typing import Any, BinaryIO, Callable

import numpy as np

from . import format as fmt


class ReconstructError(ValueError):
    pass


def unpack_ternary_bits(packed: np.ndarray, scales: np.ndarray, k: int,
                        zero_signs: np.ndarray | None = None) -> np.ndarray:
    """Inverse of ``convert.pack_ternary_bits``: raw BF16 bits ``uint16 [N, K]``.

    Without ``zero_signs`` every zero comes back as +0.0 (what production's
    ``unpack`` gives); with it, the recorded sign of each zero is restored.
    """
    n = packed.shape[0]
    if packed.shape != (n, k // 16) or scales.shape != (n, k // fmt.GROUP):
        raise ReconstructError(f"shape mismatch: codes {packed.shape}, scales {scales.shape}, K={k}")
    words = packed.view("<u4")
    codes = np.empty((n, k), dtype=np.uint16)
    for o in range(16):
        codes[:, o::16] = (words >> np.uint32(2 * o)) & np.uint32(3)
    if (codes == 3).any():
        raise ReconstructError("code 3 is not a valid ternary code")
    expanded = np.repeat(scales.view("<u2"), fmt.GROUP, axis=1)
    out = np.where(codes == 0, np.uint16(0), expanded | ((codes == 2).astype(np.uint16) << np.uint16(15)))
    out = out.astype("<u2", copy=False)
    if zero_signs is not None:
        zero = codes == 0
        nzero = int(np.count_nonzero(zero))
        signs = np.asarray(zero_signs, dtype=np.uint8).reshape(-1)
        if signs.size != (nzero + 7) // 8:
            raise ReconstructError(f"zero_signs has {signs.size} bytes for {nzero} zeros")
        bits = np.unpackbits(signs, bitorder="little")
        if bits[nzero:].any():
            raise ReconstructError("zero_signs padding bits are not zero")
        out[zero] = bits[:nzero].astype(np.uint16) << np.uint16(15)
    return out


def tensor_bytes(kt: fmt.Kt2nFile, t: dict[str, Any]) -> bytes | memoryview:
    """The source bytes of one tensor."""
    if t["kind"] == fmt.KIND_DENSE:
        return kt.seg_bytes(t["data"])
    if t["kind"] == fmt.KIND_TERNARY:
        k = int(t["shape"][1])
        zs = kt.seg_array(t["zero_signs"]) if "zero_signs" in t else None
        bits = unpack_ternary_bits(kt.seg_array(t["codes"]), kt.seg_array(t["scales"]), k, zs)
        return memoryview(bits).cast("B")
    raise ReconstructError(f"unknown kind {t['kind']!r}")


def reconstruct(kt: fmt.Kt2nFile, sink: Callable[[bytes | memoryview], Any],
                progress: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """Stream the reconstructed safetensors into ``sink``; returns hashes and checks."""
    src = kt.meta["source"]["safetensors"]
    header_bytes = base64.b64decode(src["header_b64"])
    if hashlib.sha256(header_bytes).hexdigest() != src["header_sha256"]:
        raise ReconstructError("stored header does not match its sha256")
    header = json.loads(header_bytes)
    header.pop("__metadata__", None)
    order = sorted(header.items(), key=lambda kv: tuple(kv[1]["data_offsets"]))
    if sorted(header) != sorted(kt.by_name):
        raise ReconstructError("tensor names in the stored header and in the metadata differ")
    total = hashlib.sha256()

    def emit(data: bytes | memoryview) -> None:
        total.update(data)
        sink(data)

    emit(struct.pack("<Q", len(header_bytes)))
    emit(header_bytes)
    cursor = 0
    bad: list[str] = []
    size = 8 + len(header_bytes)
    for i, (name, h) in enumerate(order):
        begin, end = h["data_offsets"]
        if begin != cursor:
            raise ReconstructError(f"stored header has a hole/overlap at {name}")
        t = kt.by_name[name]
        data = tensor_bytes(kt, t)
        if len(data) != end - begin:
            raise ReconstructError(f"{name}: {len(data)} bytes, header says {end - begin}")
        if hashlib.sha256(data).hexdigest() != t["src_sha256"]:
            bad.append(name)
        emit(data)
        size += len(data)
        cursor = end
        if progress and (i + 1) % 50 == 0:
            progress(f"  {i + 1}/{len(order)} tensors")
    digest = total.hexdigest()
    return {
        "sha256": digest,
        "size": size,
        "expected_sha256": src["sha256"],
        "expected_size": src["size"],
        "tensor_sha_mismatches": bad,
        "ok": digest == src["sha256"] and size == src["size"] and not bad,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kt2n_assets.reconstruct", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kt2n", help="input .kt2n file")
    ap.add_argument("out", help="output .safetensors path, or - for stdout")
    args = ap.parse_args(argv)
    say = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731
    with fmt.Kt2nFile(args.kt2n) as kt:
        if args.out == "-":
            out: BinaryIO = sys.stdout.buffer
            result = reconstruct(kt, out.write, say)
            out.flush()
        else:
            tmp = args.out + ".partial"
            try:
                with open(tmp, "wb") as fh:
                    result = reconstruct(kt, fh.write, say)
                if result["ok"]:
                    os.replace(tmp, args.out)
                else:
                    os.unlink(tmp)
            except BaseException:
                # Same as the converter: an exception (corrupt engine, full disk, Ctrl-C) leaves no partial file.
                if os.path.lexists(tmp):
                    os.unlink(tmp)
                raise
    say(json.dumps(result, sort_keys=True))
    if not result["ok"]:
        say("ERROR: reconstruction does not match the source")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
