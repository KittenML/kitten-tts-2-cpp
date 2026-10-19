"""Verify a .kt2n model against source safetensors from the source checkout.

    python -m kt2n_assets.verify model.kt2n --src SOURCE_DIR --report report.json

Checks file structure, source identity, dense bytes, exact ternary packing,
signed-zero reconstruction and the retained packing reference's integrity.
NumPy and CPU PyTorch are required for the complete verification.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

from . import format as fmt
from .convert import PROJECTION_RE, find_runtime_config, pack_ternary_bits
from .reconstruct import reconstruct, unpack_ternary_bits
from .safetensors_raw import DTYPES, SafetensorsFile, sha256_bytes, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = REPO_ROOT / "tests" / "reference"
VENDORED = REFERENCE_DIR / "upstream_packed_ternary.py"
PROVENANCE = REFERENCE_DIR / "PROVENANCE.json"
AUDIT_NAMES = REFERENCE_DIR / "ternary_audit_names.json"
MAX_LISTED = 10


class Check:
    def __init__(self, name: str, required: bool = True):
        self.name = name
        self.required = required
        self.failures: list[str] = []
        self.count = 0
        self.skipped: str | None = None
        self.details: dict[str, Any] = {}
        self.t0 = time.time()
        self.seconds = 0.0

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def done(self) -> "Check":
        self.seconds = round(time.time() - self.t0, 2)
        return self

    @property
    def ok(self) -> bool:
        return self.skipped is None and not self.failures

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"ok": self.ok, "required": self.required, "checked": self.count,
                             "seconds": self.seconds,
                             "failures": len(self.failures), "first_failures": self.failures[:MAX_LISTED]}
        if self.skipped:
            d["skipped"] = self.skipped
        d.update(self.details)
        return d


# ---------------------------------------------------------------------------
def load_vendored() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("kt2n_vendored_packed_ternary", VENDORED)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {VENDORED}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def to_torch_bf16(bits: np.ndarray) -> Any:
    import torch

    return torch.from_numpy(np.ascontiguousarray(bits).view(np.int16).copy()).view(torch.bfloat16)


def torch_bits(t: Any) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


# ---------------------------------------------------------------------------
def verify(kt2n_path: str, src_dir: str, *, runtime_config: str | None = None,
           expect_sha256: str | None = None, label: str | None = None,
           audit_mode: str = "auto", say: Any = None) -> dict[str, Any]:
    say = say or (lambda msg: print(msg, file=sys.stderr, flush=True))
    t_start = time.time()
    src = Path(src_dir)
    checks: dict[str, Check] = {}
    report: dict[str, Any] = {"tool": "kt2n.verify", "label": label}

    try:
        import torch  # noqa: F401

        torch_version = torch.__version__
    except Exception as exc:  # pragma: no cover - depends on environment
        torch_version = None
        torch_error = repr(exc)
    report["env"] = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch_version,
                     "machine": platform.machine(), "cpu_count": os.cpu_count()}

    # -- structure ---------------------------------------------------------
    c = checks["structure"] = Check("structure")
    say("structure: header, layout, segment sha256 ...")
    kt = fmt.Kt2nFile(kt2n_path)  # raises on structural errors (reported as a crash)
    h = kt.header
    c.details.update({"file_len": h.file_len, "meta_len": h.meta_len, "data_off": h.data_off,
                      "data_len": h.data_len, "tensors": len(kt.tensors)})
    if not kt.padding_is_zero():
        c.fail("non-zero bytes in segment padding")
    for t in kt.tensors:
        for role, seg in fmt.tensor_segments(t):
            c.count += 1
            if fmt.sha256_hex(kt.seg_bytes(seg)) != seg["sha256"]:
                c.fail(f"{t['name']}.{role}: sha256 mismatch")
    report["kt2n"] = {"file": os.path.basename(kt2n_path), "size": kt.size,
                      "sha256": sha256_file(kt2n_path),
                      "converter": kt.meta["source"].get("converter")}
    c.done()

    # -- source identity ---------------------------------------------------
    c = checks["source_identity"] = Check("source_identity")
    say("source_identity: source sha256, header, side files ...")
    meta_src = kt.meta["source"]
    st_meta = meta_src["safetensors"]
    st_path = src / st_meta["file"]
    st_sha = sha256_file(st_path)
    c.details["source_sha256"] = st_sha
    c.count += 1
    if st_sha != st_meta["sha256"]:
        c.fail(f"source sha256 {st_sha} != recorded {st_meta['sha256']}")
    if expect_sha256 and st_sha != expect_sha256.lower():
        c.fail(f"source sha256 {st_sha} != --expect-sha256 {expect_sha256}")
    st = SafetensorsFile(st_path)
    c.count += 1
    if st.size != st_meta["size"]:
        c.fail("source size differs")
    if base64.b64decode(st_meta["header_b64"]) != st.header_bytes:
        c.fail("stored safetensors header bytes differ from the source header")
    if sha256_bytes(st.header_bytes) != st_meta["header_sha256"]:
        c.fail("header sha256 differs")
    rc_path = find_runtime_config(src, runtime_config)
    for name, entry in sorted(meta_src["files"].items()):
        path = rc_path if name == "runtime_config.json" else src / name
        c.count += 1
        if not path.is_file():
            c.fail(f"side file {name} missing from the source")
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != entry["sha256"] or len(data) != entry["size"]:
            c.fail(f"side file {name} differs")
        if "b64" in entry and base64.b64decode(entry["b64"]) != data:
            c.fail(f"verbatim copy of {name} differs")
    extra = sorted(p.name for p in src.iterdir() if p.is_file() and p.name != st_meta["file"]
                   and p.name not in meta_src["files"])
    if extra:
        c.details["unrecorded_side_files"] = extra
    if kt.meta["config"] != json.loads(base64.b64decode(meta_src["files"]["config.json"]["b64"])):
        c.fail("config object != verbatim config.json")
    if kt.meta["runtime_config"] != json.loads(base64.b64decode(meta_src["files"]["runtime_config.json"]["b64"])):
        c.fail("runtime_config object != verbatim runtime_config.json")
    c.done()

    # -- tensor set ----------------------------------------------------------
    c = checks["tensor_set"] = Check("tensor_set")
    src_names = {t.name for t in st.tensors}
    kt_names = set(kt.by_name)
    ternary = {t["name"] for t in kt.tensors if t["kind"] == fmt.KIND_TERNARY}
    projections = {n for n in src_names if PROJECTION_RE.match(n)}
    c.count = len(kt_names)
    if src_names != kt_names:
        c.fail(f"tensor names differ: only in source {sorted(src_names - kt_names)[:5]}, "
               f"only in kt2n {sorted(kt_names - src_names)[:5]}")
    if ternary != projections:
        c.fail("ternary set != projection set")
    c.details.update({"tensors": len(kt_names), "ternary": len(ternary), "dense": len(kt_names - ternary)})
    c.done()

    # -- audit fixture (production checkpoint only) ------------------------------
    # Production's ternary audit fixture describes one checkpoint (its sha256 is
    # recorded in the fixture). It applies when the source is that checkpoint, or
    # always with --audit require.
    audit = json.loads(AUDIT_NAMES.read_text()) if AUDIT_NAMES.is_file() else None
    applies = audit_mode == "require" or (
        audit_mode == "auto" and audit is not None and st_sha == audit.get("checkpoint_sha256"))
    c = checks["audit_set"] = Check("audit_set", required=applies)
    if not applies:
        c.skipped = ("--audit off" if audit_mode == "off" else
                     "source is not the audited checkpoint" if audit is not None else f"{AUDIT_NAMES} not found")
    elif audit is None:
        c.skipped = f"{AUDIT_NAMES} not found"
    else:
        c.count = len(audit["exactly_ternary_names"])
        c.details["audit_vendored"] = {"names": len(audit["exactly_ternary_names"]),
                                       "equal": set(audit["exactly_ternary_names"]) == ternary,
                                       "checkpoint_sha256": audit.get("checkpoint_sha256")}
        if set(audit["exactly_ternary_names"]) != ternary:
            c.fail("ternary set != vendored audit fixture names")
        if set(audit["not_exactly_ternary_names"]) & ternary:
            c.fail("a tensor the audit calls non-ternary was packed")
    c.done()

    # -- dense bytes -----------------------------------------------------------
    c = checks["dense_bytes"] = Check("dense_bytes")
    say("dense_bytes ...")
    dense_bytes_total = 0
    for t in kt.tensors:
        if t["kind"] != fmt.KIND_DENSE or t["name"] not in st.by_name:
            continue
        c.count += 1
        a = kt.seg_bytes(t["data"])
        b = st.raw(t["name"])
        dense_bytes_total += len(b)
        info = st.by_name[t["name"]]
        if len(a) != len(b) or not np.array_equal(np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)):
            c.fail(f"{t['name']}: bytes differ")
        if list(info.shape) != t["shape"] or t["data"]["shape"] != t["shape"]:
            c.fail(f"{t['name']}: shape differs")
        if t["dtype"] != DTYPES[info.dtype][0] or t["data"]["dtype"] != t["dtype"]:
            c.fail(f"{t['name']}: dtype differs")
        if t["src_sha256"] != t["data"]["sha256"]:
            c.fail(f"{t['name']}: src_sha256 != data sha256")
    c.details["bytes"] = dense_bytes_total
    c.done()

    # -- retained reference integrity -------------------------------------------
    c = checks["reference_integrity"] = Check("reference_integrity")
    c.count = 1
    try:
        expected = json.loads(PROVENANCE.read_text())["symbols"]
        text = VENDORED.read_text()
        found = {}
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef):
                found[node.name] = ast.get_source_segment(text, node)
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                found[node.targets[0].id] = ast.get_source_segment(text, node)
        if set(found) != set(expected):
            c.fail("packing reference symbols differ from provenance")
        for name, source in found.items():
            if name not in expected or hashlib.sha256(source.encode()).hexdigest() != expected[name]["sha256"]:
                c.fail(f"packing reference symbol differs from its recorded source hash: {name}")
    except (OSError, KeyError, ValueError, SyntaxError) as exc:
        c.fail(f"cannot verify retained reference: {exc!r}")
    c.done()

    # -- ternary checks ----------------------------------------------------------
    c_np = checks["ternary_numpy"] = Check("ternary_numpy")
    c_ven = checks["pack_exact_vendored"] = Check("pack_exact_vendored")
    c_unp = checks["unpack_lossless"] = Check("unpack_lossless")
    vendored = None
    if not checks["reference_integrity"].ok:
        c_ven.skipped = "retained reference integrity failed"
    elif torch_version is None:
        c_ven.skipped = f"torch not importable: {torch_error}"
        c_unp.details["torch_unpack"] = "skipped (no torch)"
    else:
        try:
            vendored = load_vendored()
        except Exception as exc:
            c_ven.skipped = f"cannot load vendored reference: {exc!r}"
    stats = {"groups": 0, "zero_groups": 0, "zeros": 0, "neg_zero": 0, "weights": 0, "zero_signs_bytes": 0}
    tern_entries = [t for t in kt.tensors if t["kind"] == fmt.KIND_TERNARY and t["name"] in st.by_name]
    say(f"ternary: {len(tern_entries)} matrices (numpy repack, pack_exact reference, unpack) ...")
    for i, t in enumerate(tern_entries):
        name = t["name"]
        w = st.array(name)
        n, k = w.shape
        codes = kt.seg_array(t["codes"])
        scales = kt.seg_array(t["scales"])
        zs = kt.seg_array(t["zero_signs"]) if "zero_signs" in t else None
        neg_pos = np.flatnonzero(w.reshape(-1) == 0x8000)  # -0.0 positions, from the source
        stats["groups"] += int(scales.size)
        stats["zero_groups"] += int((scales == 0).sum())
        stats["zeros"] += int(t.get("zeros", 0))
        stats["neg_zero"] += int(neg_pos.size)
        stats["weights"] += int(w.size)
        stats["zero_signs_bytes"] += int(zs.size) if zs is not None else 0
        if t["src_sha256"] != sha256_bytes(st.raw(name)):
            c_np.fail(f"{name}: src_sha256 differs")
        src_dtype = DTYPES[st.by_name[name].dtype][0]
        if t["shape"] != [n, k] or t["dtype"] != src_dtype:
            c_np.fail(f"{name}: declared {t['dtype']} {t['shape']} != source {src_dtype} {[n, k]}")
        # numpy repack
        c_np.count += 1
        p2, s2, zs2, neg2 = pack_ternary_bits(np.asarray(w), name)
        if not (np.array_equal(p2, codes) and np.array_equal(s2, scales)):
            c_np.fail(f"{name}: numpy repack differs")
        if neg2 != t.get("neg_zero") or neg2 != neg_pos.size:
            c_np.fail(f"{name}: neg_zero count differs")
        if (zs is None) != (neg2 == 0) or (zs is not None and not np.array_equal(zs2, zs)):
            c_np.fail(f"{name}: zero_signs differ from a repack")
        if t.get("zeros") != int(np.count_nonzero((np.asarray(w) & 0x7FFF) == 0)):
            c_np.fail(f"{name}: zero count differs")
        # numpy unpack -> exact source bits (zero signs restored); without them only -0.0 differs
        c_unp.count += 1
        try:
            if not np.array_equal(unpack_ternary_bits(codes, scales, k, zs), w):
                c_unp.fail(f"{name}: numpy unpack (zero signs restored) != source bits")
            plain = unpack_ternary_bits(codes, scales, k)
            if not np.array_equal(np.flatnonzero(plain.reshape(-1) != w.reshape(-1)), neg_pos):
                c_unp.fail(f"{name}: numpy unpack without zero signs differs beyond the -0.0 positions")
        except Exception as exc:
            c_unp.fail(f"{name}: numpy unpack raised {exc!r}")
        if vendored is None:
            continue
        import torch

        wt = to_torch_bf16(w)
        for chk, mod in ((c_ven, vendored),):
            if mod is None:
                continue
            chk.count += 1
            try:
                packed_t, scales_t = mod.pack_exact(wt)
            except Exception as exc:
                chk.fail(f"{name}: pack_exact raised {exc!r}")
                continue
            if packed_t.dtype != torch.int32 or tuple(packed_t.shape) != (n, k // 16):
                chk.fail(f"{name}: pack_exact codes dtype/shape {packed_t.dtype} {tuple(packed_t.shape)}")
            elif packed_t.numpy().tobytes() != kt.seg_bytes(t["codes"]).tobytes():
                chk.fail(f"{name}: codes bytes differ from pack_exact")
            if scales_t.dtype != torch.bfloat16 or tuple(scales_t.shape) != (n, k // fmt.GROUP):
                chk.fail(f"{name}: pack_exact scales dtype/shape {scales_t.dtype} {tuple(scales_t.shape)}")
            elif not np.array_equal(torch_bits(scales_t), scales):
                chk.fail(f"{name}: scale bits differ from pack_exact")
        if vendored is not None:
            codes_t = torch.from_numpy(np.array(codes))
            scales_t = torch.from_numpy(np.array(scales).view(np.int16)).view(torch.bfloat16)
            un = vendored.unpack(codes_t, scales_t, k)
            if not torch.equal(un, wt):
                c_unp.fail(f"{name}: vendored unpack values != source")
            else:
                diff = np.flatnonzero(torch_bits(un).reshape(-1) != w.reshape(-1))
                if not np.array_equal(diff, neg_pos):
                    c_unp.fail(f"{name}: unpack bit differences are not exactly the -0.0 positions")
        if (i + 1) % 28 == 0:
            say(f"  {i + 1}/{len(tern_entries)} ({time.time() - t_start:.0f}s)")
    c_unp.details["torch_unpack"] = "vendored unpack: torch.equal values + bit diff == -0.0 positions" \
        if vendored is not None else c_unp.details.get("torch_unpack", "skipped")
    for chk in (c_np, c_ven, c_unp):
        chk.done()
    report["ternary_stats"] = stats

    # -- reconstruct -------------------------------------------------------------
    c = checks["reconstruct"] = Check("reconstruct")
    say("reconstruct: streaming sha256 ...")
    c.count = 1
    try:
        res = reconstruct(kt, lambda _b: None)
    except Exception as exc:
        res = {"ok": False, "sha256": None, "size": None, "expected_sha256": meta_src["safetensors"]["sha256"],
               "tensor_sha_mismatches": [f"raised {exc!r}"]}
    c.details.update({"sha256": res["sha256"], "size": res["size"]})
    if not res["ok"]:
        c.fail(f"reconstructed sha256 {res['sha256']} != {res['expected_sha256']} "
               f"(tensor mismatches: {res['tensor_sha_mismatches'][:5]})")
    if res["sha256"] != st_sha:
        c.fail("reconstructed sha256 != actual source file sha256")
    if expect_sha256 and res["sha256"] != expect_sha256.lower():
        c.fail("reconstructed sha256 != --expect-sha256")
    c.done()

    st.close()
    kt.close()
    report["checks"] = {k: v.as_dict() for k, v in checks.items()}
    report["seconds"] = round(time.time() - t_start, 1)
    report["ok"] = all(v.ok or (not v.required and v.skipped and not v.failures) for v in checks.values())
    report["verdict"] = "PASS" if report["ok"] else "FAIL"
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kt2n_assets.verify", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kt2n")
    ap.add_argument("--src", required=True, help="source directory (model.safetensors, config.json, ...)")
    ap.add_argument("--runtime-config", default=None)
    ap.add_argument("--expect-sha256", default=None)
    ap.add_argument("--label", default=None, help="free-form label stored in the report (e.g. laptop, box)")
    ap.add_argument("--audit", choices=("auto", "require", "off"), default="auto",
                    help="compare the ternary set with production's audit fixture "
                         "(auto: only when the source is the audited checkpoint)")
    ap.add_argument("--report", default=None, help="write the JSON report here")
    args = ap.parse_args(argv)
    report = verify(args.kt2n, args.src, runtime_config=args.runtime_config,
                    expect_sha256=args.expect_sha256, label=args.label, audit_mode=args.audit)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        Path(args.report).write_text(text + "\n")
    for name, chk in report["checks"].items():
        state = ("skip" if not chk["required"] else "SKIP") if chk.get("skipped") else ("ok" if chk["ok"] else "FAIL")
        print(f"{state:4} {name:22} checked={chk['checked']:<5} failures={chk['failures']} "
              f"({chk['seconds']}s){'  ' + chk['skipped'] if chk.get('skipped') else ''}")
        for f in chk["first_failures"]:
            print(f"       - {f}")
    print(f"{report['verdict']} in {report['seconds']}s")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
