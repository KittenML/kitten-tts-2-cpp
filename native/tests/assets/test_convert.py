"""End to end on a synthetic checkpoint: layout, metadata, reconstruction, verification, refusals."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kt2n_assets import format as fmt
from kt2n_assets.convert import ConvertError, NotTernaryError, convert, pack_ternary_bits
from kt2n_assets.reconstruct import reconstruct
from kt2n_assets.safetensors_raw import SafetensorsFile, sha256_file
from synth import SynthSpec, make_checkpoint, make_tensors, qwen3_config

TOP_KEYS = {"format", "version", "source", "config", "runtime_config", "quant", "tensors"}
QUIET = {"log": lambda _m: None}


# --- header / layout ----------------------------------------------------------------
def test_header_roundtrip_and_size():
    h = fmt.Header.for_meta(meta_len=5000, data_len=8192)
    raw = h.pack()
    assert len(raw) == 64 and raw[:8] == b"KT2NENG1" and raw[56:] == b"\0" * 8
    assert fmt.Header.unpack(raw) == h
    assert h.meta_off == 64 and h.data_off == 8192 and h.file_len == 16384


@pytest.mark.parametrize("pos,value", [(0, b"X"), (8, b"\x02"), (12, b"\x01"), (60, b"\x01")])
def test_header_rejects_bad_fields(pos, value):
    raw = bytearray(fmt.Header.for_meta(10, 4096).pack())
    raw[pos : pos + 1] = value
    with pytest.raises(fmt.FormatError):
        fmt.Header.unpack(bytes(raw))


def test_alignment_and_padding(synth_kt2n):
    path, _ = synth_kt2n
    with fmt.Kt2nFile(path) as kt:
        h = kt.header
        assert h.data_off % 4096 == 0 and h.data_len % 4096 == 0 and h.file_len % 4096 == 0
        assert h.data_off == fmt.align_up(64 + h.meta_len)
        for t in kt.tensors:
            for _, seg in fmt.tensor_segments(t):
                assert seg["off"] % 4096 == 0
        assert kt.padding_is_zero()
        raw = path.read_bytes()
        assert not any(raw[64 + h.meta_len : h.data_off])


def test_metadata_schema(synth_kt2n, synth_src):
    path, summary = synth_kt2n
    with fmt.Kt2nFile(path) as kt:
        meta = kt.meta
        assert set(meta) == TOP_KEYS
        assert meta["format"] == "kt2n" and meta["version"] == 1
        assert meta["quant"] == {"group": 128, "codes": {"0": 0, "1": "+1", "2": "-1"},
                                 "word": "int32 LSB-first, column 16j+o at bits 2o..2o+1"}
        assert kt.meta_bytes == json.dumps(meta, sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=False).encode()
        st = SafetensorsFile(synth_src / "model.safetensors")
        assert [t["name"] for t in meta["tensors"]] == [t.name for t in st.by_offset]
        spec = SynthSpec()
        for t in meta["tensors"]:
            assert {"name", "kind", "dtype", "shape", "src_sha256"} <= set(t)
            assert t["src_sha256"] == st.tensor_sha256(t["name"])
            if t["kind"] == "ternary2":
                n, k = t["shape"]
                assert t["codes"]["shape"] == [n, k // 16] and t["codes"]["dtype"] == "i32"
                assert t["codes"]["len"] == n * k // 4
                assert t["scales"]["shape"] == [n, k // 128] and t["scales"]["dtype"] == "bf16"
                assert t["neg_zero"] == spec.neg_zero_per_matrix
                assert t["zero_signs"]["dtype"] == "u8"
                assert t["zero_signs"]["shape"] == [(t["zeros"] + 7) // 8]
                bits = st.array(t["name"])
                assert t["zeros"] == int(((bits & 0x7FFF) == 0).sum())
            else:
                assert t["kind"] == "dense"
                assert t["data"]["shape"] == t["shape"] and t["data"]["dtype"] == "bf16"
                assert t["data"]["sha256"] == t["src_sha256"]
        st.close()
        assert summary["ternary"] == 7 * spec.layers and summary["dense"] == 1 + 4 * spec.layers + 1 + 4


def test_metadata_source_roundtrip(synth_kt2n, synth_src):
    path, _ = synth_kt2n
    with fmt.Kt2nFile(path) as kt:
        src = kt.meta["source"]
        st = SafetensorsFile(synth_src / "model.safetensors")
        assert base64.b64decode(src["safetensors"]["header_b64"]) == st.header_bytes
        assert src["safetensors"]["header_len"] == st.header_len
        assert src["safetensors"]["sha256"] == sha256_file(synth_src / "model.safetensors")
        assert src["safetensors"]["size"] == (synth_src / "model.safetensors").stat().st_size
        st.close()
        assert kt.meta["config"] == json.loads((synth_src / "config.json").read_text())
        rc = synth_src.parent / "runtime_config.json"
        assert kt.meta["runtime_config"] == json.loads(rc.read_text())
        assert base64.b64decode(src["files"]["config.json"]["b64"]) == (synth_src / "config.json").read_bytes()
        assert base64.b64decode(src["files"]["runtime_config.json"]["b64"]) == rc.read_bytes()
        for name in ("tokenizer.json", "vocab.json", "merges.txt"):
            data = (synth_src / name).read_bytes()
            assert src["files"][name] == {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        assert src["converter"]["name"] == "kt2n.convert" and src["converter"]["version"]
        assert src["production_packing"]["scope"] == "k2048" and src["production_packing"]["max_rows"] == 1


def test_codes_and_dense_bytes(synth_kt2n, synth_src):
    path, _ = synth_kt2n
    st = SafetensorsFile(synth_src / "model.safetensors")
    with fmt.Kt2nFile(path) as kt:
        for t in kt.tensors:
            if t["kind"] == "dense":
                assert kt.seg_bytes(t["data"]).tobytes() == st.raw(t["name"]).tobytes()
            else:
                packed, scales, zero_signs, _ = pack_ternary_bits(np.asarray(st.array(t["name"])))
                assert np.array_equal(kt.seg_array(t["codes"]), packed)
                assert np.array_equal(kt.seg_array(t["scales"]), scales)
                assert np.array_equal(kt.seg_array(t["zero_signs"]), zero_signs)
    st.close()


def test_conversion_is_deterministic(synth_src, synth_kt2n, tmp_path):
    path, _ = synth_kt2n
    again = tmp_path / "again.kt2n"
    convert(synth_src, again, **QUIET)
    assert again.read_bytes() == path.read_bytes()


# --- reconstruction -------------------------------------------------------------------
def test_reconstruct_in_process(synth_kt2n, synth_src):
    path, _ = synth_kt2n
    chunks: list[bytes] = []
    with fmt.Kt2nFile(path) as kt:
        res = reconstruct(kt, lambda b: chunks.append(bytes(b)))
    assert res["ok"] and not res["tensor_sha_mismatches"]
    assert b"".join(chunks) == (synth_src / "model.safetensors").read_bytes()


def test_cli_convert_verify_reconstruct(synth_src, tmp_path, cli_env):
    out = tmp_path / "cli.kt2n"
    run = lambda *a: subprocess.run([sys.executable, "-m", *a], env=cli_env, capture_output=True)  # noqa: E731
    p = run("kt2n_assets.convert", str(synth_src), str(out))
    assert p.returncode == 0, p.stderr.decode()
    assert json.loads(p.stdout)["ternary"] == 14
    p = run("kt2n_assets.verify", str(out), "--src", str(synth_src), "--report", str(tmp_path / "r.json"))
    report = json.loads((tmp_path / "r.json").read_text())
    if importlib.util.find_spec("torch") is not None:
        assert p.returncode == 0, p.stdout.decode() + p.stderr.decode()
        assert report["verdict"] == "PASS"
    else:  # without torch the required pack_exact check cannot run: the verdict must be FAIL
        assert p.returncode == 1 and report["verdict"] == "FAIL"
        failing = {k for k, v in report["checks"].items() if v["required"] and not v["ok"]}
        assert failing == {"pack_exact_vendored"}
    p = run("kt2n_assets.reconstruct", str(out), str(tmp_path / "re.safetensors"))
    assert p.returncode == 0, p.stderr.decode()
    assert (tmp_path / "re.safetensors").read_bytes() == (synth_src / "model.safetensors").read_bytes()
    p = run("kt2n_assets.reconstruct", str(out), "-")
    assert p.returncode == 0
    assert hashlib.sha256(p.stdout).hexdigest() == sha256_file(synth_src / "model.safetensors")


# --- verification ------------------------------------------------------------------------
def test_verify_passes(synth_kt2n, synth_src):
    pytest.importorskip("torch")
    from kt2n_assets.verify import verify

    path, _ = synth_kt2n
    rep = verify(str(path), str(synth_src), say=lambda _m: None)
    assert rep["verdict"] == "PASS", json.dumps(rep["checks"], indent=1)
    checks = rep["checks"]
    for name in ("structure", "source_identity", "tensor_set", "dense_bytes", "ternary_numpy",
                 "pack_exact_vendored", "unpack_lossless", "reconstruct"):
        assert checks[name]["ok"] and checks[name]["required"], name
    assert checks["pack_exact_vendored"]["checked"] == 14
    assert not checks["audit_set"]["required"]
    assert checks["reference_integrity"]["ok"]
    # The synthetic checkpoint is not the audited one: forcing the audit must fail.
    rep = verify(str(path), str(synth_src), audit_mode="require", say=lambda _m: None)
    assert rep["verdict"] == "FAIL" and not rep["checks"]["audit_set"]["ok"]


def _flip(path: Path, pos: int) -> None:
    with open(path, "r+b") as fh:
        fh.seek(pos)
        b = fh.read(1)
        fh.seek(pos)
        fh.write(bytes([b[0] ^ 0x01]))


def test_verify_rejects_changed_reference(synth_kt2n, synth_src, tmp_path, monkeypatch):
    from kt2n_assets import verify as verifier

    changed = tmp_path / "reference.py"
    changed.write_text(verifier.VENDORED.read_text().replace("GROUP = 128", "GROUP = 256"))
    monkeypatch.setattr(verifier, "VENDORED", changed)
    report = verifier.verify(str(synth_kt2n[0]), str(synth_src), say=lambda _m: None)
    assert report["verdict"] == "FAIL"
    assert not report["checks"]["reference_integrity"]["ok"]


def test_verify_detects_corrupted_codes(synth_kt2n, synth_src, tmp_path):
    pytest.importorskip("torch")
    from kt2n_assets.verify import verify

    path, _ = synth_kt2n
    bad = tmp_path / "bad.kt2n"
    shutil.copy(path, bad)
    with fmt.Kt2nFile(bad) as kt:
        t = next(t for t in kt.tensors if t["kind"] == "ternary2")
        pos = kt.header.data_off + t["codes"]["off"] + 5
    _flip(bad, pos)
    rep = verify(str(bad), str(synth_src), say=lambda _m: None)
    assert rep["verdict"] == "FAIL"
    for name in ("structure", "ternary_numpy", "pack_exact_vendored", "reconstruct"):
        assert not rep["checks"][name]["ok"], name


def test_verify_detects_dirty_padding(synth_kt2n, synth_src, tmp_path):
    pytest.importorskip("torch")
    from kt2n_assets.verify import verify

    path, _ = synth_kt2n
    bad = tmp_path / "pad.kt2n"
    shutil.copy(path, bad)
    with fmt.Kt2nFile(bad) as kt:
        t = kt.tensors[-1]
        seg = fmt.tensor_segments(t)[-1][1]
        pos = kt.header.data_off + seg["off"] + seg["len"]  # first padding byte after the last segment
        assert pos < kt.header.file_len
    _flip(bad, pos)
    rep = verify(str(bad), str(synth_src), say=lambda _m: None)
    assert not rep["checks"]["structure"]["ok"]


def test_reader_rejects_truncation_and_bad_meta(synth_kt2n, tmp_path):
    path, _ = synth_kt2n
    short = tmp_path / "short.kt2n"
    short.write_bytes(path.read_bytes()[:-4096])
    with pytest.raises(fmt.FormatError):
        fmt.Kt2nFile(short)
    meta = tmp_path / "meta.kt2n"
    shutil.copy(path, meta)
    _flip(meta, 64)  # '{' -> 'z'
    with pytest.raises(fmt.FormatError):
        fmt.Kt2nFile(meta)


def _rewrite_meta(src: Path, dst: Path, edit) -> None:
    """Copy a .kt2n with its metadata edited (canonical JSON; the length may change, data moves with it)."""
    with fmt.Kt2nFile(src) as kt:
        meta = json.loads(kt.meta_bytes)
        data = bytes(kt._mm[kt.header.data_off : kt.header.file_len])
    edit(meta)
    meta_bytes = fmt.dumps_meta(meta)
    h = fmt.Header.for_meta(len(meta_bytes), len(data))
    dst.write_bytes(h.pack() + meta_bytes + b"\0" * (h.data_off - 64 - len(meta_bytes)) + data)


@pytest.mark.parametrize("field,value", [("off", -4096), ("len", -2), ("shape", [-1, -2]), ("off", 4096.0),
                                         ("len", True)])
def test_reader_rejects_negative_or_non_integer_segment_fields(synth_kt2n, tmp_path, field, value):
    path, _ = synth_kt2n
    bad = tmp_path / "neg.kt2n"

    def edit(meta):
        t = next(t for t in meta["tensors"] if t["kind"] == "dense" and t["data"]["off"] >= 8192)
        t["data"][field] = value
        if field == "shape":  # numel(-1 x -2) * 2 == len: only the sign is wrong
            t["data"]["len"] = 4

    _rewrite_meta(path, bad, edit)
    with pytest.raises(fmt.FormatError, match="non-negative integers"):
        fmt.Kt2nFile(bad)


def test_reader_rejects_duplicate_metadata_keys(synth_kt2n, tmp_path):
    path, _ = synth_kt2n
    raw = bytearray(path.read_bytes())
    meta_len = int.from_bytes(raw[24:32], "little")
    meta = bytes(raw[64 : 64 + meta_len])
    at = meta.index(b'"source":')
    raw[64 + at : 64 + at + 9] = b'"format":'  # a second top-level "format" (json.loads keeps the last)
    bad = tmp_path / "dup.kt2n"
    bad.write_bytes(bytes(raw))
    with pytest.raises(fmt.FormatError):
        fmt.Kt2nFile(bad)


@pytest.mark.parametrize("what", ["rows", "dtype", "dense_dtype"])
def test_verify_checks_declared_shape_and_dtype(synth_kt2n, synth_src, tmp_path, what):
    pytest.importorskip("torch")
    from kt2n_assets.verify import verify

    path, _ = synth_kt2n
    bad = tmp_path / "decl.kt2n"

    def edit(meta):
        if what == "dense_dtype":
            t = next(t for t in meta["tensors"] if t["kind"] == "dense")
            t["dtype"] = t["data"]["dtype"] = "u16"  # same item size, same bytes and sha256
            return
        t = next(t for t in meta["tensors"] if t["kind"] == "ternary2")
        if what == "rows":
            t["shape"] = [t["shape"][0] + 1, t["shape"][1]]
        else:
            t["dtype"] = "f16"

    _rewrite_meta(path, bad, edit)
    rep = verify(str(bad), str(synth_src), say=lambda _m: None)
    assert rep["verdict"] == "FAIL"
    failing = {k for k, v in rep["checks"].items() if v["required"] and not v["ok"]}
    assert failing == ({"dense_bytes"} if what == "dense_dtype" else {"ternary_numpy"}), failing


def test_reconstruct_cli_leaves_no_partial_file_on_error(synth_kt2n, tmp_path, cli_env):
    path, _ = synth_kt2n
    bad = tmp_path / "code3.kt2n"
    shutil.copy(path, bad)
    with fmt.Kt2nFile(bad) as kt:
        t = next(t for t in kt.tensors if t["kind"] == "ternary2")
        pos = kt.header.data_off + t["codes"]["off"]
    with open(bad, "r+b") as fh:  # code 3 in the first word: unpack raises ReconstructError
        fh.seek(pos)
        fh.write(b"\xff")
    out = tmp_path / "re.safetensors"
    p = subprocess.run([sys.executable, "-m", "kt2n_assets.reconstruct", str(bad), str(out)], env=cli_env,
                       capture_output=True)
    assert p.returncode != 0 and b"code 3" in p.stderr, p.stderr.decode()
    assert not out.exists() and not (tmp_path / "re.safetensors.partial").exists()


# --- converter refusals -----------------------------------------------------------------------
def _tensors_with(name: str, bits: np.ndarray):
    tensors = make_tensors(SynthSpec())
    tensors[name] = ("BF16", tuple(bits.shape), bits.astype("<u2").tobytes())
    return tensors


def test_refuses_non_ternary_projection_and_leaves_no_file(tmp_path):
    tensors = make_tensors(SynthSpec())
    name = "model.layers.1.mlp.up_proj.weight"
    bits = np.frombuffer(tensors[name][2], "<u2").reshape(tensors[name][1]).copy()
    nz = np.flatnonzero(bits[7])
    bits[7, nz[0]] = (bits[7, nz[0]] & 0x8000) | 0x3F00  # a second magnitude in one group (0.5)
    src = make_checkpoint(tmp_path, tensors=_tensors_with(name, bits))
    out = tmp_path / "x.kt2n"
    with pytest.raises(NotTernaryError, match=name.replace(".", r"\.")):
        convert(src, out, **QUIET)
    assert not out.exists() and not (tmp_path / "x.kt2n.partial").exists()


def test_refuses_projection_with_bad_k(tmp_path):
    spec = SynthSpec()
    tensors = make_tensors(spec)
    name = "model.layers.0.self_attn.o_proj.weight"
    bits = np.zeros((spec.hidden, 384), np.uint16)
    tensors[name] = ("BF16", bits.shape, bits.tobytes())
    src = make_checkpoint(tmp_path, tensors=tensors)
    with pytest.raises(ConvertError, match="divisible by 256"):
        convert(src, tmp_path / "x.kt2n", **QUIET)


def test_refuses_projection_count_mismatch(tmp_path):
    spec = SynthSpec()
    tensors = make_tensors(spec)
    del tensors["model.layers.1.self_attn.v_proj.weight"]
    src = make_checkpoint(tmp_path, tensors=tensors)
    with pytest.raises(ConvertError, match="expected 2 layers"):
        convert(src, tmp_path / "x.kt2n", **QUIET)


def test_refuses_misplaced_projection_with_right_count(tmp_path):
    # Layer 1's v_proj missing, the same projection under a non-existent layer 2: still 14 names.
    tensors = make_tensors(SynthSpec())
    tensors["model.layers.2.self_attn.v_proj.weight"] = tensors.pop("model.layers.1.self_attn.v_proj.weight")
    src = make_checkpoint(tmp_path, tensors=tensors)
    with pytest.raises(ConvertError, match=r"expected 2 layers.*missing \['model\.layers\.1\.self_attn\.v_proj"
                                           r"\.weight'\].*unexpected \['model\.layers\.2\.self_attn\.v_proj"):
        convert(src, tmp_path / "x.kt2n", **QUIET)
    assert not (tmp_path / "x.kt2n").exists()


def test_refuses_whole_layer_shifted(tmp_path):
    # Two layers declared, projections for layers 1 and 2 only (layer 0's renamed to 2).
    tensors = make_tensors(SynthSpec())
    for name in [n for n in tensors if n.startswith("model.layers.0.") and n.endswith("_proj.weight")]:
        tensors[name.replace("model.layers.0.", "model.layers.2.")] = tensors.pop(name)
    src = make_checkpoint(tmp_path, tensors=tensors)
    with pytest.raises(ConvertError, match="missing"):
        convert(src, tmp_path / "x.kt2n", **QUIET)


@pytest.mark.parametrize("layers", [None, 0, "2", 2.0])
def test_refuses_missing_or_bad_layer_count(tmp_path, layers):
    src = make_checkpoint(tmp_path)
    cfg = json.loads((src / "config.json").read_text())
    if layers is None:
        del cfg["num_hidden_layers"]
    else:
        cfg["num_hidden_layers"] = layers
    (src / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ConvertError, match="num_hidden_layers"):
        convert(src, tmp_path / "x.kt2n", **QUIET)


def _tree_digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_refuses_output_that_aliases_a_source_file(tmp_path):
    src = make_checkpoint(tmp_path)
    before = _tree_digest(tmp_path)
    (tmp_path / "link.kt2n").symlink_to(src / "model.safetensors")
    (tmp_path / "hard.kt2n").hardlink_to(src / "config.json")
    for out in (src / "model.safetensors", src / "config.json", src / "tokenizer.json",
                src.parent / "runtime_config.json", src / ".." / "effective-bf16" / "model.safetensors",
                tmp_path / "link.kt2n", tmp_path / "hard.kt2n"):
        with pytest.raises(ConvertError, match="would overwrite a source file"):
            convert(src, out, **QUIET)
    (src / "model.safetensors").rename(src / "w.partial")  # OUT.partial would be the safetensors
    with pytest.raises(ConvertError, match="would overwrite a source file"):
        convert(src, src / "w", safetensors_name="w.partial", **QUIET)
    (src / "w.partial").rename(src / "model.safetensors")
    (tmp_path / "link.kt2n").unlink()
    (tmp_path / "hard.kt2n").unlink()
    assert _tree_digest(tmp_path) == before  # nothing was touched
    convert(src, tmp_path / "ok.kt2n", **QUIET)  # a fresh path next to the checkpoint is fine


def test_requires_runtime_config(tmp_path):
    src = make_checkpoint(tmp_path)
    (src.parent / "runtime_config.json").unlink()
    with pytest.raises(ConvertError, match="runtime_config.json"):
        convert(src, tmp_path / "x.kt2n", **QUIET)


def test_runtime_config_next_to_weights(tmp_path):
    src = make_checkpoint(tmp_path, runtime_config_in_parent=False)
    summary = convert(src, tmp_path / "x.kt2n", **QUIET)
    assert summary["ternary"] == 14
    with fmt.Kt2nFile(tmp_path / "x.kt2n") as kt:
        assert "location" not in kt.meta["source"]["files"]["runtime_config.json"]


def test_no_neg_zero_means_no_index_segment(tmp_path):
    src = make_checkpoint(tmp_path, SynthSpec(neg_zero_per_matrix=0, seed=5))
    convert(src, tmp_path / "x.kt2n", **QUIET)
    with fmt.Kt2nFile(tmp_path / "x.kt2n") as kt:
        for t in kt.tensors:
            if t["kind"] == "ternary2":
                assert t["neg_zero"] == 0 and "zero_signs" not in t
        res = reconstruct(kt, lambda _b: None)
    assert res["ok"]


def test_config_layers_drive_projection_count(tmp_path):
    spec = SynthSpec(layers=3, hidden=512, intermediate=1536, heads=8, kv_heads=4, seed=9)
    src = make_checkpoint(tmp_path, spec)
    assert json.loads((src / "config.json").read_text()) == qwen3_config(spec)
    summary = convert(src, tmp_path / "x.kt2n", **QUIET)
    assert summary["ternary"] == 21
