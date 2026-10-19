"""Cloning preparation integrity and separation from ordinary inference assets."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from kt2n_assets import clone_assets, convert_s3gen
from kt2n_assets.prepare import main
from kt2n_assets.safetensors_raw import SafetensorsFile, write_safetensors


@pytest.fixture
def exports(tmp_path, monkeypatch):
    data = b"\x00\x00\x80\x3f"
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(clone_assets, "CONTRACTS", {
        "consts": {"window": ([1], digest)}, "pyannote": {"weight": ([1], digest)}})
    consts, pyannote, acoustic = (tmp_path / n for n in ("consts", "pyannote", "acoustic"))
    write_safetensors(consts, {"window": ("F32", (1,), data)}, {"source": "private-source"})
    write_safetensors(pyannote, {"weight": ("F32", (1,), data),
        "bank.embeddings": ("F32", (1,), data)}, {"voice_ids": "private-bank"})
    acoustic.write_bytes(b"pinned acoustic")
    monkeypatch.setattr(convert_s3gen, "PINNED_SHA256", hashlib.sha256(acoustic.read_bytes()).hexdigest())
    return acoustic, consts, pyannote


def test_clone_exports_keep_exact_network_bytes_and_strip_bank(exports, tmp_path):
    destination = tmp_path / "out"
    provenance = clone_assets.prepare_clone_files(*exports, destination)
    assert provenance["legacy_speaker_bank"] == "removed"
    assert (destination / "s3gen.safetensors").read_bytes() == exports[0].read_bytes()
    with SafetensorsFile(destination / "voicebuild_pyannote.safetensors") as sf:
        assert set(sf.by_name) == {"weight"}
        assert bytes(sf.raw("weight")) == b"\x00\x00\x80\x3f"
        assert "voice_ids" not in sf.metadata
    with SafetensorsFile(destination / "voicebuild_consts.safetensors") as sf:
        assert sf.metadata["source"] == "native-cloning-constants-v1"


@pytest.mark.parametrize("dtype,shape,data", [
    ("F32", (1,), b"\x00\x00\x00\x40"),
    ("I32", (1,), b"\x00\x00\x80\x3f"),
    ("F32", (1, 1), b"\x00\x00\x80\x3f")])
def test_changed_weight_dtype_or_shape_rejected(exports, tmp_path, dtype, shape, data):
    write_safetensors(exports[2], {"weight": (dtype, shape, data)})
    with pytest.raises(ValueError, match="checksum/shape"):
        clone_assets.prepare_clone_files(*exports, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_network_tensor_rejected(exports, tmp_path):
    write_safetensors(exports[2], {})
    with pytest.raises(ValueError, match="tensor names"):
        clone_assets.prepare_clone_files(*exports, tmp_path / "out")


def test_acoustic_mismatch_rejected(exports, tmp_path):
    exports[0].write_bytes(b"modified")
    with pytest.raises(ValueError, match="acoustic source checksum"):
        clone_assets.prepare_clone_files(*exports, tmp_path / "out")


def test_existing_clone_assets_not_overwritten(exports, tmp_path):
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "keep").write_bytes(b"existing")
    with pytest.raises(ValueError, match="already exists"):
        clone_assets.prepare_clone_files(*exports, destination)
    assert (destination / "keep").read_bytes() == b"existing"


def test_cloning_rejects_incomplete_explicit_sources(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--out", str(tmp_path), "--with-cloning", "--clone-consts", "constants"])
    assert exc.value.code == 2


def test_original_clone_tensor_contract_is_complete():
    assert len(clone_assets.CONTRACTS["consts"]) == 9
    assert len(clone_assets.CONTRACTS["pyannote"]) == 49
    assert all(not key.startswith("bank.") for key in clone_assets.CONTRACTS["pyannote"])


@pytest.mark.parametrize("download", [False, True])
def test_automatic_export_uses_authorized_pinned_source(exports, tmp_path, monkeypatch, download):
    import huggingface_hub
    from kt2n_assets import export_clone, prepare
    import shutil
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "voices.kt2n").write_bytes(b"catalogue")
    monkeypatch.setattr(prepare.voices, "read", lambda _: SimpleNamespace(
        voices=[{"id": "Bruno", "kind": "builtin"}],
        arrays={"Bruno": {"raw_speaker": np.ones(512, dtype="f4")}}))
    checkpoint_dir = tmp_path / "authorized-model"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "pytorch_model.bin"
    checkpoint.write_bytes(b"approved pinned model fixture")
    calls = []
    def download_model(repo, filename, **kwargs):
        assert repo == clone_assets.PYANNOTE_REPO
        assert filename == "pytorch_model.bin"
        assert kwargs["revision"] == clone_assets.PYANNOTE_REVISION
        assert kwargs["token"] is True and kwargs["local_files_only"] is True
        calls.append("download")
        return str(checkpoint)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download_model)
    def export_model(source, acoustic, destination):
        assert source == checkpoint and acoustic == exports[0]
        calls.append("export")
        destination.mkdir()
        shutil.copyfile(exports[1], destination / "voicebuild_consts.safetensors")
        shutil.copyfile(exports[2], destination / "voicebuild_pyannote.safetensors")
        return {}
    monkeypatch.setattr(export_clone, "export", export_model)
    args = SimpleNamespace(out=bundle, clone_consts=None, clone_pyannote=None,
        pyannote_model=None if download else checkpoint_dir, cache_dir=tmp_path / "cache", offline=True)
    result = prepare.prepare_cloning(args, exports[0], {"voices": {"file": "voices.kt2n"}})
    assert calls == (["download", "export"] if download else ["export"])
    assert result["version"] == 1 and result["speaker_bank_voices"] == 1
    assert len(result["files"]) == 3
