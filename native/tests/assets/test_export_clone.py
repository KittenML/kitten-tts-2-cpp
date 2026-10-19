"""Portable clone preparation: trusted input gate and exact tensor export."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from kt2n_assets import export_clone
from kt2n_assets.clone_assets import CONTRACTS, checked_tensors
from kt2n_assets.safetensors_raw import SafetensorsFile


def test_unrecognized_checkpoint_is_rejected_before_loading_pickle(tmp_path, monkeypatch):
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"untrusted checkpoint")
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="pyannote checkpoint checksum"):
        export_clone.checkpoint_state(checkpoint)


def test_unrecognized_acoustic_is_rejected_before_importing_preparation_stack(tmp_path, monkeypatch):
    checkpoint = tmp_path / "s3gen.safetensors"
    checkpoint.write_bytes(b"changed checkpoint")
    monkeypatch.setitem(sys.modules, "torchaudio", None)
    with pytest.raises(ValueError, match="acoustic checkpoint checksum"):
        export_clone.frontend_tensors(checkpoint)


def test_tensor_mismatch_leaves_no_export_directory(tmp_path, monkeypatch):
    def wrong(kind):
        return {name: np.zeros(shape, dtype=np.float32)
                for name, (shape, _) in CONTRACTS[kind].items()}
    monkeypatch.setattr(export_clone, "speaker_tensors", lambda _: wrong("pyannote"))
    monkeypatch.setattr(export_clone, "frontend_tensors", lambda _: wrong("consts"))
    destination = tmp_path / "export"
    with pytest.raises(ValueError, match="checksum/shape mismatch"):
        export_clone.export(tmp_path / "model", tmp_path / "acoustic", destination)
    assert not destination.exists()
    assert not list(tmp_path.iterdir())


def test_local_offline_inputs_do_not_import_download_client(tmp_path, monkeypatch, capsys):
    model = tmp_path / "model"
    model.mkdir()
    seen = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setattr(export_clone, "export", lambda *args: seen.append(args) or {"ok": True})
    assert export_clone.main(["--pyannote-model", str(model), "--acoustic-source", str(tmp_path / "s3"),
                              "--offline", "--out", str(tmp_path / "exports")]) == 0
    assert seen == [(model / "pytorch_model.bin", tmp_path / "s3", tmp_path / "exports")]
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_existing_destination_is_never_replaced(tmp_path):
    target = tmp_path / "exports"
    target.mkdir()
    (target / "keep").write_bytes(b"existing content")
    with pytest.raises(ValueError, match="already exists"):
        export_clone.export(tmp_path / "absent", tmp_path / "absent", target)
    assert (target / "keep").read_bytes() == b"existing content"


@pytest.mark.skipif(not (os.environ.get("KT2N_CLONE_PYANNOTE_MODEL") and
                        os.environ.get("KT2N_CLONE_ACOUSTIC_SOURCE")),
                    reason="set the two upstream checkpoint paths to verify reproducible exports")
def test_pinned_upstream_checkpoints_reproduce_all_58_tensors(tmp_path):
    model = Path(os.environ["KT2N_CLONE_PYANNOTE_MODEL"])
    if model.is_dir():
        model /= "pytorch_model.bin"
    acoustic = Path(os.environ["KT2N_CLONE_ACOUSTIC_SOURCE"])
    out = tmp_path / "exports"
    report = export_clone.export(model, acoustic, out)
    assert sum(row["tensors"] for row in report["files"]) == 58
    for filename, kind in (("voicebuild_consts.safetensors", "consts"),
                           ("voicebuild_pyannote.safetensors", "pyannote")):
        assert set(checked_tensors(out / filename, kind)) == set(CONTRACTS[kind])
        with SafetensorsFile(out / filename) as source:
            assert not any(name.startswith("bank.") for name in source.by_name)
            assert "voice_ids" not in (source.metadata or {})
