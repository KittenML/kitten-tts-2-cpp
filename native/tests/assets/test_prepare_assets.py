"""Public voice compatibility, asset integrity and offline preparation contracts."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from kt2n_assets import voices
from kt2n_assets.prepare import TOKEN_MAP, check_config, local_file, prepare_voices, record, word_tables
from kt2n_assets.convert import source_files


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert text == 'Reference sentence.'
        assert add_special_tokens is False
        return SimpleNamespace(ids=[10, 20, 30])


def reference():
    return {'transcript': 'Reference sentence.', 'reference_tokens': [1, 4, 6560],
            'speaker': [0.25] * 2048, 'prompt_token': [[9, 10]],
            'prompt_feat': [[[0.1] * 80] * 4], 'embedding': [[0.2] * 192]}


def test_prepared_voice_roundtrip_without_raw_embedding():
    items = voices.from_cpp_voices({'My voice': reference()}, Tokenizer(), TOKEN_MAP)
    assert items[0].kind == 'prepared'
    assert items[0].tensors['raw_speaker'].size == 0
    assert items[0].tensors['lm_prefix'].tolist() == [151673, 158239, 10, 20, 30,
                                                   158240, 158237, 151676, 151679, 158235, 158238]
    blob = voices.write(items, checkpoint='a' * 64, s3gen_sha256='b' * 64)
    loaded = voices.read(blob)
    assert loaded.voices[0]['public_names'] == ['My voice']
    assert loaded.arrays['My voice']['raw_speaker'].shape == (0,)
    np.testing.assert_array_equal(loaded.arrays['My voice']['projected_speaker'], np.full(2048, .25, dtype='f4'))


def test_nonprepared_voice_cannot_omit_embedding():
    voice = voices.from_cpp_voices({'Bruno': reference()}, Tokenizer(), TOKEN_MAP)[0]
    voice.kind = 'builtin'
    with pytest.raises(voices.VoicesFormatError, match='speaker/embedding'):
        voices.write([voice], checkpoint='a' * 64)


def test_builtin_keeps_real_raw_embedding_for_production_compatibility():
    raw = np.linspace(-0.5, 0.5, 512, dtype='f4')
    items = voices.from_cpp_voices({'Bruno': reference()}, Tokenizer(), TOKEN_MAP,
                                   raw_embeddings={'Bruno': raw})
    assert items[0].kind == 'builtin'
    loaded = voices.read(voices.write(items, checkpoint='a' * 64))
    np.testing.assert_array_equal(loaded.arrays['Bruno']['raw_speaker'], raw)


def test_only_unchanged_public_voice_inherits_raw_vector(tmp_path, monkeypatch):
    import tokenizers
    monkeypatch.setattr(tokenizers.Tokenizer, 'from_file', lambda _: Tokenizer())
    (tmp_path / 'voices').mkdir()
    (tmp_path / 'cpp/default').mkdir(parents=True)
    public = tmp_path / 'voices/voices.json'
    public.write_text(json.dumps({'Bruno': {'artifacts': 'bruno.npz'}}))
    raw = np.linspace(-0.5, 0.5, 512, dtype='f4')
    np.savez(tmp_path / 'voices/bruno.npz', embedding=raw)
    source = tmp_path / 'cpp/default/voices.json'
    source.write_text(json.dumps({'Bruno': reference()}))
    cfg = {'token_map': TOKEN_MAP, 'voices': 'voices/voices.json',
           'cpp': {'decoders': {'default': {'voices': {'file': 'cpp/default/voices.json'}}}}}
    out = tmp_path / 'voices.kt2n'
    prepare_voices(source, tmp_path / 'tokenizer.json', cfg, out, 'a' * 64, 'b' * 64, model_dir=tmp_path)
    loaded = voices.read(out.read_bytes())
    assert loaded.voices[0]['kind'] == 'builtin'
    np.testing.assert_array_equal(loaded.arrays['Bruno']['raw_speaker'], raw)
    override = reference()
    override['speaker'][0] = 0.5
    custom = tmp_path / 'custom.json'
    custom.write_text(json.dumps({'Bruno': override}))
    prepare_voices(custom, tmp_path / 'tokenizer.json', cfg, out, 'a' * 64, 'b' * 64, model_dir=tmp_path)
    loaded = voices.read(out.read_bytes())
    assert loaded.voices[0]['kind'] == 'prepared'
    assert loaded.arrays['Bruno']['raw_speaker'].size == 0


def test_single_unpaired_reference_frame_is_recorded():
    row = reference()
    row['prompt_feat'] = [[[0.1] * 80] * 5]
    items = voices.from_cpp_voices({'Bruno': row}, Tokenizer(), TOKEN_MAP)
    loaded = voices.read(voices.write(items, checkpoint='a' * 64))
    assert loaded.arrays['Bruno']['prompt_feat'].shape == (4, 80)
    entry = next(t for t in loaded.meta['tensors'] if t['name'] == 'voices/Bruno/prompt_feat')
    assert entry['transform'] == 'trim_one_unpaired_terminal_frame'
    row['prompt_feat'] = [[[0.1] * 80] * 6]
    with pytest.raises(voices.VoicesFormatError, match='prompt_feat'):
        voices.from_cpp_voices({'Bruno': row}, Tokenizer(), TOKEN_MAP)


@pytest.mark.parametrize('field,value', [('speaker', [1.0]), ('reference_tokens', [6561]),
                                        ('prompt_token', [[-1]]), ('embedding', [[float('nan')] * 192])])
def test_invalid_prepared_voice_is_rejected(field, value):
    row = reference()
    row[field] = value
    with pytest.raises((voices.VoicesFormatError, ValueError)):
        voices.from_cpp_voices({'bad': row}, Tokenizer(), TOKEN_MAP)


def test_tampered_voice_tensor_rejected():
    blob = bytearray(voices.write(voices.from_cpp_voices({'v': reference()}, Tokenizer(), TOKEN_MAP), checkpoint='a' * 64))
    data_off = int.from_bytes(blob[32:40], 'little')
    blob[data_off] ^= 1
    with pytest.raises(voices.VoicesFormatError, match='sha256'):
        voices.read(bytes(blob))


def test_model_contract_and_path_traversal(tmp_path):
    check_config({'type': 'KITTEN2', 'sample_rate': 24000, 'token_map': TOKEN_MAP})
    with pytest.raises(ValueError, match='token map'):
        check_config({'type': 'KITTEN2', 'sample_rate': 24000, 'token_map': {}})
    out = tmp_path / 'model'
    out.mkdir()
    outside = tmp_path / 'secret'
    outside.write_text('synthetic')
    with pytest.raises(ValueError, match='unsafe'):
        local_file(out, '../secret')
    (out / 'escape').symlink_to(outside)
    with pytest.raises(ValueError, match='unsafe'):
        local_file(out, 'escape')


@pytest.mark.parametrize('shared', [False, True])
def test_huggingface_snapshot_blob_symlinks(tmp_path, shared):
    model = tmp_path / 'models--example--tts' / 'snapshots' / ('a' * 40)
    model.mkdir(parents=True)
    blobs = (tmp_path if shared else model.parent.parent) / 'blobs'
    blobs.mkdir()
    (blobs / 'weight-sha').write_bytes(b'weights')
    (model / 'model.safetensors').symlink_to(blobs / 'weight-sha')
    assert local_file(model, 'model.safetensors') == model / 'model.safetensors'


def test_converter_side_metadata_excludes_alternate_weights(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'runtime_config.json').write_text('{}')
    (tmp_path / 'model-ternary.safetensors').write_bytes(b'not a side file')
    files = source_files(tmp_path, 'model.safetensors', tmp_path / 'runtime_config.json')
    assert set(files) == {'config.json', 'runtime_config.json'}


def test_small_asset_integrity(tmp_path):
    source = tmp_path / 'model.kt2n'
    source.write_bytes(b'example')
    assert record(source, tmp_path) == {'file': 'model.kt2n', 'size': 7,
                                     'sha256': hashlib.sha256(b'example').hexdigest()}
    with pytest.raises(ValueError, match='checksum'):
        word_tables(b'wrong package')
