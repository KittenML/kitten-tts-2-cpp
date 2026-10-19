"""voices.kt2n writer/reader (tools/kt2n_assets/voices.py): round trip, canonical bytes, strict rejection."""
from __future__ import annotations

import struct
from functools import partial

import numpy as np
import pytest

from kt2n_assets import voices as V

write = partial(V.write, checkpoint="a" * 64)


def _voice(vid: str, tp: int = 7, p: int = 30, seed: int = 0, **kw) -> V.Voice:
    r = np.random.default_rng(seed)
    raw = r.standard_normal(512).astype(np.float32)
    raw /= np.linalg.norm(raw)
    t = {"lm_prefix": r.integers(0, V.VOCAB_SIZE, p).astype("<i4"), "raw_speaker": raw,
         "projected_speaker": r.standard_normal(2048).astype(np.float32),
         "prompt_token": r.integers(0, V.S3_TOKEN_VOCAB, tp).astype("<i4"),
         "prompt_feat": r.standard_normal((2 * tp, 80)).astype(np.float32),
         "embedding": r.standard_normal(192).astype(np.float32)}
    return V.Voice(id=vid, tensors=t, **kw)


def test_round_trip_and_alignment():
    vs = [_voice("a", seed=1), _voice("b_hidden", tp=250, p=327, seed=2, hidden=True, peak_gain_db=-1.25),
          _voice("voice_x", seed=3, kind="saved", identity={"use_icl": True, "nearest_flagship_voice_id": "a"})]
    blob = write(vs, source={"test": 1})
    assert len(blob) % 4096 == 0 and blob[:8] == b"KT2NVOX1"
    f = V.read(blob)
    assert [v["id"] for v in f.voices] == ["a", "b_hidden", "voice_x"]
    assert f.voices[1]["hidden"] is True and f.voices[1]["prompt_frames"] == 500
    assert f.voices[2]["identity"]["nearest_flagship_voice_id"] == "a"
    for v in vs:
        for name, a in v.tensors.items():
            assert np.array_equal(f.arrays[v.id][name], a), (v.id, name)
    for t in f.meta["tensors"]:
        assert t["data"]["off"] % 4096 == 0 and t["transform"] == "copy" and t["src_sha256"] == t["data"]["sha256"]
    assert write(vs, source={"test": 1}) == blob  # deterministic


@pytest.mark.parametrize("mutate", ["magic", "reserved", "meta_byte", "data_byte", "padding", "truncate"])
def test_rejects_corruption(mutate):
    blob = bytearray(write([_voice("a")]))
    meta_len = struct.unpack_from("<Q", blob, 24)[0]
    data_off = struct.unpack_from("<Q", blob, 32)[0]
    if mutate == "magic":
        blob[7:8] = b"2"
    elif mutate == "reserved":
        blob[60] = 1
    elif mutate == "meta_byte":
        i = blob.index(b'"kind":"builtin"', 64)
        blob[i + 8:i + 15] = b"saved\x20\x20"  # breaks canonical form / JSON
    elif mutate == "data_byte":
        blob[data_off + 5] ^= 1
    elif mutate == "padding":
        blob[64 + meta_len + 1] = 1
    elif mutate == "truncate":
        blob = blob[:-4096]
    with pytest.raises(V.VoicesFormatError):
        V.read(bytes(blob))


def test_rejects_bad_tensors():
    v = _voice("a")
    v.tensors["prompt_feat"] = v.tensors["prompt_feat"][:-1]
    with pytest.raises(V.VoicesFormatError):
        write([v])
    v = _voice("a")
    v.tensors["prompt_token"][0] = V.S3_TOKEN_VOCAB
    with pytest.raises(V.VoicesFormatError):
        write([v])
    v = _voice("a")
    v.tensors["embedding"][3] = np.nan
    with pytest.raises(V.VoicesFormatError):
        write([v])
    with pytest.raises(V.VoicesFormatError):
        write([_voice("a"), _voice("a")])


def test_language_is_the_only_metadata_change():
    vs = [_voice("a", seed=1), _voice("b", seed=2)]
    before = write(vs, source={"test": 1})
    vs[1].language = "hi"
    vs[0].language = "en"
    after = write(vs, source={"test": 1})
    fb, fa = V.read(before), V.read(after)
    assert [v["language"] for v in fa.voices] == ["en", "hi"]
    for vb, va in zip(fb.voices, fa.voices):
        assert {k: v for k, v in vb.items() if k != "language"} == {k: v for k, v in va.items() if k != "language"}
    hb, ha = struct.unpack("<8sIIQQQQQ8s", before[:64]), struct.unpack("<8sIIQQQQQ8s", after[:64])
    assert before[hb[5]:] == after[ha[5]:]  # the data section is byte-identical
