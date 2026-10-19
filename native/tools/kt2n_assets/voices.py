"""Read and write Kitten TTS native voice catalogues.

Public prepared voices use their existing projected speaker vector. A missing
raw embedding is represented by an empty tensor, never a fabricated embedding.
Such voices support inference and are excluded from the cloning speaker bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .format import ALIGN, HEADER_SIZE, align_up, dumps_meta, numel

MAGIC = b"KT2NVOX1"
VERSION = 1
HEADER_STRUCT = struct.Struct("<8sIIQQQQQ8s")

S3GEN_SHA256 = "d65cb687a2ed581ee6cc297e919ffefa63386944f42364ae13b78a594945514f"
S3GEN_REVISION = "1e4698ca7cbb41ff030c4185f0927a3b42d76924"
VOCAB_SIZE = 158_241
S3_TOKEN_VOCAB = 6_561
MAX_PREFIX = 8_192
PEAK_NORM = {"schema": "kitten-public-voices.v1", "default_gain_db": 0.0}
KINDS = ("builtin", "inline", "saved", "prepared")
# (field, dtype, rank) in file order per voice.
FIELDS = (
    ("lm_prefix", "i32", 1),
    ("raw_speaker", "f32", 1),
    ("projected_speaker", "f32", 1),
    ("prompt_token", "i32", 1),
    ("prompt_feat", "f32", 2),
    ("embedding", "f32", 1),
)
NP = {"i32": np.dtype("<i4"), "f32": np.dtype("<f4")}
VOICE_META_KEYS = {"id", "public_names", "language", "hidden", "prompt_tokens", "prompt_frames", "lm_prefix_len",
                   "peak_gain_db", "signature", "kind"}
OPTIONAL_VOICE_META_KEYS = {"identity"}  # saved / inline voices: production's ArtifactIdentity, verbatim


class VoicesFormatError(ValueError):
    pass


def _sha(b: bytes | memoryview) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass
class Voice:
    id: str
    tensors: dict[str, np.ndarray]  # FIELDS, already in the stored dtype/shape
    kind: str = "builtin"
    public_names: list[str] = field(default_factory=list)
    language: str = ""
    hidden: bool = False
    peak_gain_db: float = PEAK_NORM["default_gain_db"]
    signature: str = ""
    identity: dict[str, Any] | None = None
    # (transform, [(source name, sha256 of source bytes)]) per field
    provenance: dict[str, tuple[str, list[tuple[str, str]]]] = field(default_factory=dict)

    def meta(self) -> dict[str, Any]:
        tp = int(self.tensors["prompt_token"].shape[0])
        m = {"id": self.id, "public_names": list(self.public_names), "language": self.language,
             "hidden": bool(self.hidden), "prompt_tokens": tp, "prompt_frames": 2 * tp,
             "lm_prefix_len": int(self.tensors["lm_prefix"].shape[0]), "peak_gain_db": float(self.peak_gain_db),
             "signature": self.signature, "kind": self.kind}
        if self.identity is not None:
            m["identity"] = dict(self.identity)
        return m


def check_voice_tensors(vid: str, t: dict[str, np.ndarray], vocab: int = VOCAB_SIZE, *, kind: str = "builtin") -> None:
    """The per-voice checks every reader and writer applies (FORMAT.md + VoiceArtifact.__post_init__)."""
    if set(t) != {f for f, _, _ in FIELDS}:
        raise VoicesFormatError(f"{vid}: fields {sorted(t)}")
    for name, dt, rank in FIELDS:
        a = t[name]
        if a.dtype != NP[dt] or a.ndim != rank:
            raise VoicesFormatError(f"{vid}/{name}: dtype {a.dtype} rank {a.ndim}, want {dt} rank {rank}")
        if dt == "f32" and not np.isfinite(a).all():
            raise VoicesFormatError(f"{vid}/{name}: non-finite values")
    p, tp = t["lm_prefix"].shape[0], t["prompt_token"].shape[0]
    if not 1 <= p <= MAX_PREFIX or (t["lm_prefix"] < 0).any() or (t["lm_prefix"] >= vocab).any():
        raise VoicesFormatError(f"{vid}/lm_prefix: length {p} or ids out of [0, {vocab})")
    raw_shapes = ((0,), (512,)) if kind == "prepared" else ((512,),)
    if t["raw_speaker"].shape not in raw_shapes or t["projected_speaker"].shape != (2048,) or t["embedding"].shape != (192,):
        raise VoicesFormatError(f"{vid}: speaker/embedding shapes")
    if tp < 1 or t["prompt_feat"].shape != (2 * tp, 80):
        raise VoicesFormatError(f"{vid}: prompt_feat {t['prompt_feat'].shape} != [2*{tp}, 80]")
    if (t["prompt_token"] < 0).any() or (t["prompt_token"] >= S3_TOKEN_VOCAB).any():
        raise VoicesFormatError(f"{vid}/prompt_token: values outside [0, {S3_TOKEN_VOCAB})")


def _narrow_i32(a: np.ndarray, what: str) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.kind not in "iu":
        raise VoicesFormatError(f"{what}: not an integer array ({a.dtype})")
    if a.size and (a.min() < -(2**31) or a.max() >= 2**31):
        raise VoicesFormatError(f"{what}: does not fit int32")
    return np.ascontiguousarray(a.astype("<i4"))


def _src(name: str, a: np.ndarray) -> tuple[str, str]:
    return name, _sha(np.ascontiguousarray(a).tobytes())


def write(voices: list[Voice], *, source: dict[str, Any] | None = None, checkpoint: str,
          s3gen_sha256: str = S3GEN_SHA256, peak_norm: dict[str, Any] | None = None) -> bytes:
    if not voices:
        raise VoicesFormatError("no voices")
    ids = [v.id for v in voices]
    if len(set(ids)) != len(ids) or any(not i or "/" in i for i in ids):
        raise VoicesFormatError("voice ids must be unique, non-empty, without '/'")
    tensors, blobs, cursor = [], [], 0
    for v in voices:
        if v.kind not in KINDS:
            raise VoicesFormatError(f"{v.id}: kind {v.kind!r}")
        if not math.isfinite(v.peak_gain_db):
            raise VoicesFormatError(f"{v.id}: peak_gain_db")
        check_voice_tensors(v.id, v.tensors, kind=v.kind)
        for name, dt, _ in FIELDS:
            a = np.ascontiguousarray(v.tensors[name], dtype=NP[dt])
            raw = a.tobytes()
            digest = _sha(raw)
            transform, sources = v.provenance.get(name, ("copy", [(name, digest)]))
            entry = {"name": f"voices/{v.id}/{name}", "kind": "dense", "dtype": dt, "shape": list(a.shape),
                     "data": {"off": cursor, "len": len(raw), "shape": list(a.shape), "dtype": dt, "sha256": digest},
                     "group": "voice", "transform": transform,
                     "sources": [{"name": n, "sha256": s} for n, s in sources]}
            if transform == "copy":
                if sources[0][1] != digest:
                    raise VoicesFormatError(f"{v.id}/{name}: copy source digest differs from the stored bytes")
                entry["src_sha256"] = digest
            tensors.append(entry)
            blobs.append((cursor, raw))
            cursor = align_up(cursor + len(raw))
    data_len = align_up(cursor) if cursor else 0
    meta = {"format": "kt2n", "version": VERSION, "checkpoint": checkpoint, "s3gen_sha256": s3gen_sha256,
            "peak_norm": dict(peak_norm or PEAK_NORM), "voices": [v.meta() for v in voices], "tensors": tensors}
    if source is not None:
        meta["source"] = source
    mb = dumps_meta(meta)
    data_off = align_up(HEADER_SIZE + len(mb))
    out = bytearray(data_off + data_len)
    out[:HEADER_SIZE] = HEADER_STRUCT.pack(MAGIC, VERSION, 0, HEADER_SIZE, len(mb), data_off, data_len,
                                           data_off + data_len, b"\0" * 8)
    out[HEADER_SIZE:HEADER_SIZE + len(mb)] = mb
    for off, raw in blobs:
        out[data_off + off:data_off + off + len(raw)] = raw
    return bytes(out)


@dataclass
class VoicesFile:
    meta: dict[str, Any]
    voices: list[dict[str, Any]]
    arrays: dict[str, dict[str, np.ndarray]]  # voice id -> field -> array (read-only views)


def read(data: bytes, *, verify_sha256: bool = True) -> VoicesFile:
    """Strict reader (the C++ reader src/voice/voice_blob.cc applies the same rules)."""
    data = bytes(data)
    if len(data) < HEADER_SIZE:
        raise VoicesFormatError("shorter than the header")
    magic, version, flags, meta_off, meta_len, data_off, data_len, file_len, reserved = HEADER_STRUCT.unpack(
        data[:HEADER_SIZE])
    if magic != MAGIC or version != VERSION or flags != 0 or reserved != b"\0" * 8:
        raise VoicesFormatError("bad magic/version/flags/reserved")
    if meta_off != HEADER_SIZE or data_off != align_up(HEADER_SIZE + meta_len) or data_len % ALIGN \
            or file_len != len(data) or data_off + data_len != file_len:
        raise VoicesFormatError("bad offsets/lengths")
    mb = data[meta_off:meta_off + meta_len]
    try:
        meta = json.loads(mb.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise VoicesFormatError(f"metadata: {e}") from e
    if dumps_meta(meta) != mb:
        raise VoicesFormatError("metadata is not canonical")
    if any(data[meta_off + meta_len:data_off]):
        raise VoicesFormatError("non-zero padding after the metadata")
    if meta.get("format") != "kt2n" or meta.get("version") != VERSION:
        raise VoicesFormatError("metadata format/version")
    for k in ("checkpoint", "s3gen_sha256"):
        if not isinstance(meta.get(k), str) or len(meta[k]) != 64:
            raise VoicesFormatError(f"metadata {k}")
    voices = meta.get("voices")
    if not isinstance(voices, list) or not voices:
        raise VoicesFormatError("metadata voices")
    by_name = {t["name"]: t for t in meta["tensors"]}
    if len(by_name) != len(meta["tensors"]) or len(meta["tensors"]) != 6 * len(voices):
        raise VoicesFormatError("tensor list does not match the voices")
    spans, arrays, seen = [], {}, set()
    for vm in voices:
        keys = set(vm)
        if not VOICE_META_KEYS <= keys or keys - VOICE_META_KEYS - OPTIONAL_VOICE_META_KEYS:
            raise VoicesFormatError(f"voice metadata keys {sorted(keys)}")
        vid = vm["id"]
        if not isinstance(vid, str) or not vid or "/" in vid or vid in seen or vm["kind"] not in KINDS:
            raise VoicesFormatError(f"voice id/kind {vid!r}")
        seen.add(vid)
        t = {}
        for name, dt, _ in FIELDS:
            e = by_name.get(f"voices/{vid}/{name}")
            if e is None or e.get("kind") != "dense" or e.get("dtype") != dt:
                raise VoicesFormatError(f"{vid}/{name}: missing or wrong entry")
            s = e["data"]
            if s["dtype"] != dt or s["shape"] != e["shape"] or not all(type(d) is int and d >= 0 for d in s["shape"]):
                raise VoicesFormatError(f"{vid}/{name}: segment shape/dtype")
            if type(s["off"]) is not int or s["off"] < 0 or s["off"] % ALIGN \
                    or s["len"] != numel(s["shape"]) * 4 or s["off"] + s["len"] > data_len:
                raise VoicesFormatError(f"{vid}/{name}: segment bounds")
            raw = memoryview(data)[data_off + s["off"]:data_off + s["off"] + s["len"]]
            if verify_sha256 and _sha(raw) != s["sha256"]:
                raise VoicesFormatError(f"{vid}/{name}: sha256 mismatch")
            if e.get("transform") == "copy" and e.get("src_sha256") != s["sha256"]:
                raise VoicesFormatError(f"{vid}/{name}: src_sha256")
            spans.append((s["off"], s["off"] + s["len"]))
            arr = np.frombuffer(raw, dtype=NP[dt]).reshape(s["shape"])
            t[name] = arr
        check_voice_tensors(vid, t, kind=vm["kind"])
        if vm["prompt_tokens"] != t["prompt_token"].shape[0] or vm["prompt_frames"] != 2 * vm["prompt_tokens"] \
                or vm["lm_prefix_len"] != t["lm_prefix"].shape[0]:
            raise VoicesFormatError(f"{vid}: metadata lengths disagree with the tensors")
        arrays[vid] = t
    spans.sort()
    cursor = 0
    for a, b in spans:
        if a < cursor:
            raise VoicesFormatError("segments overlap")
        if any(data[data_off + cursor:data_off + a]):
            raise VoicesFormatError("non-zero padding between segments")
        cursor = b
    if any(data[data_off + cursor:]):
        raise VoicesFormatError("non-zero padding at the end")
    return VoicesFile(meta=meta, voices=voices, arrays=arrays)


# ---- public CPP voice conversion -------------------------------------------
def from_cpp_voices(rows: dict[str, Any], tokenizer: Any, token_map: dict[str, int],
                    *, raw_embeddings: dict[str, np.ndarray] | None = None) -> list[Voice]:
    """Convert the existing CPP voices.json, including prepare_reference.py output."""
    languages = {"Arabic": "ar", "Chinese": "zh", "French": "fr", "German": "de",
                 "Hindi": "hi", "Italian": "it", "Portuguese": "pt", "Russian": "ru", "Spanish": "es"}
    if not isinstance(rows, dict) or not rows:
        raise VoicesFormatError("voices.json must contain a nonempty voice mapping")
    result = []
    for name, row in rows.items():
        transcript = row["transcript"]
        if not isinstance(transcript, str) or not transcript.strip():
            raise VoicesFormatError(f"{name}: transcript must be nonempty")
        ref = _narrow_i32(np.asarray(row["reference_tokens"]), "reference_tokens").reshape(-1)
        if not ref.size or (ref < 0).any() or (ref >= token_map["num_audio_tokens"]).any():
            raise VoicesFormatError(f"{name}: reference tokens out of range")
        ids = tokenizer.encode(transcript.strip(), add_special_tokens=False).ids
        prefix = [token_map["start_id"], token_map["reference_text_start_id"], *ids,
                  token_map["reference_text_end_id"], token_map["reference_speech_start_id"],
                  *(int(t) + token_map["audio_id_base"] for t in ref), token_map["reference_speech_end_id"]]
        tensors = {
            "lm_prefix": np.asarray(prefix, dtype="<i4"),
            "raw_speaker": np.asarray((raw_embeddings or {}).get(name, row.get("raw_speaker", [])),
                                      dtype="<f4").reshape(-1),
            "projected_speaker": np.asarray(row["speaker"], dtype="<f4").reshape(-1),
            "prompt_token": _narrow_i32(np.asarray(row["prompt_token"]), "prompt_token").reshape(-1),
            "prompt_feat": np.asarray(row["prompt_feat"], dtype="<f4").reshape(-1, 80),
            "embedding": np.asarray(row["embedding"], dtype="<f4").reshape(-1),
        }
        # A reference ending between token boundaries can have one unmatched
        # final mel frame. The streaming renderer uses exact 2:1 conditioning.
        # Preserve this explicit transform in the file's tensor provenance.
        original_features = tensors["prompt_feat"]
        paired_frames = 2 * tensors["prompt_token"].size
        if original_features.shape[0] == paired_frames + 1:
            tensors["prompt_feat"] = np.ascontiguousarray(original_features[:paired_frames])
        kind = "builtin" if name in (raw_embeddings or {}) else "prepared"
        voice = Voice(id=name, tensors=tensors, kind=kind, public_names=[name],
                      language=row.get("language", languages.get(name, "en")),
                      peak_gain_db=float(row.get("peak_gain_db", 0.0)))
        if original_features.shape[0] == paired_frames + 1:
            voice.provenance["prompt_feat"] = ("trim_one_unpaired_terminal_frame",
                                                [_src("cpp.prompt_feat", original_features)])
        check_voice_tensors(name, tensors, kind=kind)
        result.append(voice)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("verify")
    check.add_argument("file", type=pathlib.Path)
    public = sub.add_parser("cpp", help="Convert built-in or prepared CPP voices")
    public.add_argument("--voices", type=pathlib.Path, required=True)
    public.add_argument("--tokenizer", type=pathlib.Path, required=True)
    public.add_argument("--config", type=pathlib.Path, required=True)
    public.add_argument("--checkpoint-sha256", required=True)
    public.add_argument("--acoustic-sha256", required=True)
    public.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = read(args.file.read_bytes())
        print(json.dumps({"ok": True, "voices": [v["id"] for v in result.voices]}))
        return
    from tokenizers import Tokenizer
    rows = json.loads(args.voices.read_text())
    config = json.loads(args.config.read_text())
    voices = from_cpp_voices(rows, Tokenizer.from_file(str(args.tokenizer)), config["token_map"])
    blob = write(voices, checkpoint=args.checkpoint_sha256, s3gen_sha256=args.acoustic_sha256,
                 source={"voices_sha256": _sha(args.voices.read_bytes())})
    read(blob)
    args.out.write_bytes(blob)
    print(json.dumps({"voices": len(voices), "sha256": _sha(blob)}))


if __name__ == "__main__":
    main()
