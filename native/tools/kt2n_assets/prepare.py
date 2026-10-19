"""Prepare a self-contained native bundle from the public Kitten TTS 2 model.

Only preparation needs Python model tooling. Runtime consumes the checked native
manifest, model files and prepared voices without network access.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

from . import convert, convert_s3gen, voices
from .safetensors_raw import sha256_file

ROOT = Path(__file__).resolve().parents[2]
ACOUSTIC_REPO = "ResembleAI/chatterbox-turbo"
ACOUSTIC_FILE = "s3gen_meanflow.safetensors"
NEMO_WHEEL_URL = "https://files.pythonhosted.org/packages/b6/08/ca954a6b77d259561f56580adfd52eb69f54dc2469e1184157708480e7ed/nemo_text_processing-1.2.0-py3-none-any.whl"
NEMO_WHEEL_SHA256 = "dd78657a0f031023de8b2c8fee6b55d71f444485826f7b507881e407afa5d804"
WHITELIST_TABLES = ("whitelist/tts.tsv", "whitelist/alternatives.tsv",
                    "whitelist/alternatives_all_format.tsv", "whitelist/lj_speech.tsv",
                    "whitelist/symbol.tsv", "suppletive.tsv")
TOKEN_MAP = {"audio_id_base": 151675, "num_audio_tokens": 6561, "speech_start_id": 151669,
             "speech_end_id": 151670, "text_start_id": 151671, "text_end_id": 151672,
             "start_id": 151673, "stop_id": 151674, "final_seg_id": 158236,
             "reference_speech_start_id": 158237, "reference_speech_end_id": 158238,
             "reference_text_start_id": 158239, "reference_text_end_id": 158240}


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)


def record(path: Path, root: Path) -> dict:
    return {"file": path.relative_to(root).as_posix(), "size": path.stat().st_size,
            "sha256": sha256_file(path)}


def check_config(config: dict) -> None:
    if config.get("type") != "KITTEN2" or config.get("sample_rate") != 24000:
        raise ValueError("native inference requires the KITTEN2 24 kHz model")
    if config.get("token_map") != TOKEN_MAP:
        raise ValueError("model token map is incompatible with this native engine")


def local_file(root: Path, name: str) -> Path:
    root = root.resolve()
    lexical = root / name
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"missing or unsafe model file: {name}")
    path = lexical.resolve()
    # Hugging Face snapshots use symlinks to content-addressed sibling blobs.
    # Permit that specific cache layout, while still rejecting arbitrary escapes.
    cached = (root.parent.name == "snapshots" and re.fullmatch(r"[a-f0-9]{40}", root.name)
              and any(path.is_relative_to(p) for p in
                      (root.parent.parent / "blobs", root.parent.parent.parent / "blobs")))
    if not (path.is_relative_to(root) or cached) or not path.is_file():
        raise ValueError(f"missing or unsafe model file: {name}")
    # Keep the snapshot path: callers also resolve sibling model/voice assets.
    path = lexical
    return path


def word_tables(wheel: bytes) -> tuple[bytes, bytes]:
    if hashlib.sha256(wheel).hexdigest() != NEMO_WHEEL_SHA256:
        raise ValueError("NeMo data wheel checksum mismatch")
    base = "nemo_text_processing/text_normalization/en/data/"
    whitelist, words = set(), set()
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for name in WHITELIST_TABLES:
            for line in archive.read(base + name).decode("utf-8").splitlines():
                left = line.split("\t", 1)[0].strip()
                if left:
                    whitelist.add(left.casefold())
        for name in sorted(archive.namelist()):
            if not name.startswith(base) or not name.endswith(".tsv"):
                continue
            for line in archive.read(name).decode("utf-8").splitlines():
                for word in re.split(r"[\s\-./]+", line.split("\t", 1)[0]):
                    word = word.strip(",;:'\"()").casefold()
                    if word:
                        words.add(word)
    if not whitelist or not words:
        raise ValueError("empty normalization word tables")
    encode = lambda values: ("\n".join(sorted(values)) + "\n").encode("utf-8")
    return encode(whitelist), encode(words)


def prepare_text(grammar_dir: Path, converter: Path, wheel: bytes, destination: Path) -> dict:
    """Map public English compact graphs to native OpenFst and exact word tables."""
    source = grammar_dir / "en" if (grammar_dir / "en").is_dir() else grammar_dir
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("language") != "en" or metadata.get("nemo_version") != "1.2.0":
        raise ValueError("native grammar conversion requires English NeMo 1.2.0 data")
    whitelist, words = word_tables(wheel)
    destination.mkdir(parents=True, exist_ok=True)
    sources = {}
    with tempfile.TemporaryDirectory(prefix="kt2n-grammar-") as scratch:
        for name, target in (("tagger", "tagger"), ("verbalizer", "verbalizer"), ("post", "post_processor")):
            compressed = (source / (name + ".fst.gz")).read_bytes()
            digest = hashlib.sha256(compressed).hexdigest()
            if digest != metadata["graphs"][name]["sha256"]:
                raise ValueError(f"public grammar checksum mismatch: {name}")
            inp = Path(scratch) / (name + ".compact")
            inp.write_bytes(gzip.decompress(compressed))
            out = destination / (target + ".fst.partial")
            subprocess.run([str(converter.resolve()), str(inp), str(out)], check=True)
            out.replace(destination / (target + ".fst"))
            sources[name] = digest
    atomic_bytes(destination / "nemo_whitelist.txt", whitelist)
    atomic_bytes(destination / "whitelist_words.txt", whitelist)
    atomic_bytes(destination / "data_words.txt", words)
    return {"grammar_sha256": sources, "nemo_data_sha256": NEMO_WHEEL_SHA256}


def prepare_voices(voice_file: Path, tokenizer_file: Path, config: dict, destination: Path,
                   checkpoint_sha: str, acoustic_sha: str, *, model_dir: Path | None = None) -> None:
    from tokenizers import Tokenizer
    import numpy as np
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    rows = json.loads(voice_file.read_text())
    raw = {}
    if model_dir is not None:
        public_file = local_file(model_dir, config["voices"])
        public = json.loads(public_file.read_text())
        export_file = local_file(model_dir, config["cpp"]["decoders"]["default"]["voices"]["file"])
        original = json.loads(export_file.read_text())
        for name, row in rows.items():
            # Only the unchanged public builtin carries the catalogue embedding;
            # a prepared/custom override must not inherit somebody else's raw vector.
            if name in public and row == original.get(name):
                artifact = local_file(model_dir, str(Path(config["voices"]).parent / public[name]["artifacts"]))
                with np.load(artifact, allow_pickle=False) as data:
                    raw[name] = np.asarray(data["embedding"], dtype="<f4").reshape(-1).copy()
    catalogue = voices.from_cpp_voices(rows, tokenizer, config["token_map"], raw_embeddings=raw)
    blob = voices.write(catalogue, checkpoint=checkpoint_sha, s3gen_sha256=acoustic_sha,
                        source={"voices_sha256": sha256_file(voice_file)},
                        peak_norm={"schema": "kitten-public-voices.v1", "default_gain_db": 0.0})
    voices.read(blob)
    atomic_bytes(destination, blob)


def acoustic_source(args) -> Path:
    acoustic = args.acoustic_source
    if acoustic is None:
        from huggingface_hub import hf_hub_download
        acoustic = Path(hf_hub_download(ACOUSTIC_REPO, ACOUSTIC_FILE, revision=convert_s3gen.PINNED_REVISION,
                        cache_dir=args.cache_dir, local_files_only=args.offline, token=False))
    if sha256_file(acoustic) != convert_s3gen.PINNED_SHA256:
        raise ValueError("full decoder source checksum mismatch")
    return acoustic


def prepare_cloning(args, acoustic: Path, manifest: dict) -> dict:
    from .clone_assets import prepare_clone_files
    catalogue = voices.read(local_file(args.out, manifest["voices"]["file"]).read_bytes())
    bank = [v for v in catalogue.voices if v["kind"] == "builtin"]
    if not bank:
        raise ValueError("cloning requires builtin voices with their original raw speaker embeddings")
    for voice in bank:
        raw = catalogue.arrays[voice["id"]]["raw_speaker"]
        if raw.shape != (512,) or not (raw != 0).any():
            raise ValueError("cloning speaker bank contains an empty or zero vector")
    destination = args.out / "native/cloning"
    if args.clone_consts and args.clone_pyannote:
        source = prepare_clone_files(acoustic, args.clone_consts, args.clone_pyannote, destination)
    else:
        # Only an explicitly requested cloning bundle imports these CPU model
        # preparation dependencies or uses authorized gated-model access.
        from .clone_assets import PYANNOTE_REPO, PYANNOTE_REVISION
        from .export_clone import export
        checkpoint = args.pyannote_model
        if checkpoint is None:
            from huggingface_hub import hf_hub_download
            checkpoint = Path(hf_hub_download(PYANNOTE_REPO, "pytorch_model.bin",
                revision=PYANNOTE_REVISION, cache_dir=args.cache_dir,
                local_files_only=args.offline, token=True))
        elif checkpoint.is_dir():
            checkpoint = checkpoint / "pytorch_model.bin"
        with tempfile.TemporaryDirectory(prefix="kt2n-clone-export-") as scratch:
            exports = Path(scratch) / "exports"
            export(checkpoint, acoustic, exports)
            source = prepare_clone_files(acoustic, exports / "voicebuild_consts.safetensors",
                                         exports / "voicebuild_pyannote.safetensors", destination)
    return {"version": 1, "directory": "native/cloning",
            "files": [record(path, args.out) for path in sorted(destination.iterdir())],
            "source": source, "speaker_bank_voices": len(bank)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, help="Existing public model checkout/cache (no model download)")
    p.add_argument("--repo", default="KittenML/kitten-tts-2")
    p.add_argument("--revision", default="main")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--offline", action="store_true", help="Use local files/cache only")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--voices", type=Path, help="CPP default voices.json, including prepared references")
    p.add_argument("--voices-only", action="store_true", help="Update voice catalogue in an existing native bundle")
    p.add_argument("--with-cloning", "--cloning", action="store_true", help="Include the original optional cloning networks")
    p.add_argument("--cloning-only", action="store_true", help="Add cloning to an existing native bundle")
    p.add_argument("--clone-consts", type=Path, help="Original voicebuild_consts.safetensors export")
    p.add_argument("--clone-pyannote", type=Path, help="Original authorized voicebuild_pyannote.safetensors export")
    p.add_argument("--pyannote-model", type=Path,
                   help="Authorized pinned pyannote checkpoint/directory; otherwise cloning uses your HF login")
    p.add_argument("--acoustic-source", type=Path, help="Pinned full decoder safetensors; otherwise use HF cache/download")
    p.add_argument("--grammar-dir", type=Path, default=ROOT.parent / "vendor/kitten-text-processing/kitten_text_processing/data")
    p.add_argument("--text-converter", type=Path, help="Built kt2n_prepare_text_data executable")
    p.add_argument("--nemo-wheel", type=Path, help="Pinned NeMo 1.2.0 wheel (read data only; never installed)")
    p.add_argument("--weight-norm-device", choices=("cpu", "cuda"), default="cuda",
                   help="Fold decoder weight norms on the same device family as the reference")
    a = p.parse_args(argv)
    a.out = a.out.resolve()
    config_path = a.out / "config.json"
    if a.cloning_only:
        a.with_cloning = True
    if bool(a.clone_consts) != bool(a.clone_pyannote):
        p.error("--clone-consts and --clone-pyannote must be supplied together")
    if a.pyannote_model and a.clone_pyannote:
        p.error("choose --pyannote-model or the two prepared clone exports")
    if (a.pyannote_model or a.clone_consts) and not a.with_cloning:
        p.error("clone sources require --with-cloning or --cloning-only")
    if a.voices_only and a.with_cloning:
        p.error("update voices and cloning in separate commands")
    if a.cloning_only:
        cfg = json.loads(config_path.read_text())
        check_config(cfg)
        manifest = cfg["native"]
        if manifest.get("version") != 1 or manifest.get("decoder") != "default" or "cloning" in manifest:
            raise ValueError("incompatible bundle or cloning already prepared")
        for key in ("lm", "acoustic", "voices", "tokenizer"):
            expected = manifest[key]
            if record(local_file(a.out, expected["file"]), a.out) != expected:
                raise ValueError(f"bundle checksum mismatch: {key}")
        acoustic = acoustic_source(a)
        manifest["cloning"] = prepare_cloning(a, acoustic, manifest)
        atomic_bytes(config_path, (json.dumps(cfg, indent=2) + "\n").encode())
        print(config_path)
        return 0
    if a.voices_only:
        if not a.voices:
            p.error("--voices-only requires --voices")
        cfg = json.loads(config_path.read_text())
        check_config(cfg)
        manifest = cfg["native"]
        if manifest.get("version") != 1 or manifest.get("decoder") != "default":
            raise ValueError("incompatible native bundle")
        for key in ("lm", "acoustic", "tokenizer"):
            expected = manifest[key]
            got = record(local_file(a.out, expected["file"]), a.out)
            if got != expected:
                raise ValueError(f"bundle checksum mismatch: {key}")
        target = local_file(a.out, manifest["voices"]["file"])
        prepare_voices(a.voices, local_file(a.out, manifest["tokenizer"]["file"]), cfg, target,
                       manifest["lm"]["sha256"], manifest["acoustic"]["sha256"])
        manifest["voices"] = record(target, a.out)
        atomic_bytes(config_path, (json.dumps(cfg, indent=2) + "\n").encode())
        print(config_path)
        return 0
    if config_path.exists():
        raise ValueError("output bundle already exists; choose a new --out or use --voices-only")
    if not a.text_converter or not a.text_converter.is_file():
        p.error("--text-converter must point to the built kt2n_prepare_text_data executable")
    if a.model_dir:
        model = a.model_dir.resolve()
    else:
        from huggingface_hub import snapshot_download
        model = Path(snapshot_download(a.repo, revision=a.revision, cache_dir=a.cache_dir,
                      local_files_only=a.offline, token=False,
                      allow_patterns=["config.json", "lm/config.json", "lm/model.safetensors",
                                      "lm/tokenizer.json", "cpp/default/voices.json", "voices/voices.json", "voices/*.npz"]))
    config = json.loads(local_file(model, "config.json").read_text())
    check_config(config)
    lm_dir = local_file(model, config["lm_dir"] + "/model.safetensors").parent
    voice_file = a.voices or local_file(model, config["cpp"]["decoders"]["default"]["voices"]["file"])
    acoustic = acoustic_source(a)
    if a.nemo_wheel:
        wheel = a.nemo_wheel.read_bytes()
    else:
        cache = (a.cache_dir or Path.home() / ".cache/kitten-tts") / "native-conversion"
        cached = cache / "nemo_text_processing-1.2.0.whl"
        if not cached.exists():
            if a.offline:
                raise ValueError("offline mode needs --nemo-wheel or the cached NeMo data wheel")
            with urllib.request.urlopen(NEMO_WHEEL_URL, timeout=60) as response:
                atomic_bytes(cached, response.read())
        wheel = cached.read_bytes()
    # Validate small dependencies before starting expensive model conversion.
    word_tables(wheel)
    dest = a.out / "native"
    dest.mkdir(parents=True, exist_ok=True)
    source_text = prepare_text(a.grammar_dir, a.text_converter, wheel, dest / "text")
    with tempfile.TemporaryDirectory(prefix="kt2n-config-") as scratch:
        runtime = Path(scratch) / "runtime_config.json"
        runtime.write_text(json.dumps({"type": config["type"], "token_map": config["token_map"],
                                      "generation": config.get("generation", {})}))
        convert.convert(lm_dir, dest / "model.kt2n", runtime_config=str(runtime))
    convert_s3gen.convert(str(acoustic), str(dest / "s3gen.kt2n"),
                         manifest_path=Path(__file__).with_name("s3gen_tensors.tsv"),
                         expect_sha256=convert_s3gen.PINNED_SHA256, revision=convert_s3gen.PINNED_REVISION,
                         device=a.weight_norm_device, extra_groups=[])
    tokenizer = local_file(model, config["lm_dir"] + "/tokenizer.json")
    shutil.copyfile(tokenizer, dest / "tokenizer.json")
    prepare_voices(voice_file, tokenizer, config, dest / "voices.kt2n",
                   sha256_file(dest / "model.kt2n"), sha256_file(dest / "s3gen.kt2n"), model_dir=model)
    manifest = {"version": 1, "decoder": "default",
                "lm": record(dest / "model.kt2n", a.out), "acoustic": record(dest / "s3gen.kt2n", a.out),
                "voices": record(dest / "voices.kt2n", a.out), "tokenizer": record(dest / "tokenizer.json", a.out),
                "text": {"directory": "native/text", "files": [record(f, a.out) for f in sorted((dest / "text").iterdir())]},
                "source": {"model_repo": a.repo,
                           "revision": model.name if re.fullmatch(r"[a-f0-9]{40}", model.name) else a.revision,
                           "lm_sha256": sha256_file(lm_dir / "model.safetensors"),
                           "acoustic_sha256": convert_s3gen.PINNED_SHA256,
                           "weight_norm_device": a.weight_norm_device, **source_text}}
    # A standalone bundle contains native files only. When merging into an HF
    # model repository, publishers copy native/ and only the native config key.
    config.pop("cpp", None)
    if a.with_cloning:
        manifest["cloning"] = prepare_cloning(a, acoustic, manifest)
    config["native"] = manifest
    atomic_bytes(config_path, (json.dumps(config, indent=2) + "\n").encode())
    print(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
