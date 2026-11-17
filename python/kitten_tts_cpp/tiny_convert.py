#!/usr/bin/env python3
"""Convert a Kitten TTS 2 mini / nano / pico repository (config.json type KITTEN2_TINY) to GGUF.

    python tools/kitten-tts-tiny/convert_kitten_tiny_to_gguf.py KittenML/kitten-tts-2-nano --outdir out --outtype q8_0 [--model kitten_tts_2_nano_en]
    python -m kitten_tts_cpp.tiny_convert ...   (the same, from the installed package)

The input is a Hugging Face repository id or a local directory with config.json, vocab.json, the fp32
`*.safetensors` weights and the `voices_<lang>.npz` voice packs. One GGUF is written per model entry of
config.json["models"] (mini: one model for all languages; nano / pico: one model per language).

What the converter changes (the C++ graph in kitten_tts_tiny.cpp expects these layouts):
- weight norm (weight_g, weight_v) is folded into one weight;
- Conv1d weights [Cout, Cin, K] become k-major [Cout, K, Cin] (ggml ne [Cin, K, Cout]), so one matrix product
  over a sliding window of the [C, T] activation computes the convolution; strided convolutions are rewritten
  as stride-1 convolutions over a space-to-depth input (s time steps folded into the channels);
- ConvTranspose1d weights [Cin, Cout, K] become [Cout, K, Cin] (a matrix product then col2im_1d);
- the depthwise transposed convolutions ("pool") become [3, C]; snake alphas become [C] plus their inverse;
- each bidirectional LSTM becomes stacked input / hidden weights [2, 4H, In] and one summed bias [2, 4H];
- ALBERT's single shared layer is named bert.layer.* (ggml tensor names are shorter than 64 bytes);
- for q8_0, input channels are zero-padded to a multiple of 32 (the C++ side pads activations to match).
Voices are stored as f32 tensors voice.<lang>.<key> [rows, 256] whatever the weight type.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import uuid
from pathlib import Path

import numpy as np

try:
    import gguf  # the gguf package from PyPI, or gguf-py of this repository
except ImportError:
    _gguf_py = Path(__file__).resolve().parents[2] / "gguf-py"
    if not _gguf_py.is_dir():
        raise
    sys.path.insert(0, str(_gguf_py))
    import gguf  # noqa: E402

ARCH = "kitten-tts-tiny"
QK = 32


def _load_safetensors(path):
    from safetensors.numpy import load_file
    return {k: np.ascontiguousarray(v.astype(np.float32)) for k, v in load_file(str(path)).items()}


def _fold_weight_norm(sd):
    out = {}
    for k, v in sd.items():
        if k.endswith(".weight_v"):
            base = k[: -len(".weight_v")]
            g = sd[base + ".weight_g"]
            axes = tuple(range(1, v.ndim))
            norm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=axes, keepdims=True))
            out[base + ".weight"] = (v.astype(np.float64) * (g.astype(np.float64) / norm)).astype(np.float32)
        elif k.endswith(".weight_g"):
            continue
        else:
            out[k] = v
    return out


def _conv_kmajor(w, stride=1):
    """[Cout, Cin, K] -> [Cout, K', Cin*stride] (space-to-depth for stride > 1)."""
    cout, cin, k = w.shape
    if stride == 1:
        return np.ascontiguousarray(w.transpose(0, 2, 1))
    kp = -(-k // stride)
    out = np.zeros((cout, kp, stride * cin), np.float32)
    for kk in range(k):
        kb, j = divmod(kk, stride)
        out[:, kb, j * cin:(j + 1) * cin] = w[:, :, kk]
    return out


def _strided_convs(arch):
    rates = arch["decoder"]["upsample_rates"]
    s = int(np.prod(rates[1:])) if len(rates) > 1 else 1
    return {"decoder.F0_conv.weight": 2, "decoder.N_conv.weight": 2,
            "decoder.generator.noise_convs.0.weight": s}


def convert_tensors(sd, arch):
    """-> dict name -> (array, kind) where kind says which tensors may be quantised."""
    sd = _fold_weight_norm(sd)
    strided = _strided_convs(arch)
    out = {}
    lstm_groups = {}
    for k, v in sd.items():
        if k.startswith("bert.pooler."):
            continue
        k = k.replace("bert.encoder.albert_layer_groups.0.albert_layers.0.", "bert.layer.")   # ggml names < 64 bytes
        m = re.match(r"(.*)\.(weight_ih|weight_hh|bias_ih|bias_hh)_l0(_reverse)?$", k)
        if m:
            lstm_groups.setdefault(m.group(1), {})[m.group(2) + (m.group(3) or "")] = v
            continue
        if re.search(r"\.alpha[12]\.\d+$", k):
            a = v.reshape(-1).astype(np.float32)
            out[k] = (a, "vec")
            out[k + ".inv"] = ((np.float32(1) / a).astype(np.float32), "vec")
            continue
        if k.endswith(".pool.weight"):
            out[k] = (np.ascontiguousarray(v.reshape(v.shape[0], v.shape[2]).T), "vec")
            continue
        if re.match(r"decoder\.generator\.ups\.\d+\.weight$", k):
            out[k] = (np.ascontiguousarray(v.transpose(1, 2, 0)), "convt")
            continue
        if v.ndim == 3:
            out[k] = (_conv_kmajor(v, strided.get(k, 1)), "conv")
            continue
        if v.ndim == 2 and k.endswith(".weight") and "embeddings" not in k and "embedding.weight" not in k:
            out[k] = (v, "linear")
            continue
        out[k] = (v, "vec" if v.ndim == 1 else "embd")
    for base, g in lstm_groups.items():
        out[base + ".w_ih"] = (np.stack([g["weight_ih"], g["weight_ih_reverse"]]), "linear")
        out[base + ".w_hh"] = (np.stack([g["weight_hh"], g["weight_hh_reverse"]]), "linear")
        out[base + ".b"] = (np.stack([g["bias_ih"] + g["bias_hh"], g["bias_ih_reverse"] + g["bias_hh_reverse"]]).astype(np.float32), "vec")
    return out


def _pad_last(a, mult):
    n = a.shape[-1]
    p = (-n) % mult
    if p == 0:
        return a
    pad = [(0, 0)] * (a.ndim - 1) + [(0, p)]
    return np.pad(a, pad)


def _encode(name, arr, kind, outtype, f16_prefixes=()):
    """-> (data, raw_dtype or None, raw_shape or None)."""
    big = arr.size >= 4096
    if outtype == "q8_0" and any(name.startswith(p) for p in f16_prefixes):
        outtype = "f16"
    if outtype == "f32" or kind in ("vec", "embd") or not big:
        return arr.astype(np.float32), None, None
    if outtype == "f16":
        return arr.astype(np.float16), None, None
    # q8_0: rows along the last numpy axis (ggml ne0) in blocks of 32
    if kind == "conv":
        arr = _pad_last(arr, QK)          # pad input channels; the C++ side pads activations to the weight width
    if arr.shape[-1] % QK != 0:
        return arr.astype(np.float16), None, None
    q = gguf.quants.quantize(arr.astype(np.float32), gguf.GGMLQuantizationType.Q8_0)
    return q, gguf.GGMLQuantizationType.Q8_0, None


def _resolve(repo, cache_dir=None, model=None, revision=None, token=None):
    if os.path.isdir(repo):
        return Path(repo)
    from huggingface_hub import snapshot_download
    cfg_dir = Path(snapshot_download(repo_id=repo, cache_dir=cache_dir, allow_patterns=["config.json"], revision=revision,
                                     token=token))
    cfg = json.loads((cfg_dir / "config.json").read_text())
    files = ["config.json", cfg.get("vocab", "vocab.json")]
    for name, m in cfg["models"].items():
        if model and name != model:
            continue
        files.append(m["safetensors"])
        for lang in m["languages"]:
            files.append(cfg["languages"][lang]["voices"])
    return Path(snapshot_download(repo_id=repo, cache_dir=cache_dir, allow_patterns=sorted(set(files)), revision=revision,
                                  token=token))


def convert(repo, outdir, outtype="f32", model=None, cache_dir=None, name_template="{model}-{outtype}.gguf", f16_prefixes=(),
            revision=None, token=None):
    """Write one GGUF per model of the repository; returns their paths. Each file is written under a temporary name
    and renamed when complete, so an interrupted conversion never leaves a truncated GGUF behind."""
    src = _resolve(repo, cache_dir, model, revision, token)
    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("type") != "KITTEN2_TINY":
        raise ValueError(f"config type {cfg.get('type')!r} is not KITTEN2_TINY")
    needed = [cfg.get("vocab", "vocab.json")]
    for mname, m in cfg["models"].items():
        if not model or mname == model:
            needed += [m["safetensors"]] + [cfg["languages"][lang]["voices"] for lang in m["languages"]]
    missing = sorted({f for f in needed if not (src / f).is_file()})
    if missing:
        raise FileNotFoundError(f"{src} is missing {', '.join(missing)}. Download the whole model repository "
                                f"(every voices_<language>.npz file is needed), or pass the Hugging Face repo id instead.")
    vocab_text = (src / cfg.get("vocab", "vocab.json")).read_text(encoding="utf-8")
    os.makedirs(outdir, exist_ok=True)
    written = []
    for mname, m in cfg["models"].items():
        if model and mname != model:
            continue
        arch = m["architecture"]
        sd = _load_safetensors(src / m["safetensors"])
        tensors = convert_tensors(sd, arch)
        path = Path(outdir) / name_template.format(model=mname, outtype=outtype)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        w = gguf.GGUFWriter(str(tmp), ARCH)
        w.add_name(f"{cfg.get('name', mname)} ({mname})")
        w.add_string("kitten.model", mname)
        w.add_string("kitten.family", cfg.get("family", ""))
        w.add_string("kitten.phonemizer", m.get("phonemizer", cfg.get("phonemizer", "v1")))
        w.add_string("kitten.architecture", json.dumps(arch))
        w.add_string("kitten.config", json.dumps(cfg, ensure_ascii=False))
        w.add_string("kitten.vocab", vocab_text)
        w.add_array("kitten.languages", list(m["languages"]))
        w.add_uint32("kitten.sample_rate", int(cfg.get("sample_rate", 24000)))
        w.add_string("kitten.source_repo", str(repo))
        w.add_string("kitten.weight_type", outtype if not f16_prefixes else f"{outtype} (f16: {','.join(f16_prefixes)})")
        dec = arch["decoder"]
        w.add_string("kitten.magnitude_activation", "softplus" if dec["type"] == "istftnet" else "exp")
        if m.get("source"):
            w.add_string("kitten.source", json.dumps(m["source"]))
        for name in sorted(tensors):
            if len(name.encode()) >= 64:
                raise ValueError(f"tensor name too long for ggml: {name}")
            arr, kind = tensors[name]
            data, raw_dtype, raw_shape = _encode(name, arr, kind, outtype, f16_prefixes)
            if raw_dtype is not None:
                w.add_tensor(name, data, raw_shape=raw_shape, raw_dtype=raw_dtype)
            else:
                w.add_tensor(name, data)
        voices_meta = {}
        for lang in m["languages"]:
            vfile = src / cfg["languages"][lang]["voices"]
            with np.load(vfile) as z:
                for key in z.files:
                    pack = z[key].astype(np.float32).reshape(-1, z[key].shape[-1])
                    w.add_tensor(f"voice.{lang}.{key}", np.ascontiguousarray(pack))
            voices_meta[lang] = [v["key"] for v in cfg["voices"].get(lang, [])]
        w.add_string("kitten.voices", json.dumps(voices_meta))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        os.replace(tmp, path)
        written.append(str(path))
        print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    if not written:
        raise ValueError(f"no model named {model!r}; models: {list(cfg['models'])}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="Hugging Face repo id (KittenML/kitten-tts-2-nano) or local directory")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--outtype", choices=["f32", "f16", "q8_0"], default="f32")
    ap.add_argument("--model", help="one model of config.json['models'] (default: all)")
    ap.add_argument("--cache-dir")
    ap.add_argument("--f16-prefixes", default="", help="with q8_0: comma-separated tensor name prefixes kept in f16 (e.g. decoder.generator.)")
    ap.add_argument("--name-template", default="{model}-{outtype}.gguf")
    ap.add_argument("--revision", help="Hugging Face revision (branch, tag or commit)")
    a = ap.parse_args()
    pre = tuple(p for p in a.f16_prefixes.split(",") if p)
    try:
        convert(a.repo, a.outdir, a.outtype, a.model, a.cache_dir, a.name_template, pre, revision=a.revision)
    except ValueError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
