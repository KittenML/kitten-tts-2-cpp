#!/usr/bin/env python3
"""Compare the ggml model (GGUF) with kittenml's PyTorch fp32 model on the same phonemes, voice and noise.

    python parity_kitten_tiny.py --repo DIR_OR_REPO --gguf a-f32.gguf [b-q8_0.gguf ...] --texts texts.json --out res.json

texts.json: [{"lang": "en", "text": "...", "voice": "eleanor"}, ...] (voice optional).
For each text the first phoneme chunk is synthesized by PyTorch with recorded Gaussian noise (seeded), then by
the GGUF model with the same tokens, style row, PyTorch's durations and the same noise. Reported per GGUF:
duration agreement of its own prediction, SNR (global and per clip) and log-mel mean absolute difference (dB).
Needs kittenml==0.9.8 (torch backend), librosa, numpy, and libkitten_tts_tiny (KITTEN_TTS_TINY_LIB).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def torch_run(tm, tokens, style, speed, gen, record):
    """kittenml TorchModel.__call__ with the SineGen noise drawn from `gen` (recorded) or zero."""
    import torch
    orig = torch.randn_like

    def fake(x, *a, **k):
        if x.dim() == 3 and x.shape[-1] == 9:
            z = torch.randn(x.shape, generator=gen, dtype=torch.float32) if gen is not None else torch.zeros(x.shape)
            record["noise"] = z[0].numpy().copy()
            return z.to(x.device)
        return orig(x, *a, **k)

    torch.randn_like = fake
    try:
        wav, dur = tm(tokens, style, speed)
    finally:
        torch.randn_like = orig
    return wav, dur


def logmel_db(x, sr=24000):
    import librosa
    m = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=sr, n_fft=1024, hop_length=256, n_mels=80, power=2.0)
    return 10 * np.log10(np.maximum(m, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav-dir")
    ap.add_argument("--dequantize", type=int, default=1, help="1: f16 / q8_0 weights converted to f32 at load (default)")
    ap.add_argument("--allow-tf32", action="store_true", help="GPUs: allow TF32 / fp16 matrix products (lossy)")
    ap.add_argument("--floors", action="store_true",
                    help="also measure PyTorch against itself (1 thread vs --threads, same noise; and another noise seed) "
                         "and the official int8 ONNX model against PyTorch")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    from kittenml.kittentts2_tiny.model import KittenTTS2Tiny as Ref
    from kitten_tts_cpp.tiny import TinyModel

    ref = Ref(a.repo, backend="torch", device="cpu", threads=a.threads)
    texts = json.loads(Path(a.texts).read_text())
    models = [(p, TinyModel(p, a.device, a.threads, bool(a.dequantize), a.allow_tf32)) for p in a.gguf]
    rows = []
    for i, t in enumerate(texts):
        lang, key = ref.resolve_voice(t.get("voice"), t.get("lang"))
        if models[0][1].info["model"] != ref.config["languages"][lang]["model"]:
            continue
        ps = ref.phonemize(t["text"], lang)[0]
        tok = ref._tokens(ps)
        pack = ref._pack(lang, key)
        style = pack[min(len(tok) - 1, pack.shape[0] - 1)][None].astype(np.float32)
        tm = ref._torch_model(lang)
        gen = torch.Generator().manual_seed(a.seed + i)
        rec = {}
        t0 = time.time()
        wav_t, dur_t = torch_run(tm, tok, style, a.speed, gen, rec)
        torch_s = time.time() - t0
        row = {"i": i, "lang": lang, "voice": key, "n_tokens": len(tok), "frames": int(dur_t.sum()),
               "audio_s": len(wav_t) / 24000, "torch_s": torch_s, "models": {}}

        def compare(wav_x):
            n = min(len(wav_t), len(wav_x))
            r, g = wav_t[:n].astype(np.float64), wav_x[:n].astype(np.float64)
            err, sig = float(((r - g) ** 2).sum()), float((r ** 2).sum())
            return {"len_equal": len(wav_t) == len(wav_x), "snr_db": 10 * np.log10(sig / max(err, 1e-30)), "sig": sig, "err": err,
                    "logmel_mad_db": float(np.mean(np.abs(logmel_db(wav_t[:n]) - logmel_db(wav_x[:n])))),
                    "max_abs_diff": float(np.abs(r - g).max())}
        if a.floors:
            torch.set_num_threads(1)
            rec1 = {}
            w1, d1 = torch_run(tm, tok, style, a.speed, torch.Generator().manual_seed(a.seed + i), rec1)
            torch.set_num_threads(a.threads)
            row["floor_threads"] = dict(compare(w1), dur_equal=bool(np.array_equal(d1, dur_t)))
            w2, _ = torch_run(tm, tok, style, a.speed, torch.Generator().manual_seed(a.seed + 10_000 + i), {})
            row["floor_noise"] = compare(w2)
            wo, do = ref._session(lang).run(None, {"input_ids": np.array([tok], dtype=np.int64), "style": style,
                                                  "speed": np.array([a.speed], dtype=np.float32)})
            row["onnx_int8"] = dict(compare(wo.reshape(-1)), dur_equal=bool(np.array_equal(np.asarray(do).reshape(-1), dur_t)))
        for path, gm in models:
            name = Path(path).name
            _, dur_g = gm.synthesize(tok, style.reshape(-1), a.speed, noise="none")
            t0 = time.time()
            wav_g, _ = gm.synthesize(tok, style.reshape(-1), a.speed, noise=rec["noise"], durations=dur_t.astype(np.int32))
            g_s = time.time() - t0
            row["models"][name] = dict(compare(wav_g), dur_equal=bool(np.array_equal(dur_g, dur_t)),
                                       dur_mismatch_tokens=int((dur_g != dur_t).sum()), ggml_s=g_s, report=gm.last_report())
            if a.wav_dir:
                import soundfile as sf
                os.makedirs(a.wav_dir, exist_ok=True)
                sf.write(f"{a.wav_dir}/{i:03d}_{lang}_{key}_torch.wav", wav_t, 24000, subtype="FLOAT")
                sf.write(f"{a.wav_dir}/{i:03d}_{lang}_{key}_{name}.wav", wav_g, 24000, subtype="FLOAT")
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "models"}),
              {k: (round(v["snr_db"], 1), round(v["logmel_mad_db"], 3), v["dur_equal"]) for k, v in row["models"].items()}, flush=True)
    summary = {}
    keys = [Path(p).name for p, _ in models] + (["floor_threads", "floor_noise", "onnx_int8"] if a.floors else [])
    for name in keys:
        ms = [r["models"][name] if name in r["models"] else r[name] for r in rows]
        if not ms:
            continue
        sig = sum(m["sig"] for m in ms)
        err = sum(m["err"] for m in ms)
        summary[name] = {"clips": len(ms), "snr_global_db": 10 * np.log10(sig / max(err, 1e-30)),
                         "snr_median_db": float(np.median([m["snr_db"] for m in ms])),
                         "snr_min_db": float(np.min([m["snr_db"] for m in ms])),
                         "logmel_mad_db_mean": float(np.mean([m["logmel_mad_db"] for m in ms])),
                         "logmel_mad_db_max": float(np.max([m["logmel_mad_db"] for m in ms])),
                         "durations_equal": sum(m.get("dur_equal", False) for m in ms),
                         "duration_tokens_differing": sum(m.get("dur_mismatch_tokens", 0) for m in ms)}
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
