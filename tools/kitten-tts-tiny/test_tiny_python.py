#!/usr/bin/env python3
"""Functional test of kitten_tts_cpp.KittenTTS2Tiny against kittenml's KittenTTS2Tiny (same front end and voices).

    KITTEN_TTS_TINY_LIB=build/bin/libkitten_tts_tiny.so python test_tiny_python.py [REPO_OR_DIR ...] [--device cpu]

For each repository: the GGUF is converted on first use (f32), a few sentences are spoken in each tested language,
and the output is checked against kittenml's PyTorch model run with the same phonemes: same chunking and length
(durations are deterministic), finite, not silent, sample rate 24 kHz. Needs kittenml==0.9.8 for the comparison.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

TEXTS = {
    "en": "Hello there! This is Kitten TTS running on ggml. It reads two sentences.",
    "de": "Guten Morgen, wie geht es dir heute?",
    "fr": "Bonjour, comment allez-vous aujourd'hui ?",
    "es": "Buenos días, ¿cómo estás hoy?",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*", default=["KittenML/kitten-tts-2-nano"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default="f32")
    ap.add_argument("--cache-dir")
    ap.add_argument("--gguf-repo", help="load GGUF files from this Hugging Face repository instead of converting")
    ap.add_argument("--hf-token-file", help="file holding a Hugging Face token (for a private GGUF repository)")
    a = ap.parse_args()
    token = open(a.hf_token_file).read().strip() if a.hf_token_file else None
    from kitten_tts_cpp import KittenTTS2Tiny
    try:
        from kittenml.kittentts2_tiny.model import KittenTTS2Tiny as Ref
    except ImportError:
        Ref = None
    failures = 0
    for repo in a.repos:
        t0 = time.time()
        m = KittenTTS2Tiny(repo, device=a.device, weights=a.weights, cache_dir=a.cache_dir, seed=0, gguf_repo=a.gguf_repo,
                           gguf_token=token)
        ref = Ref(repo, backend="torch", device="cpu") if Ref else None
        print(f"{repo}: languages {m.languages}, {sum(len(v) for v in m.available_voices.values())} voices")
        for lang, text in TEXTS.items():
            if lang not in m.languages:
                continue
            voice = m.voices(lang)[0]["name"]
            audio = m.generate(text, voice=voice, language=lang)
            ok = audio.dtype == np.float32 and np.isfinite(audio).all() and np.abs(audio).max() > 0.01 and m.sample_rate == 24000
            msg = f"  {lang} {voice}: {len(audio) / 24000:.2f} s, peak {np.abs(audio).max():.2f}, report {m.last_report.get('rtf', 0):.3f} RTF"
            if ref is not None:
                ra = ref.generate(text, voice=voice, language=lang)
                same_len = len(ra) == len(audio)
                ok = ok and same_len
                msg += f", PyTorch length {'equal' if same_len else f'differs ({len(ra)} vs {len(audio)})'}"
            print(("PASS" if ok else "FAIL") + msg)
            failures += not ok
        # bad input to the raw C API must come back as an exception, not abort the process
        lang0 = m.languages[0]
        tm = m._model(lang0)
        style = tm.style(lang0, m.voices(lang0)[0]["key"], 10)
        for bad, why in (([0, 5, 10**6, 0], "token id out of range"), ([0, -3, 0], "negative token id"), ([0], "one token")):
            try:
                tm.synthesize(np.array(bad, dtype=np.int32), style)
                print(f"FAIL  bad input accepted: {why}")
                failures += 1
            except (RuntimeError, ValueError) as e:
                print(f"PASS  bad input rejected ({why}): {e}")
        try:
            tm.synthesize(np.array([0, 5, 6, 0], dtype=np.int32), style, durations=[1, 2])
            print("FAIL  wrong durations length accepted")
            failures += 1
        except ValueError as e:
            print(f"PASS  wrong durations length rejected: {e}")
        for bad in (0, -1.0, float("nan"), float("inf"), "fast", 0.1, 5.0):
            try:
                m.generate("Hi.", language=lang0, speed=bad)
                print(f"FAIL  speed {bad!r} accepted")
                failures += 1
            except ValueError as e:
                print(f"PASS  speed {bad!r} rejected: {e}")
            if isinstance(bad, str):
                continue
            try:   # the C library checks the same range (0.25 to 4)
                tm.generate_chunk("a", lang0, m.voices(lang0)[0]["key"], speed=bad)
                print(f"FAIL  C API accepted speed {bad!r}")
                failures += 1
            except (RuntimeError, ValueError) as e:
                print(f"PASS  C API rejected speed {bad!r}: {e}")
        for bad in (1e-4, 0.2499, 4.001):
            try:
                tm.synthesize(np.array([0, 5, 6, 0], dtype=np.int32), style, speed=bad)
                print(f"FAIL  C API kt_tiny_synthesize accepted speed {bad!r}")
                failures += 1
            except (RuntimeError, ValueError) as e:
                print(f"PASS  C API kt_tiny_synthesize rejected speed {bad!r}: {e}")
        for good in (0.25, 4.0):
            ok = len(tm.generate_chunk("a", lang0, m.voices(lang0)[0]["key"], speed=good)) > 0
            print(("PASS" if ok else "FAIL") + f"  C API speed {good} accepted")
            failures += not ok
        # default threads: the physical cores this process may use (affinity mask), at most 8
        info = m.model_info(lang0)
        allowed = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        ok = 1 <= int(info.get("threads", 0)) <= min(8, allowed) and "library" in info
        print(("PASS" if ok else "FAIL") + f"  default threads {info.get('threads')} (CPUs allowed: {allowed}), library {info.get('library')}")
        failures += not ok
        try:
            m.generate("Hi.", voice="no-such-voice", language=lang0)
            print("FAIL  unknown voice accepted")
            failures += 1
        except ValueError as e:
            ok = m.voices(lang0)[0]["name"] in str(e)
            print(("PASS" if ok else "FAIL") + f"  unknown voice error lists the voices: {str(e)[:100]}")
            failures += not ok
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            try:
                m.generate_to_file("Hi.", os.path.join(td, "x.wav"), language=lang0, sample_rate=16000)
                print("FAIL  generate_to_file accepted sample_rate=16000 (the audio is 24 kHz)")
                failures += 1
            except ValueError as e:
                print(f"PASS  generate_to_file rejects another sample rate: {str(e)[:80]}")
            import soundfile as sf
            m.generate_to_file("Hi.", os.path.join(td, "y.wav"), language=lang0, sample_rate=24000)
            ok = sf.info(os.path.join(td, "y.wav")).samplerate == 24000
            print(("PASS" if ok else "FAIL") + "  generate_to_file writes 24 kHz")
            failures += not ok
        try:
            from kitten_tts_cpp import KittenTTS
        except ImportError as e:   # KittenTTS2 needs this package's LibTorch extension, which a library-only build lacks
            print(f"SKIP  KittenTTS name check ({e})")
        else:
            try:
                KittenTTS(repo)
                print("FAIL  KittenTTS accepted a mini / nano / pico repository")
                failures += 1
            except ValueError as e:
                ok = "KittenTTS2Tiny" in str(e)
                print(("PASS" if ok else "FAIL") + f"  KittenTTS points to KittenTTS2Tiny: {e}")
                failures += not ok
        if "de" in m.languages:   # the text front end is shared by the process: several threads must get the same phonemes
            import concurrent.futures as cf
            sents = ["Guten Morgen, wie geht es dir heute?", "Das Wetter ist schön, wir gehen im Park spazieren.",
                     "Ich habe gestern drei Bücher gelesen.", "Kannst du mir bitte beim Kochen helfen?"]
            want = {t: m.phonemize(t, "de") for t in sents}
            with cf.ThreadPoolExecutor(6) as ex:
                got = list(ex.map(lambda t: (t, m.phonemize(t, "de")), sents * 25))
            bad = sum(g != want[t] for t, g in got)
            print(("PASS" if bad == 0 else "FAIL") + f"  German front end from 6 threads: {bad} of {len(got)} differ from one thread")
            failures += bad != 0
        with KittenTTS2Tiny(repo, device=a.device, weights=a.weights, cache_dir=a.cache_dir, seed=0, gguf_repo=a.gguf_repo,
                            gguf_token=token) as cm:
            ok = len(cm.generate(TEXTS[lang0], language=lang0)) > 0
        ok = ok and not cm._models
        print(("PASS" if ok else "FAIL") + "  context manager generates and closes")
        failures += not ok
        stream = list(m.generate_stream(TEXTS["en"] if "en" in m.languages else TEXTS[m.languages[0]]))
        print(f"  stream: {len(stream)} chunk(s); first load + tests {time.time() - t0:.1f} s")
        m.close()
    print("FAILURES", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
