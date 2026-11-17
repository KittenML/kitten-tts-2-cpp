"""Kitten TTS 2 mini / nano / pico (config type KITTEN2_TINY) on the ggml runtime.

    from kitten_tts_cpp import KittenTTS2Tiny
    m = KittenTTS2Tiny("KittenML/kitten-tts-2-nano")            # CPU or the first GPU ggml finds
    audio = m.generate("Hello there.", voice="Eleanor")          # float32 at m.sample_rate (24 kHz)
    audio = m.generate("Guten Morgen.", language="de")
    m.voices("de")

The weights come from the official repository's fp32 safetensors, converted to GGUF on first use and cached
(no other download is needed), or from a GGUF repository / folder (gguf_repo=, gguf_dir=).
weights: "f32" (matches the PyTorch model), "f16" or "q8_0" (smaller download, lossy).
Text goes through kittenml's own front end (vendored in tiny_frontend): misaki's lexicon, CMUdict and a small BART
model (run in numpy) for English, misaki's Japanese reader with fugashi for Japanese (both Python 3.10 to 3.14), and
espeak-ng (phonemizer, espeakng_loader) for the other languages. The package installs all of them.
The native library is libkitten_tts_tiny (built with -DLLAMA_BUILD_KITTEN_TTS_TINY=ON); set
KITTEN_TTS_TINY_LIB to its path if it is not next to this file.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000
GAP_S = 0.15
END_TOKEN = 10
NOISE = {"random": 0, "none": 1, "external": 2}


class _Params(ctypes.Structure):
    _fields_ = [("device", ctypes.c_char_p), ("n_threads", ctypes.c_int), ("dequantize", ctypes.c_int),
                ("allow_tf32", ctypes.c_int)]


class _Options(ctypes.Structure):
    _fields_ = [("speed", ctypes.c_float), ("seed", ctypes.c_uint64), ("noise_mode", ctypes.c_int),
                ("noise", ctypes.POINTER(ctypes.c_float)), ("noise_len", ctypes.c_size_t),
                ("durations", ctypes.POINTER(ctypes.c_int32))]


_libs = {}  # "main" or "x86_64_v3" -> (ctypes library, path)
_lib_lock = threading.Lock()


def _x86_64_v3():
    """True on an x86-64 CPU with AVX2, FMA, F16C and BMI2 (what the x86-64-v3 build of the library needs)."""
    import platform
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="ascii", errors="replace") as f:
                for line in f:
                    if line.startswith("flags"):
                        flags = set(line.split(":", 1)[1].split())
                        return {"avx2", "fma", "f16c", "bmi2"} <= flags
            return False
        if sys.platform == "win32":
            k = ctypes.windll.kernel32
            # PF_AVX2_INSTRUCTIONS_AVAILABLE (40); every CPU with AVX2 also has FMA, F16C and BMI2
            return bool(k.IsProcessorFeaturePresent(40))
    except (OSError, AttributeError):
        return False
    return False


def _cuda_wheel(here):
    try:
        return bool(json.loads((here / "_build_info.json").read_text()).get("cuda", False))
    except (OSError, ValueError):
        return False


def _library(device="auto"):
    """The tiny library for `device`, loaded once per build, and its path.

    x86-64 wheels carry two builds: the main one (portable; in the CUDA wheel also the GPU code) and an x86-64-v3 one
    (AVX2 / FMA / F16C / BMI2, CPU only, several times faster on the CPU). The CPU wheel uses the x86-64-v3 build on
    every CPU that has those extensions. The CUDA wheel uses it for device="cpu", and for device="auto" when PyTorch
    finds no CUDA GPU; GPU devices use the main build.
    """
    with _lib_lock:
        here = Path(__file__).resolve().parent
        cuda_wheel = _cuda_wheel(here)
        if cuda_wheel:
            # Load torch's CUDA libraries (cuBLAS, nvJitLink, ...) first, as kitten_tts_cpp.model does: otherwise this
            # library can pick up older copies from LD_LIBRARY_PATH (a CUDA toolkit) and a later `import torch` fails.
            try:
                import torch  # noqa: F401
            except ImportError:
                torch = None
        dev = str(device or "auto").lower()
        if cuda_wheel:
            cpu = dev == "cpu" or (dev == "auto" and not (torch is not None and torch.cuda.is_available()))
        else:
            cpu = True
        kind = "x86_64_v3" if cpu and _x86_64_v3() else "main"
        if kind in _libs:
            return _libs[kind]
        names = {"win32": ["kitten_tts_tiny.dll"], "darwin": ["libkitten_tts_tiny.dylib"]}.get(sys.platform, ["libkitten_tts_tiny.so"])
        if kind == "x86_64_v3":
            names = [n.replace("kitten_tts_tiny", "kitten_tts_tiny_x86_64_v3") for n in names] + names
        cands = [os.environ["KITTEN_TTS_TINY_LIB"]] if os.environ.get("KITTEN_TTS_TINY_LIB") else []
        cands += [str(here / n) for n in names] + [str(here / "lib" / n) for n in names]
        err = None
        for c in cands:
            if os.path.exists(c):
                try:
                    lib = ctypes.CDLL(c)
                    path = c
                    break
                except OSError as e:
                    err = e
        else:
            raise ImportError(f"libkitten_tts_tiny not found (tried {cands}); build it with "
                              f"-DLLAMA_BUILD_KITTEN_TTS_TINY=ON or set KITTEN_TTS_TINY_LIB" + (f": {err}" if err else ""))
        P = ctypes.POINTER
        lib.kt_tiny_load.argtypes = [ctypes.c_char_p, P(_Params)]
        lib.kt_tiny_load.restype = ctypes.c_void_p
        lib.kt_tiny_free.argtypes = [ctypes.c_void_p]
        lib.kt_tiny_last_error.restype = ctypes.c_char_p
        lib.kt_tiny_info_json.argtypes = [ctypes.c_void_p]
        lib.kt_tiny_info_json.restype = ctypes.c_char_p
        lib.kt_tiny_last_report.argtypes = [ctypes.c_void_p]
        lib.kt_tiny_last_report.restype = ctypes.c_char_p
        lib.kt_tiny_voice_style.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, P(ctypes.c_float)]
        lib.kt_tiny_synthesize.argtypes = [ctypes.c_void_p, P(ctypes.c_int32), ctypes.c_int, P(ctypes.c_float), P(_Options),
                                           P(P(ctypes.c_float)), P(ctypes.c_size_t), P(P(ctypes.c_int32))]
        lib.kt_tiny_generate_chunk.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float,
                                               ctypes.c_int, ctypes.c_uint64, P(P(ctypes.c_float)), P(ctypes.c_size_t)]
        lib.kt_tiny_free_buffer.argtypes = [ctypes.c_void_p]
        _libs[kind] = (lib, path)
        return _libs[kind]


def _error(lib):
    return lib.kt_tiny_last_error().decode("utf-8", "replace")


class TinyModel:
    """One GGUF model (low level): token ids / phonemes in, waveform out."""

    def __init__(self, path, device="auto", threads=0, dequantize=True, allow_tf32=False):
        lib, lib_path = _library(device)
        self._lib = lib
        p = _Params(str(device).encode(), int(threads or 0), int(bool(dequantize)), int(bool(allow_tf32)))
        self._h = lib.kt_tiny_load(os.fspath(path).encode(), ctypes.byref(p))
        if not self._h:
            raise RuntimeError(f"cannot load {path}: {_error(lib)}")
        self.info = json.loads(lib.kt_tiny_info_json(self._h).decode("utf-8"))
        self.info["library"] = os.path.basename(lib_path)
        self._lock = threading.Lock()
        self._report = ""

    def close(self):
        if getattr(self, "_h", None):
            self._lib.kt_tiny_free(self._h)
            self._h = None

    __del__ = close

    def last_report(self):
        """Timings (ms) and buffer sizes of this model's last call (copied while that call held the model)."""
        with self._lock:
            r = self._report
        return json.loads(r) if r else {}

    def style(self, language, voice, n_tokens):
        out = (ctypes.c_float * 256)()
        if self._lib.kt_tiny_voice_style(self._h, language.encode(), (voice or "").encode(), int(n_tokens), out) != 0:
            raise ValueError(_error(self._lib))
        return np.frombuffer(out, dtype=np.float32).copy()

    def synthesize(self, tokens, style, speed=1.0, seed=0, noise="random", durations=None):
        """Raw model call -> (waveform float32, durations int32). noise: "random", "none" or an array [n_samples, 9]."""
        lib = self._lib
        tok = np.ascontiguousarray(tokens, dtype=np.int32)
        st = np.ascontiguousarray(style, dtype=np.float32).reshape(-1)
        if st.size != 256:
            raise ValueError("style must have 256 values")
        o = _Options(float(speed), int(seed), 0, None, 0, None)
        keep = []
        if isinstance(noise, str):
            o.noise_mode = NOISE[noise]
        else:
            z = np.ascontiguousarray(noise, dtype=np.float32).reshape(-1)
            keep.append(z)
            o.noise_mode, o.noise, o.noise_len = 2, z.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), z.size
        if durations is not None:
            d = np.ascontiguousarray(durations, dtype=np.int32).reshape(-1)
            if d.size != tok.size:
                raise ValueError(f"durations must have one value per token ({tok.size}), got {d.size}")
            keep.append(d)
            o.durations = d.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        audio = ctypes.POINTER(ctypes.c_float)()
        dur = ctypes.POINTER(ctypes.c_int32)()
        n = ctypes.c_size_t()
        with self._lock:
            rc = lib.kt_tiny_synthesize(self._h, tok.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), tok.size,
                                        st.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.byref(o),
                                        ctypes.byref(audio), ctypes.byref(n), ctypes.byref(dur))
            if rc == 0:
                self._report = (lib.kt_tiny_last_report(self._h) or b"").decode("utf-8", "replace")
        if rc != 0:
            raise RuntimeError(_error(lib))
        try:
            wav = np.ctypeslib.as_array(audio, shape=(n.value,)).copy() if n.value else np.zeros(0, np.float32)
            dd = np.ctypeslib.as_array(dur, shape=(tok.size,)).copy()
        finally:
            lib.kt_tiny_free_buffer(ctypes.cast(audio, ctypes.c_void_p))
            lib.kt_tiny_free_buffer(ctypes.cast(dur, ctypes.c_void_p))
        return wav, dd

    def generate_chunk(self, phonemes, language, voice, speed=1.0, keep_start=True, seed=0):
        lib = self._lib
        audio = ctypes.POINTER(ctypes.c_float)()
        n = ctypes.c_size_t()
        with self._lock:
            rc = lib.kt_tiny_generate_chunk(self._h, phonemes.encode("utf-8"), language.encode(), (voice or "").encode(),
                                            float(speed), int(bool(keep_start)), int(seed), ctypes.byref(audio), ctypes.byref(n))
            if rc == 0:   # the library's report string changes with the model's next call: copy it while holding the model
                self._report = (lib.kt_tiny_last_report(self._h) or b"").decode("utf-8", "replace")
        if rc != 0:
            raise RuntimeError(_error(lib))
        try:
            return np.ctypeslib.as_array(audio, shape=(n.value,)).copy() if n.value else np.zeros(0, np.float32)
        finally:
            lib.kt_tiny_free_buffer(ctypes.cast(audio, ctypes.c_void_p))


SPEED_MIN, SPEED_MAX = 0.25, 4.0   # the speed range of generate(); the C library checks the same range
# Official repositories load this exact revision of their files (the converter was checked against it), as in
# kitten-inference; pass revision= for another one.
REVISIONS = {
    "KittenML/kitten-tts-2-mini": "c01797caec3967b400da457656e011eb81f530a4",
    "KittenML/kitten-tts-2-nano": "54526b8d8e45333cd720d3a082c106aacf416064",
    "KittenML/kitten-tts-2-pico": "da8c9d029e16d9302f1827adb303d92316e25f73",
}
# The English (spaCy, regex, addict) and Japanese (fugashi, jaconv, addict) text front end: the package installs these on
# CPython 3.10 to 3.14 (their markers in setup.py; fugashi not on Windows on Arm, which has no fugashi package).
_FRONT_END_PYTHONS = ((3, 10), (3, 14))
_FRONT_END_MODULES = ("spacy", "regex", "addict", "fugashi", "jaconv")


def _front_end_unavailable(missing, lang="en"):
    """The error text when a package of the English or Japanese front end is missing."""
    import platform
    major, minor = sys.version_info[:2]
    language = {"en": "English", "ja": "Japanese"}.get(lang, repr(lang))
    if not _FRONT_END_PYTHONS[0] <= (major, minor) <= _FRONT_END_PYTHONS[1]:
        return (f"{language} text with Kitten TTS 2 mini, nano and pico is not supported on Python {major}.{minor}: "
                "its text front end (spaCy) installs only on Python 3.10 to 3.14. "
                "The other languages work on this Python.")
    if missing.split(".")[0] == "fugashi" and platform.system() == "Windows" and platform.machine().upper() == "ARM64":
        return ("Japanese text with Kitten TTS 2 mini is not supported on Windows on Arm: its Japanese tokenizer "
                "(fugashi) has no package for it. The other languages work here.")
    try:
        from importlib.metadata import packages_distributions
        dist = (packages_distributions().get("kitten_tts_cpp") or ["kitten-tts-cpp"])[0]
    except Exception:
        dist = "kitten-tts-cpp"
    return (f"{language} text with Kitten TTS 2 mini, nano and pico needs {missing}, which {dist} installs on "
            f"Python {major}.{minor} ({platform.system()} {platform.machine()}), but it is missing: "
            f"pip install --force-reinstall {dist}")


def _cache_root(cache_dir):
    if cache_dir:
        return Path(cache_dir)
    base = os.environ.get("KITTEN_TTS_CPP_CACHE") or os.path.join(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"), "kitten-tts-cpp")
    return Path(base) / "tiny"


class KittenTTS2Tiny:
    """A Kitten TTS 2 nano / pico / mini repository on the ggml runtime (same API as kittenml's KittenTTS2Tiny).

    device: "auto" (first GPU ggml finds, else CPU), "cpu", "cuda", "metal", "vulkan". On Metal the text and prosody stages
    run on the CPU and the decoder on the GPU.
    weights: "f32" (default, matches PyTorch fp32 to rounding), "f16", "q8_0" or "q8_0-genf16" (lossy; q8_0-genf16 keeps the
    waveform generator in f16). gguf_repo / gguf_dir: read <model>-<weights>.gguf from there instead of converting.
    allow_tf32 (default False): on GPUs, allow TF32 (CUDA) or fp16 (Vulkan, Metal) matrix products: faster on some GPUs, lossy.
    dequantize (default True): f16 / q8_0 weights are converted to f32 when loading, so only the weights are rounded
    (smaller download, f32 arithmetic); False computes in the file's type (less memory, also rounds activations).
    threads: CPU threads (default: the physical cores this process may use, at most 8).
    seed: noise seed of the harmonic source. None (default) draws a new seed for every call, so two calls give slightly
    different audio, as the official model does; an int gives the same audio for the same input, device and build.
    cache_dir: where downloads and converted GGUF files go (default: Hugging Face's cache for downloads and
    ~/.cache/kitten-tts-cpp/tiny, or $KITTEN_TTS_CPP_CACHE/tiny, for GGUF files). revision: the repository revision
    (default: the tested revision for the official repositories, else "main"); converted files are cached per commit.
    hf_token: a Hugging Face token, for a private repository.
    """

    def __init__(self, model_name="KittenML/kitten-tts-2-nano", cache_dir=None, device="auto", hf_token=None, *,
                 weights="f32", threads=None, gguf_repo=None, gguf_dir=None, revision=None, seed=None, gguf_token=None,
                 dequantize=True, allow_tf32=False):
        if weights not in ("f32", "f16", "q8_0", "q8_0-genf16"):
            raise ValueError('weights must be "f32", "f16", "q8_0" or "q8_0-genf16"')
        self.model_name, self.cache_dir, self.hf_token, self.revision = model_name, cache_dir, hf_token, revision
        self.weights, self.device, self.threads = weights, device or "auto", threads or 0
        self.gguf_repo, self.gguf_dir, self.gguf_token = gguf_repo, gguf_dir, gguf_token
        self.dequantize = dequantize
        self.allow_tf32 = allow_tf32
        self._local = os.path.isdir(model_name)
        self.repo_id = None if self._local else (model_name if "/" in model_name else f"KittenML/{model_name}")
        if not self._local and revision is None:
            self.revision = REVISIONS.get(self.repo_id)
        config_path = self._fetch("config.json")
        self._commit = None
        if not self._local:
            # the commit the revision resolved to (a branch such as "main" can move): it keys the converted files
            parts = os.path.normpath(config_path).split(os.sep)
            if "snapshots" in parts and parts.index("snapshots") + 1 < len(parts):
                self._commit = parts[parts.index("snapshots") + 1]
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if self.config.get("type") != "KITTEN2_TINY":
            raise ValueError(f"{model_name} is not a Kitten TTS 2 mini / nano / pico repository (type {self.config.get('type')!r})")
        self.vocab = json.loads(Path(self._fetch(self.config.get("vocab", "vocab.json"))).read_text(encoding="utf-8"))
        self.sample_rate = int(self.config.get("sample_rate", SAMPLE_RATE))
        self.languages = list(self.config["languages"])
        self.default_language = self.config.get("default_language", self.languages[0])
        self._voices = {lang: list(vs) for lang, vs in self.config["voices"].items()}
        self._models = {}
        self._frontend = None
        self._seed = seed
        self._calls = 0
        self.last_report = None

    # ---- files ------------------------------------------------------------------------------------
    def _fetch(self, filename):
        if self._local:
            return os.path.join(self.model_name, filename)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=self.repo_id, filename=filename, cache_dir=self.cache_dir, token=self.hf_token,
                               revision=self.revision)

    def _g2p_dir(self):
        d = self.config.get("g2p_dir", "g2p")
        if self._local:
            return os.path.join(self.model_name, d)
        from huggingface_hub import snapshot_download
        root = snapshot_download(repo_id=self.repo_id, cache_dir=self.cache_dir, token=self.hf_token, revision=self.revision,
                                 allow_patterns=[f"{d}/*", f"{d}/**"])
        return os.path.join(root, d)

    def _gguf_path(self, model):
        fname = f"{model}-{self.weights}.gguf"
        if self.gguf_dir:
            return os.path.join(self.gguf_dir, fname)
        if self.gguf_repo:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=self.gguf_repo, filename=fname, cache_dir=self.cache_dir,
                                   token=self.gguf_token or self.hf_token)
        slug = (self.repo_id or Path(self.model_name).resolve().name).replace("/", "--")
        rev = self._commit or self.revision
        if rev and not self._local:
            slug += "@" + re.sub(r"[^A-Za-z0-9._-]", "_", str(rev))   # one cache entry per commit of the repository
        out = _cache_root(self.cache_dir) / slug
        path = out / fname
        if not path.exists():
            from .tiny_convert import convert
            outtype, pre = (("q8_0", ("decoder.generator.",)) if self.weights == "q8_0-genf16" else (self.weights, ()))
            # written to a temporary file and renamed, so a cancelled conversion is redone next time
            convert(self.model_name if self._local else self.repo_id, str(out), outtype, model=model,
                    cache_dir=self.cache_dir, name_template="{model}-" + self.weights + ".gguf", f16_prefixes=pre,
                    revision=None if self._local else (self._commit or self.revision), token=self.hf_token)
        return str(path)

    def _model(self, lang):
        name = self.config["languages"][lang]["model"]
        if name not in self._models:
            self._models[name] = TinyModel(self._gguf_path(name), self.device, self.threads, self.dequantize, self.allow_tf32)
        return self._models[name]

    @property
    def frontend(self):
        if self._frontend is None:
            from .tiny_frontend import Frontend
            self._frontend = Frontend(self.config.get("phonemizer", "v1"), self.vocab, self._g2p_dir())
        return self._frontend

    # ---- voices -----------------------------------------------------------------------------------
    @property
    def available_voices(self):
        return {lang: [v["name"] for v in vs] for lang, vs in self._voices.items()}

    def voices(self, language=None):
        langs = [self._language(language)] if language else self.languages
        return [dict(v, language=lang) for lang in langs for v in self._voices[lang]]

    def _language(self, language):
        if language is None:
            return self.default_language
        if language not in self.config["languages"]:
            raise ValueError(f"{self.config.get('name', 'this model')} does not speak {language!r}; languages: {self.languages}")
        return language

    def resolve_voice(self, voice=None, language=None):
        """-> (language, voice key), with kittenml's rules (name in any case, description, or the first voice)."""
        if voice is None:
            lang = self._language(language)
            return lang, self._voices[lang][0]["key"]
        langs = [self._language(language)] if language else self.languages
        want = str(voice).casefold()
        hits = [(lang, v["key"]) for lang in langs for v in self._voices[lang]
                if want in {str(v.get(f, "")).casefold() for f in ("key", "name", "description", "id") if f in v}]
        if not hits:
            raise ValueError(f"no voice {voice!r}" + (f" for {language!r}" if language else "") + "; voices: "
                             + str(self.available_voices if not language else self.available_voices[langs[0]]))
        if len(hits) > 1:
            if language is None and any(lang == self.default_language for lang, _ in hits):
                return next(h for h in hits if h[0] == self.default_language)
            raise ValueError(f"voice {voice!r} exists in {[lang for lang, _ in hits]}: pass language=")
        return hits[0]

    # ---- synthesis --------------------------------------------------------------------------------
    def _front(self, text, lang):
        try:
            return self.frontend(text, lang)[0]
        except ModuleNotFoundError as exc:
            name = (exc.name or "").split(".")[0]
            if name in _FRONT_END_MODULES:
                raise ImportError(_front_end_unavailable(exc.name, lang)) from exc
            raise

    def phonemize(self, text, language=None):
        return self._front(text, self._language(language))

    def generate_stream(self, text, voice=None, language=None, speed=1.0):
        try:
            ok = SPEED_MIN <= float(speed) <= SPEED_MAX
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"speed must be a number from {SPEED_MIN} to {SPEED_MAX} (1.0 is normal; higher is faster), got {speed!r}")
        speed = float(speed)
        lang, key = self.resolve_voice(voice, language)
        # text first: if the front end imports torch (directly or through a dependency), torch's CUDA libraries must load before this library's
        # (a CUDA build of ggml linked against another NCCL otherwise breaks `import torch` in the same process)
        chunks = self._front(text, lang)
        model = self._model(lang)
        for i, ps in enumerate(chunks):
            seed = self._seed if self._seed is not None else int.from_bytes(os.urandom(8), "little")
            yield model.generate_chunk(ps, lang, key, speed, keep_start=(i == 0), seed=seed + i)
            self.last_report = model.last_report()

    def generate(self, text, voice=None, language=None, speed=1.0):
        """Synthesize text. Returns float32 audio at sample_rate (24 kHz). speed: 0.25 to 4 (1.0 normal, higher is faster)."""
        gap = np.zeros(int(GAP_S * self.sample_rate), dtype=np.float32)
        out = []
        for w in self.generate_stream(text, voice, language, speed):
            out += [w, gap]
        return np.concatenate(out[:-1]) if out else np.zeros(0, dtype=np.float32)

    def generate_to_file(self, text, output_path, voice=None, language=None, speed=1.0, sample_rate=None):
        """Synthesize `text` into an audio file (WAV, FLAC, ... by the file name) at `sample_rate` (24 kHz).

        sample_rate is accepted for compatibility and must be the model's rate (24000): the audio is not resampled."""
        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            raise ValueError(f"sample_rate must be {self.sample_rate} (the model's rate; the audio is not resampled), "
                             f"got {sample_rate!r}: resample the array from generate() yourself for another rate")
        import soundfile as sf
        audio = self.generate(text, voice=voice, language=language, speed=speed)
        sf.write(str(output_path), audio, self.sample_rate)
        return str(output_path)

    def model_info(self, language=None):
        """Information about the native model that speaks `language` (loaded if needed): device, CPU threads, weight
        type, and "library", the build of the native library in use (on x86-64, the AVX2 build or the portable one)."""
        return dict(self._model(self._language(language)).info)

    def close(self):
        for m in self._models.values():
            m.close()
        self._models.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


__all__ = ["KittenTTS2Tiny", "TinyModel"]
