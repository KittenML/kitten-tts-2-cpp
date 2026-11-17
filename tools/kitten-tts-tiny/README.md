# Kitten TTS 2 mini, nano and pico on ggml

Runs `KittenML/kitten-tts-2-mini` (16 languages), `kitten-tts-2-nano` and `kitten-tts-2-pico` (8 languages, one
model per language) with ggml on the CPU (x86 and Arm) and on GPUs with CUDA (NVIDIA), Metal (Apple silicon) or
Vulkan. On Metal the text and prosody stages run on the CPU and the waveform decoder on the GPU; the audio matches the
CPU's to 90 dB signal-to-noise ratio or better, with identical durations. No PyTorch or ONNX Runtime is needed for
synthesis.

The f32 model matches the PyTorch fp32 model to rounding, not bit for bit: given the same input and noise, every test
sentence gets identical durations, and the audio differs by 0.006 to 0.03 dB of log-mel (signal-to-noise ratio 41 to
54 dB) on the CPU (x86 and Arm), on CUDA and on Vulkan (NVIDIA RTX 5070). PyTorch on 1 thread against PyTorch on 16
threads agrees more closely: 44 dB for mini (0.6 to 2 dB closer than this library), 70 dB for nano and 57 dB for pico
(11 to 18 dB closer). Two PyTorch runs with different noise seeds differ far more: 0.18 to 0.52 dB of log-mel. In
German, Spanish, French, Italian, Portuguese, Russian and Hindi (CUDA, 3 or 4 sentences per language and model),
durations were identical and the log-mel difference 0.015 to 0.035 dB. On 454 CMU
ARCTIC clips (English), the f32 files score the same as PyTorch fp32 within the confidence interval: word error rate
(Whisper large-v3) differs by +0.03 to +0.05 points, UTMOS by less than 0.002 (rendered on CUDA). f16 and q8_0 files are lossy (below).

## Build

Needs CMake and a C++17 compiler; nothing else. On macOS, `xcode-select --install` gives the compiler and git but not
CMake; get CMake in a virtual environment, together with the converter's packages (below):
`python3 -m venv .venv && . .venv/bin/activate && python -m pip install cmake numpy safetensors huggingface_hub gguf`.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_KITTEN_TTS_TINY=ON
cmake --build build --config Release --target kitten-tts-tiny -j 8
```

Add `-DGGML_CUDA=ON` or `-DGGML_VULKAN=ON` for a GPU. On macOS, Metal is on by default (its shaders are compiled
when the library loads, so the Command Line Tools are enough; no Xcode), and add `-DGGML_NATIVE=OFF` to the first
line: on an Apple M4 the native build ran 1.7 to 2.3 times slower than the pip package. The build produces the command line tool
`build/bin/kitten-tts-tiny` and the C library `build/bin/libkitten_tts_tiny.so` (`.dylib`), whose API is in
`kitten_tts_tiny.h`. With Visual Studio on Windows, `--config Release` picks the release build (the default would be
a slow debug build), and the files are `build\bin\Release\kitten-tts-tiny.exe` and `kitten_tts_tiny.dll`.

## Get a model

Convert the official repository (needs `numpy`, `safetensors`, `huggingface_hub` and `gguf`, or this repository's
`gguf-py`, in place of `gguf`):

```sh
python tools/kitten-tts-tiny/convert_kitten_tiny_to_gguf.py KittenML/kitten-tts-2-nano --revision 54526b8d8e45333cd720d3a082c106aacf416064 --model kitten_tts_2_nano_en --outtype f32 --outdir models
```

`--revision` pins the tested commit of the repository (without it, the latest version is converted); the tested
commits of mini, nano and pico are `REVISIONS` in `python/kitten_tts_cpp/tiny.py`.

`--outtype` is `f32` (the reference), `f16` or `q8_0` (smaller, lossy). `--f16-prefixes decoder.generator.` with `q8_0` keeps the waveform
generator in f16. Without `--model`, every language of the repository is converted. Each file holds its voices.

## Use it

Python (text in, audio out; the text front end is kittenml's own, vendored):

```python
from kitten_tts_cpp import KittenTTS2Tiny
m = KittenTTS2Tiny("KittenML/kitten-tts-2-nano")            # converts to GGUF on first use and caches it
audio = m.generate("Hello there.", voice="Eleanor")          # float32, 24 kHz
audio = m.generate("Guten Morgen.", language="de")
m.voices("de")
```

The x86-64 wheels hold two builds of the library: one for any x86-64 CPU and one for CPUs with AVX2 (5 to 9 times
faster on an AMD EPYC 7282 at 4 threads); the package loads the fast one when the CPU supports it
(`m.model_info()["library"]` says which).

Options: `device="auto" | "cpu" | "cuda" | "metal" | "vulkan"`, `weights="f32" | "f16" | "q8_0" | "q8_0-genf16"`,
`threads=` (default: the physical cores the process may use, at most 8; on Apple silicon, every core except the
efficiency cores),
`seed=` (default `None`: a new noise seed for every call, so repeated calls sound slightly different, as with the
official model; an integer gives the same audio for the same input on the same device), `cache_dir=` (downloads and
converted GGUF files; by default the Hugging Face cache and `~/.cache/kitten-tts-cpp/tiny`, or
`$KITTEN_TTS_CPP_CACHE/tiny`), `revision=` (default: the tested revision of the official repositories; converted
files are cached per commit), `hf_token=` (private repositories), `gguf_dir=` or `gguf_repo=` to read prepared GGUF
files. `KittenTTS2Tiny` also works as a context manager
(`with KittenTTS2Tiny(...) as m:`). `speed=` goes from 0.25 to 4 (1.0 is normal, higher is faster). Set `KITTEN_TTS_TINY_LIB` to the library path
if it is not installed next to the package. `pip install kitten-tts-cpp` installs the whole text front end, from
ready-made packages (nothing to compile): `spacy` for English and `fugashi` for Japanese (the package needs Python 3.10 to 3.14; spaCy's small
English model comes inside the package, and the fallback model for unknown English words runs in numpy),
and `phonemizer` and `espeakng_loader` for the other languages (espeak-ng is GPL-3.0; it comes as a separate package and
is not built into this one). The first Japanese use downloads a Japanese dictionary (unidic-lite, 47 MB) once.

For an NVIDIA GPU you need the GPU build of the package instead of `kitten-tts-cpp`: `kitten-tts-cpp-cu12` on Linux or
`kitten-tts-cpp-cu13` on Windows. It is not on PyPI yet: build it with the CUDA toolkit as in
[Building](../../packaging/README.md#building) and install it after the matching PyTorch build as in
[Installing an artifact](../../packaging/README.md#installing-an-artifact). `device="auto"` (the default) then picks
the GPU, and `device="cuda"` asks for it. The pip packages do not include Vulkan; for it, build from source (above) and set `KITTEN_TTS_TINY_LIB` to the built
library.

Command line (phonemes in, as the front end produces them):

```sh
build/bin/kitten-tts-tiny -m models/kitten_tts_2_nano_en-f32.gguf --phonemes "həlˈoʊ wˈɜːld." --voice eleanor -o hello.wav
build/bin/kitten-tts-tiny -m models/kitten_tts_2_nano_en-f32.gguf --info
```

On Windows the command line may not pass IPA characters through; put the phonemes in a UTF-8 text file and use
`--phonemes-file hello.txt` instead of `--phonemes`.

## Which weights

By default the library converts f16 and q8_0 weights to f32 when it loads them: the file is smaller, only the weights
are rounded, and the arithmetic is f32 (on the Arm CPU we measured, computing in f16 or q8_0 was not faster than f32). Pass `--no-dequantize` (CLI),
`dequantize=False` (Python) or `kt_tiny_params.dequantize = 0` to compute in the file's type instead; ggml then also
rounds the activations, which costs more quality.

| file | size (mini / nano / pico) | quality against PyTorch fp32 (log-mel difference, dB; mini / nano / pico), default loading | with `--no-dequantize` |
|---|---|---|---|
| f32 | 386 / 71 / 31 MB | 0.019 / 0.006 / 0.028 | same |
| f16 | 225 / 38 / 17 MB | 0.20 / 0.15 / 0.10 | 0.17 to 0.32 |
| q8_0-genf16 | 168 / 25 / 11 MB | 0.73 / 0.76 / 0.85, supported, not optimal | 0.75 to 1.05 |
| q8_0 | 149 / 23 / 10 MB | 0.79 / 0.80 / 1.45, supported, not optimal | 1.56 to 3.93 |

Measured on an x86 CPU with 12 English sentences, PyTorch's durations and noise fed in. For scale: two PyTorch runs
with different noise seeds differ by 0.18 to 0.52 dB, and the official int8 ONNX model differs from PyTorch fp32 by
1.5 to 2.5 dB. f16 files change a duration in 1 of 12 sentences (nano), q8_0 files in 2 to 4 of 12. On ARCTIC, the q8_0 file
loaded as f32 adds 0.04 to 0.08 points of word error rate (mini: +0.08, confidence interval +0.01 to +0.15) and changes
UTMOS by -0.011 (pico) to +0.004 (nano): small, but measurable.

On GPUs, matrix products run in full f32: no TF32 on CUDA (each product asks ggml for strict f32), and no fp16 or
cooperative-matrix products on Vulkan. `allow_tf32=True` (Python), `--allow-tf32` (CLI) or
`kt_tiny_params.allow_tf32 = 1` allows them: a little faster, and lossy. On CUDA it gains only 6 to 8 % (RTX 5070: mini 0.0055 to 0.0051 real-time factor,
pico 0.0016 to 0.0015), because the decoder convolutions, most of the work, run in this library's own f32 kernel
either way; it costs 2.8 dB of log-mel on pico (more than the official int8 model), 0.36 dB on nano, where 1 of the
12 sentences gets a different duration, and 0.22 dB on mini (same 12 sentences as the table above). We do not
recommend it. Known limitation on Vulkan: ggml's Vulkan backend has no per-product strict f32 option, so the
library sets the environment variables `GGML_VK_DISABLE_F16`, `GGML_VK_DISABLE_COOPMAT` and `GGML_VK_DISABLE_COOPMAT2`
(unless already set) just before it creates the Vulkan device. They apply to everything in the process that uses
Vulkan through ggml, and setting them is not thread-safe: load the first Vulkan model before starting other threads, or
pass `allow_tf32=True` to leave them alone.

## Speed

Arm CPU (Cortex-X925 / A725, 20 cores), f32 weights; first audio is a 2.3 s sentence, the real-time factor (processing
time / audio time) is measured on a 10.7 s chunk; mean of two runs, each the median of 4 warm runs (the runs agree
within 6 %):

| model | threads | first audio | real-time factor | peak memory |
|---|---|---|---|---|
| pico | 1 | 154 ms | 0.076 | 105 MB |
| pico | 4 | 54 ms | 0.023 | 105 MB |
| nano | 1 | 324 ms | 0.149 | 169 MB |
| nano | 4 | 99 ms | 0.044 | 169 MB |
| mini | 4 | 623 ms | 0.281 | 605 MB |
| mini | 16 | 320 ms | 0.115 | 604 MB |

mini does far more arithmetic than nano and pico (its generator alone about 48 GFLOP per second of audio), which is
why it gains the most from more threads. On an x86 server CPU (AMD EPYC 7282,
Zen 2), 4 threads, indicative, the wheel's AVX2 build: real-time factor pico 0.059, nano 0.093, mini 0.33 (the
wheel's portable build, used on CPUs without AVX2: pico 0.32, nano 0.80). On an x86 server CPU with AVX-512 (Intel
Xeon Gold 6152, a native build, same method as the Arm table, two runs within 8 %): real-time factor at 4 threads pico
0.047, nano 0.068, mini 0.25; at 16 threads 0.023, 0.033, 0.11. The default is at most 8 threads, but on CPUs with
more cores more threads still help: with the wheels' x86-64-v3 build on that 22-core Xeon, mini goes from 0.16 at 8
threads to 0.11 at 16 and 0.10 at 22 (pico 0.031, 0.022, 0.021). Pass `threads=` (up to the number of physical cores)
there, unless several models share the CPU.

Apple M4 (4 performance and 6 efficiency cores, macOS), the pip package from Python, f32 weights, English, median of 3
fresh processes (each the median of 5 warm passes over one chunk of 8.6 to 9.4 s): real-time factor at the default 4
threads (the performance cores) pico 0.016, nano 0.029, mini 0.18; at 1 thread 0.049, 0.089, 0.58; at 10 threads
0.012, 0.022, 0.12. Peak memory of the Python process, text front end included: 530 MB (pico), 600 MB (nano), 1.06 GB
(mini).

Apple M6 (2 + 4 performance and 6 efficiency cores, macOS 27), same method (one chunk of 9.5 to 10.7 s, median of 3
fresh processes in a balanced order): Metal (`device="auto"` on a Mac) against the CPU at the default 6 threads:

| model | Metal, real-time factor | CPU, real-time factor | Metal / CPU, first audio | Metal / CPU, peak memory |
|---|---|---|---|---|
| pico | 0.0086 | 0.0103 | 20 / 24 ms | 535 / 546 MB |
| nano | 0.0158 | 0.0191 | 35 / 45 ms | 577 / 609 MB |
| mini | 0.050 | 0.125 | 97 / 272 ms | 913 / 1093 MB |

The first sentence after starting Python takes 0.9 to 1.0 s (pico, nano) and 1.0 to 1.2 s (mini) on either device,
the Metal shader compile included. `allow_tf32=True` on Metal (half-precision matrix products) is no faster there and
lossier.

CUDA, f32, warm, against kittenml's PyTorch fp32 model on the same GPU and host (torch 2.9.1, CUDA 12.8). Real-time
factor over 4 English texts (6 chunks, 69 s of audio), median of 3 passes; first audio = model time for the first
chunk (a 6.7 s sentence); peak memory of the process (RSS). One session per GPU that alternates this library,
PyTorch with its default settings and PyTorch in strict f32, each in a fresh process (RTX 5070 with an Intel Xeon
Gold 6152 host: 3 rounds, table = mean; L4 with an Intel Xeon Gold 6526Y host: 3 rounds, table = mean); the rounds agree
within 4 %.

| GPU | model | real-time factor, this library | PyTorch fp32, TF32 off | PyTorch default (TF32 convolutions) | first audio, this library / PyTorch default | peak memory (RSS), this library / PyTorch |
|---|---|---|---|---|---|---|
| RTX 5070 | pico | 0.0016 | 0.0042 | 0.0043 | 13 / 43 ms | 0.63 / 2.3 GB |
| RTX 5070 | nano | 0.0021 | 0.0045 | 0.0046 | 18 / 47 ms | 0.67 / 2.4 GB |
| RTX 5070 | mini | 0.0055 | 0.0115 | 0.0101 | 41 / 64 ms | 0.68 / 2.4 GB |
| L4 | pico | 0.0018 | 0.0034 | 0.0033 | 15 / 32 ms | 0.56 / 1.8 GB |
| L4 | nano | 0.0026 | 0.0037 | 0.0037 | 21 / 36 ms | 0.60 / 1.8 GB |
| L4 | mini | 0.0088 | 0.0125 | 0.0098 | 64 / 60 ms | 0.62 / 1.9 GB |

This library computes in strict f32 throughout; PyTorch's default uses TF32 for convolutions, which is faster on
these GPUs and less exact. mini's first audio on the L4 is 64 ms against 80 ms for PyTorch in strict f32. The first call after loading takes about 140 to 310 ms.

PyTorch's time at these sizes depends on the host CPU, because it launches many small GPU kernels one after another:
with the same RTX 5070 in a desktop with an AMD Ryzen 5 5500, PyTorch's default ran pico at 0.0033 and nano at
0.0036, while this library's pico (0.0015 to 0.0016) and nano (0.0021) were as fast as on the Xeon host (an earlier
build). So the speed-up over PyTorch on this GPU is a range: pico 2.1 to 2.7 times, nano 1.7 to 2.2 times.

Vulkan (RTX 5070, NVIDIA driver, same method and host, 3 rounds) is slower, because the LSTM and the decoder's fused
steps have CUDA kernels only, so far more graph nodes run: real-time factor pico 0.015, nano 0.018, mini 0.024; first
audio 110 / 162 / 198 ms. Vulkan's time also depends on the host CPU, which records and submits the GPU work node by
node: an earlier build gave the same numbers on this host and pico 0.009, nano 0.011, mini 0.019 on the Ryzen host.

## How it works

Three ggml graphs run per chunk of text: tokens to durations (PL-BERT, text encoder, BiLSTM duration predictor),
frames to pitch and energy (BiLSTM, AdaIN residual blocks), and the iSTFTNet decoder and generator. The inverse STFT
is a fixed-basis matrix product followed by `ggml_col2im_1d`. Between the graphs the CPU expands durations into
frames and builds the harmonic source (sine generator and forward STFT) with PyTorch's float operations in the same
order. On the CPU backend, AdaIN plus activation and the LSTM recurrence run as fused operations.

With CUDA, three parts run as this library's own CUDA kernels (`kitten_tts_tiny_cuda.cu`): the LSTM recurrence (each
direction split over several thread blocks that keep their part of the recurrent weights in shared memory), the
harmonic source with its STFT, and, before each decoder convolution, AdaIN + activation + zero padding in one pass.
The decoder and generator convolutions (input channels a multiple of 16, output channels a multiple of 4: all of them
in these models) then run as one implicit-GEMM kernel that reads the overlapping input windows in place and adds the
bias and, in the generator's residual units, the unit's input, in f32 fused multiply-adds (no window copy, no TF32;
it differs from the cuBLAS path by rounding only, 116 dB SNR on a 13 s mini chunk); other channel counts would copy
their sliding window and use cuBLAS. The LSTM, AdaIN and convolution kernels are custom graph nodes that ggml's CUDA backend
runs on its own stream (an opt-in hook in ggml-cuda), so they stay inside CUDA graphs; the harmonic source uses a
placeholder node after which the ggml scheduler stops and the kernel fills it in. Without them, a GPU backend runs every
LSTM time step as separate graph nodes (about 19,000 nodes for a 250-token chunk instead of about 650). The random
noise of the harmonic source then comes from a GPU random number generator: the same seed gives the same audio on
the same device, not the same audio as on the CPU.

Builds with `-DKITTEN_TTS_TINY_DEBUG_ENV=ON` also read debugging variables (`KT_TINY_DUMP=dir` writes intermediate
tensors); release builds ignore them. `KT_TINY_PROFILE=1` (read when the first model loads) prints the time per graph
operation.

Test tools: `parity_kitten_tiny.py` (against kittenml's PyTorch model, with the same noise and durations) and
`test_tiny_python.py` (the Python class end to end).
