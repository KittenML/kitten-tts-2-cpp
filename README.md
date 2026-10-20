# KittenTTS2 C++

C++ inference for KittenTTS2 on the CPU, with an optional NVIDIA GPU (CUDA) path, built-in voices, expression controls, and prepared reference voices. The runtime downloads prepared model assets and runs synthesis natively in C++. Users do not need to run the Python exporter.

The speech language model runs through llama.cpp/GGML, text normalization uses the [kitten-text-processing](https://github.com/KittenML/kitten-text-processing) submodule, and the S3 waveform decoder uses LibTorch on the CPU, or on the GPU with a locally exported CUDA decoder. Only KittenTTS2 checkpoints are supported.

## Features

- Built-in voices, including Bruno, Bella, Jasper, Luna, Rosie, Hugo, Kiki, and Leo.
- Emotion tags, vocal events, and emphasis: `[joyful]`, `<laugh>`, and `(((really)))`.
- Stable and expressive sampling presets, sentence-aware chunking, and audio joins.
- A 1.03 GB GGUF with lossless packing of the checkpoint's ternary weights, plus an FP16 reference export.
- Default S3 and optional `student_w4` / `student_w8` decoder exports.
- Mono float32 WAV output at 24 kHz.

## Python interface

Precompiled wheel builds are configured for Linux x86-64/ARM64, Apple Silicon macOS, Windows x86-64, and experimental Windows ARM64. See the [wheel matrix and installation guide](packaging/README.md). These builds remain local configuration until the workflow is pushed and run; cross-platform artifacts have not been published.

The `py-interface` branch includes a native Python package with a KittenTTS-style API. `setup.py` invokes CMake during installation and bundles the normalizer grammar data. Install a CPU PyTorch build first, then build against that same installation:

```sh
git submodule update --init vendor/kitten-text-processing
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install setuptools wheel cmake numpy soundfile
python -m pip install . --no-build-isolation
```

A C++17 compiler and OpenSSL development headers are required for HTTPS downloads (`libssl-dev` on Debian/Ubuntu). Set `CMAKE_BUILD_PARALLEL_LEVEL=8` to control build parallelism. `CMAKE_ARGS` accepts extra CMake flags, such as `-DOPENSSL_ROOT_DIR=/path/to/openssl`. Builds optimize for the local CPU; use `CMAKE_ARGS="-DGGML_NATIVE=OFF"` when building a wheel for other machines. Source-built wheels require the exact PyTorch release used at build time and the linked system OpenSSL libraries. The portable CI wheels instead link LibreSSL statically.

```python
from kitten_tts_cpp import KittenTTS  # KittenTTS2 and get_model are also available

with KittenTTS("KittenML/kitten-tts-2", decoder="student_w4", threads=8) as model:
    print(model.available_voices)
    audio = model.generate("Hello again.", voice="Bruno", advanced={"seed": 1234})
    model.generate_to_file("Hello again.", "hello.wav", voice="Bruno")
    for chunk in model.generate_stream("A longer passage goes here.", voice="Bruno"):
        pass  # Each chunk is a float32 NumPy array at model.sample_rate.
```

Weights and the decoder stay loaded between calls. The GIL is released during native loading and inference. Native inference calls are serialized because the TorchScript decoder uses process-wide random and thread settings. Streaming yields untrimmed sentence chunks as they are decoded; close the iterator when stopping early. `generate` applies the same chunk joins as the CLI. `last_report` contains token IDs and generation timings.

The constructor also accepts `cache_dir`, `hf_token`, `revision`, `offline`, `decoder_threads`, `repack`, `model_path` (a local GGUF override), `data_dir`, and `ca_file` (a custom HTTPS trust bundle). A local prepared asset directory can replace the repository ID. Generation supports the Python KittenTTS2 preset, sampling, normalization, reference-prompt, and `advanced` options, including `use_emotion`.

Only KittenTTS2 CPU inference and `weights="packed"` are supported. This package uses the separate `kitten_tts_cpp` import name so it can coexist with `kittenml`. Raw-audio `reference=` cloning is not implemented; prepare a reference with `prepare_reference.py` and select its voice name. It does not provide the older ONNX models or guarantee identical sampled output to Python's original PyTorch backend.

For development, `python setup.py build_ext --inplace` builds the extension; set `PYTHONPATH=python` to import it directly. `python setup.py bdist_wheel` produces an installable wheel, and `python setup.py sdist` includes the native sources and normalizer submodule contents.

## Build

Requires a C++17 compiler, CMake, and CPU LibTorch. A CPU PyTorch installation can provide LibTorch's headers and libraries. Use the same LibTorch version for export and inference, with compatible torch/torchaudio versions for asset preparation.

```sh
git submodule update --init vendor/kitten-text-processing

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF \
  -DLLAMA_BUILD_KITTEN_TTS=ON \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

cmake --build build --target kitten-tts -j 8
```

The normalizer prepares its grammar data using Python at build time. To use existing grammar data, configure with `-DKITTEN_DATA_DIR=/path/to/data`.

## GPU (CUDA) build

The speech LM can also run on an NVIDIA GPU. Configure with one CUDA 12.8 toolkit and `-DGGML_CUDA=ON`. To run the S3 decoder on the GPU too, use the single configure command in the [CUDA decoder section](tools/kitten-tts/README.md#cuda-waveform-decoder-optional-local-export) of the detailed guide instead, which also points CMake at a CUDA LibTorch.

```sh
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DGGML_CUDA=ON -DGGML_METAL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES='75-real;86-real;89-real;120a-real' \
  -DLLAMA_BUILD_KITTEN_TTS=ON \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-cuda --target kitten-tts -j 8
build-cuda/bin/kitten-tts --gpu-layers 99 --timings --text 'Hello from the GPU.' --output gpu.wav
```

`--gpu-layers` defaults to 0, so CPU runs give the same tokens, audio and logits as before unless you ask for the GPU; they use less memory, because each context reserves one output row and is freed before the decoder runs. With `--gpu-layers 99` the 28 transformer layers and the output head run on the GPU. The TQ2_1 weights are repacked losslessly to the CUDA-supported Q2_0 format at load time. Text normalization, tokenization, the token embedding lookup, sampling and WAV writing stay on the CPU. The S3 decoder stays on the CPU unless you export a CUDA decoder and pass `--decoder-device cuda` (see the detailed guide). If any TQ2_1 weight would stay on the CPU, the runtime stops with an error unless you pass `--allow-hybrid`. Other options are `--main-gpu N` and `--flash-attn auto|on|off`.

On an RTX 4090 the whole LM used about 1,670 MiB of GPU memory for each of 10 test cases (5 sentences x 2 voices, up to 12.2 s of audio), including about 395 MiB of CUDA context. With the CUDA decoder, one request of one of these sentences peaked at 2,219-2,265 MiB. Later requests in the same process, a 20.9 s chunk and multi-chunk texts peak higher, because LibTorch keeps its cached blocks while the next LM context is allocated: up to about 2,495 MiB for repeated requests and for one 20.9 s chunk, and about 2,876 MiB for a 4-chunk text (about 2,462 MiB with `KITTEN_DECODER_EMPTY_CACHE=1`). GPU memory was measured on that card only. The Q2_0 GPU kernels are validated (kernel tests and logits) on that card (SM89) only; builds of this source also ran full generations on an RTX PRO 6000 (12.0), a Tesla T4 (7.5) and a GTX 1080 Ti (6.1), without a logits comparison. On the RTX 4090 machine (a shared host, with the CPU rows on 8 threads of one 8-core CCD) the LM ran about 455 forward passes/s (37 on the CPU), or 177 to 184 generated tokens/s including CPU sampling (33 on the CPU), and the RTF was 0.14-0.16 with the CUDA decoder, 0.41 with the CPU decoder and 1.28 on the CPU alone. Two setups are supported but are not the fastest, because part of the work stays on the CPU. The CPU decoder next to a GPU LM (the default) makes the decoder the slowest stage: on that RTX 4090 its RTF was 0.28 on the CPU against 0.008 on the GPU. Partial offload with `--allow-hybrid` runs the layers left on the CPU at CPU speed: on a GTX 1080 Ti with `--gpu-layers 14`, the LM took 16.4 s for a 5.2 s sentence, against 1.4 s with every layer on the GPU and 36 s on the CPU alone (8 threads, with the portable CPU code of the CI wheels, which is slower than a native build's). Use it only when the whole LM does not fit in GPU memory. Precision differs from CPU runs, so logits differ slightly: GPU activations are quantized to Q8_1 instead of the CPU's Q8_K (Q8_0 with AMX), the F16 output head uses FP16 products and partial sums on CUDA, and flash attention sums in a different precision (see the detailed guide). `--timings` prints per-stage times, tokens per second, sampling share, RTF and peak GPU memory. See the [GPU section of the detailed guide](tools/kitten-tts/README.md#gpu-cuda-build) for the memory breakdown and the stage list.

## Download and run

The runtime fetches `config.json` from `KittenML/kitten-tts-2` first, then downloads its GGUF, selected TorchScript decoder, and matching voice conditioning. Assets are cached under `$XDG_CACHE_HOME/kitten-tts` (or `~/.cache/kitten-tts`).

```sh
build/bin/kitten-tts --text 'Hello from Kitten T T S.' --output hello.wav
build/bin/kitten-tts --decoder student_w4 --text 'Hello again.' --output w4.wav
```

Use `--repo OWNER/REPO` to select another KittenTTS2 repository, `--revision COMMIT` to pin a model version, `--cache-dir DIR` to change the cache location, and `--offline` to use cached files without network access. `HF_TOKEN` supports private repositories. `--download-only --decoder student_w4` prepares the cache without synthesis. Only assets needed for the selected mode are downloaded.

The model repository must publish a `cpp` manifest and prepared files as described in the [publisher guide](tools/kitten-tts/README.md#publish-native-assets). Local exports and model repository directories remain supported through `--assets DIR`; `--model FILE` overrides the GGUF.

The default GGUF uses this fork's `TQ2_1` format: 256 ternary weights packed into 64 bytes, plus two FP16 scales for the checkpoint's 128-weight groups. The exporter requires exact reconstruction. This reduces the model from 1.45 GB with Q4_0 to 1.03 GB. Embeddings remain FP16 and one-dimensional tensors remain FP32. GGML's quantized activation arithmetic can still introduce small logit differences.

`TQ2_1` requires this fork; it is not standard `TQ2_0` and stock llama.cpp cannot load it. The fork has CPU kernels for it, and GPU builds repack it losslessly to Q2_0 at load time. Use `--outtype ternary-q4_0` for the previous standard Q4_0 storage format, or `--outtype f16` for the reference export.

On supported Intel CPUs, the runtime losslessly repacks TQ2_1 weights into the existing AMX Q4_0 layout at load time. The GGUF stays 1.03 GB, but runtime transformer-weight storage grows from about 357 MiB to 756 MiB. Use `--no-repack` to keep the compact CPU representation; it affects only CPU buffers, not the GPU repack to Q2_0. CPUs without AMX use the existing compact kernels.

See the [detailed guide](tools/kitten-tts/README.md) for FP16 exports, student decoders, reference voice preparation, and the normalizer submodule setup.

## Generate speech

```sh
build/bin/kitten-tts \
  --text 'Hello there. This is Kitten T T S running on the CPU.' \
  --voice Bruno --threads 8 --seed 1234 \
  --output hello.wav --report hello.json
```

For expressive speech:

```sh
build/bin/kitten-tts \
  --text '[joyful] We won the grant <laugh> I can (((hardly))) believe it!' \
  --voice Kiki --preset expressive \
  --output expressive.wav
```

Use `--help` for sampling, repetition penalties, chunking, and decoder thread settings. `--tokens-only` skips waveform decoding; `--repeat 3` benchmarks repeated synthesis with the model and decoder kept loaded.

## Python parity and CPU performance

The implementation is checked against `kittenml.kittentts2`:

| Check | Recorded result |
| --- | --- |
| Text normalization and chunking | 228 cases passed |
| Sampling rules | 12 cases passed |
| Audio joins | 3 cases passed |
| Packed weights | All 310 tensors reconstruct the FP16 reference exactly |
| Native waveform decoding | Sample-for-sample agreement with Python in tested cases for all three decoders |
| Packed LM logits | Close numerical agreement with the Python reference |

On a Xeon Platinum 8462Y+ with eight pinned CPU cores, the AMX path reduced a 256-token forced-sequence workload from **6.40 to 5.60 seconds** (about **14% higher throughput**). Prompt evaluation fell from **1.09 to 0.58 seconds**. TQ2_1 reached **45.7 generated tokens/second**, comparable to Q4_0 at **44.3** in the same test. These are medians of three warmed runs, including prompt evaluation and sampling but excluding model loading. Performance varies with hardware and workload.

Run the differential checks with:

```sh
python tools/kitten-tts/check_parity.py \
  --binary build/bin/kitten-tts \
  --python-package ../KittenTTS \
  --repo ../kitten-tts-2 --assets models/kitten2 \
  --model models/kitten2/model-tq2_1.gguf \
  --lm --decoder
```

Measurements and source revisions are recorded in [validation.json](tools/kitten-tts/validation.json).

## Current scope

Sampled output is not bit-identical to Python: BF16 arithmetic, GGML activation quantization, and sampling RNGs differ. Matching seeds do not guarantee matching generated tokens across runtimes.

Voice cloning requires an offline Python preparation step and a reference transcript. The Python interface supports sentence-chunk streaming; sample-by-sample streaming is not implemented. Waveform decoding currently depends on LibTorch: on the CPU, or on CUDA with a locally exported decoder.

## Credits

KittenTTS2 and kitten-text-processing are from KittenML. See the [implementation guide](tools/kitten-tts/README.md#attribution) for decoder attribution and license details.

Built on [DeepGrove's llama.cpp fork](https://github.com/deepgrove-ai/llama.cpp).
