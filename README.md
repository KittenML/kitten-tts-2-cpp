# KittenTTS2 C++

CPU-focused C++ inference for KittenTTS2, with built-in voices, expression controls, and prepared reference voices. The runtime downloads prepared model assets and runs synthesis natively in C++. Users do not need to run the Python exporter.

The speech language model runs through llama.cpp/GGML, text normalization uses the [kitten-text-processing](https://github.com/KittenML/kitten-text-processing) submodule, and the S3 waveform decoder uses CPU LibTorch. Only KittenTTS2 checkpoints are supported.

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

## Download and run

The runtime fetches `config.json` from `KittenML/kitten-tts-2` first, then downloads its GGUF, selected TorchScript decoder, and matching voice conditioning. Assets are cached under `$XDG_CACHE_HOME/kitten-tts` (or `~/.cache/kitten-tts`).

```sh
build/bin/kitten-tts --text 'Hello from Kitten T T S.' --output hello.wav
build/bin/kitten-tts --decoder student_w4 --text 'Hello again.' --output w4.wav
```

Use `--repo OWNER/REPO` to select another KittenTTS2 repository, `--revision COMMIT` to pin a model version, `--cache-dir DIR` to change the cache location, and `--offline` to use cached files without network access. `HF_TOKEN` supports private repositories. `--download-only --decoder student_w4` prepares the cache without synthesis. Only assets needed for the selected mode are downloaded.

The model repository must publish a `cpp` manifest and prepared files as described in the [publisher guide](tools/kitten-tts/README.md#publish-native-assets). Local exports and model repository directories remain supported through `--assets DIR`; `--model FILE` overrides the GGUF.

The default GGUF uses this fork's `TQ2_1` format: 256 ternary weights packed into 64 bytes, plus two FP16 scales for the checkpoint's 128-weight groups. The exporter requires exact reconstruction. This reduces the model from 1.45 GB with Q4_0 to 1.03 GB. Embeddings remain FP16 and one-dimensional tensors remain FP32. GGML's quantized activation arithmetic can still introduce small logit differences.

`TQ2_1` requires this fork's CPU runtime; it is not standard `TQ2_0` and stock llama.cpp cannot load it. Use `--outtype ternary-q4_0` for the previous standard Q4_0 storage format, or `--outtype f16` for the reference export.

On supported Intel CPUs, the runtime losslessly repacks TQ2_1 weights into the existing AMX Q4_0 layout at load time. The GGUF stays 1.03 GB, but runtime transformer-weight storage grows from about 357 MiB to 756 MiB. Use `--no-repack` to keep the compact CPU representation. CPUs without AMX use the existing compact kernels.

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

Voice cloning requires an offline Python preparation step and a reference transcript. The Python interface supports sentence-chunk streaming; sample-by-sample streaming is not implemented. Waveform decoding currently depends on CPU LibTorch.

## Credits

KittenTTS2 and kitten-text-processing are from KittenML. See the [implementation guide](tools/kitten-tts/README.md#attribution) for decoder attribution and license details.

Built on [DeepGrove's llama.cpp fork](https://github.com/deepgrove-ai/llama.cpp).
