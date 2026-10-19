# Kitten-TTS 2 C++

C++ inference for KittenTTS2, with CPU and native NVIDIA CUDA backends in one repository, built-in voices, expression controls, and prepared reference voices.

Model weights come from [KittenML/kitten-tts-2 on Hugging Face](https://huggingface.co/KittenML/kitten-tts-2). Speech is generated locally on your machine. Follow the [GPU quick start](#native-nvidia-inference) or [CPU setup](#build) below.

The CPU backend runs the speech language model through llama.cpp/GGML and the S3 waveform decoder through CPU LibTorch. Its text normalization uses the [kitten-text-processing](https://github.com/KittenML/kitten-text-processing) submodule. The [native CUDA backend](native/README.md) provides the full inference engine and streaming APIs without LibTorch. Only KittenTTS2 checkpoints are supported.

## Native NVIDIA inference

The CUDA backend targets NVIDIA **Ampere, Ada, Hopper, and Blackwell** GPUs, with builds for compute capabilities 8.0, 8.6, 8.9, 9.0, 10.0, and 12.0. It has currently been tested on **five GPUs: A100, RTX 3090, RTX 4090, RTX 5090, and RTX PRO 6000 Blackwell**. Other GPUs with these compute capabilities have not yet been validated.

You need Linux, at least 24 GB VRAM for ordinary inference, CUDA Toolkit 12.8 or newer with a compatible NVIDIA driver, GCC 11 or newer, Make, CMake 3.28 or newer, Python 3.10 or newer for setup, and the FFmpeg development libraries. The tested voice-cloning profile requires extra model files and at least 32 GB VRAM; see [voice cloning](native/README.md#voice-cloning).

Run these three steps from the repository root.

### 1. Build

```sh
# Debian/Ubuntu audio decoder dependencies
sudo apt-get install libavformat-dev libavcodec-dev libavutil-dev libswresample-dev
git submodule update --init vendor/kitten-text-processing
python3 native/tools/build_text_deps.py --prefix "$PWD/build-deps/text" --jobs 8

cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
  -DLLAMA_BUILD_KITTEN_TTS=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF \
  -DKITTEN_CPU=OFF -DKITTEN_CUDA=ON \
  -DKT2N_BUILD_CLONING=ON -DKT2N_VOICE_AV=ON -DKT2N_AV=OFF \
  -DKT2N_TEXT_DEPS="$PWD/build-deps/text"
cmake --build build-cuda --target kitten-tts kt2n_prepare_text_data -j 8
```

The build supports all five GPUs above and includes the voice builder. You do not need to choose an architecture or change engine settings. Cloning networks are loaded only when cloning is requested.

### 2. Download and prepare the model once

Prepared CUDA files are not yet available on Hugging Face. This command downloads the source weights and voices automatically and converts them for the engine:

```sh
python3 -m venv .venv-convert
. .venv-convert/bin/activate
python -m pip install -r native/requirements-convert.txt
python native/tools/prepare_assets.py \
  --revision f2eee74f4467855ad58fe16f1fb7b7df473263fd \
  --text-converter build-cuda/bin/kt2n_prepare_text_data \
  --cache-dir models/source-cache --out models/kitten-native
```

Keep `models/kitten-native` for future runs. The prepared model is about 1.98 GB; setup also needs space for the downloads and Python packages. You only need to run this step once.

### 3. Generate speech

```sh
build-cuda/bin/kitten-tts --backend cuda --assets models/kitten-native \
  --text 'Hello from Kitten T T S.' --voice Bruno --output hello.wav
```

This writes `hello.wav`. Change `--text` and `--voice` for your own speech, or add `--device 1` to use a second GPU. Inference uses local files and needs no network connection, PyTorch, or LibTorch. Keep the default one-stream setting on RTX 3090 and RTX 4090.

Add `--stream false` to use the native engine's complete-generation quality mode. To create a voice from a recording, see the optional [voice cloning setup and examples](native/README.md#voice-cloning).

See [voice cloning and offline use](native/README.md).

### Measured time to first audio byte

| GPU | Median TTFB | p95 TTFB |
|---|---:|---:|
| RTX PRO 6000 Blackwell 96 GB | **9.65 ms** | 17.59 ms |
| RTX 5090 32 GB | **10.47 ms** | 19.05 ms |
| RTX 4090 24 GB | **16.33 ms** | 31.76 ms |
| RTX 3090 24 GB | **25.73 ms** | 50.67 ms |
| A100 PCIe 40 GB | **27.49 ms** | 54.71 ms |

Warm, single-stream benchmark: 1,500 requests per GPU using Bruno and the default decoder, cycling through the same 12 plain-English prompts with normalization enabled.

Use one stream on the tested 24 GB cards. A100, RTX 5090, and RTX PRO 6000 also passed two- and four-stream functional checks.

## Features

- Built-in voices, including Bruno, Bella, Jasper, Luna, Rosie, Hugo, Kiki, and Leo.
- Emotion tags, vocal events, and emphasis: `[joyful]`, `<laugh>`, and `(((really)))`.
- Stable and expressive sampling presets, sentence-aware chunking, and audio joins.
- A 1.03 GB GGUF with lossless packing of the checkpoint's ternary weights, plus an FP16 reference export.
- Default S3 and optional `student_w4` / `student_w8` decoder exports.
- Mono WAV output at 24 kHz: float32 on CPU and PCM16 through the CUDA CLI.

## Build

The following builds the CPU backend. For CUDA, use the [GPU quick start](#native-nvidia-inference).

Requires a C++20 compiler, CMake, and CPU LibTorch. A CPU PyTorch installation can provide LibTorch's headers and libraries. Use the same LibTorch version for export and inference, with compatible torch/torchaudio versions for asset preparation.

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

### CPU

The CPU runtime fetches `config.json` from [KittenML/kitten-tts-2](https://huggingface.co/KittenML/kitten-tts-2) first, then downloads its GGUF, selected TorchScript decoder, and matching voice conditioning. You do not need to download files manually or run an exporter. Assets are cached under `$XDG_CACHE_HOME/kitten-tts` (or `~/.cache/kitten-tts`).

```sh
build/bin/kitten-tts --text 'One day, a little girl named Lily found a needle in her room.' --output hello.wav
build/bin/kitten-tts --decoder student_w4 --text 'One day, a little girl named Lily found a needle in her room.' --output w4.wav
```

Use `--repo OWNER/REPO` to select another KittenTTS2 repository, `--revision COMMIT` to pin a model version, `--cache-dir DIR` to change the cache location, and `--offline` to use cached files without network access. `HF_TOKEN` supports private repositories. `--download-only --decoder student_w4` prepares the cache without synthesis. Only assets needed for the selected mode are downloaded.

Local exports and model repository directories remain supported through `--assets DIR`; `--model FILE` overrides the GGUF. Model publishers can find the file layout in the [CPU publisher guide](tools/kitten-tts/README.md#publish-native-assets).

### CUDA

Follow the [GPU quick start](#native-nvidia-inference). The CLI uses the prepared model directory. For cached downloads, offline preparation, and publishing CUDA model files, see the [model file instructions](native/README.md#model-files-and-offline-use).

## CPU model formats

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

## Current scope

Sampled output is not bit-identical to Python: BF16 arithmetic, GGML activation quantization, and sampling RNGs differ. Matching seeds do not guarantee matching generated tokens across runtimes.

The CPU backend provides batch synthesis and uses LibTorch for waveform decoding; its reference voices need an offline Python preparation step. The native CUDA backend provides streaming audio, complete-generation mode, incremental text APIs, and optional voice cloning directly from audio and its transcript. Cloning needs additional model assets and GPU memory.

## Credits

KittenTTS2 and kitten-text-processing are from KittenML. See the [implementation guide](tools/kitten-tts/README.md#attribution) for decoder attribution and license details.

Built on [DeepGrove's llama.cpp fork](https://github.com/deepgrove-ai/llama.cpp).

The llama.cpp code retains its [MIT license](LICENSE). Original native KittenML material uses [Apache-2.0](native/LICENSE); the native library also contains LGPL-2.1-or-later-derived `atempo` code and other third-party components. See [NOTICE](native/NOTICE) for attribution. Model weights retain their publishers' licenses.
