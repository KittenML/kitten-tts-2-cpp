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

## Build

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

The runtime fetches `config.json` from `KittenML/kitten-tts-2` first, then downloads its GGUF, selected TorchScript decoder, and matching voice conditioning. Assets are cached under `$XDG_CACHE_HOME/kitten-tts` (or `~/.cache/kitten-tts`).

```sh
build/bin/kitten-tts --text 'One day, a little girl named Lily found a needle in her room.' --output hello.wav
build/bin/kitten-tts --decoder student_w4 --text 'One day, a little girl named Lily found a needle in her room.' --output w4.wav
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

## Current scope

Sampled output is not bit-identical to Python: BF16 arithmetic, GGML activation quantization, and sampling RNGs differ. Matching seeds do not guarantee matching generated tokens across runtimes.

Voice cloning requires an offline Python preparation step and a reference transcript. The runtime is a batch CLI; Python's streaming API is not implemented. Waveform decoding currently depends on CPU LibTorch.

## Credits

KittenTTS2 and kitten-text-processing are from KittenML. See the [implementation guide](tools/kitten-tts/README.md#attribution) for decoder attribution and license details.

Built on [DeepGrove's llama.cpp fork](https://github.com/deepgrove-ai/llama.cpp).
