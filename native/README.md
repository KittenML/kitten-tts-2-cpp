# Native CUDA inference

The NVIDIA backend for [Kitten TTS 2 C++](../README.md) runs locally without PyTorch or LibTorch. Start with the [GPU quick start](../README.md#native-nvidia-inference) to build, download the model from Hugging Face, and generate speech.

## Generate speech

```sh
build-cuda/bin/kitten-tts --backend cuda --assets models/kitten-native \
  --voice Bruno --text 'A complete passage with natural pauses.' \
  --stream false --output complete.wav
```

`--stream false` selects complete-generation quality mode, including its sentence chunking, generation budgets, whole-segment rendering, and audio processing. It waits for the completed audio. The default uses streaming generation. The CLI accepts `--no-normalize` to disable text normalization.

To write mono PCM16 audio at 24 kHz to standard output as it arrives:

```sh
build-cuda/bin/kitten-tts --backend cuda --assets models/kitten-native \
  --voice Bruno --text 'Streaming speech.' --format pcm --output -
```

## Voice cloning

The default CUDA build includes voice cloning. Prepare the extra assets once, then provide a recording and its transcript. The tested profile needs at least 32 GB VRAM; ordinary inference works on the tested 24 GB cards.

Cloning uses [pyannote/embedding](https://huggingface.co/pyannote/embedding), which requires accepted access on Hugging Face. With that access, use Python 3.12 for preparation:

```sh
python3.12 -m venv .venv-clone-export
. .venv-clone-export/bin/activate
python -m pip install torch==2.7.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r native/requirements-cloning-export.txt
hf auth login
python native/tools/prepare_assets.py --out models/kitten-native \
  --cloning-only --cache-dir models/source-cache
```

Generate speech from a recording:

```sh
build-cuda/bin/kitten-tts --backend cuda --assets models/kitten-native \
  --reference-audio reference.wav --reference-transcript 'The words in the recording.' \
  --stream false --text 'New speech using this voice.' --output cloned.wav
```

Recordings must contain 2 to 30 seconds of speech and fit within 10 MiB. Supply the transcript yourself. Cloned voices support complete generation and audio streaming; incremental text sessions require a built-in voice. Cloning is disabled at engine startup unless requested.

## Model files and offline use

Preparation downloads the model and voices from [KittenML/kitten-tts-2](https://huggingface.co/KittenML/kitten-tts-2), the acoustic checkpoint from [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo), and normalization data. Keep the entire prepared directory, including `config.json`. Inference uses these files without network access. The tested 47-voice bundle occupies about 1.98 GB before optional cloning assets.

To prepare from local source files, pass `--offline --model-dir DIR --acoustic-source FILE --nemo-wheel FILE --grammar-dir DIR` to `native/tools/prepare_assets.py`, along with `--text-converter` and `--out` from the quick start. `--cache-dir` selects the download cache. Run the preparation tool with `--help` for optional cloning source files.

## C and C++

The [C API](include/kt2n/engine.h) supports generation, cancellation, incremental text, and voice creation. Use `kt2n_voice_export` and `kt2n_voice_load` to save and reuse a cloned voice with the same model bundle. [examples/stream.c](examples/stream.c) shows submission, event polling, and cleanup. Keep the engine alive across requests to reuse the loaded model. Install the standalone library with `cmake --install build-native --prefix PREFIX`; consumers can use `find_package(kt2n CONFIG REQUIRED)` and link `kt2n::engine`.

For a standalone build, configure `cmake -S native -B build-native -DKT2N_CUDA=ON -DKT2N_BUILD_TEXT=ON -DKT2N_TEXT_DEPS="$PWD/build-deps/text"`. Add `-DKT2N_BUILD_CLONING=ON -DKT2N_VOICE_AV=ON` for cloning, which requires the FFmpeg development libraries.

These are local library and CLI interfaces. The hosted HTTP, SSE, and WebSocket server routes are not included.

## Tests and licenses

Enable `-DKT2N_BUILD_TESTS=ON`, build, then run `ctest --test-dir build-native -L cpu --output-on-failure`. GPU tests use `KT2N_REQUIRE_GPU=1 ctest --test-dir build-native -L gpu --output-on-failure` and need a CUDA device. With the conversion dependencies and `pytest` installed, run `python -m pytest native/tests/assets`.

Original KittenML code uses [Apache-2.0](LICENSE). The tempo stretcher derives from LGPL-2.1-or-later FFmpeg code; other components retain their own licenses. See [NOTICE](NOTICE) and [licenses/](licenses/). The repository's llama.cpp code retains the root [MIT license](../LICENSE). Model weights retain their publishers' terms.
