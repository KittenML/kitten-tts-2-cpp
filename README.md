# Kitten-TTS 2 C++

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

On Ubuntu/Debian, install the build dependencies and activate a Python 3.10+ environment before running the commands below. OpenSSL development files are required for HTTPS downloads.

```sh
sudo apt install build-essential python3-venv libssl-dev curl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install cmake
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

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


### Windows CPU build (PowerShell, Visual Studio 2022)

Tested on Windows 11 x64 with CPython 3.12, MSVC 14.41, Visual Studio's bundled CMake 3.29.5, and CPU PyTorch 2.14.1. Install Git, standard 64-bit CPython 3.12, and Visual Studio 2022 or Build Tools 2022 with the **Desktop development with C++** workload, a Windows SDK, and the **C++ CMake tools for Windows** component.

Start in the repository root after cloning `https://github.com/KittenML/kitten-tts-2-cpp`. Paste these commands into PowerShell. They locate the Visual Studio CMake executable, so CMake does not need to be on your ordinary PATH. The dependency environment is separate from the Python KittenTTS setup.

```powershell
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsRoot = & $vswhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$cmakeExe = Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
if (-not (Test-Path -LiteralPath $cmakeExe)) { throw 'Install the Visual Studio C++ CMake tools component.' }
git submodule update --init vendor/kitten-text-processing
if ($LASTEXITCODE -ne 0) { throw 'Submodule initialization failed.' }
py -3.12 -m venv .venv-windows-cpp
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
& .\.venv-windows-cpp\Scripts\python.exe -m pip install 'torch==2.14.1' 'huggingface_hub==1.33.0'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
$cppPython = (Resolve-Path .\.venv-windows-cpp\Scripts\python.exe).Path
$torchPrefix = & $cppPython -c 'import torch; print(torch.utils.cmake_prefix_path)'
$torchLib = & $cppPython -c "from pathlib import Path; import torch; print(Path(torch.__file__).parent / 'lib')"
$env:PATH = "$torchLib;$env:PATH"
$cmakeArgs = @(
    '-S', '.', '-B', 'build-windows',
    '-G', 'Visual Studio 17 2022', '-A', 'x64',
    '-DCMAKE_CXX_STANDARD=20',
    '-DCMAKE_CXX_STANDARD_REQUIRED=ON',
    '-DCMAKE_CXX_FLAGS=/Zc:__cplusplus /utf-8',
    '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF',
    '-DLLAMA_BUILD_KITTEN_TTS=ON',
    '-DLLAMA_BUILD_SERVER=OFF',
    '-DLLAMA_BUILD_TESTS=OFF',
    '-DLLAMA_BUILD_EXAMPLES=OFF',
    '-DLLAMA_BUILD_LIBRESSL=ON',
    "-DCMAKE_PREFIX_PATH=$torchPrefix",
    "-DPython3_EXECUTABLE=$cppPython"
)
& $cmakeExe @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
& $cmakeExe --build build-windows --config Release --target kitten-tts -j 8
if ($LASTEXITCODE -ne 0) { throw 'C++ build failed.' }
```

`/Zc:__cplusplus` enables accurate C++ standard reporting for the existing MSVC Unicode-string macros; `/utf-8` selects UTF-8 source and execution encodings. `LLAMA_BUILD_LIBRESSL=ON` builds the bundled TLS dependency if system OpenSSL development files are unavailable. It does not resolve the download failure described below.

The Visual Studio generator requires `--config Release`; `CMAKE_BUILD_TYPE` alone does not select its configuration. The executable from this Windows build is `build-windows/bin/Release/kitten-tts.exe`. Keep the build directory and the dependency environment available while using the executable. PyTorch's `torch/lib` directory must be on the current process PATH; the run commands below set it.

## Download and run

The runtime fetches `config.json` from `KittenML/kitten-tts-2` first, then downloads its GGUF, selected TorchScript decoder, and matching voice conditioning. Assets are cached under `$XDG_CACHE_HOME/kitten-tts` (or `~/.cache/kitten-tts`).

If the model reports `model config has no cpp assets`, its native asset manifest has not been published. Use this manual-download fallback for the default decoder instead of the automatic-download examples below:

```sh
base=https://huggingface.co/KittenML/kitten-tts-2/resolve/main
mkdir -p models/kitten2
curl -fL "$base/config.json" -o models/kitten2/config.json
curl -fL "$base/cpp/model-tq2_1.gguf" -o models/kitten2/model-tq2_1.gguf
for file in decoder.pt voices.json; do
  curl -fL "$base/cpp/default/$file" -o "models/kitten2/$file"
done
build/bin/kitten-tts --assets models/kitten2 --text 'Hello.' --output hello.wav
```

When using this fallback, also add `--assets models/kitten2` to subsequent synthesis commands. Other decoders require their matching asset bundles.

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


### Windows local download and run (default decoder)

At the tested model revision, the C++ automatic downloader failed on Windows with `HTTP -1`. Independently, the published `config.json` lacked the `cpp` manifest required by the automatic workflow. The published native files themselves worked. Use the Python Hugging Face client to download those files, then use the C++ runtime's local `--assets` mode. No Python exporter is required.

Save the following as `download_windows_assets.py` in the repository root:

```python
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

destination = Path("native-assets")
destination.mkdir(exist_ok=True)
files = {
    "config.json": "config.json",
    "cpp/model-tq2_1.gguf": "model-tq2_1.gguf",
    "cpp/default/decoder.pt": "decoder.pt",
    "cpp/default/voices.json": "voices.json",
}
for published, filename in files.items():
    downloaded = hf_hub_download(
        repo_id="KittenML/kitten-tts-2",
        filename=published,
        revision="baa41e5d2c5f64be0095365672a7858542261271",
        token=False,
    )
    shutil.copyfile(downloaded, destination / filename)
    print(f"Saved {destination / filename}", flush=True)
```

The pinned revision is `baa41e5d2c5f64be0095365672a7858542261271`. Its native files total about 1.60 GB, excluding LibTorch and grammar data. The download cache and the copied `native-assets` directory each retain the model files, so allow additional space for both.

From the same repository root, run:

```powershell
$cppPython = (Resolve-Path .\.venv-windows-cpp\Scripts\python.exe).Path
$torchLib = & $cppPython -c "from pathlib import Path; import torch; print(Path(torch.__file__).parent / 'lib')"
$env:PATH = "$torchLib;$env:PATH"
& $cppPython download_windows_assets.py
if ($LASTEXITCODE -ne 0) { throw 'Asset download failed.' }
& .\build-windows\bin\Release\kitten-tts.exe --assets .\native-assets --text 'Hello from Windows. This is a speech test.' --voice Bruno --output windows-cpp.wav --report windows-cpp.json
if ($LASTEXITCODE -ne 0) { throw 'Speech generation failed.' }
```

This produces mono 24 kHz `windows-cpp.wav` and a synthesis report. For subsequent runs, skip the Python download command, repeat the Python-path and DLL-PATH assignments in any new PowerShell session, and keep `--assets .\native-assets` on the C++ command.

This directory contains the default decoder. Do not add `--decoder student_w4` to this command; that requires its own decoder and matching voice-conditioning assets. Publishers must supply the missing `cpp` manifest to restore automatic model selection. The cause of the Windows `HTTP -1` failure was not established by this test.

## Generate speech

For the Windows local-assets setup above, use `build-windows/bin/Release/kitten-tts.exe`, set PyTorch's DLL PATH in the current session, and include `--assets .\native-assets` in every synthesis command below. The samples below use POSIX shell syntax; in PowerShell, write the command on one line or use PowerShell line continuations.

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
