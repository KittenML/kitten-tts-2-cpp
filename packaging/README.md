# Precompiled wheels

The `Python wheels` workflow runs on pushes to `py-interface` and supports manual dispatch. It builds and tests wheels as GitHub Actions artifacts; it does not publish to PyPI or create a release.

| Target | Python versions | Runner | Status |
| --- | --- | --- | --- |
| Linux x86-64, glibc 2.34+ | 3.10-3.14 | ubuntu-24.04 + manylinux_2_34 | CI validation in progress; local x86-64 build tested |
| Linux ARM64, glibc 2.34+ | 3.10-3.14 | ubuntu-24.04-arm + manylinux_2_34 | CI validation in progress |
| macOS 14+, Apple Silicon | 3.10-3.14 | macos-14 | CI validation in progress |
| Windows x86-64 | 3.10-3.14 | windows-2022 | CI validation in progress |
| Windows ARM64 | 3.11-3.13 | windows-11-arm + ClangCL | Experimental; CI validation in progress |

These are 23 CPython builds. This is not universal architecture support: Intel macOS, 32-bit platforms, Alpine/musl, mobile, WebAssembly, PyPy, and free-threaded Python are outside this matrix. The pinned PyTorch 2.13.0 distribution does not provide compatible wheels for most of these targets. The CPU index has s390x builds, but this project does not yet have an s390x build/test runner.

Sources for the matrix: [PyTorch CPU wheels](https://download.pytorch.org/whl/cpu/torch/), [cibuildwheel configuration](https://cibuildwheel.pypa.io/en/stable/options/), and [GitHub runner availability](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

### NVIDIA GPU (CUDA) wheels

The same workflow also builds the package with NVIDIA GPU support: `kitten-tts-cpp-cu12` for Linux and `kitten-tts-cpp-cu13` for Windows. The import name stays `kitten_tts_cpp`. It runs the speech model on the CPU by default, and on an NVIDIA GPU when you ask for it (see [Using the GPU](#using-the-gpu)).

| Target | Package | Python versions | Built with | Uses the CUDA libraries of | Tested |
| --- | --- | --- | --- | --- | --- |
| Linux x86-64, glibc 2.34+ | `kitten-tts-cpp-cu12` | 3.10-3.14 | ubuntu-24.04 + manylinux_2_34, CUDA 12.8 compiler | PyTorch 2.13.0+cu129 | CI without a GPU: import, CPU smoke test, clear "no CUDA device" error. A local build of this package (Python 3.12) also ran on an RTX PRO 6000 (12.0), a Tesla T4 (7.5) and a GTX 1080 Ti (6.1) |
| Windows x86-64 | `kitten-tts-cpp-cu13` | 3.10-3.14 | windows-2022, CUDA 13.0 compiler | PyTorch 2.13.0+cu130 | CI without a GPU: the same checks. Not tested on a Windows machine with a GPU |

These are 10 CPython builds, one workflow job each. Each wheel is compiled with the same CUDA release as the PyTorch build whose CUDA runtime and cuBLAS it uses. Windows uses CUDA 13 because PyTorch 2.13.0 has no CUDA 12.9 build for Windows.

| | Linux (`kitten-tts-cpp-cu12`) | Windows (`kitten-tts-cpp-cu13`) |
| --- | --- | --- |
| Compiled GPU code | compute capability 6.1 (GTX 10xx), 7.5 (GTX 16xx, RTX 20xx, T4), 8.6 (RTX 30xx, A2000, A10), 8.9 (RTX 40xx, L4) and 12.0 (RTX 50xx, RTX PRO 6000) | the same without 6.1: CUDA 13 cannot compile for 6.1, so GTX 10xx cards need the Linux package or the CPU package |
| Portable GPU code | 8.0, which the driver compiles when the model first loads. On 8.0 cards (A100, A30) this gives code for their own architecture. Newer cards without compiled code (for example H100) run code made for 8.0, which was 5 to 8 times slower per token on the card where it was measured (see [Supported, but not the fastest setup](#supported-but-not-the-fastest-setup)) | the same |
| Waveform decoder on the GPU (`decoder_device="cuda"`) | 7.5 and newer; the PyTorch 2.13.0 CUDA 12.9 build has no code for 6.1 | 7.5 and newer |
| NVIDIA driver | one for CUDA 12; 570 or newer for 12.0 cards and for cards that use the portable code | 580 or newer (CUDA 13) |

The CUDA wheels do not bundle CUDA libraries. They use the CUDA runtime and cuBLAS that the matching PyTorch build installs, so install that PyTorch build first. Without an NVIDIA driver or GPU, the CUDA wheel still imports and runs on the CPU, and asking for the GPU gives a clear error.

## Building

Push to `py-interface` to start the build. Manual dispatch is also available once the workflow file exists on the default branch; select `py-interface` as the run branch. If the normalizer repository is private, provide `KITTEN_SUBMODULE_TOKEN` with read access to both repositories. Download the per-platform artifacts after the jobs pass.

Every wheel is installed into a separate test environment. The smoke test checks the Torch ABI version, packaged grammar data, native normalization, TorchScript loading/decoding, NumPy output, and an HTTPS config download without model credentials. Select the `model_test` workflow input for an additional real W4 generation on every wheel; private models require an `HF_TOKEN` Actions secret. A full release should pass that test on each platform first.

To build a single target locally:

```sh
python -m pip install cibuildwheel==4.3.0
python -m cibuildwheel --only cp311-manylinux_x86_64 --output-dir wheelhouse
```

Linux cibuildwheel builds require Docker or Podman. macOS and Windows builds run on those operating systems with their compiler toolchains. Cross-platform wheel filenames cannot be obtained by renaming an existing wheel.

To build a single CUDA target locally, set the same options as the workflow's CUDA jobs (see `.github/workflows/python-wheels.yml`), for example `KITTEN_CUDA=1` and `PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu129` in `CIBW_ENVIRONMENT_LINUX`, and `CIBW_BEFORE_ALL_LINUX="sh {project}/packaging/install_cuda_linux.sh"`.

For a native portable build without containers:

```sh
KITTEN_PORTABLE=1 CMAKE_ARGS='-DLLAMA_BUILD_LIBRESSL=ON' \
  python setup.py build --build-base build-portable bdist_wheel
```

For the CUDA package, install the CUDA toolkit and the CUDA build of PyTorch that belong together (CUDA 12.8 and `cu129` on Linux; CUDA 13.0 and `cu130` on Windows), then add `KITTEN_CUDA=1`. On Linux:

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install setuptools wheel cmake ninja numpy soundfile certifi
KITTEN_CUDA=1 KITTEN_PORTABLE=1 CMAKE_ARGS='-DLLAMA_BUILD_LIBRESSL=ON' \
  python setup.py build --build-base build-cuda bdist_wheel
```

The package name follows the CUDA major version of the installed PyTorch (`kitten-tts-cpp-cu12` or `kitten-tts-cpp-cu13`). `KITTEN_CUDA_ARCHITECTURES` overrides the GPU architectures (default `61-real;75-real;80-virtual;86-real;89-real;120a-real` for CUDA 12, the same without `61-real` for CUDA 13).

This produces a wheel for the build host. On Linux, use `packaging/repair_wheel.py` with auditwheel/patchelf installed to repair and assign its actual glibc compatibility tag. A wheel built on a newer host cannot simply be labeled manylinux_2_28. The local CPython 3.11 test produced a manylinux_2_35_x86_64 wheel, not the CI matrix's manylinux_2_34 target.

## Installing an artifact

Install the matching CPU PyTorch runtime, then the wheel for your platform and Python version:

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install /path/to/kitten_tts_cpp-0.1.0-<python>-<abi>-<platform>.whl
```

For the GPU package, install the CUDA build of PyTorch instead, then the GPU wheel. On Linux:

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install /path/to/kitten_tts_cpp_cu12-0.1.0-<python>-<abi>-<platform>.whl
```

On Windows:

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install /path/to/kitten_tts_cpp_cu13-0.1.0-<python>-<abi>-win_amd64.whl
```

Install only one of the packages in an environment; all of them provide `kitten_tts_cpp`. The GPU package stops at import with an explanation if the installed PyTorch is a CPU-only build or was built for another CUDA major version.

No compiler, CMake, or exporter is needed to install the wheel. Model assets are still downloaded from the model repository on first use. The wheel includes the normalizer data and statically linked LibreSSL. It uses the separately installed PyTorch libraries rather than bundling another copy. Repair tools handle other native dependencies. HTTPS uses the certifi CA bundle by default; `SSL_CERT_FILE` or the constructor's `ca_file` parameter can select a custom trust bundle. This is why the wheel requires the exact PyTorch release it was compiled against and checks it before loading the native extension.

Portable builds disable host-specific instruction flags, including AVX/AVX2/AVX-512 and AMX on x86, and use baseline GGML kernels. They prioritize compatibility and may be slower than local CPU-optimized builds. LibTorch retains its own CPU optimizations. Rebuild from source without `KITTEN_PORTABLE=1` for native CPU optimizations; quantized arithmetic can differ between kernel paths.

## Using the GPU

With the GPU package, pass `gpu_layers=99` to run the whole speech model on the GPU:

```python
from kitten_tts_cpp import KittenTTS

with KittenTTS("KittenML/kitten-tts-2", decoder="student_w4", gpu_layers=99) as model:
    audio = model.generate("Hello from the GPU.", voice="Bruno", advanced={"seed": 1234})
    print(model.last_report["lm"]["placement"])  # "gpu"
```

The options match the command-line runner's flags, and invalid values give the same errors:

| Option | Default | Meaning |
| --- | --- | --- |
| `gpu_layers` | `0` | `0` keeps the speech model on the CPU (same output as the CPU package). `99` or `"all"` puts the whole model on the GPU. Smaller values need `allow_hybrid=True`. |
| `main_gpu` | `0` | Which GPU to use. |
| `flash_attn` | `"auto"` | `"auto"`, `"on"` or `"off"`. |
| `allow_hybrid` | `False` | Accept a model that is only partly on the GPU. |
| `decoder_device` | `"cpu"` | `"cuda"` runs the waveform decoder on the GPU. It needs a decoder exported for CUDA in a local asset directory (see the [detailed guide](../tools/kitten-tts/README.md#cuda-waveform-decoder-optional-local-export)); the published decoders run on the CPU. |

`last_report["lm"]` lists where each part runs. On a machine without an NVIDIA GPU or driver, `gpu_layers` above 0 stops with "needs a visible CUDA GPU" and `gpu_layers=0` runs on the CPU. GPU results differ slightly from CPU results because the GPU computes with different precision; see the detailed guide.

### Supported, but not the fastest setup

These setups work and give correct audio, but they are slower than running both the speech model and the waveform decoder on the GPU. The numbers are for one short sentence (about 5 seconds of audio, voice Bruno, seed 1234, default decoder, second call, 8 CPU threads) with the Linux GPU wheel on otherwise idle machines; "real-time factor" is generation time divided by audio length, so below 1 is faster than real time.

- **Waveform decoder on the CPU.** This is the default (`decoder_device="cpu"`), because the published decoders are prepared for the CPU, and it is the only choice on compute capability 6.1 (GTX 10xx), where the PyTorch CUDA build has no GPU code. The speech model still runs on the GPU, but the decoder becomes the slowest step. Tesla T4: decoder 5.0 s on the CPU against 0.2 s on the GPU, real-time factor 1.23 against 0.31 (speech model 1.5 s and 1.4 s). GTX 1080 Ti: speech model 1.4 s and decoder 5.0 s, real-time factor 1.18; with everything on the CPU the speech model alone takes 36 s (real-time factor 7.9).
- **Speech model only partly on the GPU** (`gpu_layers` below 29 with `allow_hybrid=True`): the layers left on the CPU run at CPU speed. GTX 1080 Ti with `gpu_layers=14`: speech model 16.4 s, against 1.4 s with every layer on the GPU and 36 s on the CPU (real-time factor 4.1). Use it only when the whole model does not fit in GPU memory.
- **Cards that run the portable code** (cards newer than 8.0 without compiled code in the wheel, for example H100): the driver compiles the 8.0 code when the model first loads, and the card then runs code made for 8.0, not for its own architecture. This costs speed on every call, not only at the start. An RTX PRO 6000 forced onto the portable code (`CUDA_FORCE_PTX_JIT=1`) ran the speech model 5 to 8 times slower per token in two runs: 19.2 and 18.9 ms per token on the second call (about 52 tokens per second) against 3.7 ms (272 tokens per second) with its compiled 12.0 code, and 16.3 and 18.7 ms against 2.4 ms on the first call. With CPU sampling included, the speech model took 3.0 and 2.8 s against 1.1 s on the second call. Loading took about as long as with the compiled code. These numbers come from a shared machine (load average 26-33, decoder `student_w4`), unlike the others in this section, and nothing was measured on an H100. On 8.0 cards (A100, A30) the compiled result is code for their own architecture, so only the first load can be slower (not measured). For full speed on a card without compiled code, build the package from source with its architecture, for example `KITTEN_CUDA_ARCHITECTURES=90-real` for an H100 (see [Building](#building)).

