# Precompiled wheels

The `Python wheels` workflow runs on pushes to `py-interface` and supports manual dispatch. It builds and tests wheels as GitHub Actions artifacts; it does not publish to PyPI or create a release.

CI status below is from the [2026-10-22 run](https://github.com/haaziq-sl/kitten-tts-2-cpp/actions/runs/39751306869).

| Target | Python versions | Runner | Status |
| --- | --- | --- | --- |
| Linux x86-64, glibc 2.34+ | 3.10-3.14 | ubuntu-24.04 + manylinux_2_34 | CI smoke passed |
| Linux ARM64, glibc 2.34+ | 3.10-3.14 | ubuntu-24.04-arm + manylinux_2_34 | CI smoke passed |
| macOS 14+, Apple Silicon | 3.10-3.14 | macos-14 | CI smoke passed |
| Windows x86-64 | 3.10-3.14 | windows-2022 | CI smoke passed |
| Windows ARM64 | 3.11-3.13 | windows-11-arm + ClangCL | Experimental; CI smoke passed |

These are 23 CPython builds. This is not universal architecture support: Intel macOS, 32-bit platforms, Alpine/musl, mobile, WebAssembly, PyPy, and free-threaded Python are outside this matrix. The pinned PyTorch 2.13.0 distribution does not provide compatible wheels for most of these targets. The CPU index has s390x builds, but this project does not yet have an s390x build/test runner.

Sources for the matrix: [PyTorch CPU wheels](https://download.pytorch.org/whl/cpu/torch/), [cibuildwheel configuration](https://cibuildwheel.pypa.io/en/stable/options/), and [GitHub runner availability](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

### NVIDIA GPU (CUDA) wheels

The same workflow also builds the package with NVIDIA GPU support: `kitten-tts-cpp-cu12` for Linux and `kitten-tts-cpp-cu13` for Windows. The import name stays `kitten_tts_cpp`. It runs the speech model on the CPU by default, and on an NVIDIA GPU when you ask for it (see [Using the GPU](#using-the-gpu)).

| Target | Package | Python versions | Built with | Uses the CUDA libraries of |
| --- | --- | --- | --- | --- |
| Linux x86-64, glibc 2.34+ | `kitten-tts-cpp-cu12` | 3.10-3.14 | ubuntu-24.04 + manylinux_2_34, CUDA 12.8 compiler | PyTorch 2.13.0+cu129 |
| Windows x86-64 | `kitten-tts-cpp-cu13` | 3.10-3.14 | windows-2022, CUDA 13.0 compiler | PyTorch 2.13.0+cu130 |

These are 10 CPython builds, one workflow job each. The Linux compiler is CUDA 12.8 and the PyTorch runtime is CUDA 12.9; both use CUDA 12 runtime and cuBLAS libraries. Windows uses CUDA 13.0 for both because PyTorch 2.13.0 has no CUDA 12.9 build for Windows.

| | Linux (`kitten-tts-cpp-cu12`) | Windows (`kitten-tts-cpp-cu13`) |
| --- | --- | --- |
| Compiled GPU code | compute capability 6.1 (GTX 10xx), 7.5 (GTX 16xx, RTX 20xx, T4), 8.6 (RTX 30xx, A2000, A10), 8.9 (RTX 40xx, L4) and 12.0 (RTX 50xx, RTX PRO 6000) | the same without 6.1: CUDA 13 cannot compile for 6.1, so GTX 10xx cards need the Linux package or the CPU package |
| Portable GPU code | 8.0 PTX, compiled by the driver for GPUs without a matching binary and normally cached for reuse | the same |
| Waveform decoder on the GPU (`decoder_device="cuda"`) | 7.5 and newer; the PyTorch 2.13.0 CUDA 12.9 build has no code for 6.1 | 7.5 and newer |
| NVIDIA driver | one for CUDA 12; 570 or newer for 12.0 cards and for cards that use the portable code | 580 or newer (CUDA 13) |

H100 uses the PTX fallback; its performance has not been benchmarked here. To build native H100 kernels, set `KITTEN_CUDA_ARCHITECTURES=90-real` (see [Building](#building)).

The CUDA wheels do not bundle CUDA libraries. They use the CUDA runtime and cuBLAS that the matching PyTorch build installs, so install that PyTorch build first. Without an NVIDIA driver or GPU, the CUDA wheel still imports and runs on the CPU, and asking for the GPU gives a clear error.

## Building

Push to `py-interface` to start the build. Manual dispatch is also available once the workflow file exists on the default branch; select `py-interface` as the run branch. If the normalizer repository is private, provide `KITTEN_SUBMODULE_TOKEN` with read access to both repositories. Download the per-platform artifacts after the jobs pass.

Every CI-built wheel is installed into a separate test environment. The smoke test checks the Torch ABI version, packaged grammar data, native normalization, TorchScript loading/decoding, NumPy output, and an HTTPS config download without model credentials. Select the `model_test` workflow input for an additional real W4 generation on every wheel; private models require an `HF_TOKEN` Actions secret. On a GPU machine, this also tests both published CUDA decoders and offline reuse. GitHub-hosted runners have no GPU; run those checks on GPU hardware before release. For local smoke tests, `KITTEN_WHEEL_MODEL_REVISION` pins the model commit and `KITTEN_WHEEL_MODEL_CACHE` selects the cache.

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

For the CUDA package, install the tested CUDA toolkit and PyTorch combination (CUDA 12.8 and `cu129` on Linux; CUDA 13.0 and `cu130` on Windows), then add `KITTEN_CUDA=1`. On Linux:

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install setuptools wheel "cmake>=3.28" ninja numpy soundfile certifi
KITTEN_CUDA=1 KITTEN_PORTABLE=1 CMAKE_ARGS='-DLLAMA_BUILD_LIBRESSL=ON' \
  python setup.py build --build-base build-cuda bdist_wheel
```

The package name follows the CUDA major version of the installed PyTorch (`kitten-tts-cpp-cu12` or `kitten-tts-cpp-cu13`). `KITTEN_CUDA_ARCHITECTURES` overrides the GPU architectures (default `61-real;75-real;80-virtual;86-real;89-real;120a-real` for CUDA 12, the same without `61-real` for CUDA 13).

This produces a wheel for the build host. On Linux, use `packaging/repair_wheel.py` with auditwheel/patchelf installed to repair and assign its actual glibc compatibility tag. A wheel built on a newer host cannot simply be labeled manylinux_2_28. The local CPython 3.11 test produced a manylinux_2_35_x86_64 wheel, not the CI matrix's manylinux_2_34 target.

## Installing an artifact

Install the matching CPU PyTorch runtime, then the wheel for your platform and Python version:

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install /path/to/kitten_tts_cpp-<version>-<python>-<abi>-<platform>.whl
```

For the GPU package, install the CUDA build of PyTorch instead, then the GPU wheel. Use a virtual environment: `python3 -m venv .venv` and `. .venv/bin/activate` on Linux (recent distributions such as Ubuntu 24.04 do not let pip install into the system Python), or `py -m venv .venv` and `.venv\Scripts\activate` on Windows. On Linux:

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install /path/to/kitten_tts_cpp_cu12-<version>-<python>-<abi>-<platform>.whl
```

On Windows, PyTorch needs the [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe). A new Windows installation may not have it; PyTorch then fails to import with `WinError 126` for `c10.dll`. Then:

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install /path/to/kitten_tts_cpp_cu13-<version>-<python>-<abi>-win_amd64.whl
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

The first CUDA call on Windows can take over a minute; keep the model open for faster repeated requests.

The options match the command-line runner's flags, and invalid values give the same errors:

| Option | Default | Meaning |
| --- | --- | --- |
| `gpu_layers` | `0` | `0` keeps the speech model on the CPU (same output as the CPU package). `99` or `"all"` puts the whole model on the GPU. Smaller values need `allow_hybrid=True`. |
| `main_gpu` | `0` | Which GPU to use. |
| `flash_attn` | `"auto"` | `"auto"`, `"on"` or `"off"`. |
| `allow_hybrid` | `False` | Accept a model that is only partly on the GPU. |
| `decoder_device` | `"cpu"` | `"cuda"` runs the waveform decoder on the GPU, with the decoder's CUDA export downloaded from the model repository (`default` and `student_w4`; `student_w8` has none). |

For the recommended full GPU setup, add `decoder_device="cuda"`: `KittenTTS("KittenML/kitten-tts-2", decoder="student_w4", gpu_layers=99, decoder_device="cuda")`. The CUDA decoder downloads automatically; no export step is needed.

`last_report["lm"]` lists where each part runs. On a machine without an NVIDIA GPU or driver, `gpu_layers` above 0 stops with "needs a visible CUDA GPU" and `gpu_layers=0` runs on the CPU. GPU results differ slightly from CPU results because the GPU computes with different precision; see the detailed guide.

### Supported, but not the fastest setup

These setups work and give correct audio, but they are slower than running both the speech model and the waveform decoder on the GPU. Real-time factor (RTF) is generation time divided by audio length, for one short sentence with the Linux GPU wheel on an RTX 6000 Ada (default decoder, 8 CPU threads); lower is faster.

- **Waveform decoder on the CPU** (`decoder_device="cpu"`, the default): the decoder becomes the slowest step. RTF 0.48 against 0.12 with the decoder on the GPU. It is the only choice on compute capability 6.1 (GTX 10xx).
- **Speech model only partly on the GPU** (`gpu_layers` below 29 with `allow_hybrid=True`): the layers left on the CPU run at CPU speed. Use it only when the whole model does not fit in GPU memory.
- **CPU only** (`gpu_layers=0`): RTF 3.41 on the same machine.
