# Precompiled wheels

The local `py-interface` branch contains a manually triggered wheel workflow. It builds and tests wheels as GitHub Actions artifacts; it does not publish to PyPI or create a release. The workflow has not been run remotely yet.

| Target | Python versions | Runner | Status |
| --- | --- | --- | --- |
| Linux x86-64, glibc 2.28+ | 3.10-3.14 | ubuntu-24.04 + manylinux_2_28 | Configured; local x86-64 build tested separately |
| Linux ARM64, glibc 2.28+ | 3.10-3.14 | ubuntu-24.04-arm + manylinux_2_28 | Configured; not yet runner-tested |
| macOS 14+, Apple Silicon | 3.10-3.14 | macos-14 | Configured; not yet runner-tested |
| Windows x86-64 | 3.10-3.14 | windows-2022 | Configured; not yet runner-tested |
| Windows ARM64 | 3.11-3.13 | windows-11-arm + ClangCL | Experimental; not yet runner-tested |

These are 23 CPython builds. This is not universal architecture support: Intel macOS, 32-bit platforms, Alpine/musl, mobile, WebAssembly, PyPy, and free-threaded Python are outside this matrix. The pinned PyTorch 2.13.0 distribution does not provide compatible wheels for most of these targets. The CPU index has s390x builds, but this project does not yet have an s390x build/test runner.

Sources for the matrix: [PyTorch CPU wheels](https://download.pytorch.org/whl/cpu/torch/), [cibuildwheel configuration](https://cibuildwheel.pypa.io/en/stable/options/), and [GitHub runner availability](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

## Building

After the workflow is available in GitHub, run **Python wheels** manually. If the normalizer repository is private, provide `KITTEN_SUBMODULE_TOKEN` with read access to both repositories. Download the per-platform artifacts after the jobs pass. No workflow has been pushed or triggered as part of this implementation.

Every wheel is installed into a separate test environment. The smoke test checks the Torch ABI version, packaged grammar data, native normalization, TorchScript loading/decoding, NumPy output, and an HTTPS config download without model credentials. Select the `model_test` workflow input for an additional real W4 generation on every wheel; private models require an `HF_TOKEN` Actions secret. A full release should pass that test on each platform first.

To build a single target locally:

```sh
python -m pip install cibuildwheel==4.3.0
python -m cibuildwheel --only cp311-manylinux_x86_64 --output-dir wheelhouse
```

Linux cibuildwheel builds require Docker or Podman. macOS and Windows builds run on those operating systems with their compiler toolchains. Cross-platform wheel filenames cannot be obtained by renaming an existing wheel.

For a native portable build without containers:

```sh
KITTEN_PORTABLE=1 CMAKE_ARGS='-DLLAMA_BUILD_LIBRESSL=ON' \
  python setup.py build --build-base build-portable bdist_wheel
```

This produces a wheel for the build host. On Linux, use `packaging/repair_wheel.py` with auditwheel/patchelf installed to repair and assign its actual glibc compatibility tag. A wheel built on a newer host cannot simply be labeled manylinux_2_28. The local CPython 3.11 test produced a manylinux_2_35_x86_64 wheel, not the CI matrix's manylinux_2_28 target.

## Installing an artifact

Install the matching CPU PyTorch runtime, then the wheel for your platform and Python version:

```sh
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install /path/to/kitten_tts_cpp-0.1.0-<python>-<abi>-<platform>.whl
```

No compiler, CMake, or exporter is needed to install the wheel. Model assets are still downloaded from the model repository on first use. The wheel includes the normalizer data and statically linked LibreSSL. It uses the separately installed PyTorch libraries rather than bundling another copy. Repair tools handle other native dependencies. HTTPS uses the certifi CA bundle by default; `SSL_CERT_FILE` or the constructor's `ca_file` parameter can select a custom trust bundle. This is why the wheel requires the exact PyTorch release it was compiled against and checks it before loading the native extension.

Portable builds disable host-specific instruction flags, including AVX/AVX2/AVX-512 and AMX on x86, and use baseline GGML kernels. They prioritize compatibility and may be slower than local CPU-optimized builds. LibTorch retains its own CPU optimizations. Rebuild from source without `KITTEN_PORTABLE=1` for native CPU optimizations; quantized arithmetic can differ between kernel paths.
