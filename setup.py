"""Build the KittenTTS2 extension with CMake during package installation.

KITTEN_CUDA=1 builds the CUDA package, kitten-tts-cpp-cu<N>, against the CUDA build of torch installed for the
build; N is that torch's CUDA major version.
"""
import os
import platform
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
TORCH_VERSION = importlib.metadata.version('torch').split('+')[0]
CUDA = os.environ.get('KITTEN_CUDA', '0') == '1'
if CUDA:
    import torch
    CUDA_MAJOR = (torch.version.cuda or '12').split('.')[0]  # a CPU-only torch is rejected in build_extension
else:
    CUDA_MAJOR = None
# Real cubins: 6.1 Pascal (CUDA 12 only; CUDA 13 cannot compile for it), 7.5 Turing, 8.6 Ampere, 8.9 Ada, 12.0 Blackwell.
# 8.0 PTX lets other cards from 8.0 up (for example A100, H100) compile the kernels at first start.
CUDA_ARCHITECTURES = os.environ.get('KITTEN_CUDA_ARCHITECTURES',
                                    ('61-real;' if CUDA_MAJOR == '12' else '') + '75-real;80-virtual;86-real;89-real;120a-real')


class CMakeBuild(build_ext):
    def run(self):
        super().run()
        if self.inplace:
            for directory in ('data', 'licenses'):
                source = Path(self.build_lib) / 'kitten_tts_cpp' / directory
                destination = Path(self.get_ext_fullpath('kitten_tts_cpp._native')).resolve().parent / directory
                if source.resolve() != destination.resolve():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
            shutil.copyfile(Path(self.build_lib) / 'kitten_tts_cpp/_build_info.json',
                            Path(self.get_ext_fullpath('kitten_tts_cpp._native')).resolve().parent / '_build_info.json')

    def build_extension(self, ext):
        import torch
        if CUDA and torch.version.cuda is None:
            raise RuntimeError(f'KITTEN_CUDA=1 needs a CUDA build of torch=={TORCH_VERSION}; '
                               'packaging/prepare_wheel.py installs the one the release wheels use')
        if not (ROOT / 'vendor/kitten-text-processing/CMakeLists.txt').is_file():
            raise RuntimeError('Initialize vendor/kitten-text-processing with git submodule update --init before installing')
        output = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        build = Path(self.build_temp).resolve() / 'kitten'
        build.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        if importlib.util.find_spec('cmake'):
            cmake = [sys.executable, '-m', 'cmake']
        elif shutil.which('cmake'):
            cmake = [shutil.which('cmake')]
        else:
            raise RuntimeError('CMake is required; install it with pip install cmake')
        args = [
            '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=OFF',
            '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', '-DBUILD_TESTING=OFF',
            '-DLLAMA_BUILD_COMMON=ON', '-DLLAMA_BUILD_TOOLS=ON',
            '-DLLAMA_BUILD_KITTEN_TTS=ON', '-DKITTEN_BUILD_PYTHON=ON', '-DLLAMA_BUILD_KITTEN_TTS_TINY=ON',
            '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF',
            '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_MTMD=OFF', '-DLLAMA_BUILD_APP=OFF',
            f'-DGGML_CUDA={"ON" if CUDA else "OFF"}', '-DGGML_METAL=OFF', '-DGGML_BACKEND_DL=OFF',
            '-DGGML_NATIVE=ON', '-DLLAMA_OPENSSL=ON',
            f'-DPython3_EXECUTABLE={sys.executable}',
            f'-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}',
            f'-DKITTEN_PYTHON_OUTPUT_DIR={output}',
        ]
        if CUDA:
            # GGML_CUDA_NO_VMM: no link to the driver library, so the wheel also imports and runs on the CPU
            # on machines without an NVIDIA driver. The CUDA runtime and cuBLAS come from torch's dependencies.
            # NCCL is only for multi-GPU splits, which the engine does not use; a system NCCL would be bundled.
            args += ['-DGGML_CUDA_NO_VMM=ON', '-DGGML_CUDA_NCCL=OFF', f'-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCHITECTURES}']
        if (Path(sys.base_prefix) / 'include/openssl/ssl.h').is_file():
            args.append(f'-DOPENSSL_ROOT_DIR={sys.base_prefix}')
        args += shlex.split(os.environ.get('CMAKE_ARGS', ''))
        portable = os.environ.get('KITTEN_PORTABLE', '0') == '1'
        if portable:
            args += ['-DGGML_NATIVE=OFF', '-DGGML_OPENMP=OFF']
            for feature in ('SSE42', 'AVX', 'AVX2', 'AVX_VNNI', 'AVX512', 'AVX512_VBMI',
                            'AVX512_VNNI', 'AVX512_BF16', 'FMA', 'F16C', 'BMI2',
                            'AMX_TILE', 'AMX_INT8', 'AMX_BF16'):
                args.append(f'-DGGML_{feature}=OFF')
        subprocess.run([*cmake, '-S', str(ROOT), '-B', str(build), *args], check=True)
        subprocess.run([*cmake, '--build', str(build), '--config', 'Release', '--target', '_native', 'kitten-tts-tiny-lib',
                        '--parallel', os.environ.get('CMAKE_BUILD_PARALLEL_LEVEL', '4')], check=True)
        # Kitten TTS 2 mini / nano / pico: one self-contained library (ggml linked in statically), loaded with ctypes
        tiny = [p for p in build.rglob('*kitten_tts_tiny*') if p.suffix in ('.so', '.dylib', '.dll') and p.is_file()]
        if not tiny:
            raise RuntimeError('libkitten_tts_tiny was not built')
        for p in tiny:
            shutil.copyfile(p, output / p.name)
        if sys.platform == 'darwin' and platform.machine().lower() == 'arm64':
            # Apple silicon: the tiny library is built a second time with ggml's Metal backend (its shaders embedded in
            # the library and compiled at run time, so no Xcode is needed on the user's Mac) and replaces the copy above.
            # On device="metal" (and "auto") the decoder runs on the GPU; device="cpu" runs the same CPU code as before.
            # The Kitten TTS 2 (Qwen3) library above stays CPU-only.
            mtl = build.parent / (build.name + '-metal')
            mtl_args = [a for a in args if not a.startswith(('-DGGML_METAL=', '-DLLAMA_BUILD_KITTEN_TTS=', '-DKITTEN_BUILD_PYTHON='))]
            mtl_args += ['-DGGML_METAL=ON', '-DGGML_METAL_EMBED_LIBRARY=ON', '-DLLAMA_BUILD_KITTEN_TTS=OFF',
                         '-DKITTEN_BUILD_PYTHON=OFF']
            subprocess.run([*cmake, '-S', str(ROOT), '-B', str(mtl), *mtl_args], check=True)
            subprocess.run([*cmake, '--build', str(mtl), '--config', 'Release', '--target', 'kitten-tts-tiny-lib',
                            '--parallel', os.environ.get('CMAKE_BUILD_PARALLEL_LEVEL', '4')], check=True)
            built = [p for p in mtl.rglob('*kitten_tts_tiny*') if p.suffix == '.dylib' and p.is_file()]
            if not built:
                raise RuntimeError('libkitten_tts_tiny (Metal) was not built')
            for p in built:
                shutil.copyfile(p, output / p.name)
        if portable and platform.machine().lower() in ('x86_64', 'amd64'):
            # The portable build has no x86 SIMD extensions, which makes the Kitten TTS 2 mini / nano / pico models about
            # 5x slower on the CPU. Ship a second copy of the (small, CPU-only) tiny library built for x86-64-v3 (AVX2,
            # FMA, F16C, BMI2); kitten_tts_cpp.tiny loads it only on CPUs that have those extensions (in the CUDA wheel:
            # for device="cpu", and for device="auto" without a CUDA GPU).
            v3 = build.parent / (build.name + '-x86-64-v3')
            v3_args = [a for a in args if not a.startswith(('-DGGML_', '-DLLAMA_BUILD_KITTEN_TTS=', '-DKITTEN_BUILD_PYTHON=',
                                                            '-DCMAKE_CUDA_ARCHITECTURES='))]
            v3_args += ['-DGGML_CUDA=OFF', '-DGGML_METAL=OFF', '-DGGML_BACKEND_DL=OFF', '-DGGML_NATIVE=OFF', '-DGGML_OPENMP=OFF',
                        '-DLLAMA_BUILD_KITTEN_TTS=OFF', '-DKITTEN_BUILD_PYTHON=OFF', '-DLLAMA_OPENSSL=OFF', '-DLLAMA_BUILD_LIBRESSL=OFF']
            for feature in ('SSE42', 'AVX', 'AVX2', 'FMA', 'F16C', 'BMI2'):
                v3_args.append(f'-DGGML_{feature}=ON')
            for feature in ('AVX_VNNI', 'AVX512', 'AVX512_VBMI', 'AVX512_VNNI', 'AVX512_BF16', 'AMX_TILE', 'AMX_INT8', 'AMX_BF16'):
                v3_args.append(f'-DGGML_{feature}=OFF')
            subprocess.run([*cmake, '-S', str(ROOT), '-B', str(v3), *v3_args], check=True)
            subprocess.run([*cmake, '--build', str(v3), '--config', 'Release', '--target', 'kitten-tts-tiny-lib',
                            '--parallel', os.environ.get('CMAKE_BUILD_PARALLEL_LEVEL', '4')], check=True)
            for p in v3.rglob('*kitten_tts_tiny*'):
                if p.suffix in ('.so', '.dll') and p.is_file():
                    shutil.copyfile(p, output / p.name.replace('kitten_tts_tiny', 'kitten_tts_tiny_x86_64_v3'))
        data = build / 'tools/kitten-tts/text-processing/data'
        if not (data / 'en/tagger.fst').is_file():
            raise RuntimeError('Native normalizer grammar data was not generated')
        shutil.copytree(data, output / 'data', dirs_exist_ok=True)
        info = {'torch_version': TORCH_VERSION, 'portable': portable, 'cuda': CUDA}
        if CUDA:
            cache = (build / 'CMakeCache.txt').read_text(errors='replace')
            nvcc = next(line.split('=', 1)[1] for line in cache.splitlines() if line.startswith('CMAKE_CUDA_COMPILER:'))
            version = subprocess.run([nvcc, '--version'], check=True, capture_output=True, text=True).stdout
            release = next(line for line in version.splitlines() if 'release' in line)  # "..., release X.Y, VX.Y.Z"
            toolkit = release.rsplit(',', 1)[-1].strip().lstrip('V')
            info.update({'cuda_runtime': torch.version.cuda, 'cuda_toolkit': toolkit,
                         'cuda_architectures': CUDA_ARCHITECTURES})
        (output / '_build_info.json').write_text(json.dumps(info, indent=2) + '\n')
        licenses = output / 'licenses'
        licenses.mkdir(exist_ok=True)
        for relative in ('LICENSE', 'tools/kitten-tts/LICENSE-KITTENTTS',
                         'vendor/kitten-text-processing/LICENSE',
                         'vendor/kitten-text-processing/LICENSE-SACREMOSES',
                         'vendor/kitten-text-processing/NOTICE'):
            shutil.copyfile(ROOT / relative, licenses / relative.replace('/', '_'))
        for backend in ('libressl', 'boringssl'):
            for filename in ('COPYING', 'LICENSE'):
                source = build / '_deps' / f'{backend}-src' / filename
                if source.is_file():
                    shutil.copyfile(source, licenses / f'{backend}-{filename}')


setup(
    name=f'kitten-tts-cpp-cu{CUDA_MAJOR}' if CUDA else 'kitten-tts-cpp', version='0.2.1',
    description=(f'KittenTTS2 and Kitten TTS 2 mini, nano and pico inference with native C++ bindings, NVIDIA GPU '
                 f'(CUDA {CUDA_MAJOR}) build' if CUDA
                 else 'CPU KittenTTS2 and Kitten TTS 2 mini, nano and pico inference with native C++ bindings'),
    long_description=(ROOT / 'README.md').read_text(), long_description_content_type='text/markdown',
    author='KittenML', url='https://github.com/KittenML/kitten-tts-2-cpp',
    python_requires='>=3.10',
    packages=['kitten_tts_cpp', 'kitten_tts_cpp.tiny_frontend', 'kitten_tts_cpp.tiny_frontend.misaki',
              'kitten_tts_cpp.tiny_frontend.misaki.data', 'kitten_tts_cpp.tiny_frontend.num2words'],
    package_dir={'': 'python'},
    # en_core_web_sm/: spaCy's small English model (3.8.0, MIT), loaded from the package by the English front end.
    package_data={'kitten_tts_cpp.tiny_frontend': ['LICENSE-kittenml', 'NOTICE', 'en_core_web_sm/*',
                                                   'en_core_web_sm/*/*', 'en_core_web_sm/*/*/*'],
                  'kitten_tts_cpp.tiny_frontend.misaki.data': ['*.json', '*.txt'],
                  'kitten_tts_cpp.tiny_frontend.num2words': ['COPYING']},
    ext_modules=[Extension('kitten_tts_cpp._native', sources=[])],
    cmdclass={'build_ext': CMakeBuild},
    install_requires=['numpy>=1.26', f'torch=={TORCH_VERSION}', 'soundfile>=0.12', 'certifi>=2024.7.4',
                      # Kitten TTS 2 mini, nano and pico (KittenTTS2Tiny): weights download and conversion to GGUF,
                      # espeak-ng text front end (every language but English and Japanese)
                      'huggingface_hub>=0.25,<2', 'safetensors>=0.4', 'gguf>=0.19',
                      'phonemizer>=3.3', 'espeakng_loader>=0.2.4',
                      # English and Japanese front end (kittenml 0.9.8's, with the misaki 0.9.4 and num2words parts
                      # it uses vendored in tiny_frontend; its fallback BART runs in numpy): all from wheels on
                      # CPython 3.10 to 3.14; fugashi has no Windows on Arm package. Japanese downloads its
                      # dictionary on first use (tiny_frontend/unidic.py). spaCy's small English model
                      # (en_core_web_sm 3.8.0, in tiny_frontend) works with spaCy 3.8 only.
                      'spacy>=3.8,<3.9; python_version < "3.15"', 'regex>=2023.10.3; python_version < "3.15"',
                      'addict>=2.4; python_version < "3.15"', 'jaconv>=0.3; python_version < "3.15"',
                      'fugashi>=1.3.2; python_version < "3.15" and platform_machine != "ARM64"'],
    include_package_data=True, zip_safe=False,
)
