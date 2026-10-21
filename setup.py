"""Build the KittenTTS2 CPU extension with CMake during package installation."""
import os
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
            '-DLLAMA_BUILD_KITTEN_TTS=ON', '-DKITTEN_BUILD_PYTHON=ON',
            '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF',
            '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_MTMD=OFF', '-DLLAMA_BUILD_APP=OFF',
            '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF', '-DGGML_BACKEND_DL=OFF',
            '-DGGML_NATIVE=ON', '-DLLAMA_OPENSSL=ON',
            f'-DPython3_EXECUTABLE={sys.executable}',
            f'-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}',
            f'-DKITTEN_PYTHON_OUTPUT_DIR={output}',
        ]
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
        subprocess.run([*cmake, '--build', str(build), '--config', 'Release', '--target', '_native',
                        '--parallel', os.environ.get('CMAKE_BUILD_PARALLEL_LEVEL', '4')], check=True)
        data = build / 'tools/kitten-tts/text-processing/data'
        if not (data / 'en/tagger.fst').is_file():
            raise RuntimeError('Native normalizer grammar data was not generated')
        shutil.copytree(data, output / 'data', dirs_exist_ok=True)
        (output / '_build_info.json').write_text(json.dumps({
            'torch_version': TORCH_VERSION, 'portable': portable,
        }, indent=2) + '\n')
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
    name='kitten-tts-cpp', version='0.1.0',
    description='CPU KittenTTS2 inference with native C++ bindings',
    long_description=(ROOT / 'README.md').read_text(), long_description_content_type='text/markdown',
    author='KittenML', url='https://github.com/KittenML/kitten-tts-2-cpp',
    python_requires='>=3.10',
    packages=['kitten_tts_cpp'], package_dir={'': 'python'},
    ext_modules=[Extension('kitten_tts_cpp._native', sources=[])],
    cmdclass={'build_ext': CMakeBuild},
    install_requires=['numpy>=1.26', f'torch=={TORCH_VERSION}', 'soundfile>=0.12', 'certifi>=2024.7.4'],
    include_package_data=True, zip_safe=False,
)
