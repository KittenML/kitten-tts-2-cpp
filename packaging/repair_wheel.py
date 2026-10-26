"""Repair system dependencies, leaving LibTorch supplied by the pinned torch wheel.

CUDA wheels (KITTEN_CUDA=1) also leave out the CUDA runtime and cuBLAS, which the CUDA build of torch installs,
and the driver library, which comes with the NVIDIA driver.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

wheel, destination = sys.argv[1:]
torch_lib = Path(importlib.util.find_spec('torch').origin).parent / 'lib'
cuda = os.environ.get('KITTEN_CUDA') == '1'
env = os.environ.copy()
if sys.platform == 'linux':
    paths = [torch_lib]
    if cuda:
        nvidia = torch_lib.parent.parent / 'nvidia'
        paths += [nvidia / 'cuda_runtime/lib', nvidia / 'cublas/lib']
    env['LD_LIBRARY_PATH'] = os.pathsep.join(map(str, paths)) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    subprocess.run(['auditwheel', 'show', wheel], env=env, check=True)
    command = ['auditwheel', 'repair', '-w', destination]
    for library in sorted(torch_lib.glob('*.so*')):
        command += ['--exclude', library.name]
    if cuda:
        import torch
        major = torch.version.cuda.split('.')[0]
        for library in ('libcuda.so.1', f'libcudart.so.{major}', f'libcublas.so.{major}', f'libcublasLt.so.{major}'):
            command += ['--exclude', library]
    command.append(wheel)
elif sys.platform == 'darwin':
    env['DYLD_LIBRARY_PATH'] = str(torch_lib) + os.pathsep + env.get('DYLD_LIBRARY_PATH', '')
    command = [sys.executable, '-m', 'delocate.cmd.delocate_wheel', '--require-archs', 'arm64', '--exclude', '/torch/lib/',
               '-w', destination, wheel]
elif sys.platform == 'win32':
    # A CUDA build of torch ships the CUDA runtime and cuBLAS DLLs in torch/lib.
    excluded = ';'.join([p.name for p in sorted(torch_lib.glob('*.dll'))] + (['nvcuda.dll'] if cuda else []))
    command = [sys.executable, '-m', 'delvewheel', 'repair', '--add-path', str(torch_lib),
               '--exclude', excluded, '-w', destination, wheel]
else:
    raise RuntimeError(f'Unsupported wheel platform: {sys.platform}')
subprocess.run(command, env=env, check=True)
