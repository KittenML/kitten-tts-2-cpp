"""Repair system dependencies, leaving LibTorch supplied by the pinned torch wheel."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

wheel, destination = sys.argv[1:]
torch_lib = Path(importlib.util.find_spec('torch').origin).parent / 'lib'
env = os.environ.copy()
if sys.platform == 'linux':
    env['LD_LIBRARY_PATH'] = str(torch_lib) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    subprocess.run(['auditwheel', 'show', wheel], env=env, check=True)
    command = ['auditwheel', 'repair', '-w', destination]
    for library in sorted(torch_lib.glob('*.so*')):
        command += ['--exclude', library.name]
    command.append(wheel)
elif sys.platform == 'darwin':
    env['DYLD_LIBRARY_PATH'] = str(torch_lib) + os.pathsep + env.get('DYLD_LIBRARY_PATH', '')
    command = [sys.executable, '-m', 'delocate.cmd.delocate_wheel', '--require-archs', 'arm64', '--exclude', '/torch/lib/',
               '-w', destination, wheel]
elif sys.platform == 'win32':
    excluded = ';'.join(p.name for p in sorted(torch_lib.glob('*.dll')))
    command = [sys.executable, '-m', 'delvewheel', 'repair', '--add-path', str(torch_lib),
               '--exclude', excluded, '-w', destination, wheel]
else:
    raise RuntimeError(f'Unsupported wheel platform: {sys.platform}')
subprocess.run(command, env=env, check=True)
