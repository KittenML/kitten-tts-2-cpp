"""Install the same CPU Torch version for the build and the resulting wheel."""
import subprocess
import sys

subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.13.0',
                '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'setuptools>=68', 'wheel',
                'cmake>=3.21', 'numpy>=1.26', 'soundfile>=0.12', 'certifi>=2024.7.4'], check=True)
if sys.platform == 'win32':
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'delvewheel>=1.9'], check=True)
