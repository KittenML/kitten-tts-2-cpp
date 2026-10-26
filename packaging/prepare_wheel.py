"""Install the same Torch version for the build and the resulting wheel; with KITTEN_CUDA=1, its CUDA build."""
import os
import subprocess
import sys

cuda = os.environ.get('KITTEN_CUDA') == '1'
index = ('cu130' if sys.platform == 'win32' else 'cu129') if cuda else 'cpu'
subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.13.0',
                '--index-url', 'https://download.pytorch.org/whl/' + index], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'setuptools>=68', 'wheel',
                'cmake>=3.21', 'numpy>=1.26', 'soundfile>=0.12', 'certifi>=2024.7.4'] + (['ninja'] if cuda else []), check=True)
if sys.platform == 'win32':
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'delvewheel>=1.9'], check=True)
