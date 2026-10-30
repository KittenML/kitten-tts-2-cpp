#!/usr/bin/env python3
"""Exercise native asset downloads against a local model repository."""
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

binary = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    repo = root / 'server/org/model/resolve/main'
    repo.mkdir(parents=True)
    requests = []
    heads = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def end_headers(self):
            path = Path(self.translate_path(self.path))
            if path.is_file():
                self.send_header('ETag', '"' + hashlib.sha256(path.read_bytes()).hexdigest() + '"')
            super().end_headers()

        def do_HEAD(self):
            heads.append(self.path)
            super().do_HEAD()

        def do_GET(self):
            requests.append(self.path)
            super().do_GET()

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(root / 'server')))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = dict(os.environ, HF_ENDPOINT=f'http://127.0.0.1:{server.server_port}', HF_TOKEN='')
    config = {'type': 'KITTEN2', 'cpp': {'version': 1, 'gguf': {'file': 'cpp/model.gguf', 'size': 4}, 'decoders': {}}}
    for name in ('default', 'student_w4', 'student_w8'):
        config['cpp']['decoders'][name] = {
            'torchscript': {'file': f'cpp/{name}/decoder.pt', 'size': 4},
            'voices': {'file': f'cpp/{name}/voices.json', 'size': 4},
        }
        if name != 'student_w8':
            config['cpp']['decoders'][name]['torchscript_cuda'] = {'file': f'cpp/{name}/decoder-cuda.pt', 'size': 4}
    for entry in [config['cpp']['gguf']] + [entry for decoder in config['cpp']['decoders'].values() for entry in decoder.values()]:
        path = repo / entry['file']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'test')

    def run(*args, ok=True, cache='cache'):
        (repo / 'config.json').write_text(json.dumps(config))
        result = subprocess.run([binary, '--repo', 'org/model', '--cache-dir', str(root / cache), '--download-only', *args], env=env, capture_output=True, text=True)
        assert (result.returncode == 0) == ok, result.stderr
        return result

    try:
        run('--decoder', 'student_w4')
        assert requests == ['/org/model/resolve/main/' + f for f in ('config.json', 'cpp/student_w4/voices.json', 'cpp/model.gguf', 'cpp/student_w4/decoder.pt')], requests
        requests.clear()
        heads.clear()
        run('--decoder', 'student_w4', '--offline')
        assert not requests and not heads
        profile = root / 'profile'
        result = subprocess.run([binary, '--repo', 'org/model', '--download-only', '--decoder', 'student_w4'],
                                env={**env, 'HOME': '', 'XDG_CACHE_HOME': '', 'USERPROFILE': str(profile)},
                                cwd=root, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert (profile / '.cache/kitten-tts/org/model/main/config.json').is_file()
        run('--decoder', 'student_w8', '--offline', ok=False)
        run('--decoder', 'missing', ok=False)
        requests.clear()
        run('--decoder', 'student_w4', '--decoder-device', 'cuda', cache='cuda')
        assert requests == ['/org/model/resolve/main/' + f for f in ('config.json', 'cpp/student_w4/voices.json', 'cpp/model.gguf', 'cpp/student_w4/decoder-cuda.pt')], requests
        requests.clear()
        heads.clear()
        run('--decoder', 'student_w4', '--decoder-device', 'cuda', '--offline', cache='cuda')
        assert not requests and not heads
        cuda_entry = config['cpp']['decoders']['student_w4'].pop('torchscript_cuda')
        run('--decoder', 'student_w4', cache='upgrade')
        config['cpp']['decoders']['student_w4']['torchscript_cuda'] = cuda_entry
        run('--decoder', 'student_w4', '--decoder-device', 'cuda', cache='upgrade')
        assert (root / 'upgrade/org/model/main/cpp/student_w4/decoder-cuda.pt').read_bytes() == b'test'
        cuda_entry['size'] = 5
        result = run('--decoder', 'student_w4', '--decoder-device', 'cuda', cache='cuda-size', ok=False)
        assert 'asset size mismatch' in result.stderr, result.stderr
        cuda_entry['size'] = 4
        requests.clear()
        result = run('--decoder', 'student_w8', '--decoder-device', 'cuda', cache='cuda-missing', ok=False)
        assert 'no CUDA export' in result.stderr and requests == ['/org/model/resolve/main/config.json'], (result.stderr, requests)
        requests.clear()
        run('--tokens-only', cache='tokens')
        assert not any(path.endswith('decoder.pt') for path in requests)
        requests.clear()
        run('--decode-tokens', 'unused.json', cache='decode')
        assert not any(path.endswith('.gguf') for path in requests)
        requests.clear()
        run('--model', '/unused/override.gguf', cache='override')
        assert not any(path.endswith('.gguf') for path in requests)
        run('--revision', '../escape', ok=False)
        config['cpp']['gguf']['file'] = '../escape'
        run(cache='unsafe', ok=False)
        config['cpp']['gguf']['file'] = 'cpp/missing.gguf'
        run(cache='missing', ok=False)
        config['cpp']['gguf']['file'] = 'cpp/model.gguf'
        config['cpp']['gguf']['size'] = 5
        run(cache='size', ok=False)
        config['cpp']['version'] = 2
        run(cache='version', ok=False)
        del config['cpp']
        run(cache='legacy', ok=False)
        print('Asset download tests passed')
    finally:
        server.shutdown()
        server.server_close()
