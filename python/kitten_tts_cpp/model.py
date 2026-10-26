import json
import math
import numbers
import os
from pathlib import Path
import queue
import sys
import threading

import numpy as np
import certifi
import torch  # Load LibTorch's shared dependencies before the extension.

_build_info = Path(__file__).with_name('_build_info.json')
_info = json.loads(_build_info.read_text()) if _build_info.is_file() else {}
if _info:
    _required_torch = _info['torch_version']
    if torch.__version__.split('+')[0] != _required_torch:
        raise ImportError(f'This kitten-tts-cpp wheel requires torch=={_required_torch}; install that version or rebuild from source')
    if _info.get('cuda'):
        # The extension uses the CUDA runtime and cuBLAS that a CUDA build of torch installs.
        _major = _info.get('cuda_runtime', '12').split('.')[0]
        _package = f'kitten-tts-cpp-cu{_major}'
        _index = 'cu' + _info.get('cuda_runtime', f'{_major}.0').replace('.', '')
        _install = (f'pip install torch=={_required_torch} --index-url https://download.pytorch.org/whl/{_index}, '
                    'or use the CPU package kitten-tts-cpp')
        if torch.version.cuda is None:
            raise ImportError(f'{_package} needs a CUDA {_major} build of torch=={_required_torch}, but the installed '
                              f'torch {torch.__version__} is CPU-only. Install the CUDA build with {_install}')
        if torch.version.cuda.split('.')[0] != _major:
            raise ImportError(f'{_package} needs a CUDA {_major} build of torch=={_required_torch}, but the installed '
                              f'torch {torch.__version__} uses CUDA {torch.version.cuda}. Install the CUDA {_major} build with {_install}')
if sys.platform == 'win32':
    # LibTorch's DLLs, and for a CUDA torch the CUDA runtime and cuBLAS, are in torch/lib.
    _dll_directory = os.add_dll_directory(str(Path(torch.__file__).parent / 'lib'))

from . import _native

_ADVANCED = {
    'seed': None, 'repetition_penalty': 1.1, 'repetition_window': None,
    'token_run_penalty': 1.3, 'token_run_grace': 10, 'chunk_chars': None,
    'chunk_min_chars': None, 'chunk_gap_s': 0.16, 'use_emotion': None,
}
_OPTIONS = {
    'seed': 'seed', 'repetition_penalty': 'repetition-penalty',
    'repetition_window': 'repetition-window', 'token_run_penalty': 'run-penalty',
    'token_run_grace': 'run-grace', 'chunk_chars': 'chunk-chars',
    'chunk_min_chars': 'chunk-min-chars', 'chunk_gap_s': 'chunk-gap',
    'use_emotion': 'use-emotion',
}


def _integer(name, value, minimum=0, maximum=2147483647):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in {minimum}..{maximum}')


def _finite(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')


class KittenTTS2:
    """Persistent model. Uses the same native engine as the kitten-tts CLI.

    model_name is a Hugging Face repository or a local native asset directory.
    Raw recording cloning and Python checkpoint weight formats are not supported.

    The speech LM runs on the CPU by default. The CUDA package (kitten-tts-cpp-cu12 on Linux,
    kitten-tts-cpp-cu13 on Windows) can run it on an NVIDIA GPU: gpu_layers=99 (or 'all') offloads
    the whole LM, main_gpu selects the GPU, flash_attn is
    'auto', 'on' or 'off', and allow_hybrid=True accepts weights left on the CPU when gpu_layers is
    below 29. decoder_device='cuda' runs the waveform decoder on the GPU with the decoder's CUDA export
    (downloaded like the other assets; student_w8 has none).
    These options are checked like the CLI's --gpu-layers, --main-gpu, --flash-attn, --allow-hybrid
    and --decoder-device. last_report['lm']['placement'] is 'cpu', 'gpu' or 'hybrid'.
    """

    def __init__(self, model_name='KittenML/kitten-tts-2', cache_dir=None,
                 device=None, hf_token=None, decoder=None, weights='packed', *,
                 threads=8, decoder_threads=None, revision=None, offline=False,
                 model_path=None, data_dir=None, repack=True, ca_file=None,
                 gpu_layers=0, main_gpu=0, decoder_device='cpu', flash_attn='auto', allow_hybrid=False):
        if device not in (None, 'cpu', torch.device('cpu')):
            raise ValueError('device must be None or "cpu"; use gpu_layers and decoder_device="cuda" for an NVIDIA GPU')
        if weights != 'packed':
            raise ValueError('Only weights="packed" is supported; use model_path for a custom GGUF')
        _integer('threads', threads, 1)
        decoder_threads = threads if decoder_threads is None else decoder_threads
        _integer('decoder_threads', decoder_threads, 1)
        args = {'--threads': str(threads), '--decoder-threads': str(decoder_threads),
                '--data': str(data_dir or Path(__file__).parent / 'data'),
                '--ca-file': str(ca_file or os.environ.get('SSL_CERT_FILE') or certifi.where())}
        model_name = os.fspath(model_name)
        if Path(model_name).is_dir():
            if revision is not None:
                raise ValueError('revision applies only to a Hugging Face repository')
            args['--assets'] = model_name
        else:
            args['--repo'] = model_name
            if revision is not None:
                args['--revision'] = revision
        for key, value in [('cache-dir', cache_dir), ('decoder', decoder),
                           ('hf-token', hf_token), ('model', model_path)]:
            if value is not None:
                args['--' + key] = str(value)
        if offline:
            args['--offline'] = '1'
        if not repack:
            args['--no-repack'] = '1'
        if not isinstance(allow_hybrid, bool):
            raise ValueError('allow_hybrid must be a boolean')
        # Passed as text, so the native engine checks the values with the CLI's rules and messages.
        args.update({'--gpu-layers': str(gpu_layers), '--main-gpu': str(main_gpu),
                     '--decoder-device': str(decoder_device), '--flash-attn': str(flash_attn)})
        if allow_hybrid:
            args['--allow-hybrid'] = '1'
        self._streams_lock = threading.Lock()
        self._stops = set()
        self._lock = threading.RLock()
        self._engine = _native.Engine(args)
        metadata = json.loads(self._engine.metadata())
        self.config = metadata['config']
        self._voices = metadata['voices']
        self.placement = metadata.get('lm', {}).get('placement', 'cpu')
        self.decoder_device = metadata.get('decoder_device', 'cpu')
        self.device = 'cuda' if self.placement != 'cpu' or self.decoder_device == 'cuda' else 'cpu'
        self.decoder = decoder or self.config.get('cpp_decoder', self.config.get('default_decoder', 'default'))
        self.last_report = None

    @property
    def sample_rate(self):
        return self.config.get('sample_rate', 24000)

    @property
    def available_voices(self):
        return list(self._voices)

    @property
    def available_decoders(self):
        return list(self.config.get('cpp', {}).get('decoders', {self.decoder: None}))

    @property
    def available_weights(self):
        return ['packed']

    @property
    def advanced_options(self):
        result = dict(_ADVANCED)
        generation = self.config.get('generation', {})
        for name, fallback in [('repetition_window', 0), ('chunk_chars', 380), ('chunk_min_chars', 130)]:
            result[name] = generation.get(name, fallback)
        return result

    def _options(self, text, voice=None, reference=None, reference_text=None,
                 preset=None, max_new_tokens=1000, normalize=True,
                 use_reference_prompt=None, temperature=None, top_k=None,
                 top_p=None, min_p=None, advanced=None):
        if reference is not None and voice is not None:
            raise ValueError('give either voice= or reference=, not both')
        if reference_text is not None and reference is None:
            raise ValueError('reference_text requires reference')
        if reference is not None:
            raise NotImplementedError('Raw-audio cloning requires offline prepare_reference.py; select the prepared voice by name')
        if not isinstance(text, str):
            raise TypeError('text must be a string')
        if not text.strip():
            raise ValueError('no text to speak')
        voice = voice or self.config.get('default_voice', 'Bruno')
        if voice not in self._voices:
            raise ValueError(f'unknown voice: {voice}')
        preset = preset or self.config.get('default_preset', 'stable')
        if preset not in self.config['decode_presets']:
            raise ValueError(f'unknown preset: {preset}')
        _integer('max_new_tokens', max_new_tokens, 1)
        opts = dict(self.advanced_options)
        if advanced is not None:
            unknown = set(advanced) - set(opts)
            if unknown:
                raise ValueError(f'unknown advanced options: {sorted(unknown)}')
            opts.update({key: value for key, value in advanced.items() if value is not None})
        for name in ('chunk_chars', 'chunk_min_chars'):
            if opts[name] is None:
                opts[name] = self.advanced_options[name]
        if opts['repetition_window'] is None:
            opts['repetition_window'] = 0
        for name, minimum in [('seed', 0), ('repetition_window', 0), ('token_run_grace', 1), ('chunk_chars', 1), ('chunk_min_chars', 0)]:
            if opts[name] is not None:
                _integer(name, opts[name], minimum)
        for name in ('repetition_penalty', 'token_run_penalty', 'chunk_gap_s'):
            _finite(name, opts[name])
        if opts['repetition_penalty'] <= 0 or opts['token_run_penalty'] < 0 or not 0 <= opts['chunk_gap_s'] <= 60:
            raise ValueError('invalid repetition, run penalty, or chunk gap')
        for name, value in [('normalize', normalize), ('use_reference_prompt', use_reference_prompt), ('use_emotion', opts['use_emotion'])]:
            if (value is None and name != 'normalize') or isinstance(value, bool):
                continue
            raise ValueError(f'{name} must be a boolean' + ('' if name == 'normalize' else ' or None'))
        if top_k is not None:
            _integer('top_k', top_k)
        for name, value in [('temperature', temperature), ('top_p', top_p), ('min_p', min_p)]:
            if value is not None:
                _finite(name, value)
        if temperature is not None and temperature <= 0:
            raise ValueError('temperature must be positive')
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError('top_p must be in (0, 1]')
        if min_p is not None and not 0 <= min_p <= 1:
            raise ValueError('min_p must be in [0, 1]')
        args = {'--text': text, '--voice': voice, '--preset': preset, '--max-tokens': str(max_new_tokens)}
        if not normalize:
            args['--no-normalize'] = '1'
        if use_reference_prompt is not None:
            args['--use-reference'] = str(int(use_reference_prompt))
        for name, value in [('temperature', temperature), ('top-k', top_k), ('top-p', top_p), ('min-p', min_p)]:
            if value is not None:
                args['--' + name] = str(value)
        for name, value in opts.items():
            if value is not None:
                args['--' + _OPTIONS[name]] = str(int(value)) if isinstance(value, bool) else str(value)
        return args

    def _run(self, args, callback=None):
        with self._lock:
            if self._engine is None:
                raise RuntimeError('model is closed')
            audio, report = self._engine.generate(args, callback)
            self.last_report = json.loads(report)
            return audio

    def generate(self, text, voice=None, reference=None, reference_text=None,
                 preset=None, max_new_tokens=1000, normalize=True,
                 use_reference_prompt=None, temperature=None, top_k=None,
                 top_p=None, min_p=None, advanced=None):
        """Return a mono float32 NumPy array at sample_rate."""
        args = self._options(text, voice, reference, reference_text, preset, max_new_tokens,
                             normalize, use_reference_prompt, temperature, top_k, top_p, min_p, advanced)
        return self._run(args)

    def generate_stream(self, text, voice=None, reference=None, reference_text=None,
                        preset=None, max_new_tokens=1000, normalize=True,
                        use_reference_prompt=None, temperature=None, top_k=None,
                        top_p=None, min_p=None, advanced=None):
        """Yield untrimmed audio chunks as decoded, like KittenTTS2's Python API.

        Close the iterator if stopping early. Cancellation completes the current chunk.
        """
        args = self._options(text, voice, reference, reference_text, preset, max_new_tokens,
                             normalize, use_reference_prompt, temperature, top_k, top_p, min_p, advanced)
        messages = queue.Queue()
        stopped = threading.Event()

        def put(kind, value):
            if stopped.is_set():
                return False
            messages.put((kind, value))
            return True

        def worker():
            try:
                self._run(args, lambda audio: put('audio', audio))
            except Exception as exc:
                put('error', exc)
            finally:
                put('done', None)

        with self._streams_lock:
            self._stops.add(stopped)
        thread = threading.Thread(target=worker, name='kitten-tts-cpp-stream', daemon=True)
        thread.start()
        try:
            while True:
                try:
                    kind, value = messages.get(timeout=0.05)
                except queue.Empty:
                    if stopped.is_set():
                        break
                    continue
                if kind == 'done':
                    break
                if kind == 'error':
                    raise value
                yield value
        finally:
            stopped.set()
            thread.join()
            with self._streams_lock:
                self._stops.discard(stopped)

    def generate_to_file(self, text, output_path, **kwargs):
        import soundfile as sf
        audio = self.generate(text, **kwargs)
        sf.write(str(output_path), audio, self.sample_rate)
        return str(output_path)

    def normalize_text(self, text):
        with self._lock:
            if self._engine is None:
                raise RuntimeError('model is closed')
            return self._engine.normalize(text)

    def close(self):
        with self._streams_lock:
            for stopped in self._stops:
                stopped.set()
        with self._lock:
            self._engine = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


KittenTTS = KittenTTS2


def get_model(model_name='KittenML/kitten-tts-2', cache_dir=None, device=None, hf_token=None,
              decoder=None, weights='packed', **kwargs):
    return KittenTTS2(model_name, cache_dir, device, hf_token, decoder, weights, **kwargs)
