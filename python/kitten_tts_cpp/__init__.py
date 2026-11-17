"""KittenTTS2's Python interface backed by the C++ runtime (CPU, or NVIDIA GPU with the CUDA package).

KittenTTS2Tiny runs Kitten TTS 2 mini / nano / pico on ggml and does not need LibTorch for synthesis.
Names are imported on first use, so importing one model family does not load the other's native library.
"""

__version__ = '0.2.1'
__all__ = ['KittenTTS', 'KittenTTS2', 'KittenTTS2Tiny', 'get_model']


def __getattr__(name):
    if name in ('KittenTTS', 'KittenTTS2', 'get_model'):
        from . import model
        return getattr(model, name)
    if name == 'KittenTTS2Tiny':
        from .tiny import KittenTTS2Tiny
        return KittenTTS2Tiny
    if name in ('model', 'tiny', 'tiny_convert'):
        # submodules as attributes, as before (`import kitten_tts_cpp; kitten_tts_cpp.model.X`)
        import importlib
        return importlib.import_module(f'{__name__}.{name}')
    raise AttributeError(f"module 'kitten_tts_cpp' has no attribute {name!r}")
