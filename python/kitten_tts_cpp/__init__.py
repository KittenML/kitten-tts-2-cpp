"""KittenTTS2's Python interface backed by the CPU C++ runtime."""
from .model import KittenTTS2, KittenTTS, get_model

__version__ = '0.2.0'
__all__ = ['KittenTTS', 'KittenTTS2', 'get_model']
