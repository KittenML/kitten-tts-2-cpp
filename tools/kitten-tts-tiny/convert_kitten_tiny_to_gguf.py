#!/usr/bin/env python3
"""Convert Kitten TTS 2 mini / nano / pico (KITTEN2_TINY) safetensors to GGUF; see python/kitten_tts_cpp/tiny_convert.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "kitten_tts_cpp"))
from tiny_convert import main  # noqa: E402

if __name__ == "__main__":
    main()
