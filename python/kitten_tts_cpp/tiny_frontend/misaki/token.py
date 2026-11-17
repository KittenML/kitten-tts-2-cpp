# Vendored from misaki 0.9.4 (misaki/token.py, https://github.com/hexgrad/misaki), Apache License 2.0.
# Unchanged except this header.
from dataclasses import dataclass
from typing import Optional
import addict

@dataclass
class MToken:
    text: str
    tag: str
    whitespace: str
    phonemes: Optional[str] = None
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None

    class Underscore(addict.Dict):
        def __getattr__(self, key):
            return super().__getattr__(key) if key in self else None

    _: Optional[Underscore] = None
