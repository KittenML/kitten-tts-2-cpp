"""Shared fixtures for the converter tests (CI-safe: synthetic weights only)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO / "tools"
REFERENCE_DIR = REPO / "tests" / "reference"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


@pytest.fixture(scope="session")
def vendored():
    """Production's pack_exact/unpack, vendored verbatim (needs torch)."""
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("upstream_packed_ternary", REFERENCE_DIR / "upstream_packed_ternary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def synth_src(tmp_path_factory):
    """A synthetic Qwen3-shaped checkpoint (2 layers, hidden 256, inter 768, vocab 1024)."""
    from synth import make_checkpoint

    return make_checkpoint(tmp_path_factory.mktemp("synth"))


@pytest.fixture(scope="session")
def synth_kt2n(synth_src, tmp_path_factory):
    from kt2n_assets.convert import convert

    out = tmp_path_factory.mktemp("kt2n") / "model.kt2n"
    summary = convert(synth_src, out, log=lambda _m: None)
    return out, summary


@pytest.fixture
def cli_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TOOLS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    return env
