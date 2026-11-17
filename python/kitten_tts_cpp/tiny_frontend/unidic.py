# From kitten-inference (KittenML, Apache License 2.0); the cache folder is kitten-tts-cpp's.
"""The Japanese dictionary of the mini model's Japanese front end: UniDic 2.1.2 as packaged by unidic-lite 1.0.8.

unidic-lite ships only as a 47 MB source archive that unpacks to 260 MB, so kitten-tts-cpp does not install it.
If it is installed, its dictionary is used. Otherwise the same archive is downloaded from PyPI on the first Japanese
use, checked against its SHA-256, and its dictionary folder is unpacked into the cache once.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import threading
import uuid

VERSION = "1.0.8"
URL = ("https://files.pythonhosted.org/packages/55/2b/8cf7514cb57d028abcef625afa847d60ff1ffbf0049c36b78faa7c35046f/"
       "unidic-lite-1.0.8.tar.gz")
SHA256 = "db9d4572d9fdd4d00a97949d4b0741ec480ee05a7e7e2e32f547500dae27b245"
_PREFIX = f"unidic-lite-{VERSION}/unidic_lite/dicdir/"
_lock = threading.Lock()


def cache_root():
    base = os.environ.get("KITTEN_TTS_CPP_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "kitten-tts-cpp")
    return os.path.join(base, f"unidic-lite-{VERSION}")


def _installed():
    try:
        import unidic_lite
    except ImportError:
        return None
    path = unidic_lite.DICDIR
    return path if os.path.isfile(os.path.join(path, "sys.dic")) else None


def _download(path):
    import ssl
    import urllib.request
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    digest = hashlib.sha256()
    with urllib.request.urlopen(URL, context=context, timeout=60) as r, open(path, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            digest.update(block)
            f.write(block)
    if digest.hexdigest() != SHA256:
        raise OSError(f"the downloaded file has SHA-256 {digest.hexdigest()}, not {SHA256}")


def _unpack(archive, out):
    os.makedirs(out)
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar:
            name = m.name[len(_PREFIX):] if m.name.startswith(_PREFIX) else ""
            if not m.isfile() or not name or "/" in name or "\\" in name or name.startswith("."):
                continue
            with tar.extractfile(m) as src, open(os.path.join(out, name), "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    if not os.path.isfile(os.path.join(out, "sys.dic")):
        raise OSError("the archive has no dictionary")


def dictionary_dir():
    """-> the unidic-lite dictionary folder, downloading it on first use if unidic-lite is not installed."""
    installed = _installed()
    if installed:
        return installed
    root = cache_root()
    final = os.path.join(root, "dicdir")
    if os.path.isfile(os.path.join(final, "sys.dic")):
        return final
    with _lock:
        if os.path.isfile(os.path.join(final, "sys.dic")):
            return final
        os.makedirs(root, exist_ok=True)
        tag = f"{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}"
        archive, staging = os.path.join(root, f"download.{tag}.tmp"), os.path.join(root, f"dicdir.{tag}.tmp")
        print(f"Japanese text: downloading the unidic-lite {VERSION} dictionary (47 MB, 260 MB unpacked) once, "
              f"into {root}", file=sys.stderr)
        try:
            _download(archive)
            _unpack(archive, staging)
            try:
                os.replace(staging, final)          # atomic; another process may have finished first
            except OSError:
                if not os.path.isfile(os.path.join(final, "sys.dic")):
                    raise
        except Exception as exc:
            raise RuntimeError(
                f"Japanese text needs the unidic-lite {VERSION} dictionary, which is downloaded on first use, but "
                f"the download failed ({exc}). Check the network (or a proxy blocking files.pythonhosted.org), or "
                f"install it yourself with `{sys.executable} -m pip install unidic-lite=={VERSION}`, then try "
                "again.") from exc
        finally:
            for p in (archive, staging):
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.exists(p):
                    os.remove(p)
    return final


def tagger_args():
    """The fugashi Tagger arguments that load the unidic-lite dictionary."""
    path = dictionary_dir()
    return f'-r "{os.path.join(path, "mecabrc")}" -d "{path}"'
