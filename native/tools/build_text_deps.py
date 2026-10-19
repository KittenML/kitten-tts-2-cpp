#!/usr/bin/env python3
"""Build the pinned native normalization dependencies without root privileges."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request

SOURCES = (
    ("openfst-1.8.3", "https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.3.tar.gz?download=1",
     "077714159d5cf3e38a80b6c6656d3ccc2c8b8b6c50bb41bb65c5fec10796bf53"),
    ("pcre2-10.42", "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.42/pcre2-10.42.tar.gz",
     "c33b418e3b936ee3153de2c61cc638e7e4fe3156022a5c77d0711bcbb9d64f1f"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    cache = (args.cache or prefix / "sources").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CFLAGS="-O3 -fPIC", CXXFLAGS="-O3 -fPIC -std=c++17")
    for name, url, digest in SOURCES:
        archive = cache / (name + ".tar.gz")
        if not archive.exists():
            if args.offline:
                raise RuntimeError(f"missing cached source: {archive}")
            temporary = archive.with_suffix(".part")
            request = urllib.request.Request(url, headers={"User-Agent": "KittenTTS-build/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                temporary.write_bytes(response.read())
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                temporary.unlink()
                raise RuntimeError(f"source checksum mismatch: {name}")
            temporary.replace(archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"cached source checksum mismatch: {archive}")
        source = cache / name
        if not source.exists():
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    if not (cache / member.name).resolve().is_relative_to(cache):
                        raise RuntimeError("unsafe source archive path")
                tar.extractall(cache)
        if name.startswith("openfst"):
            # OpenFst 1.8.3 copy assignment is diagnosed by GCC 15 before instantiation.
            header = source / "src/include/fst/fst.h"
            text = header.read_text()
            for symbol in ("isymbols_", "osymbols_"):
                old = f"{symbol} = impl.{symbol} ? impl.{symbol}->Copy() : nullptr;"
                new = f"{symbol}.reset(impl.{symbol} ? impl.{symbol}->Copy() : nullptr);"
                text = text.replace(old, new)
            header.write_text(text)
            table = source / "src/include/fst/bi-table.h"
            table.write_text(table.read_text().replace("selector_(table.s_)", "selector_(table.selector_)"))
        library = prefix / "lib" / ("libfst.a" if name.startswith("openfst") else "libpcre2-8.a")
        if library.exists():
            continue
        flags = [f"--prefix={prefix}", "--enable-static", "--disable-shared"]
        if name.startswith("openfst"):
            flags += ["--disable-bin"]
        else:
            flags += ["--enable-jit", "--disable-pcre2grep", "--disable-pcre2test"]
        subprocess.run(["./configure", *flags], cwd=source, env=env, check=True)
        subprocess.run(["make", f"-j{args.jobs}"], cwd=source, env=env, check=True)
        subprocess.run(["make", "install"], cwd=source, env=env, check=True)
    print(f"Configure with -DKT2N_TEXT_DEPS={prefix}")


if __name__ == "__main__":
    main()
