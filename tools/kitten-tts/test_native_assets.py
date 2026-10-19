#!/usr/bin/env python3
"""Test native downloads, offline reuse, hashes and decoder rejection without inference."""
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


def main():
    binary = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "server/org/model/resolve/main"
        repo.mkdir(parents=True)
        seen = []

        class Handler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                seen.append(self.path)
                super().do_GET()

            def do_HEAD(self):
                seen.append(self.path)
                super().do_HEAD()

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(root / "server")))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        env = dict(os.environ, HF_ENDPOINT=f"http://127.0.0.1:{server.server_port}", HF_TOKEN="")

        def entry(name):
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test")
            return {"file": name, "size": 4, "sha256": hashlib.sha256(b"test").hexdigest()}

        native = {"version": 1, "decoder": "default"}
        for key in ("lm", "acoustic", "voices", "tokenizer"):
            native[key] = entry("native/" + key)
        native["text"] = {"directory": "native/text", "files": [entry("native/text/rules")]}
        config = {"type": "KITTEN2", "native": native}

        def run(*options, ok=True, cache="cache"):
            (repo / "config.json").write_text(json.dumps(config))
            result = subprocess.run([binary, "--backend", "cuda", "--repo", "org/model", "--cache-dir", str(root / cache), "--download-only", *options], env=env, capture_output=True, text=True)
            assert (result.returncode == 0) == ok, result.stderr
            return result

        try:
            run()
            assert all("cpp/" not in path and not path.endswith(".gguf") for path in seen)
            seen.clear()
            run("--offline")
            assert not seen
            run("--offline", "--stream", "true")
            run("--offline", "--stream", "false")
            for value in ("0", "1", "False", "no"):
                assert "--stream must be true or false" in run("--stream", value, ok=False).stderr
            assert not seen
            reference = ("--reference-audio", "reference.wav", "--reference-transcript", "A reference recording.")
            assert "no native cloning assets" in run(*reference, ok=False).stderr
            assert "cannot be combined" in run(*reference, "--voice", "Bruno", ok=False).stderr
            assert "--reference-transcript is required" in run("--reference-audio", "reference.wav", ok=False).stderr
            assert "require --reference-audio" in run("--reference-transcript", "Reference.", ok=False).stderr
            assert "unknown option --reference-language" in run("--reference-language", "en", ok=False).stderr
            native["cloning"] = {"version": 1, "directory": "native/cloning", "files": [entry("native/cloning/" + name) for name in ("s3gen.safetensors", "voicebuild_consts.safetensors", "voicebuild_pyannote.safetensors")]}
            run(*reference, cache="cloning")
            seen.clear()
            run("--offline", *reference, cache="cloning")
            assert not seen
            (root / "cloning/org/model/main/native/cloning/voicebuild_consts.safetensors").write_bytes(b"evil")
            assert "SHA256" in run("--offline", *reference, cache="cloning", ok=False).stderr
            run("--offline", cache="cloning")
            native["cloning"]["files"] = native["cloning"]["files"][:1]
            assert "missing native cloning asset" in run(*reference, cache="clone-missing", ok=False).stderr
            native["cloning"]["directory"] = "elsewhere"
            assert "outside its directory" in run(*reference, cache="clone-escape", ok=False).stderr
            del native["cloning"]
            for name in ("student_w4", "student_w8"):
                result = run("--decoder", name, ok=False)
                assert "default decoder" in result.stderr
            (root / "cache/org/model/main/native/lm").write_bytes(b"evil")
            assert "SHA256" in run("--offline", ok=False).stderr
            native["lm"]["file"] = "../escape"
            run(cache="escape", ok=False)
            native["lm"]["file"] = "native/lm"
            native["text"]["directory"] = "elsewhere"
            run(cache="text-escape", ok=False)
            print("Native asset tests passed")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
