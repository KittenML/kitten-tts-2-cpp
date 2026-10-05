"""Tests for the QA scripts. Run: python -m unittest discover -s tools/kitten-tts/qa/tests"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

QA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, QA)

import plan  # noqa: E402
import report  # noqa: E402
import run_target  # noqa: E402
from common import CHANGED, FAILED, NO_RESULT, PASSED, UNSUPPORTED, classify, exit_reason, wer  # noqa: E402

ASR = {"enabled": True, "model": "openai/whisper-small.en", "warn_above": 0.15, "fail_above": 0.5}

# Tests use this, not config.toml, so editing the real config never breaks them.
FIXTURE = """
[sample]
text = "Hello there."
voice = "Bruno"

[build]
cmake_args = ["-DA=1"]

[tests.default]
args = ["--text", "{text}", "--output", "{out}"]
wer = true

[tests.download]
assets = "hub"
args = ["--text", "{text}", "--output", "{out}"]
gating = false
reason = "no manifest"

[[target]]
name = "Linux x64"
runner = "ubuntu-24.04"
cmake_args = ["-DB=2"]

[[target]]
name = "Main only"
runner = "macos-15"
events = ["push"]
tests = ["default"]
"""


def spec(**kw):
    s = {"id": "linux-x64", "name": "Linux x64", "runner": "ubuntu-24.04", "expect": "works", "gating": True,
         "reason": "", "text": "Hello there.", "voice": "Bruno", "asr": ASR, "build": {},
         "tests": [{"key": "default", "title": "Speak", "gating": True},
                   {"key": "download", "title": "Download", "gating": False, "reason": "no manifest"}]}
    s.update(kw)
    return s


def test(key="default", status="pass", **kw):
    t = {"key": key, "status": status, "secs": 5.0, "rtf": 0.9, "lm_seconds": 2.0, "decoder_seconds": 1.0,
         "audio_s": 3.3, "generated_tokens": 80, "buffers": ["CPU_Mapped"], "wer": 0.0, "transcript": "Hello there."}
    t.update(kw)
    return t


def result(s=None, built=True, tests=None):
    r = {"spec": s or spec(), "env": {"cpu": "AMD EPYC 7763 64-Core Processor", "cpu_count": 4, "ram_gb": 15.6,
                                      "features": ["avx2"]},
         "build": {"ok": built, "stage": "done" if built else "build", "build_secs": 240, "setup_secs": 60}}
    if built:
        r["tests"] = tests if tests is not None else [test(), test("download", "fail", error="no cpp assets")]
    else:
        r["build"]["error"] = "error: no LibTorch"
    return r


class Classify(unittest.TestCase):
    def test_pass_ignores_non_gating_test(self):
        self.assertEqual(classify(result())[0], PASSED)

    def test_gating_test_failure_fails(self):
        status, reasons, failing = classify(result(tests=[test(status="fail", error="boom")]))
        self.assertEqual((status, failing), (FAILED, True))
        self.assertIn("boom", reasons[0])

    def test_wer_above_limit_fails(self):
        self.assertEqual(classify(result(tests=[test(wer=0.9)]))[0], FAILED)

    def test_build_failure(self):
        self.assertEqual(classify(result(built=False))[0::2], (FAILED, True))
        self.assertEqual(classify(result(spec(expect="build-fails"), built=False))[0::2], (UNSUPPORTED, False))
        self.assertEqual(classify(result(spec(expect="build-fails")))[0::2], (CHANGED, False))

    def test_non_gating_platform(self):
        self.assertEqual(classify(result(spec(gating=False), built=False))[0::2], (FAILED, False))

    def test_crashed_transcription_fails(self):
        r = result()
        r["asr"] = {"status": "crash", "error": "stopped"}
        self.assertEqual(classify(r)[0], FAILED)

    def test_no_build_record_is_no_result(self):
        self.assertEqual(classify({"spec": spec()})[0], NO_RESULT)


class Helpers(unittest.TestCase):
    def test_wer(self):
        self.assertEqual(wer("Hello there. This is Kitten TTS.", "hello there this is kitten t t s")[0], 0)

    def test_exit_codes(self):
        self.assertIn("illegal CPU instruction", exit_reason(3221225501))
        self.assertIn("a DLL was not found", exit_reason(-1073741515))
        self.assertIn("segmentation fault", exit_reason(-11))
        self.assertEqual(exit_reason(None), "timed out")

    def test_read_float_wav(self):
        samples = [0.0, 0.5, -0.5, 0.25]
        body = struct.pack("<4f", *samples)
        data = (b"RIFF" + struct.pack("<I", 36 + len(body)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 1, 24000,
                96000, 4, 32) + b"data" + struct.pack("<I", len(body)) + body)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(data)
        self.addCleanup(os.remove, f.name)
        a, rate = run_target.read_wav(f.name)
        self.assertEqual((rate, list(a)), (24000, samples))

    def test_fill_and_buffers(self):
        self.assertEqual(run_target.fill("--threads={threads}", {"threads": 4}), "--threads=4")
        log = "load_tensors:   CPU_Mapped model buffer size = 975.60 MiB\nload_tensors:   AMX model buffer size = 1.0 MiB"
        self.assertEqual(run_target.weight_buffers(log), ["AMX", "CPU_Mapped"])


class Plan(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.remove, f.name)
        return f.name

    def expand(self, path, **env):
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            return plan.expand(plan.load(path))
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_repo_config_is_valid(self):
        jobs = plan.expand(plan.load(os.path.join(QA, "config.toml")))
        self.assertTrue(jobs)
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs))

    def test_cmake_args_add_up_and_events_filter(self):
        path = self.write(FIXTURE)
        jobs = self.expand(path, GITHUB_EVENT_NAME="pull_request")
        self.assertEqual([j["name"] for j in jobs], ["Linux x64"])
        self.assertEqual(json.loads(jobs[0]["spec"])["build"]["cmake_args"], ["-DA=1", "-DB=2"])
        jobs = self.expand(path, GITHUB_EVENT_NAME="push", QA_TESTS="default")
        self.assertEqual([[t["key"] for t in json.loads(j["spec"])["tests"]] for j in jobs], [["default"], ["default"]])

    def test_invalid_config_is_rejected(self):
        path = self.write('[sample]\ntext="x"\nvoice="Bruno"\n[tests.a]\nargs=["{bogus}"]\ngating=false\n'
                          '[[target]]\nname="A"\nrunner="r"\ntests=["missing"]\nexpect="maybe"\n')
        with self.assertRaises(SystemExit) as e:
            plan.load(path)
        for bit in ("unknown placeholder {bogus}", "needs a reason", "unknown test 'missing'", "expect must be"):
            self.assertIn(bit, str(e.exception))


class Report(unittest.TestCase):
    def run_report(self, results, planned=None, jobs=None):
        d = tempfile.mkdtemp()
        for i, r in enumerate(results):
            os.makedirs(os.path.join(d, "results", str(i)))
            with open(os.path.join(d, "results", str(i), "result.json"), "w") as f:
                json.dump(r, f)
        with open(os.path.join(d, "plan.json"), "w") as f:
            json.dump({"report": {"slow_job_minutes": 30}, "jobs": planned or [r["spec"] for r in results]}, f)
        args = [sys.executable, os.path.join(QA, "report.py"), os.path.join(d, "results"), os.path.join(d, "out"),
                "--plan", os.path.join(d, "plan.json")]
        if jobs is not None:
            with open(os.path.join(d, "jobs.json"), "w") as f:
                json.dump(jobs, f)
            args += ["--jobs", os.path.join(d, "jobs.json")]
        subprocess.run(args, check=True, capture_output=True)
        gate = subprocess.run([sys.executable, os.path.join(QA, "report.py"), "--gate",
                               os.path.join(d, "out", "summary.json")], capture_output=True, text=True)
        with open(os.path.join(d, "out", "pr-comment.md"), encoding="utf-8") as f:
            return f.read(), gate.returncode

    def test_pass(self):
        md, code = self.run_report([result()])
        self.assertEqual(code, 0)
        self.assertIn("The supported platform passed", md)
        self.assertIn("| Linux x64 | AMD EPYC 7763<br>4 cores / 16 GB<br>AVX2 | ✅ 4 min | ✅ 1/1 | CPU_Mapped |", md)
        self.assertIn("| Speak | ✅ |", md)
        self.assertIn("| Download | ⚠️ |", md)
        self.assertIn("Download: does not fail the run. no manifest", md)
        self.assertNotIn("### Failures", md)

    def test_failure_links_log_and_keeps_footer(self):
        jobs = [{"name": "Linux x64", "html_url": "https://example.test/1", "started_at": "2026-10-05T10:00:00Z",
                 "completed_at": "2026-10-05T10:40:00Z"}]
        md, code = self.run_report([result(tests=[test(status="fail", error="kitten-tts exited", log_tail="Killed")])],
                                   jobs=jobs)
        self.assertEqual(code, 1)
        self.assertIn("**Linux x64** - Speak: kitten-tts exited - [log](https://example.test/1)", md)
        self.assertIn("````\nKilled\n````", md)
        self.assertIn("\U0001f422 40 min", md)
        self.assertTrue(md.rstrip().endswith("</details>"))

    def test_unsupported_missing_and_wer(self):
        arm = result(spec(id="win-arm", name="Windows ARM64", expect="build-fails", reason="no LibTorch"), built=False)
        missing = spec(id="mac", name="macOS")
        wer_bad = result(tests=[test(wer=0.8, transcript="something else")])
        md, code = self.run_report([wer_bad, arm], planned=[wer_bad["spec"], arm["spec"], missing])
        self.assertEqual(code, 1)
        self.assertIn("not supported, as expected: Windows ARM64 (no LibTorch)", md)
        self.assertIn("**macOS**: no result", md)
        self.assertIn("Speak: WER 80.0%, Whisper heard \"something else\"", md)

    def test_comment_size_limit(self):
        big = [result(spec(id=f"p{i}", name=f"P{i}"), tests=[test(status="fail", error="x" * 300, log_tail="t" * 5000)])
               for i in range(60)]
        md, code = self.run_report(big)
        self.assertLessEqual(len(md), report.COMMENT_LIMIT)
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
