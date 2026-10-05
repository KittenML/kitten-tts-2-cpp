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
from common import FAILED, NO_RESULT, PASSED, classify, exit_reason, wer  # noqa: E402

OK, BAD, LATE = "\u2705", "\u274c", "\u23f1\ufe0f"
ASR = {"enabled": True, "model": "openai/whisper-small.en", "fail_above": 0.5}

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
    s = {"id": "linux-x64", "name": "Linux x64", "runner": "ubuntu-24.04", "text": "Hello there.", "voice": "Bruno",
         "asr": ASR, "build": {}, "tests": [{"key": "default", "title": "Speak"}, {"key": "download", "title": "Download"}]}
    s.update(kw)
    return s


def test(key="default", status="pass", **kw):
    t = {"key": key, "status": status, "secs": 5.0, "rtf": 0.9, "lm_seconds": 2.0, "decoder_seconds": 1.0,
         "audio_s": 3.3, "generated_tokens": 80, "buffers": ["CPU_Mapped"], "wer": 0.0, "transcript": "Hello there."}
    t.update(kw)
    return t


def result(s=None, built=True, tests=None, cpu="AMD EPYC 7763 64-Core Processor"):
    r = {"spec": s or spec(), "env": {"cpu": cpu, "cpu_count": 4, "ram_gb": 15.6, "features": ["avx2"]},
         "build": {"ok": built, "stage": "done" if built else "build", "build_secs": 240, "setup_secs": 60}}
    if built:
        r["tests"] = tests if tests is not None else [test(), test("download", "fail", error="no cpp assets")]
    else:
        r["build"]["error"] = "error: no LibTorch"
    return r


class Classify(unittest.TestCase):
    def test_statuses(self):
        self.assertEqual(classify(result(tests=[test(), test("download")]))[0], PASSED)
        self.assertEqual(classify(result())[0], FAILED)                       # download fails
        self.assertEqual(classify(result(built=False)), (FAILED, ["build: error: no LibTorch"]))
        self.assertEqual(classify(result(tests=[test(wer=0.9)]))[0], FAILED)
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

    def test_log_parsing(self):
        self.assertEqual(run_target.fill("--threads={threads}", {"threads": 4}), "--threads=4")
        log = "load_tensors:   CPU_Mapped model buffer size = 975.60 MiB\nload_tensors:   AMX model buffer size = 1.0 MiB"
        self.assertEqual(run_target.weight_buffers(log), ["AMX", "CPU_Mapped"])
        make = "x.cpp:1:2: error: no type named 'strong_ordering'\n1 error generated.\ngmake: *** [all] Error 2"
        self.assertEqual(run_target.last_error(make), "x.cpp:1:2: error: no type named 'strong_ordering'")
        msvc = "llama-chat.cpp(561,16): error C2088: built-in operator '<<' cannot be applied\nfoo"
        self.assertIn("error C2088", run_target.last_error(msvc))


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
            return [json.loads(j["spec"]) for j in plan.expand(plan.load(path))]
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_repo_config_is_valid(self):
        jobs = plan.expand(plan.load(os.path.join(QA, "config.toml")))
        self.assertTrue(jobs)
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs))

    def test_cmake_args_add_up_and_events_filter(self):
        path = self.write(FIXTURE)
        specs = self.expand(path, GITHUB_EVENT_NAME="pull_request")
        self.assertEqual([s["name"] for s in specs], ["Linux x64"])
        self.assertEqual(specs[0]["build"]["cmake_args"], ["-DA=1", "-DB=2"])
        self.assertEqual(specs[0]["limits"], {"step_minutes": 10, "job_minutes": 45})
        specs = self.expand(path, GITHUB_EVENT_NAME="push", QA_TESTS="default")
        self.assertEqual([[t["key"] for t in s["tests"]] for s in specs], [["default"], ["default"]])

    def test_invalid_config_is_rejected(self):
        path = self.write('[sample]\ntext="x"\nvoice="Bruno"\n[tests.a]\nargs=["{bogus}"]\n[limits]\nstep=1\n'
                          '[[target]]\nname="A"\nrunner="r"\ntests=["missing"]\ngating=false\n')
        with self.assertRaises(SystemExit) as e:
            plan.load(path)
        for bit in ("unknown placeholder {bogus}", "unknown test 'missing'", "unknown setting 'gating'",
                    "limits: unknown setting 'step'"):
            self.assertIn(bit, str(e.exception))


class Report(unittest.TestCase):
    def run_report(self, results, baseline=None, planned=None, jobs=None):
        d = tempfile.mkdtemp()
        for name, rs in (("results", results), ("baseline", baseline or [])):
            os.makedirs(os.path.join(d, name))
            for i, r in enumerate(rs):
                os.makedirs(os.path.join(d, name, str(i)))
                with open(os.path.join(d, name, str(i), "result.json"), "w") as f:
                    json.dump(r, f)
        if baseline is not None:
            with open(os.path.join(d, "baseline", "about.json"), "w") as f:
                json.dump({"run_id": 1, "label": "main"}, f)
        with open(os.path.join(d, "plan.json"), "w") as f:
            json.dump({"report": {"slow_job_minutes": 30}, "jobs": planned or [r["spec"] for r in results]}, f)
        args = [sys.executable, os.path.join(QA, "report.py"), os.path.join(d, "results"), os.path.join(d, "out"),
                "--plan", os.path.join(d, "plan.json"), "--baseline", os.path.join(d, "baseline")]
        if jobs is not None:
            with open(os.path.join(d, "jobs.json"), "w") as f:
                json.dump(jobs, f)
            args += ["--jobs", os.path.join(d, "jobs.json")]
        subprocess.run(args, check=True, capture_output=True)
        gate = subprocess.run([sys.executable, os.path.join(QA, "report.py"), "--gate",
                               os.path.join(d, "out", "summary.json")], capture_output=True, text=True)
        with open(os.path.join(d, "out", "pr-comment.md"), encoding="utf-8") as f:
            return f.read(), gate.returncode

    def test_first_run_reports_only(self):
        md, code = self.run_report([result()])
        self.assertEqual(code, 0)
        self.assertIn("Report only: there is no earlier run to compare with yet", md)
        self.assertIn(f"| Test | Linux x64<br>AMD EPYC 7763 |", md)
        self.assertIn(f"| Speak | {OK} |", md)
        self.assertIn(f"| Download | {BAD} |", md)
        self.assertIn("- Linux x64: download: no cpp assets", md)

    def test_a_test_that_works_on_main_and_breaks_fails_the_run(self):
        now = [result(tests=[test(status="fail", error="kitten-tts exited", log_tail="Killed"),
                             test("download", "fail", error="no cpp assets")])]
        jobs = [{"name": "Linux x64", "html_url": "https://example.test/1", "started_at": "2026-10-05T10:00:00Z",
                 "completed_at": "2026-10-05T10:40:00Z"}]
        md, code = self.run_report(now, baseline=[result()], jobs=jobs)
        self.assertEqual(code, 1)
        self.assertIn("1 test broke compared with main", md)
        self.assertIn(f"| Speak | {BAD} new |", md)
        self.assertIn(f"| Download | {BAD} |", md)                  # failed on main too: listed, not failed
        self.assertIn("**Linux x64** - Speak: kitten-tts exited - [log](https://example.test/1)", md)
        self.assertIn("````\nKilled\n````", md)
        self.assertIn("\U0001f422 40 min", md)
        self.assertTrue(md.rstrip().endswith("</details>"))

    def test_failure_on_a_new_cpu_is_reported_not_failed(self):
        now = [result(tests=[test(status="fail", error="illegal CPU instruction"), test("download", "fail")],
                      cpu="INTEL(R) XEON(R) PLATINUM 8573C")]
        md, code = self.run_report(now, baseline=[result()])
        self.assertEqual(code, 0)
        self.assertIn("**On a CPU the baseline never drew**, so not counted as broken: "
                      "Linux x64 on Intel Xeon Platinum 8573C", md)

    def test_a_build_that_breaks_fails_even_on_a_new_cpu(self):
        md, code = self.run_report([result(built=False, cpu="Another CPU")], baseline=[result()])
        self.assertEqual(code, 1)

    def test_platform_that_never_built_is_listed_and_one_that_starts_is_marked(self):
        arm = spec(id="win-arm", name="Windows ARM64")
        md, code = self.run_report([result(arm, built=False), result()],
                                   baseline=[result(arm, built=False), result(built=False)])
        self.assertEqual(code, 0)
        self.assertIn("- Windows ARM64: build: error: no LibTorch", md)
        self.assertIn(f"{OK} new 4 min", md)
        self.assertIn("**Works now, did not before:** Linux x64", md)

    def test_missing_job_and_timeouts(self):
        md, code = self.run_report([result(tests=[test(status="timeout", error="took longer than 10 min")])],
                                   baseline=[result(), result(spec(id="mac", name="macOS"))],
                                   planned=[spec(), spec(id="mac", name="macOS")])
        self.assertEqual(code, 1)
        self.assertIn(f"| Speak | {LATE} new |", md)
        self.assertIn(f"{LATE} new", self.row(md, "macOS"))

    def row(self, md, name):
        return next(l for l in md.splitlines() if l.startswith(f"| {name} |"))

    def test_comment_size_limit(self):
        def job(i, **kw):
            return result(spec(id=f"p{i}", name=f"P{i}"), **kw)
        now = [job(i, tests=[test(status="fail", error="x" * 300, log_tail="t" * 5000)]) for i in range(60)]
        md, code = self.run_report(now, baseline=[job(i, tests=[test()]) for i in range(60)])
        self.assertLessEqual(len(md), report.COMMENT_LIMIT)
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
