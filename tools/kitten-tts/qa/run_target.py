"""One platform job: build kitten-tts as the README says, run its examples, write result.json.

    python tools/kitten-tts/qa/run_target.py --spec spec.json --out qa-out [--work DIR]

The spec is one job from plan.py. Standard library only until torch is installed.
Every CLI run and the transcription run in their own process with a timeout, so
a crash or hang is reported instead of ending the job.
"""
import argparse
import glob
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from common import STATUS_LABEL, classify, crash_reason, exit_reason, wer  # noqa: E402

HF_REPO = "KittenML/kitten-tts-2"
DECODERS = ("default", "student_w4", "student_w8")
EXE = ".exe" if sys.platform == "win32" else ""


# -- System info ------------------------------------------------------------------

def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def system_info():
    info = {"os": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "cpu": "", "ram_gb": None, "features": []}
    try:
        if sys.platform == "linux":
            for line in _run(["lscpu"]).splitlines():
                if line.startswith("Model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
            with open("/proc/meminfo") as f:
                info["ram_gb"] = round(int(f.readline().split()[1]) / 2**20, 1)
            with open("/proc/cpuinfo") as f:
                flags = next((l.split(":", 1)[1].split() for l in f if l.startswith(("flags", "Features"))), [])
            wanted = ["avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_tile", "asimddp", "i8mm", "sve"]
            info["features"] = [x for x in wanted if x in flags]
        elif sys.platform == "darwin":
            info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            info["ram_gb"] = round(int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1)
            feats = (_run(["sysctl", "-n", "machdep.cpu.leaf7_features"]) + " " +
                     _run(["sysctl", "-n", "machdep.cpu.features"])).lower().split()
            info["features"] = [x for x in ("avx2", "avx512f", "avx512vnni") if x in feats]
            if _run(["sysctl", "-n", "hw.optional.arm.FEAT_DotProd"]) == "1":
                info["features"].append("asimddp")
            if _run(["sysctl", "-n", "hw.optional.arm.FEAT_I8MM"]) == "1":
                info["features"].append("i8mm")
        elif sys.platform == "win32":
            import ctypes
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            info["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                                      "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                                      "ullAvailExtendedVirtual")]
            ms = MemStatus()
            ms.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            info["ram_gb"] = round(ms.ullTotalPhys / 2**30, 1)
    except Exception as e:
        info["cpu"] = info["cpu"] or f"unknown ({type(e).__name__})"
    info["cpu"] = info["cpu"] or platform.processor() or "unknown"
    return info


# -- Setup and build ----------------------------------------------------------------

def step_limit(spec):
    return int(spec.get("limits", {}).get("step_minutes", 10) * 60)


def span(secs):
    secs = int(secs)
    return f"{secs // 60} min" if secs >= 60 else f"{secs} s"


def step(cmd, log_path, timeout, cwd=None, env=None):
    """Run a command into a log file; (ok, secs, last lines). A timeout ends with a line saying so."""
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        try:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = None
            log.write(f"\nERROR: took longer than {span(timeout)}, stopped\n")
    with open(log_path, encoding="utf-8", errors="replace") as f:
        tail = f.read()[-3000:]
    return code == 0, round(time.time() - t0, 1), tail


def vs_dev_env(arch):
    """The environment of a Visual Studio developer prompt for `arch`, which docs/build.md asks for on Windows ARM."""
    vswhere = os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
    vs = _run([vswhere, "-latest", "-products", "*", "-property", "installationPath"])
    bat = os.path.join(vs, "VC", "Auxiliary", "Build", "vcvarsall.bat")
    out = subprocess.run(f'cmd /s /c ""{bat}" {arch} >nul && set"', capture_output=True, text=True, shell=True).stdout
    env = dict(os.environ)
    for line in out.splitlines():
        k, sep, v = line.partition("=")
        if sep and k:
            env[k] = v
    return env


def setup_and_build(spec, out):
    """Install torch (LibTorch) and the QA tools, then configure and build kitten-tts as the README does."""
    b = spec["build"]
    limit = step_limit(spec)
    log = os.path.join(out, "build.log")
    res = {"ok": False, "stage": "setup"}
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    t0 = time.time()
    ok, _, tail = step(pip + ["-U", "pip"], log, limit)
    torch_cmd = pip + [b["torch"]] + (["--index-url", b["torch_index"]] if b["torch_index"] else [])
    ok, _, tail = step(torch_cmd, log, limit)
    extra = ["numpy", "huggingface_hub", "cmake"] + (["ninja"] if "Ninja" in b["cmake_args"] else [])
    if spec["asr"].get("enabled"):
        # Whisper runs on this torch: transformers 5.1 and later need torch 2.4, and Intel Macs have none past 2.2.
        version = _run([sys.executable, "-c", "import torch; print(torch.__version__)"])
        old = tuple(int(x) for x in re.findall(r"\d+", version)[:2]) < (2, 4) if version else False
        extra.append("transformers<5.1" if old else "transformers")
    if ok:
        ok, _, tail = step(pip + extra, log, limit)
    res["setup_secs"] = round(time.time() - t0, 1)
    if not ok:
        res.update(error=last_error(read_log(log)), log_tail=tail)
        return res
    res["torch"] = _run([sys.executable, "-c", "import torch; print(torch.__version__)"])
    prefix = _run([sys.executable, "-c", "import torch; print(torch.utils.cmake_prefix_path)"])

    res["stage"] = "build"
    cmake = shutil.which("cmake") or os.path.join(os.path.dirname(sys.executable), "cmake" + EXE)
    build_dir = os.path.join(ROOT, "build")
    configure = [cmake, "-S", ROOT, "-B", build_dir] + b["cmake_args"] + [
        f"-DCMAKE_PREFIX_PATH={prefix}", f"-DPython3_EXECUTABLE={sys.executable}"]
    env = vs_dev_env(b["vs_dev_env"]) if b.get("vs_dev_env") else None
    ok, secs_c, tail = step(configure, log, limit, cwd=ROOT, env=env)
    if ok:
        jobs = str(os.cpu_count() or 2)
        ok, secs_b, tail = step([cmake, "--build", build_dir, "--target", "kitten-tts", "--config", "Release",
                                 "--parallel", jobs], log, limit, cwd=ROOT, env=env)
        res["build_secs"] = round(secs_c + secs_b, 1)
    if not ok:
        res.update(error=last_error(read_log(log)), log_tail=tail)
        return res
    found = [p for p in glob.glob(os.path.join(build_dir, "bin", "**", "kitten-tts" + EXE), recursive=True)
             if os.path.isfile(p)]
    if not found:
        res.update(error="kitten-tts was not found under build/bin", log_tail=tail)
        return res
    res.update(ok=True, stage="done", binary=found[0])
    return res


def read_log(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def last_error(tail):
    """The line that says what went wrong: the first compiler 'error:', else the last error line."""
    lines = [l for l in tail.strip().splitlines() if l.strip()]
    compiler = [l for l in lines if re.search(r"\berror( [A-Z]+\d+)?:", l)]
    errors = [l for l in lines if re.search(r"error|Error|ERROR|fatal", l)]
    return (compiler[0] if compiler else errors[-1] if errors else lines[-1] if lines else "failed").strip()[:300]


def run_env(binary):
    """Let the binary find LibTorch and the GGML libraries on every platform."""
    torch_lib = _run([sys.executable, "-c", "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"])
    env = dict(os.environ)
    dirs = [os.path.dirname(binary), torch_lib]
    var = {"win32": "PATH", "darwin": "DYLD_LIBRARY_PATH"}.get(sys.platform, "LD_LIBRARY_PATH")
    env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    return env


# -- Model assets -------------------------------------------------------------------

def prepare_assets(work):
    """The model repo's cpp/ files with the cpp manifest added, as prepare_model_repo.py writes it."""
    from huggingface_hub import snapshot_download
    t0 = time.time()
    root = snapshot_download(HF_REPO, allow_patterns=["config.json", "cpp/*", "cpp/*/*"],
                             local_dir=os.path.join(work, "assets"))
    path = os.path.join(root, "config.json")
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    had_manifest = "cpp" in config

    def entry(rel):
        return {"file": rel, "size": os.path.getsize(os.path.join(root, rel))}

    config["cpp"] = {"version": 1, "gguf": entry("cpp/model-tq2_1.gguf"), "decoders": {
        d: {"torchscript": entry(f"cpp/{d}/decoder.pt"), "voices": entry(f"cpp/{d}/voices.json")}
        for d in DECODERS if os.path.exists(os.path.join(root, "cpp", d, "decoder.pt"))}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=1)
    return {"dir": root, "secs": round(time.time() - t0, 1), "hub_has_manifest": had_manifest}


def download_assets(spec, out, work):
    """prepare_assets() in its own process, so a stalled download is stopped at the step limit."""
    path = os.path.join(work, "assets.json")
    if os.path.exists(path):
        os.remove(path)
    ok, secs, tail = step([sys.executable, os.path.abspath(__file__), "--spec", os.path.join(out, "spec.json"),
                           "--out", out, "--work", work, "--child", "assets"], os.path.join(out, "logs", "assets.log"),
                          step_limit(spec))
    if ok and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"error": last_error(tail), "secs": secs}


# -- Tests --------------------------------------------------------------------------

def read_wav(path):
    """(samples, sample_rate) for a PCM16 or float32 WAV; numpy array, mono."""
    import numpy as np
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    pos, fmt = 12, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
            if fmt[0] == 0xFFFE:
                fmt = (struct.unpack("<H", body[24:26])[0],) + fmt[1:]
        elif cid == b"data" and fmt:
            tag, channels, rate, _, _, bits = fmt
            if tag == 3 and bits == 32:
                a = np.frombuffer(body, "<f4").astype("f4")
            elif tag == 1 and bits == 16:
                a = np.frombuffer(body, "<i2").astype("f4") / 32768
            else:
                raise ValueError(f"unsupported WAV format {tag}/{bits}")
            return (a.reshape(-1, channels).mean(1) if channels > 1 else a), rate
        pos += 8 + size + (size & 1)
    raise ValueError("WAV has no data")


def check_audio(path, text=None):
    import numpy as np
    a, rate = read_wav(path)
    assert a.size, "no audio"
    assert np.isfinite(a).all(), "audio contains NaN or inf"
    dur = a.size / rate
    peak = float(np.abs(a).max())
    assert peak > 1e-3, f"audio is silent (peak {peak:.2g})"
    if text:
        cps = len(text) / dur
        assert 3 <= cps <= 40, f"{dur:.1f}s of audio for {len(text)} characters is implausible"
    return round(dur, 3)


def fill(arg, values):
    return re.sub(r"{(\w+)}", lambda m: str(values[m.group(1)]), arg)


def run_cli(binary, env, args, log_path, timeout_s):
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ kitten-tts " + " ".join(args) + "\n")
        log.flush()
        try:
            code = subprocess.run([binary] + args, stdout=log, stderr=subprocess.STDOUT, env=env,
                                  timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            code = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return code, round(time.time() - t0, 2), text


def weight_buffers(log_text):
    """Where the weights live: 'CPU_Mapped', 'AMX', 'CPU_REPACK' ..."""
    return sorted(set(re.findall(r"load_tensors:\s+(\S+) model buffer size", log_text)))


def run_cli_test(test, spec, binary, env, assets, out, work, timeout_s):
    key = test["key"]
    wav = os.path.join(out, "audio", f"{key}.wav")
    report = os.path.join(out, "reports", f"{key}.json")
    values = {"text": spec["text"], "voice": spec["voice"], "threads": os.cpu_count() or 4, "out": wav}
    args = [fill(a, values) for a in test["args"]]
    if test["assets"] == "local" and not (assets or {}).get("dir"):
        return {"key": key, "status": "fail", "secs": 0,
                "error": f"model files were not downloaded: {(assets or {}).get('error', 'unknown error')}"}
    where = (["--assets", assets["dir"]] if test["assets"] == "local" else
             ["--cache-dir", os.path.join(work, "hub-cache")])
    res = {"key": key, "status": "pass"}
    code, secs, log = run_cli(binary, env, where + args + ["--report", report],
                              os.path.join(out, "logs", f"{key}.log"), timeout_s)
    first, tries = None, 1
    while crashed(code) and tries < 3:
        # A crash gets two more tries: one that passes then is flaky, not broken, and is reported as such.
        first = first or crash_reason(code)
        tries += 1
        code, more, log = run_cli(binary, env, where + args + ["--report", report],
                                  os.path.join(out, "logs", f"{key}-try{tries}.log"), timeout_s)
        secs += more
        if code == 0:
            res["flaky"] = f"{first} {'the first time' if tries == 2 else 'twice'}; passed when run again"
    res["secs"] = secs
    res["buffers"] = weight_buffers(log)
    if code != 0:
        said = re.findall(r"^kitten-tts: (.+)$", log, re.M)
        error = (f"took longer than {span(timeout_s)}" if code is None else
                 crash_reason(code, said[-1] if said else ""))
        if first:
            error += f", {tries} times" if error == first else f" (first run: {first})"
        res.update(status="timeout" if code is None else "fail", error=error[:400], log_tail=log[-2500:])
        if code is None and timeout_s < step_limit(spec):
            # Cut short by the job's deadline, not by the step limit: it says nothing either way.
            res.update(status="skipped", error=f"stopped after {span(timeout_s)}: the job was near its time limit")
        return res
    try:
        with open(report, encoding="utf-8") as f:
            rep = json.load(f)
        for k in ("lm_seconds", "decoder_seconds", "audio_seconds", "rtf", "generated_tokens", "iteration"):
            if k in rep:
                res[k] = round(rep[k], 4) if isinstance(rep[k], float) else rep[k]
        res["unfinished_chunks"] = sum(not c.get("terminated", True) for c in rep.get("chunks", []))
        spoken = test.get("wer_text") or (spec["text"] if "{text}" in " ".join(test["args"]) else None)
        if test["audio"]:
            res["audio_s"] = check_audio(wav, spoken)
            res["wav"] = os.path.relpath(wav, out)
            if test["wer"]:
                res["wer_text"] = spoken
        else:
            assert rep.get("generated_tokens", 0) > 0, "no tokens were generated"
        if test["repeat_same"]:
            again_wav, again_rep = wav[:-4] + "-again.wav", report[:-5] + "-again.json"
            code2, _, log2 = run_cli(binary, env, where + [again_wav if a == wav else a for a in args] + [
                "--report", again_rep], os.path.join(out, "logs", f"{key}-again.log"), timeout_s)
            assert code2 == 0, f"second run {exit_reason(code2)}"
            with open(again_rep, encoding="utf-8") as f:
                rep2 = json.load(f)
            tokens = [c.get("audio_tokens") for c in rep.get("chunks", [])]
            tokens2 = [c.get("audio_tokens") for c in rep2.get("chunks", [])]
            assert tokens == tokens2, "the same seed and threads gave different tokens"
            with open(wav, "rb") as a, open(again_wav, "rb") as b:
                assert a.read() == b.read(), "the same seed and threads gave different audio"
            res["same_twice"] = True
    except Exception as e:
        res.update(status="fail", error=f"{type(e).__name__}: {e}"[:400], trace=traceback.format_exc()[-1500:],
                   log_tail=log[-1500:])
    return res


def crashed(code):
    """A signal or a native exception, not an ordinary error exit or a timeout."""
    return code is not None and (code < 0 or (code > 128 and sys.platform != "win32")
                                 or (code & 0xFFFFFFFF) >= 0xC0000000)


def run_repo_tests(binary, env, out, limit):
    """The repository's own kitten-tts tests."""
    res = {"key": "repo_tests", "status": "pass", "files": []}
    t0 = time.time()
    for name, cmd in (("test_assets.py", [sys.executable, os.path.join(ROOT, "tools/kitten-tts/test_assets.py"), binary]),
                      ("test_tq2.py", [sys.executable, os.path.join(ROOT, "tools/kitten-tts/test_tq2.py")])):
        log_path = os.path.join(out, "logs", f"repo-{name}.log")
        ok, secs, tail = step(cmd, log_path, limit, cwd=ROOT, env=env)
        res["files"].append({"name": name, "ok": ok, "secs": secs})
        if not ok:
            res["status"] = "fail"
            res["error"] = (res.get("error", "") + f"{name}: {last_error(tail)}; ")[:400]
            res["log_tail"] = (res.get("log_tail", "") + f"--- {name}\n{tail[-1200:]}\n")[-2500:]
    res["secs"] = round(time.time() - t0, 1)
    if res.get("error"):
        res["error"] = res["error"].rstrip("; ")
    return res


# -- Transcription ------------------------------------------------------------------

def child_asr(spec, out):
    """Transcribe every WER test's WAV with Whisper and score it against its text."""
    import numpy as np
    from transformers import pipeline
    with open(os.path.join(out, "asr-refs.json"), encoding="utf-8") as f:
        refs = json.load(f)
    pipe = pipeline("automatic-speech-recognition", model=spec["asr"]["model"], device="cpu")
    rows = []
    for ref in refs:
        try:
            a, rate = read_wav(os.path.join(out, ref["wav"]))
            n = int(len(a) * 16000 / rate)
            a16 = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype("f4")
            hyp = pipe(a16)["text"].strip()
            w, edits, words = wer(ref["text"], hyp)
            rows.append({"key": ref["key"], "wer": round(w, 4), "edits": edits, "words": words, "transcript": hyp})
            print(f"{ref['key']}: WER {w:.1%} - {hyp}", flush=True)
        except Exception as e:
            rows.append({"key": ref["key"], "wer": None, "error": f"{type(e).__name__}: {e}"[:300]})
    with open(os.path.join(out, "asr.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)


def run_asr(spec, out, refs, timeout_s):
    with open(os.path.join(out, "asr-refs.json"), "w", encoding="utf-8") as f:
        json.dump(refs, f)
    log_path = os.path.join(out, "logs", "asr.log")
    ok, secs, tail = step([sys.executable, "-X", "faulthandler", os.path.abspath(__file__), "--spec",
                           os.path.join(out, "spec.json"), "--out", out, "--child", "asr"], log_path,
                          timeout_s)
    path = os.path.join(out, "asr.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return {"status": "done", "secs": secs, "rows": json.load(f)}
    return {"status": "crash", "secs": secs, "error": "transcription stopped before reporting", "log_tail": tail}


# -- Driver -------------------------------------------------------------------------

def drive(spec_path, out, work):
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    for d in ("logs", "audio", "reports"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    os.makedirs(work, exist_ok=True)
    shutil.copy(spec_path, os.path.join(out, "spec.json"))
    t_start = time.time()
    result = {"spec": spec, "env": system_info()}
    print(json.dumps(result["env"]), flush=True)

    print("Installing LibTorch and building kitten-tts...", flush=True)
    result["build"] = setup_and_build(spec, out)
    print(json.dumps({k: v for k, v in result["build"].items() if k != "log_tail"}), flush=True)
    if result["build"]["ok"]:
        binary = result["build"]["binary"]
        env = run_env(binary)
        assets = None
        if any(t["kind"] == "cli" and t["assets"] == "local" for t in spec["tests"]):
            assets = result["assets"] = download_assets(spec, out, work)
        result["tests"], refs = [], []
        # Stop starting tests before GitHub cancels the job, so result.json is always written.
        deadline = t_start + spec["limits"].get("job_minutes", 45) * 60 - 240
        reserve = step_limit(spec) // 2 if spec["asr"].get("enabled") else 0
        for test in spec["tests"]:
            left = deadline - time.time() - reserve
            if left < 60:
                row = {"key": test["key"], "status": "skipped", "error": "not run: the job ran out of time"}
            elif test["kind"] == "repo":
                row = run_repo_tests(binary, env, out, min(step_limit(spec), left))
            else:
                row = run_cli_test(test, spec, binary, env, assets, out, work, min(step_limit(spec), int(left)))
            print(f"  {test['key']}: {row['status']} ({row.get('secs')}s) {row.get('error', '')}", flush=True)
            result["tests"].append(row)
            if test.get("wer") and row.get("wav") and row.get("wer_text"):
                refs.append({"key": test["key"], "wav": row["wav"], "text": row["wer_text"]})
        if spec["asr"].get("enabled") and refs:
            result["asr"] = run_asr(spec, out, refs, min(step_limit(spec), max(deadline - time.time(), 60)))
            for row in result["asr"].get("rows", []):
                test = next(t for t in result["tests"] if t["key"] == row["key"])
                test["wer"], test["transcript"] = row.get("wer"), row.get("transcript")
                if row.get("error"):
                    test["asr_error"] = row["error"]

    result["secs"] = round(time.time() - t_start, 1)
    status, reasons = classify(result)
    result["status"], result["reasons"] = status, reasons
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"\n{spec['name']}: {STATUS_LABEL[status]}", flush=True)
    for r in reasons:
        print(f"  - {r}", flush=True)
    # The verdict (did this break something that works on main?) is the report job's.
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=os.path.join(tempfile.gettempdir(), "kitten-tts-qa"))
    ap.add_argument("--child")
    args = ap.parse_args()
    if args.child == "asr":
        with open(args.spec, encoding="utf-8") as f:
            child_asr(json.load(f), args.out)
        return
    if args.child == "assets":
        with open(os.path.join(args.work, "assets.json"), "w", encoding="utf-8") as f:
            json.dump(prepare_assets(args.work), f)
        return
    sys.exit(drive(args.spec, os.path.abspath(args.out), args.work))


if __name__ == "__main__":
    main()
