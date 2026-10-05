"""Statuses, WER and exit codes shared by run_target.py and report.py. Standard library only."""
import re
import sys

PASSED = "passed"
FAILED = "failed"
UNSUPPORTED = "unsupported"   # the build failed, as config.toml says it should
CHANGED = "changed"           # expected not to build, but now it does
NO_RESULT = "no-result"       # the job died, timed out or was cancelled

STATUS_LABEL = {
    PASSED: "Passed",
    FAILED: "Failed",
    UNSUPPORTED: "Unsupported (expected)",
    CHANGED: "Now builds",
    NO_RESULT: "No result",
}


def test_failures(result):
    """(test, reason) for every gating test that failed, including WER above the limit."""
    spec_tests = {t["key"]: t for t in result["spec"]["tests"]}
    fail_above = result["spec"].get("asr", {}).get("fail_above")
    out = []
    for t in result.get("tests", []):
        if not spec_tests.get(t["key"], {}).get("gating", True):
            continue
        if t["status"] != "pass":
            out.append((t["key"], t.get("error") or t["status"]))
        elif fail_above is not None and t.get("wer") is not None and t["wer"] > fail_above:
            out.append((t["key"], f"WER {t['wer']:.0%} is above {fail_above:.0%}"))
    return out


def classify(result):
    """(status, reasons, failing) for one platform job's result.json."""
    spec = result["spec"]
    expect = spec.get("expect", "works")
    gating = spec.get("gating", True)
    build = result.get("build") or {}
    if not build:
        return NO_RESULT, ["the job did not record a build"], gating
    if not build.get("ok"):
        if expect == "build-fails":
            return UNSUPPORTED, [], False
        return FAILED, [f"{build.get('stage', 'build')} failed"], gating
    reasons = [f"{key}: {why}" for key, why in test_failures(result)]
    asr = result.get("asr") or {}
    if asr.get("status") in ("crash", "timeout"):
        reasons.append(f"WER transcription: {asr.get('error', asr['status'])}")
    if expect == "build-fails":
        return CHANGED, reasons, False
    if reasons:
        return FAILED, reasons, gating
    return PASSED, [], False


# -- Word error rate --------------------------------------------------------------

def normalize_words(text):
    """Words for WER: case and punctuation do not count, KittenTTS == Kitten TTS."""
    t = text.lower()
    t = re.sub(r"\bt\.?\s?t\.?\s?s\b\.?", "tts", t)
    t = re.sub(r"\bkitten[\s-]*tts\b", "kitten tts", t)
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", t)


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def wer(reference, hypothesis):
    """(wer, edits, reference word count)."""
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    edits = edit_distance(ref, hyp)
    return (edits / len(ref) if ref else 0.0), edits, len(ref)


# -- Exit codes -------------------------------------------------------------------

WINDOWS_CODES = {0xC0000005: "access violation", 0xC000001D: "illegal CPU instruction",
                 0xC00000FD: "stack overflow", 0xC0000409: "stack buffer overrun",
                 0xC0000135: "a DLL was not found", 0xC0000374: "heap corruption"}
SIGNALS = {4: "illegal CPU instruction (SIGILL)", 6: "aborted (SIGABRT)", 7: "bus error (SIGBUS)",
           8: "floating point exception (SIGFPE)", 9: "killed (SIGKILL), often out of memory",
           11: "segmentation fault (SIGSEGV)"}


def exit_reason(code):
    """'exited with code 3221225501 (0xC000001D, illegal CPU instruction)' and the like."""
    if code is None:
        return "timed out"
    if code < 0 and -code in SIGNALS:
        return f"was killed by signal {-code}: {SIGNALS[-code]}"
    if code > 128 and code - 128 in SIGNALS and sys.platform != "win32":
        return f"exited with code {code}: {SIGNALS[code - 128]}"
    unsigned = code & 0xFFFFFFFF
    if unsigned in WINDOWS_CODES:
        return f"exited with code {code} (0x{unsigned:08X}, {WINDOWS_CODES[unsigned]})"
    return f"exited with code {code}"
