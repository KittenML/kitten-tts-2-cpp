"""Statuses, WER and exit codes shared by run_target.py and report.py. Standard library only."""
import re
import sys

PASSED = "passed"
FAILED = "failed"
NO_RESULT = "no-result"       # the job died, timed out or was cancelled

STATUS_LABEL = {PASSED: "Works", FAILED: "Does not work", NO_RESULT: "No result"}


def wer_failed(row, fail_above):
    return fail_above is not None and row.get("wer") is not None and row["wer"] > fail_above


def test_ok(row, fail_above):
    """A test works when it ran without error and, where Whisper listened, it heard the text."""
    return row.get("status") == "pass" and not wer_failed(row, fail_above)


def why(row, fail_above):
    """One line on why a test does not work."""
    if row.get("status") == "pass" and wer_failed(row, fail_above):
        return f"Whisper heard \"{(row.get('transcript') or '').strip()[:100]}\" (WER {row['wer']:.0%})"
    return re.sub(r"\s+", " ", row.get("error") or row.get("status") or "failed").strip()[:220]


def classify(result):
    """(status, reasons) for one platform job. Whether a failure is new is decided in the report."""
    build = result.get("build") or {}
    if not build:
        return NO_RESULT, ["the job did not record a build"]
    if not build.get("ok"):
        return FAILED, [f"{build.get('stage', 'build')}: {build.get('error') or 'failed'}"]
    fail_above = result["spec"].get("asr", {}).get("fail_above")
    reasons = [f"{t['key']}: {why(t, fail_above)}" for t in result.get("tests", []) if not test_ok(t, fail_above)]
    return (FAILED, reasons) if reasons else (PASSED, [])


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


def crash_reason(code, said=""):
    """'crashed: segmentation fault (SIGSEGV)', or 'exited with code 1: <what it said last>'."""
    if code < 0 and -code in SIGNALS:
        return f"crashed: {SIGNALS[-code]}"
    if code > 128 and code - 128 in SIGNALS and sys.platform != "win32":
        return f"crashed: {SIGNALS[code - 128]}"
    unsigned = code & 0xFFFFFFFF
    if unsigned in WINDOWS_CODES:
        return f"crashed: {WINDOWS_CODES[unsigned]} (0x{unsigned:08X})"
    return f"exited with code {code}" + (f": {said}" if said else "")


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
