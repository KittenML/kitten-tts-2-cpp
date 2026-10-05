"""Combine every platform job's result.json into the PR report.

    python report.py RESULTS_DIR OUT_DIR [--plan plan.json] [--jobs jobs.json] [--run-started ISO]
    python report.py --gate OUT_DIR/summary.json

Writes OUT_DIR/pr-comment.md (short: platforms, tests, speed, failures),
OUT_DIR/summary.md (the same plus every job's numbers) and OUT_DIR/summary.json.
jobs.json is the run's job list from the GitHub API, for log links and job times.
--gate exits 1 when a gating platform failed or produced no result.
"""
import argparse
import datetime
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CHANGED, FAILED, NO_RESULT, PASSED, STATUS_LABEL, UNSUPPORTED, classify  # noqa: E402

COMMENT_LIMIT = 60000
OK, BAD, SKIP, WARN, NEW, LATE, SLOW = ("\u2705", "\u274c", "\u2796", "\u26a0\ufe0f", "\U0001f195", "\u23f1\ufe0f",
                                        "\U0001f422")
ICON = {PASSED: OK, FAILED: BAD, UNSUPPORTED: SKIP, CHANGED: NEW, NO_RESULT: LATE}
SPEED_TESTS = ("default", "student_w4", "student_w8", "benchmark")


# -- Loading ------------------------------------------------------------------------

def load_results(results_dir, plan):
    results = {}
    for path in glob.glob(os.path.join(results_dir, "**", "result.json"), recursive=True):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        results[r["spec"]["id"]] = r
    planned = plan.get("jobs", [])
    for spec in planned:
        if spec["id"] not in results:
            results[spec["id"]] = {"spec": spec, "env": {}, "missing": True}
    order = {s["id"]: i for i, s in enumerate(planned)}
    out = sorted(results.values(), key=lambda r: (order.get(r["spec"]["id"], 1e9), r["spec"]["id"]))
    for r in out:
        if r.get("missing"):
            r["status"], r["reasons"] = NO_RESULT, ["the job timed out, was cancelled or crashed before reporting"]
            r["failing"] = r["spec"].get("gating", True)
        else:
            r["status"], r["reasons"], r["failing"] = classify(r)
    return out


def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def attach_jobs(results, jobs):
    by_name = {j["name"]: j for j in jobs}
    for r in results:
        j = by_name.get(r["spec"]["name"])
        if not j:
            continue
        r["job_url"] = j.get("html_url")
        start, end = parse_time(j.get("started_at")), parse_time(j.get("completed_at"))
        if start and end:
            r["job_secs"] = (end - start).total_seconds()


# -- Formatting ---------------------------------------------------------------------

def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def table(header, rows, align=None):
    align = align or ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(v, digits=2):
    return "-" if v is None else f"{v:.{digits}f}"


def pct(v):
    return "-" if v is None else f"{v:.1%}"


def minutes(secs):
    return f"{secs / 60:.0f} min" if secs >= 60 else f"{secs:.0f} s"


def short_cpu(name):
    """'INTEL(R) XEON(R) PLATINUM 8573C' -> 'Intel Xeon Platinum 8573C'; drops '64-Core Processor'."""
    n = re.sub(r"\((R|TM)\)", "", name, flags=re.I)
    n = re.sub(r"\s+\d+-Core Processor|\s+CPU\s*@.*|\s+Processor$", "", n)
    words = {"INTEL": "Intel", "XEON": "Xeon", "PLATINUM": "Platinum", "GOLD": "Gold", "SILVER": "Silver"}
    return " ".join(words.get(w, w) for w in n.split())


FEATURE_NAMES = {"avx2": "AVX2", "avx512f": "AVX-512", "avx512_vnni": "VNNI", "avx512vnni": "VNNI",
                 "avx_vnni": "VNNI", "amx_tile": "AMX", "asimddp": "DotProd", "i8mm": "I8MM", "sve": "SVE"}


def cpu_label(r, specs=True):
    env = r.get("env") or {}
    if not env.get("cpu"):
        return r["spec"]["runner"]
    feats = []
    for f in env.get("features", []):
        name = FEATURE_NAMES.get(f, f)
        if name not in feats:
            feats.append(name)
    bits = ([f"{env['cpu_count']} cores"] if env.get("cpu_count") else []) + (
        [f"{round(env['ram_gb'])} GB"] if env.get("ram_gb") else [])
    label = short_cpu(env["cpu"])
    if specs:
        extra = " / ".join(feats)
        return f"{label}<br>{' / '.join(bits)}" + (f"<br>{extra}" if extra else "")
    return label


def test_row(r, key):
    return next((t for t in r.get("tests", []) if t["key"] == key), None)


def spec_test(r, key):
    return next((t for t in r["spec"].get("tests", []) if t["key"] == key), None)


def wer_failed(r, t):
    fail_above = r["spec"].get("asr", {}).get("fail_above")
    return t.get("wer") is not None and fail_above is not None and t["wer"] > fail_above


# -- Short report (the PR comment) --------------------------------------------------

def headline(results, ctx):
    failing = [r for r in results if r["failing"]]
    supported = [r for r in results if r["spec"].get("expect", "works") == "works" and r["spec"].get("gating", True)]
    if failing:
        verdict = f"{BAD} {len(failing)} of {len(supported)} supported platform{'s' if len(supported) != 1 else ''} failed"
    else:
        verdict = (f"{OK} The supported platform passed" if len(supported) == 1 else
                   f"{OK} All {len(supported)} supported platforms passed")
    source = "this PR" if ctx.get("pr") else "this commit"
    bits = [f"kitten-tts built from {source}" + (f" ({ctx['sha'][:7]})" if ctx.get("sha") else "")]
    if ctx.get("run_url"):
        took = f" took {minutes(ctx['run_secs'])}" if ctx.get("run_secs") else ""
        bits.append(f"[run {ctx['run_id']}]({ctx['run_url']}){took}")
    return f"## {verdict}\n\n{' - '.join(bits)}"


def platforms_section(results, slow_minutes):
    rows, notes, slow = [], [], []
    for r in results:
        build = r.get("build") or {}
        if r["status"] == NO_RESULT:
            built = LATE
        elif build.get("ok"):
            built = f"{OK} {minutes(build['build_secs'])}" if build.get("build_secs") else OK
        else:
            built = SKIP if r["status"] == UNSUPPORTED else (BAD if r["failing"] else WARN)
        gating = [t for t in r.get("tests", []) if (spec_test(r, t["key"]) or {}).get("gating", True)]
        passed = [t for t in gating if t["status"] == "pass" and not wer_failed(r, t)]
        tests = "-" if not gating else f"{OK if len(passed) == len(gating) else BAD} {len(passed)}/{len(gating)}"
        default = test_row(r, "default") or next((t for t in r.get("tests", []) if t.get("buffers")), {})
        weights = ", ".join(default.get("buffers") or []) or "-"
        secs = r.get("job_secs") or r.get("secs")
        took = minutes(secs) if secs else "-"
        if secs and secs > slow_minutes * 60:
            took = f"{SLOW} {took}"
            slow.append(f"{r['spec']['name']} took {minutes(secs)}")
        name = r["spec"]["name"] + ("" if r["spec"].get("gating", True) else " (non-gating)")
        rows.append([name, cpu_label(r), built, tests, weights, took])
    md = "### Platforms\n\n" + table(["Platform", "CPU", "Build", "Tests", "Weights in", "Job time"], rows,
                                     ["---", "---", "---", ":---:", "---", "---:"])
    unsupported = [r for r in results if r["status"] == UNSUPPORTED]
    if unsupported:
        notes.append(f"{SKIP} not supported, as expected: " + "; ".join(
            f"{r['spec']['name']} ({r['spec'].get('reason', '').rstrip('.')})" for r in unsupported))
    soft = [r for r in results if r["status"] == FAILED and not r["failing"]]
    if soft:
        notes.append(f"{WARN} failed, but does not fail the run: " + "; ".join(
            f"{r['spec']['name']} ({r['spec'].get('reason', '').rstrip('.')})" for r in soft))
    changed = [r for r in results if r["status"] == CHANGED]
    if changed:
        notes.append(f"{NEW} builds now, though config.toml expects it not to: "
                     + ", ".join(r["spec"]["name"] for r in changed) + " - update the config")
    if slow:
        notes.append(f"{SLOW} slower than {slow_minutes} min: " + "; ".join(slow))
    notes.append("Weights in: where GGML keeps the model weights. AMX means the Intel AMX repack is in use; "
                 "CPU_Mapped means the compact TQ2_1 kernels.")
    return md + "\n\n" + "  \n".join(notes)


def tests_section(results):
    ran = [r for r in results if r.get("tests")]
    if not ran:
        return ""
    keys, titles, soft_reasons = [], {}, {}
    for r in results:
        for t in r["spec"].get("tests", []):
            if t["key"] not in titles:
                keys.append(t["key"])
                titles[t["key"]] = t.get("title", t["key"])
            if not t.get("gating", True):
                soft_reasons[t["key"]] = t.get("reason", "")
    rows = []
    for key in keys:
        row = [titles[key]]
        for r in ran:
            t = test_row(r, key)
            if t is None:
                row.append("-")
            elif t["status"] == "pass" and not wer_failed(r, t):
                row.append(OK)
            else:
                row.append(WARN if key in soft_reasons else BAD)
        rows.append(row)
    md = ("### Tests\n\nThe README's examples, run against this build. Whisper must hear the spoken text "
          f"(WER at most {ran[0]['spec'].get('asr', {}).get('fail_above', 0):.0%}). - = not run on that platform.\n\n"
          + table(["Test"] + [r["spec"]["name"] for r in ran], rows, ["---"] + [":---:"] * len(ran)))
    shown = [k for k in soft_reasons if any((test_row(r, k) or {}).get("status") not in (None, "pass") for r in ran)]
    if shown:
        md += "\n\n" + "  \n".join(f"{WARN} {titles[k]}: does not fail the run. {soft_reasons[k]}" for k in shown)
    return md


def speed_section(results):
    ran = [r for r in results if any(test_row(r, k) for k in SPEED_TESTS)]
    if not ran:
        return ""
    labels = {"default": "default", "student_w4": "student_w4", "student_w8": "student_w8",
              "benchmark": "warm (--repeat 3)"}
    rows = []
    for r in ran:
        row = [r["spec"]["name"]]
        for key in SPEED_TESTS:
            t = test_row(r, key)
            v = (t or {}).get("rtf") if (t or {}).get("status") == "pass" else None
            row.append("-" if v is None else (f"{SLOW} {v:.2f}" if v > 1 else f"{v:.2f}"))
        t = test_row(r, "default") or {}
        tps = t["generated_tokens"] / t["lm_seconds"] if t.get("lm_seconds") and t.get("generated_tokens") else None
        row.append(fmt(tps, 1))
        rows.append(row)
    return ("### Speed\n\nReal-time factor: (LM + decoder time) / audio length, as kitten-tts reports it. "
            f"Below 1 is faster than realtime; {SLOW} is slower.\n\n"
            + table(["Platform"] + [labels[k] for k in SPEED_TESTS] + ["LM tokens/s"], rows,
                    ["---"] + ["---:"] * (len(SPEED_TESTS) + 1)))


def failure_items(results):
    items = []
    for r in results:
        if not r["failing"]:
            continue
        name = f"**{r['spec']['name']}**"
        link = f" - [log]({r['job_url']})" if r.get("job_url") else ""
        build = r.get("build") or {}
        if r["status"] == NO_RESULT:
            items.append((f"{name}: no result - {r['reasons'][0]}{link}", ""))
            continue
        if build and not build.get("ok"):
            items.append((f"{name}: {build.get('stage', 'build')} failed - `{cell(build.get('error', ''))}`{link}",
                          build.get("log_tail", "")))
            continue
        asr = r.get("asr") or {}
        if asr.get("status") in ("crash", "timeout"):
            items.append((f"{name} - WER transcription: {cell(asr.get('error', ''))}{link}", asr.get("log_tail", "")))
        for t in r.get("tests", []):
            if not (spec_test(r, t["key"]) or {}).get("gating", True):
                continue
            title = (spec_test(r, t["key"]) or {}).get("title", t["key"])
            if t["status"] != "pass":
                items.append((f"{name} - {title}: {cell(t.get('error', t['status']))}{link}",
                              t.get("trace") or t.get("log_tail") or ""))
            elif wer_failed(r, t):
                items.append((f"{name} - {title}: WER {pct(t['wer'])}, Whisper heard "
                              f"\"{cell((t.get('transcript') or '')[:120])}\"{link}", ""))
    return items


def failures_section(results, log_chars):
    items = failure_items(results)
    if not items:
        return ""
    # Logs are top-level blocks under their line; four backticks so a log cannot close the fence.
    blocks = []
    for line, log in items:
        blocks.append(f"- {BAD} {line}")
        if log.strip() and log_chars:
            blocks.append(f"<details><summary>Log</summary>\n\n````\n{log.strip()[-log_chars:]}\n````\n\n</details>")
    return "### Failures\n\n" + "\n\n".join(blocks)


def footer(results, ctx):
    spec = results[0]["spec"] if results else {}
    asr = spec.get("asr", {})
    about = [f"Sample text ({len(spec.get('text', ''))} characters, voice {spec.get('voice', '?')}): "
             f"\"{spec.get('text', '')}\"",
             "Model files: KittenML/kitten-tts-2 cpp/, with the cpp manifest added locally (see the Download test)."]
    if asr.get("enabled"):
        about.append(f"WER: `{asr.get('model')}` on the runner; case and punctuation do not count; "
                     f"flagged above {asr.get('warn_above', 0):.0%}, fails above {asr.get('fail_above', 0):.0%}.")
    about.append("What runs is set in `tools/kitten-tts/qa/config.toml`.")
    where = f"the [run summary]({ctx['run_url']})" if ctx.get("run_url") else "the run summary"
    return (f"Every job's numbers (build and test times, LM and decoder seconds, transcripts, audio downloads) "
            f"are in {where}.\n\n<details><summary>About this run</summary>\n\n"
            + "\n".join(f"- {a}" for a in about) + "\n\n</details>")


# -- Details (run summary only) -------------------------------------------------------

def details_section(results):
    out = ["### Every job\n"]
    for r in results:
        if not r.get("tests"):
            continue
        build = r.get("build") or {}
        meta = [f"setup {minutes(build.get('setup_secs') or 0)}", f"build {minutes(build.get('build_secs') or 0)}"]
        if build.get("torch"):
            meta.append(f"LibTorch {build['torch']}")
        if (r.get("assets") or {}).get("secs"):
            meta.append(f"model download {minutes(r['assets']['secs'])}")
        if r.get("audio_url"):
            meta.append(f"[audio]({r['audio_url']})")
        if r.get("job_url"):
            meta.append(f"[log]({r['job_url']})")
        rows = []
        for t in r["tests"]:
            title = (spec_test(r, t["key"]) or {}).get("title", t["key"])
            status = "Passed" if t["status"] == "pass" and not wer_failed(r, t) else (
                "Failed (WER)" if t["status"] == "pass" else t["status"].capitalize())
            rows.append([title, status, fmt(t.get("secs"), 1), fmt(t.get("lm_seconds")), fmt(t.get("decoder_seconds")),
                         fmt(t.get("audio_s")), fmt(t.get("rtf"), 3), t.get("generated_tokens", "-"), pct(t.get("wer")),
                         (t.get("transcript") or t.get("error") or "")[:160]])
        out.append(f"<details><summary><b>{r['spec']['name']}</b> - {ICON[r['status']]} {STATUS_LABEL[r['status']]}"
                   f" - {cpu_label(r).replace('<br>', ' - ')}</summary>\n\n{' - '.join(meta)}\n\n"
                   + table(["Test", "Status", "Time (s)", "LM (s)", "Decoder (s)", "Audio (s)", "RTF", "Tokens", "WER",
                            "Transcript / error"], rows,
                           ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"])
                   + "\n\n</details>\n")
    return "\n".join(out) if len(out) > 1 else ""


def build_report(results, ctx, slow_minutes):
    short = [headline(results, ctx), platforms_section(results, slow_minutes), tests_section(results),
             speed_section(results)]
    comment = "\n\n".join(p for p in short + [failures_section(results, 1500), footer(results, ctx)] if p)
    if len(comment) > COMMENT_LIMIT:
        comment = "\n\n".join(p for p in short + [failures_section(results, 0), footer(results, ctx)] if p)
    full = "\n\n".join(p for p in short + [failures_section(results, 3000), details_section(results)] if p)
    return full, comment[:COMMENT_LIMIT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--plan")
    ap.add_argument("--jobs")
    ap.add_argument("--run-started")
    ap.add_argument("--gate")
    args = ap.parse_args()

    if args.gate:
        with open(args.gate, encoding="utf-8") as f:
            summary = json.load(f)
        for j in summary["jobs"]:
            if j["failing"]:
                print(f"FAILED {j['platform']}: {STATUS_LABEL[j['status']]} - {'; '.join(j['reasons'])}")
        sys.exit(1 if summary["failing"] else 0)

    plan = {}
    if args.plan and os.path.exists(args.plan):
        with open(args.plan, encoding="utf-8") as f:
            plan = json.load(f)
    results = load_results(args.results_dir, plan)
    if args.jobs and os.path.exists(args.jobs):
        with open(args.jobs, encoding="utf-8") as f:
            attach_jobs(results, json.load(f))
    server, repo, run_id = (os.environ.get(k, "") for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    started = parse_time(args.run_started)
    ctx = {"pr": os.environ.get("GITHUB_EVENT_NAME") == "pull_request",
           "sha": os.environ.get("QA_SHA") or os.environ.get("GITHUB_SHA", ""), "run_id": run_id,
           "run_url": f"{server}/{repo}/actions/runs/{run_id}" if run_id else "",
           "run_secs": (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds() if started else None}
    full, comment = build_report(results, ctx, plan.get("report", {}).get("slow_job_minutes", 30))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(full)
    with open(os.path.join(args.out_dir, "pr-comment.md"), "w", encoding="utf-8") as f:
        f.write(comment)
    jobs = [{"platform": r["spec"]["name"], "status": r["status"], "failing": r["failing"],
             "reasons": r.get("reasons", [])} for r in results]
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"failing": sum(j["failing"] for j in jobs), "jobs": jobs}, f, indent=1)
    print(f"{len(results)} platform jobs, {sum(j['failing'] for j in jobs)} failing; "
          f"comment {len(comment)} chars, summary {len(full)} chars")


if __name__ == "__main__":
    main()
