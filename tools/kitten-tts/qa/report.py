"""Combine every platform job's result.json into the PR report.

    python report.py RESULTS_DIR OUT_DIR [--plan plan.json] [--jobs jobs.json] [--run-started ISO] [--baseline DIR]
    python report.py --gate OUT_DIR/summary.json

The report says what builds and works on which platform and CPU, and what broke:
a test that works in the baseline run (the latest run on main) and not here.
Only that fails the run; something that does not work on main either is listed.
Writes pr-comment.md (short), summary.md (plus every job's numbers) and
summary.json. --gate exits 1 when something broke.
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FAILED, NO_RESULT, PASSED, STATUS_LABEL, classify  # noqa: E402

COMMENT_LIMIT = 60000
OK, BAD, LATE, SLOW = "\u2705", "\u274c", "\u23f1\ufe0f", "\U0001f422"
ICON = {PASSED: OK, FAILED: BAD, NO_RESULT: LATE}
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
        else:
            r["status"], r["reasons"] = classify(r)
        r["outcomes"] = outcomes(r)
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


def one_line(text, limit=220):
    return cell(re.sub(r"\s+", " ", str(text)))[:limit]


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


def cpu_of(r):
    return short_cpu((r.get("env") or {}).get("cpu") or "")


FEATURE_NAMES = {"avx2": "AVX2", "avx512f": "AVX-512", "avx512_vnni": "VNNI", "avx512vnni": "VNNI",
                 "avx_vnni": "VNNI", "amx_tile": "AMX", "asimddp": "DotProd", "i8mm": "I8MM", "sve": "SVE"}


def cpu_label(r):
    env = r.get("env") or {}
    if not env.get("cpu"):
        return r["spec"]["runner"]
    feats = []
    for f in env.get("features", []):
        if FEATURE_NAMES.get(f, f) not in feats:
            feats.append(FEATURE_NAMES.get(f, f))
    bits = ([f"{env['cpu_count']} cores"] if env.get("cpu_count") else []) + (
        [f"{round(env['ram_gb'])} GB"] if env.get("ram_gb") else [])
    return f"{cpu_of(r)}<br>{' / '.join(bits)}" + (f"<br>{' / '.join(feats)}" if feats else "")


def column(r):
    return f"{r['spec']['name']}<br>{cpu_of(r)}" if cpu_of(r) else r["spec"]["name"]


def test_row(r, key):
    return next((t for t in r.get("tests", []) if t["key"] == key), None)


def title_of(r, key):
    if key == "build":
        return "install LibTorch and build"
    return next((t.get("title", key) for t in r["spec"].get("tests", []) if t["key"] == key), key)


def wer_failed(r, t):
    fail_above = r["spec"].get("asr", {}).get("fail_above")
    return t.get("wer") is not None and fail_above is not None and t["wer"] > fail_above


# -- Outcomes and the comparison with the baseline -----------------------------------

def outcomes(r):
    """{test key: True / False / "timeout"}: "build", then every test the job was asked to run."""
    keys = ["build"] + [t["key"] for t in r["spec"].get("tests", [])]
    if r["status"] == NO_RESULT:
        return {}
    if not (r.get("build") or {}).get("ok"):
        return {k: False for k in keys}
    out = {"build": True}
    for key in keys[1:]:
        t = test_row(r, key)
        if not t:
            out[key] = False
        elif t["status"] == "timeout":
            out[key] = "timeout"
        else:
            out[key] = t["status"] == "pass" and not wer_failed(r, t)
    return out


def compare(results, baseline):
    """Mark what broke and what started working since the baseline run.

    A test broke when it works in the baseline for the same platform and not here,
    on a CPU the baseline also ran on. A CPU the baseline never drew cannot tell a
    regression from a CPU-specific problem, so a failure there is reported only.
    The build does not depend on the CPU, so it always counts.
    """
    base = {b["spec"]["name"]: b for b in baseline}
    for r in results:
        r["broke"], r["fixed"], r["masked"], r["new_cpu"] = [], [], [], False
        b = base.get(r["spec"]["name"])
        if not b:
            continue
        built = (r.get("build") or {}).get("ok")
        r["new_cpu"] = bool(built and cpu_of(r) and cpu_of(b)) and cpu_of(r) != cpu_of(b)
        if r["status"] == NO_RESULT:
            if b["outcomes"].get("build") is True:
                r["broke"] = ["build"]
            continue
        for key, ok in r["outcomes"].items():
            before = b["outcomes"].get(key)
            if ok is not True and before is True and (key == "build" or not r["new_cpu"]):
                r["broke"].append(key)
            elif ok is not True and before is True:
                r["masked"].append(key)            # would have broken, but on a CPU the baseline never drew
            elif ok is True and before is not None and before is not True:
                r["fixed"].append(key)


# -- Short report -------------------------------------------------------------------

def headline(results, ctx):
    broke = sum(len(r["broke"]) for r in results)
    works = sum(r["status"] == PASSED for r in results)
    if ctx.get("baseline"):
        b = ctx["baseline"]
        against = f"[{b['label']}]({b['url']})" if b.get("url") else b["label"]
        verdict = (f"{BAD} {broke} test{'s' if broke != 1 else ''} broke compared with {against}" if broke else
                   f"{OK} Nothing broke compared with {against}")
    else:
        verdict = f"{OK} Report only: there is no earlier run to compare with yet"
    bits = [f"{works} of {len(results)} platforms work fully",
            f"kitten-tts built from {'this PR' if ctx.get('pr') else 'this commit'}"
            + (f" ({ctx['sha'][:7]})" if ctx.get("sha") else "")]
    if ctx.get("run_url"):
        took = f" took {minutes(ctx['run_secs'])}" if ctx.get("run_secs") else ""
        bits.append(f"[run {ctx['run_id']}]({ctx['run_url']}){took}")
    return f"## {verdict}\n\n{' - '.join(bits)}"


def mark(r, key, text):
    """'X new' when this changed since the baseline."""
    if key in r["broke"] or (key in r["fixed"] and text == OK):
        return f"{text} new"
    return text


def platforms_section(results, slow_minutes):
    rows, slow = [], []
    for r in results:
        build = r.get("build") or {}
        if r["status"] == NO_RESULT:
            built = mark(r, "build", LATE)
        elif build.get("ok"):
            built = mark(r, "build", OK) + (f" {minutes(build['build_secs'])}" if build.get("build_secs") else "")
        else:
            built = mark(r, "build", BAD)
        tests = [k for k in r["outcomes"] if k != "build"]
        passed = [k for k in tests if r["outcomes"][k] is True]
        summary = "-" if not build.get("ok") else (
            f"{OK if len(passed) == len(tests) else BAD}{' new' if r['broke'] else ''} {len(passed)}/{len(tests)}")
        default = test_row(r, "default") or next((t for t in r.get("tests", []) if t.get("buffers")), {})
        secs = r.get("job_secs") or r.get("secs")
        took = minutes(secs) if secs else "-"
        if secs and secs > slow_minutes * 60:
            took = f"{SLOW} {took}"
            slow.append(f"{r['spec']['name']} took {minutes(secs)}")
        rows.append([r["spec"]["name"], cpu_label(r), built, summary, ", ".join(default.get("buffers") or []) or "-",
                     took])
    md = "### Platforms\n\n" + table(["Platform", "CPU", "Build", "Tests", "Weights in", "Job time"], rows,
                                     ["---", "---", "---", ":---:", "---", "---:"])
    notes = []
    problems = [r for r in results if r["status"] == FAILED and not r["broke"]]
    if problems:
        compared = any(r.get("compared") for r in results)
        lines = [f"- {r['spec']['name']}: {one_line(reason)}" for r in problems for reason in r["reasons"][:3]]
        notes.append("**Does not work** (" + ("on the baseline too, so it does not fail the run" if compared
                                               else "nothing to compare with yet") + "):\n\n" + "\n".join(lines))
    fixed = [r for r in results if r["fixed"] and not r["broke"]]
    if fixed:
        notes.append("**Works now, did not before:** " + "; ".join(
            f"{r['spec']['name']} ({', '.join(title_of(r, k) for k in r['fixed'][:3])})" for r in fixed))
    unclear = [r for r in results if r["masked"]]
    if unclear:
        notes.append("**On a CPU the baseline never drew**, so not counted as broken: "
                     + ", ".join(f"{r['spec']['name']} on {cpu_of(r)}" for r in unclear))
    if slow:
        notes.append(f"{SLOW} slower than {slow_minutes} min: " + "; ".join(slow))
    notes.append("Weights in: where GGML keeps the model weights. AMX means the Intel AMX repack is in use; "
                 "CPU_Mapped means the compact TQ2_1 kernels.")
    return md + "\n\n" + "\n\n".join(notes)


def tests_section(results):
    keys, titles = [], {}
    for r in results:
        for t in r["spec"].get("tests", []):
            if t["key"] not in titles:
                keys.append(t["key"])
                titles[t["key"]] = t.get("title", t["key"])
    if not keys:
        return ""
    rows = []
    for key in keys:
        row = [titles[key]]
        for r in results:
            ok = r["outcomes"].get(key)
            if r["status"] == NO_RESULT:
                row.append(LATE)
            elif ok is None:
                row.append("-")
            elif ok is True:
                row.append(mark(r, key, OK))
            else:
                row.append(mark(r, key, LATE if ok == "timeout" else BAD))
        rows.append(row)
    fail_above = results[0]["spec"].get("asr", {}).get("fail_above", 0)
    return ("### Tests\n\nThe README's examples and the kitten-tts tests, run against this build, one column per "
            f"platform and the CPU it drew. {OK} works, {BAD} does not, **new** = changed since the baseline, "
            f"{LATE} took too long, - not run there. Whisper must hear the spoken text (WER at most {fail_above:.0%}).\n\n"
            + table(["Test"] + [column(r) for r in results], rows, ["---"] + [":---:"] * len(results)))


def speed_section(results):
    ran = [r for r in results if any(test_row(r, k) for k in SPEED_TESTS)]
    if not ran:
        return ""
    labels = {"default": "default", "student_w4": "student_w4", "student_w8": "student_w8",
              "benchmark": "warm (--repeat 3)"}
    rows = []
    for r in ran:
        row = [column(r).replace("<br>", " / ")]
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


def log_for(r, key):
    build = r.get("build") or {}
    if r["status"] == NO_RESULT:
        return r["reasons"][0], ""
    if key == "build" or not build.get("ok"):
        return f"{build.get('stage', 'build')} failed: {build.get('error', '')}", build.get("log_tail", "")
    t = test_row(r, key) or {}
    if t.get("status") == "pass" and wer_failed(r, t):
        return f"WER {pct(t['wer'])}, Whisper heard \"{(t.get('transcript') or '')[:120]}\"", ""
    return t.get("error") or t.get("status", "did not run"), t.get("trace") or t.get("log_tail") or ""


def failures_section(results, log_chars, only_broken):
    items = []
    for r in results:
        keys = r["broke"] if only_broken else [k for k, ok in r["outcomes"].items() if ok is not True] or (
            ["build"] if r["status"] == NO_RESULT else [])
        link = f" - [log]({r['job_url']})" if r.get("job_url") else ""
        for key in keys:
            error, log = log_for(r, key)
            items.append((f"**{r['spec']['name']}** - {title_of(r, key)}: {cell(error)}{link}", log))
            if key == "build" or not (r.get("build") or {}).get("ok"):
                break                      # one line is enough when nothing was built
    if not items:
        return ""
    # Logs are top-level blocks under their line; four backticks so a log cannot close the fence.
    blocks = []
    for line, log in items:
        blocks.append(f"- {BAD} {line}")
        if log.strip() and log_chars:
            blocks.append(f"<details><summary>Log</summary>\n\n````\n{log.strip()[-log_chars:]}\n````\n\n</details>")
    title = "### Broke since the baseline" if only_broken else "### Everything that does not work"
    return f"{title}\n\n" + "\n\n".join(blocks)


def footer(results, ctx):
    spec = results[0]["spec"] if results else {}
    asr = spec.get("asr", {})
    about = [f"Sample text ({len(spec.get('text', ''))} characters, voice {spec.get('voice', '?')}): "
             f"\"{spec.get('text', '')}\"",
             "Model files: KittenML/kitten-tts-2 cpp/, with the cpp manifest added locally; the Download test "
             "uses the README's download path as is."]
    if asr.get("enabled"):
        about.append(f"WER: `{asr.get('model')}` on the runner; case and punctuation do not count; "
                     f"fails above {asr.get('fail_above', 0):.0%}.")
    limit = spec.get("limits", {}).get("step_minutes")
    if limit:
        about.append(f"Each install, build, model download, test and transcription is stopped after {limit} min.")
    about.append("Compared with the latest finished run on the base branch (main), or this branch's previous run "
                 "when main has none. Only a test that works there and not here fails the run.")
    about.append("What runs is set in `tools/kitten-tts/qa/config.toml`.")
    where = f"the [run summary]({ctx['run_url']})" if ctx.get("run_url") else "the run summary"
    return (f"Every job's numbers (build and test times, LM and decoder seconds, transcripts, logs, audio) "
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
            status = "Passed" if t["status"] == "pass" and not wer_failed(r, t) else (
                "Failed (WER)" if t["status"] == "pass" else t["status"].capitalize())
            rows.append([title_of(r, t["key"]), status, fmt(t.get("secs"), 1), fmt(t.get("lm_seconds")),
                         fmt(t.get("decoder_seconds")), fmt(t.get("audio_s")), fmt(t.get("rtf"), 3),
                         t.get("generated_tokens", "-"), pct(t.get("wer")),
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
    comment = "\n\n".join(p for p in short + [failures_section(results, 1500, True), footer(results, ctx)] if p)
    if len(comment) > COMMENT_LIMIT:
        comment = "\n\n".join(p for p in short + [failures_section(results, 0, True), footer(results, ctx)] if p)
    full = "\n\n".join(p for p in short + [failures_section(results, 3000, True),
                                           failures_section(results, 1500, False), details_section(results)] if p)
    return full, comment[:COMMENT_LIMIT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--plan")
    ap.add_argument("--jobs")
    ap.add_argument("--run-started")
    ap.add_argument("--baseline", help="result.json files of the run to compare with, and its about.json")
    ap.add_argument("--gate")
    args = ap.parse_args()

    if args.gate:
        with open(args.gate, encoding="utf-8") as f:
            summary = json.load(f)
        for j in summary["jobs"]:
            for key in j["broke"]:
                print(f"BROKE {j['platform']}: {key}")
        sys.exit(1 if summary["failing"] else 0)

    plan = {}
    if args.plan and os.path.exists(args.plan):
        with open(args.plan, encoding="utf-8") as f:
            plan = json.load(f)
    results = load_results(args.results_dir, plan)
    server, repo, run_id = (os.environ.get(k, "") for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    baseline, about = [], None
    if args.baseline and os.path.exists(os.path.join(args.baseline, "about.json")):
        with open(os.path.join(args.baseline, "about.json"), encoding="utf-8") as f:
            about = json.load(f)
        about["url"] = f"{server}/{repo}/actions/runs/{about['run_id']}" if server and about.get("run_id") else ""
        baseline = load_results(args.baseline, {})
    compare(results, baseline)
    for r in results:
        r["compared"] = bool(baseline)
    if args.jobs and os.path.exists(args.jobs):
        with open(args.jobs, encoding="utf-8") as f:
            attach_jobs(results, json.load(f))
    started = parse_time(args.run_started)
    ctx = {"pr": os.environ.get("GITHUB_EVENT_NAME") == "pull_request",
           "sha": os.environ.get("QA_SHA") or os.environ.get("GITHUB_SHA", ""), "run_id": run_id,
           "run_url": f"{server}/{repo}/actions/runs/{run_id}" if run_id else "",
           "run_secs": (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds() if started else None,
           "baseline": about if baseline else None}
    full, comment = build_report(results, ctx, plan.get("report", {}).get("slow_job_minutes", 30))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(full)
    with open(os.path.join(args.out_dir, "pr-comment.md"), "w", encoding="utf-8") as f:
        f.write(comment)
    jobs = [{"platform": r["spec"]["name"], "status": r["status"], "broke": r["broke"], "fixed": r["fixed"],
             "reasons": r.get("reasons", [])} for r in results]
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"failing": sum(bool(j["broke"]) for j in jobs), "jobs": jobs}, f, indent=1)
    print(f"{len(results)} platform jobs, {sum(bool(j['broke']) for j in jobs)} with something broken; "
          f"comment {len(comment)} chars, summary {len(full)} chars")


if __name__ == "__main__":
    main()
