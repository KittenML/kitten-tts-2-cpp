"""Combine every platform job's result.json into the PR report.

    python report.py RESULTS_DIR OUT_DIR [--plan plan.json] [--jobs jobs.json] [--run-started ISO] [--baseline DIR]
    python report.py --gate OUT_DIR/summary.json

Laid out like the React Native SDK's report: a summary, one status row per
platform, what does not work and why, then every test on every platform. A run
fails only when a test that works in the baseline run (the latest run on main)
stops working here; everything else is reported, not failed. Writes
pr-comment.md, summary.md (the same plus every job's numbers) and summary.json.
--gate exits 1 when something broke.
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
from common import NO_RESULT, PASSED, classify, test_ok, why  # noqa: E402

COMMENT_LIMIT = 60000
# Sources stay ASCII (AGENTS.md), so the report's symbols are escapes.
OK, BAD, SLOW = "\u2705", "\u274c", "\U0001f422"
DOT, DASH, NONE, LE = "\u00b7", "\u2013", "\u2014", "\u2264"
TITLE = "# kitten-tts Platform Report"


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
            r["status"], r["reasons"] = NO_RESULT, ["GitHub did not run the job to the end (no runner, or cancelled)"]
        else:
            r["status"], r["reasons"] = classify(r)
        r["rows"] = {t["key"]: t for t in r.get("tests", [])}
        r["outcomes"] = outcomes(r)
    return out


def outcomes(r):
    """{test key: "pass" / "fail" / "timeout" / "skipped"}: "build", then every test that ran."""
    if r["status"] == NO_RESULT:
        return {}
    out = {"build": "pass" if (r.get("build") or {}).get("ok") else "fail"}
    fail_above = r["spec"].get("asr", {}).get("fail_above")
    for t in r.get("tests", []):
        s = t.get("status")
        out[t["key"]] = s if s in ("timeout", "skipped") else "pass" if test_ok(t, fail_above) else "fail"
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
    # <br> is the one tag kept: it stacks lines in one cell.
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def table(header, rows, align=None):
    align = align or ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(v, digits=2):
    return NONE if v is None else f"{v:.{digits}f}"


def pct(v):
    return NONE if v is None else f"{v:.0%}" if v in (0, 1) else f"{v:.1%}"


def minutes(secs):
    return f"{secs / 60:.0f} min" if secs >= 60 else f"{secs:.0f} s"


def rtf_text(v):
    if v is None:
        return NONE
    return f"{SLOW} {v:.2f}" if v > 1 else f"{v:.2f}"


def short_cpu(name):
    """'INTEL(R) XEON(R) PLATINUM 8573C' -> 'Intel Xeon Platinum 8573C'; drops '64-Core Processor'."""
    n = re.sub(r"\((R|TM)\)", "", name or "", flags=re.I)
    n = re.sub(r"\s+\d+-Core Processor|\s+CPU\s*@.*|\s+Processor$", "", n)
    words = {"INTEL": "Intel", "XEON": "Xeon", "PLATINUM": "Platinum", "GOLD": "Gold", "SILVER": "Silver"}
    return " ".join(words.get(w, w) for w in n.split())


def cpu_of(r):
    return short_cpu((r.get("env") or {}).get("cpu"))


FEATURE_NAMES = {"avx2": "AVX2", "avx512f": "AVX-512", "avx512_vnni": "VNNI", "avx512vnni": "VNNI",
                 "avx_vnni": "VNNI", "amx_tile": "AMX", "asimddp": "DotProd", "i8mm": "I8MM", "sve": "SVE"}


def cpu_label(r):
    """'AMD EPYC 7763<br>4 cores, 16 GB, AVX2'."""
    env = r.get("env") or {}
    if not env.get("cpu"):
        return r["spec"]["runner"]
    bits = ([f"{env['cpu_count']} cores"] if env.get("cpu_count") else []) + (
        [f"{round(env['ram_gb'])} GB"] if env.get("ram_gb") else [])
    for f in env.get("features", []):
        if FEATURE_NAMES.get(f, f) not in bits:
            bits.append(FEATURE_NAMES.get(f, f))
    return f"{cpu_of(r)}<br>{', '.join(bits)}"


def built(r):
    return (r.get("build") or {}).get("ok")


def title_of(r, key):
    if key == "build":
        return "Install LibTorch and build"
    return next((t.get("title", key) for t in r["spec"].get("tests", []) if t["key"] == key), key)


def why_of(r, key):
    if r["status"] == NO_RESULT:
        return r["reasons"][0]
    if key == "build":
        b = r.get("build") or {}
        error = re.sub(r"(?:[A-Za-z]:)?(?:[\\/][^\s:\\/]*)+[\\/]", "", b.get("error") or "failed")  # paths -> file names
        return f"{b.get('stage', 'build')}: {error}"
    return why(r["rows"].get(key) or {}, r["spec"].get("asr", {}).get("fail_above"))


# -- Comparing with the baseline run --------------------------------------------------

def compare(results, baseline):
    """A test broke when it passed in the baseline run for the same platform and does not pass
    here; it works now when it did not pass there. A job GitHub never ran says nothing either way."""
    base = {b["spec"]["name"]: b for b in baseline}
    for r in results:
        r["broke"], r["fixed"] = [], []
        b = r["base"] = base.get(r["spec"]["name"])
        if not b or NO_RESULT in (r["status"], b["status"]):
            continue
        for key, now in r["outcomes"].items():
            before = b["outcomes"].get(key)
            if now == "skipped" or before in (None, "skipped"):
                continue
            if now == "pass" and before != "pass":
                r["fixed"].append(key)
            elif now != "pass" and before == "pass":
                r["broke"].append(key)


# -- The PR comment -------------------------------------------------------------------

def against(ctx):
    b = ctx.get("baseline") or {}
    return f"[{b['label']}]({b['url']})" if b.get("url") else b.get("label", "")


def headline(results, ctx):
    broke = sum(len(r["broke"]) for r in results)
    if not ctx.get("baseline"):
        verdict = f"{OK} **Report only**: there is no earlier run to compare with yet."
    elif broke:
        verdict = f"{BAD} **{broke} test{'s' if broke != 1 else ''} broke** compared with {against(ctx)}."
    else:
        verdict = f"{OK} **Nothing broke** compared with {against(ctx)}."
    return f"{TITLE}\n\n{verdict}"


def summary_section(results, ctx):
    full = sum(r["status"] == PASSED for r in results)
    lost = sum(r["status"] == NO_RESULT for r in results)
    none = sum(not built(r) for r in results) - lost
    spec = results[0]["spec"] if results else {}
    sha = f"`{ctx['sha'][:7]}`, " if ctx.get("sha") else ""
    rows = [["Commit", f"{sha}kitten-tts built from {'this PR' if ctx.get('pr') else 'this commit'}"],
            ["Platforms", f"{len(results)}: {full} pass every test {DOT} {len(results) - full - none - lost} pass "
                          f"some {DOT} {none} do not build" + (f" {DOT} {lost} no result" if lost else "")],
            ["Sample text", f"{len(spec.get('text', ''))} characters, voice {spec.get('voice', '?')}"]]
    if ctx.get("baseline"):
        rows.append(["Compared with", against(ctx)])
    if ctx.get("run_url"):
        took = f", {minutes(ctx['run_secs'])}" if ctx.get("run_secs") else ""
        rows.append(["Run", f"[{ctx['run_id']}]({ctx['run_url']}){took}"])
    return "## Summary\n\n" + table(["Field", "Value"], rows)


def broke_section(results):
    rows = []
    for r in results:
        for key in r["broke"]:
            before = (r["base"]["rows"].get(key) or {}).get("secs") if r.get("base") else None
            rows.append([r["spec"]["name"], cpu_of(r) or NONE, title_of(r, key), why_of(r, key),
                         f"passed in {minutes(before)}" if before else "passed",
                         f"[log]({r['job_url']})" if r.get("job_url") else NONE])
    if not rows:
        return ""
    return "## Broke in This PR\n\n" + table(["Platform", "CPU", "Test", "What happened", "In the baseline", "Log"],
                                             rows)


def new_mark(r, key, text):
    return f"{text} new" if key in r["broke"] or (key in r["fixed"] and text.startswith(OK)) else text


def status_section(results, slow_minutes):
    rows = []
    for r in results:
        b = r.get("build") or {}
        if r["status"] == NO_RESULT:
            build_cell, tests_cell = "no result", NONE
        elif not b.get("ok"):
            build_cell, tests_cell = new_mark(r, "build", f"{BAD} {b.get('stage', 'build')}"), NONE
        else:
            build_cell = new_mark(r, "build", OK) + (f" {minutes(b['build_secs'])}" if b.get("build_secs") else "")
            keys = [t["key"] for t in r["spec"].get("tests", [])]
            passed = sum(r["outcomes"].get(k) == "pass" for k in keys)
            skipped = sum(r["outcomes"].get(k) == "skipped" for k in keys)
            tests_cell = f"{OK if passed == len(keys) else BAD} {passed}/{len(keys)}"
            tests_cell += f", {skipped} not run" if skipped else ""
            tests_cell += " new" if any(k != "build" for k in r["broke"]) else ""
        sample = {t["key"] for t in r["spec"].get("tests", []) if "{text}" in " ".join(t.get("args", []))}
        rtfs = {k: t["rtf"] for k, t in r["rows"].items()
                if k in sample and t.get("rtf") and r["outcomes"].get(k) == "pass"}
        wers = [t["wer"] for t in r.get("tests", []) if t.get("wer") is not None]
        secs = r.get("job_secs") or r.get("secs")
        took = minutes(secs) if secs else NONE
        if secs and secs > slow_minutes * 60:
            took = f"{SLOW} {took}"
        rows.append([r["spec"]["name"], cpu_label(r), build_cell, tests_cell,
                     rtf_text(statistics.mean(rtfs.values())) if rtfs else NONE,
                     pct(statistics.mean(wers)) if wers else NONE, took])
    legend = (f"RTF: (LM + decoder time) {chr(247)} audio length as kitten-tts reports it, averaged over the tests "
              f"that speak the sample text (each test's own is in the run summary); {SLOW} slower than realtime, or "
              f"a job over {slow_minutes} min. **new**: changed in this PR.")
    return ("## Platform Status\n\n"
            + table(["Platform", "CPU", "Build", "Tests", "RTF", "WER", "Runtime"], rows,
                    ["---", "---", "---", ":---:", "---:", "---:", "---:"])
            + f"\n\n{legend}")


def problems_section(results):
    """One row per reason something does not work, with the platforms it happens on."""
    groups = {}          # why -> {"what": [...], "where": [...]}
    for r in results:
        keys = [k for k, v in r["outcomes"].items() if v in ("fail", "timeout")]
        for key in keys:
            g = groups.setdefault(why_of(r, key), {"what": [], "where": []})
            if title_of(r, key) not in g["what"]:
                g["what"].append(title_of(r, key))
            if r["spec"]["name"] not in g["where"]:
                g["where"].append(r["spec"]["name"])
    if not groups:
        return ""
    names = [r["spec"]["name"] for r in results]
    builds = [r["spec"]["name"] for r in results if built(r)]

    def places(where):
        if len(where) == len(names) > 1:
            return "Every platform"
        if where == builds and len(builds) > 1 and "Install LibTorch and build" not in where:
            return "Every platform that builds"
        return ", ".join(where)
    rows = [["<br>".join(g["what"]), reason, places(g["where"])] for reason, g in groups.items()]
    return "## What Does Not Work\n\n" + table(["What", "Why", "Where"], rows)


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
            now = r["outcomes"].get(key)
            planned = any(t["key"] == key for t in r["spec"].get("tests", []))
            if planned and r["status"] != NO_RESULT and not built(r):
                row.append(BAD)            # it does not build there, so none of its tests work
            elif now in (None, "skipped"):
                row.append(NONE)
            elif now == "pass":
                row.append(new_mark(r, key, OK) + (" flaky" if (r["rows"].get(key) or {}).get("flaky") else ""))
            else:
                row.append(new_mark(r, key, BAD))
        rows.append(row)
    fail_above = results[0]["spec"].get("asr", {}).get("fail_above", 0)
    headers = [f"{r['spec']['name']}<br>{cpu_of(r)}" if cpu_of(r) else r["spec"]["name"] for r in results]
    unbuilt = [r["spec"]["name"] for r in results if r["status"] != NO_RESULT and not built(r)]
    note = (f"\n\n**{', '.join(unbuilt)}**: {BAD} on every test, because kitten-tts does not build there (why: What "
            "Does Not Work above).") if unbuilt else ""
    return ("## Tests\n\n" + table(["Test"] + headers, rows, ["---"] + [":---:"] * len(results)) + note
            + f"\n\nThe README's examples and the repository's kitten-tts tests, run against this build. {OK} works "
              f"{DOT} {BAD} does not (why above) {DOT} {NONE} not run there {DOT} **flaky**: crashed, then passed when "
              f"run again. Whisper must hear the spoken text (WER {LE} {fail_above:.0%}).")


def notes_section(results, slow_minutes):
    notes = []
    fixed = [f"{r['spec']['name']}: {', '.join(title_of(r, k) for k in r['fixed'])}" for r in results if r["fixed"]]
    if fixed:
        notes.append("**Works now**, did not in the baseline: " + f" {DOT} ".join(fixed))
    flaky = [f"{r['spec']['name']}: {title_of(r, t['key'])} ({t['flaky']})"
             for r in results for t in r.get("tests", []) if t.get("flaky")]
    if flaky:
        notes.append("**Flaky**, crashed and then passed when run again: " + f" {DOT} ".join(flaky))
    unrun = [f"{r['spec']['name']}: {', '.join(title_of(r, k) for k, v in r['outcomes'].items() if v == 'skipped')}"
             for r in results if "skipped" in r["outcomes"].values()]
    if unrun:
        notes.append("**Not run**, the job was near its time limit: " + f" {DOT} ".join(unrun))
    unheard = [f"{r['spec']['name']} ({r['asr'].get('error', 'failed')})" for r in results
               if (r.get("asr") or {}).get("status") not in (None, "done")]
    if unheard:
        notes.append("**WER not measured**, the transcription did not finish: " + "; ".join(unheard))
    lost = [r["spec"]["name"] for r in results if r["status"] == NO_RESULT]
    if lost:
        notes.append("**No result**, GitHub did not run the job to the end (no runner, or cancelled), so it says "
                     f"nothing about this PR: {', '.join(lost)}")
    slow = [r for r in results if (r.get("job_secs") or r.get("secs") or 0) > slow_minutes * 60]
    if slow:
        notes.append(f"{SLOW} **Slow jobs**: " + "; ".join(
            f"{r['spec']['name']} took {minutes(r.get('job_secs') or r['secs'])}" for r in slow))
    return "## Notes\n\n" + "\n".join(f"- {n}" for n in notes) if notes else ""


def footer(results, ctx):
    spec = results[0]["spec"] if results else {}
    asr, limit = spec.get("asr", {}), spec.get("limits", {}).get("step_minutes", 10)
    where = f"the [run summary]({ctx['run_url']})" if ctx.get("run_url") else "the run summary"
    lines = (f"Each setup, build, model download, test and transcription is stopped after {limit} min; a test that "
             f"crashes is run once more. Every job's numbers, transcripts, logs and audio are in {where}.")
    about = [f"Sample text: \"{spec.get('text', '')}\" (voice {spec.get('voice', '?')})",
             "Model files: KittenML/kitten-tts-2 cpp/, with the cpp manifest added locally; the Download test "
             "uses the README's download path as is."]
    if asr.get("enabled"):
        about.append(f"WER: `{asr.get('model')}` on the runner; case and punctuation do not count.")
    about += ["Compared with the latest finished run on the base branch (main), or this branch's previous run "
              "when main has none. Only a test that works there and not here fails the run.",
              "What runs is set in `tools/kitten-tts/qa/config.toml`."]
    return (lines + "\n\n<details><summary>About this run</summary>\n\n"
            + "\n".join(f"- {a}" for a in about) + "\n\n</details>")


# -- Every job's numbers (run summary only) -------------------------------------------

def details_section(results):
    out = ["## Job Details"]
    for r in results:
        b = r.get("build") or {}
        if r["status"] == NO_RESULT or not b:
            continue
        meta = [cpu_label(r).replace("<br>", ", "), f"setup {minutes(b.get('setup_secs') or 0)}"]
        if b.get("build_secs"):
            meta.append(f"build {minutes(b['build_secs'])}")
        if b.get("torch"):
            meta.append(f"LibTorch {b['torch']}")
        if (r.get("assets") or {}).get("secs"):
            meta.append(f"model download {minutes(r['assets']['secs'])}")
        buffers = sorted({x for t in r.get("tests", []) for x in t.get("buffers") or []})
        if buffers:
            meta.append(f"weights in {', '.join(buffers)}")
        meta += [f"[audio]({r['audio_url']})"] if r.get("audio_url") else []
        meta += [f"[log]({r['job_url']})"] if r.get("job_url") else []
        body = f" {DOT} ".join(meta)
        if not b.get("ok"):
            body += f"\n\n{why_of(r, 'build')}"
            if b.get("log_tail"):
                body += f"\n\n````\n{b['log_tail'].strip()[-1500:]}\n````"
        else:
            rows, logs = [], []
            fail_above = r["spec"].get("asr", {}).get("fail_above")
            for t in r.get("tests", []):
                ok = test_ok(t, fail_above)
                status = "Passed" if ok else {"timeout": "Timed out", "skipped": "Not run"}.get(t["status"], "Failed")
                note = ("heard word for word" if t.get("wer") == 0 else t.get("transcript") or "") if ok \
                    else why(t, fail_above)
                if t.get("flaky"):
                    note = f"flaky: {t['flaky']}"
                rows.append([title_of(r, t["key"]), status, fmt(t.get("secs"), 1), fmt(t.get("lm_seconds")),
                             fmt(t.get("decoder_seconds")), fmt(t.get("audio_s")), fmt(t.get("rtf"), 3),
                             t.get("generated_tokens", NONE), pct(t.get("wer")), note[:200]])
                tail = t.get("log_tail") or t.get("trace")
                if not ok and tail:
                    logs.append(f"<details><summary>{title_of(r, t['key'])} log</summary>\n\n````\n"
                                f"{tail.strip()[-1500:]}\n````\n\n</details>")
            body += "\n\n" + table(["Test", "Status", "Time (s)", "LM (s)", "Decoder (s)", "Audio (s)", "RTF",
                                    "Tokens", "WER", "Heard / why not"], rows,
                                   ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"])
            if logs:
                body += "\n\n" + "\n\n".join(logs)
        icon = OK if r["status"] == PASSED else BAD
        out.append(f"<details><summary>{icon} <b>{r['spec']['name']}</b> {NONE} {cpu_of(r) or r['spec']['runner']}"
                   f"</summary>\n\n{body}\n\n</details>")
    return "\n\n".join(out) if len(out) > 1 else ""


def build_report(results, ctx, slow_minutes):
    parts = [headline(results, ctx), summary_section(results, ctx), broke_section(results),
             status_section(results, slow_minutes), problems_section(results), tests_section(results),
             notes_section(results, slow_minutes)]
    comment = "\n\n".join(p for p in parts + [footer(results, ctx)] if p)
    full = "\n\n".join(p for p in parts + [details_section(results), footer(results, ctx)] if p)
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
