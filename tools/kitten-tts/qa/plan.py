"""Expand config.toml into the GitHub Actions job matrix.

    python tools/kitten-tts/qa/plan.py [--config FILE] [--out plan.json]

Targets with `events` run only for those triggers ($GITHUB_EVENT_NAME).
Optional filters, comma-separated (workflow_dispatch inputs):
    QA_TARGETS  keep targets whose name or runner contains one of these
    QA_TESTS    run only these tests

Writes `matrix=<json>` to $GITHUB_OUTPUT and the full plan to --out.
"""
import argparse
import json
import os
import re
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_KEYS = {"title", "kind", "args", "assets", "wer", "wer_text", "audio", "repeat_same", "gating", "reason",
             "timeout_minutes"}
TARGET_KEYS = {"name", "runner", "expect", "gating", "reason", "tests", "events", "cmake_args", "torch",
               "torch_index", "python", "timeout_minutes"}
EXPECTS = {"works", "build-fails"}
EVENTS = {"pull_request", "push", "workflow_dispatch"}
PLACEHOLDERS = {"text", "voice", "threads", "out"}


def csv_env(name):
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    errors = []
    tests = cfg.get("tests", {})
    for key, t in tests.items():
        where = f"tests.{key}"
        for k in set(t) - TEST_KEYS:
            errors.append(f"{where}: unknown setting {k!r}")
        if t.get("kind", "cli") not in ("cli", "repo"):
            errors.append(f"{where}: kind must be 'cli' or 'repo'")
        if t.get("kind", "cli") == "cli" and not t.get("args"):
            errors.append(f"{where}: needs args")
        if t.get("assets", "local") not in ("local", "hub"):
            errors.append(f"{where}: assets must be 'local' or 'hub'")
        for arg in t.get("args", []):
            for name in re.findall(r"{(\w+)}", arg):
                if name not in PLACEHOLDERS:
                    errors.append(f"{where}: unknown placeholder {{{name}}} (known: {sorted(PLACEHOLDERS)})")
        if t.get("gating", True) is False and not t.get("reason"):
            errors.append(f"{where}: a non-gating test needs a reason")
    names = set()
    for t in cfg.get("target", []):
        where = f"target {t.get('name', '?')!r}"
        for k in set(t) - TARGET_KEYS:
            errors.append(f"{where}: unknown setting {k!r}")
        for k in ("name", "runner"):
            if k not in t:
                errors.append(f"{where}: needs {k!r}")
        if t.get("name") in names:
            errors.append(f"{where}: duplicate name")
        names.add(t.get("name"))
        if t.get("expect", "works") not in EXPECTS:
            errors.append(f"{where}: expect must be one of {sorted(EXPECTS)}")
        for e in t.get("events", []):
            if e not in EVENTS:
                errors.append(f"{where}: unknown event {e!r}")
        for k in t.get("tests", []):
            if k not in tests:
                errors.append(f"{where}: unknown test {k!r}")
    if errors:
        sys.exit("config.toml is invalid:\n  " + "\n  ".join(errors))
    return cfg


def expand(cfg):
    only_targets = [v.lower() for v in csv_env("QA_TARGETS")]
    only_tests = csv_env("QA_TESTS")
    event = os.environ.get("GITHUB_EVENT_NAME")   # unset locally: plan every target
    build = cfg.get("build", {})
    jobs = []
    for t in cfg["target"]:
        if event and t.get("events") and event not in t["events"]:
            continue
        if only_targets and not any(v in t["name"].lower() or v in t["runner"] for v in only_targets):
            continue
        tests = []
        for key in t.get("tests") or list(cfg["tests"]):
            if only_tests and key not in only_tests:
                continue
            test = {"key": key, "kind": "cli", "assets": "local", "wer": False, "audio": True,
                    "repeat_same": False, "gating": True, "reason": "", "timeout_minutes": 15}
            test.update(cfg["tests"][key])
            test.setdefault("title", key)
            tests.append(test)
        spec = {
            "id": slug(t["name"]),
            "name": t["name"],
            "runner": t["runner"],
            "expect": t.get("expect", "works"),
            "gating": t.get("gating", True),
            "reason": t.get("reason", ""),
            "build": {
                "cmake_args": build.get("cmake_args", []) + t.get("cmake_args", []),
                "python": t.get("python", build.get("python", "3.12")),
                "torch": t.get("torch", build.get("torch", "torch")),
                "torch_index": t.get("torch_index", build.get("torch_index", "")),
            },
            "text": cfg["sample"]["text"],
            "voice": cfg["sample"]["voice"],
            "asr": cfg.get("asr", {"enabled": False}),
            "tests": tests,
        }
        timeout = t.get("timeout_minutes") or min(360, 40 + sum(x["timeout_minutes"] * (2 if x["repeat_same"] else 1)
                                                                for x in tests))
        jobs.append({"id": spec["id"], "name": t["name"], "runner": t["runner"], "python": spec["build"]["python"],
                     "timeout": timeout, "spec": json.dumps(spec)})
    if not jobs:
        sys.exit("No jobs left after filtering; check QA_TARGETS / QA_TESTS and the events of each target.")
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.toml"))
    ap.add_argument("--out", default="plan.json")
    args = ap.parse_args()
    cfg = load(args.config)
    jobs = expand(cfg)
    with open(args.out, "w") as f:
        json.dump({"report": cfg.get("report", {}), "jobs": [json.loads(j["spec"]) for j in jobs]}, f, indent=1)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write("matrix=" + json.dumps({"include": jobs}) + "\n")
    for j in jobs:
        print(f"{j['name']:24} {j['runner']:18} {len(json.loads(j['spec'])['tests']):>2} tests {j['timeout']:>4} min")
    print(f"{len(jobs)} jobs")


if __name__ == "__main__":
    main()
