# kitten-tts QA

`.github/workflows/kitten-tts-qa.yml` builds kitten-tts with the root README's commands on
GitHub-hosted runners (Linux, Windows and macOS; x86_64 and ARM; Intel, AMD and Apple CPUs),
runs the README's examples, and posts one report to the pull request.

It assumes nothing about what should work. It answers two questions:

- **What works where?** Every platform builds and runs every test, and the report shows
  what works, with the build's or the test's own error for everything that does not.
- **Did this change break anything?** Each run is compared with the latest finished run
  on `main` (or this branch's previous run when `main` has none). The run fails only when
  something that works there stops working here. What does not work on `main` either is
  listed, not failed; what starts working is marked **new**. The first run, with nothing
  to compare with, only reports.

Not counted as broken, but listed: a test failure on a CPU the `main` run never drew
(runner CPUs are assigned at random), a crash of a test that also crashed on `main` and
passed when run again, and a timeout on a platform that also timed out on `main` or where
`main` took over half the limit. A test that crashes is run up to twice more; one that
passes then is reported as flaky.

It runs on pull requests and pushes to `main` that touch kitten-tts, GGML, llama, the
normalizer or the build. You can also start it from the Actions tab. Every install,
build, model download, test and transcription is stopped after `[limits] step_minutes`
(10 min); no test starts when the job is near `[limits] job_minutes` (45), so every job
reports.

## What each job does

1. Installs CPU PyTorch for LibTorch, as the README says, and builds the `kitten-tts`
   target with the README's CMake line.
2. Downloads `cpp/` from `KittenML/kitten-tts-2` and adds the cpp manifest to its
   `config.json` locally, as `prepare_model_repo.py` would.
3. Runs the tests in `config.toml`: the repository's own `test_assets.py` and
   `test_tq2.py`, then the README's commands (each decoder, voice and seed, expressive
   preset, `--no-repack`, `--tokens-only`, `--repeat 3`). It also runs the plain download
   path exactly as users would.
4. Checks every WAV, then has Whisper transcribe the speech to measure word error rate.

## What the report shows

The pull request gets one comment for each commit, laid out like the React Native SDK's:

- **Summary:** the commit, how many platforms pass every test, and what it was compared with.
- **Broke in This PR:** only when something broke, one row per test with a log link.
- **Platform Status:** one row per platform: the CPU (cores, RAM and the SIMD features GGML
  can use), build, tests passed, average and worst real-time factor, WER and job time
  (flagged over `report.slow_job_minutes`).
- **What Does Not Work:** one row per reason, with every platform it happens on.
- **Tests:** one row per README example, one column per platform and the CPU it drew.
- **Notes:** what started working, what was flaky (crashed, then passed when run again) and
  what was not counted and why.

The run summary adds every test's numbers (time, LM and decoder seconds, audio length,
tokens, WER, what Whisper heard), where GGML kept the weights (AMX on Intel CPUs that have
it), and the log of every failure. Audio is attached to each job.

## Changing what is tested

Edit [`config.toml`](config.toml). The workflow needs no changes.

| To... | Do this |
|---|---|
| Add a platform or compiler | Add a `[[target]]` with a runner label, plus `cmake_args` if needed |
| Add a README example | Add a `[tests.<key>]` with its `args` ({text}, {voice}, {threads}, {out}) |
| Run some tests on one platform only | `tests = [...]` on the target |
| Run a platform only on PRs or only on `main` | `events = [...]` |
| Change the CMake line or LibTorch | `[build]`, or the same keys on a target |
| Build in a Visual Studio developer environment | `vs_dev_env = "arm64"` (or `"x64"`) on a target |
| Change the spoken text, WER limit or ASR model | `[sample]`, `[asr]` |
| Change the time limits | `[limits]`, or `timeout_minutes` on a target |

`python tools/kitten-tts/qa/plan.py` checks the config and prints the jobs. The Plan job
runs it too, with `python -m unittest discover -s tools/kitten-tts/qa/tests`.

## Running one platform locally

```sh
python tools/kitten-tts/qa/plan.py --out plan.json      # QA_TARGETS / QA_TESTS filter it
python -c "import json; json.dump(json.load(open('plan.json'))['jobs'][0], open('spec.json', 'w'))"
python tools/kitten-tts/qa/run_target.py --spec spec.json --out qa-out
python tools/kitten-tts/qa/report.py qa-out report
```

## Pull requests from forks

GitHub gives fork pull requests a read-only token, so the report job cannot comment on
them. `kitten-tts-qa-comment.yml` runs after the workflow in the base repository and posts
the report. It only works once it is on the default branch.
