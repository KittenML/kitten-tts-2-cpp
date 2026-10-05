# kitten-tts QA

`.github/workflows/kitten-tts-qa.yml` builds kitten-tts with the root README's commands on
GitHub-hosted runners (Linux, Windows and macOS; x86_64 and ARM; Intel, AMD and Apple CPUs),
runs the README's examples, and posts one report to the pull request.

It runs on pull requests and pushes to `main` that touch kitten-tts, GGML, llama, the
normalizer or the build. You can also start it from the Actions tab.

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

- **Platforms:** the CPU (cores, RAM and the SIMD features GGML can use), build time,
  tests passed, where GGML keeps the weights (AMX on Intel CPUs that have it) and job time.
  Jobs slower than `report.slow_job_minutes` are flagged.
- **Tests:** one row per README example and one column per platform.
- **Speed:** the real-time factor kitten-tts reports for each decoder, warm (`--repeat 3`),
  and LM tokens per second.
- **Failures:** one line each, with a link to the job's log and the log's tail.

The run summary also has every test's numbers: time, LM and decoder seconds, audio length,
tokens, WER and what Whisper heard. Audio is attached to each job.

The run fails when a supported platform fails to build, a gating test fails, a job produces
no result, or WER is above `asr.fail_above`.

## Changing what is tested

Edit [`config.toml`](config.toml). The workflow needs no changes.

| To... | Do this |
|---|---|
| Add a platform or compiler | Add a `[[target]]` with a runner label, plus `cmake_args` if needed |
| Add a README example | Add a `[tests.<key>]` with its `args` ({text}, {voice}, {threads}, {out}) |
| Run some tests on one platform only | `tests = [...]` on the target |
| Report a test without failing the run | `gating = false` and a `reason` on the test |
| Mark a platform known-unsupported | `expect = "build-fails"` and a `reason` |
| Run a platform only on PRs or only on `main` | `events = [...]` |
| Change the CMake line or LibTorch | `[build]`, or the same keys on a target |
| Change the spoken text, WER limits or ASR model | `[sample]`, `[asr]` |

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
