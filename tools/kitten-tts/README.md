# KittenTTS2 CPU inference

This is a local fork of DeepGrove's llama.cpp with a native C++ KittenTTS2 runner.
The speech LM runs in llama.cpp/GGML on CPU. Text normalization uses the pinned
`vendor/kitten-text-processing` submodule. The S3 waveform decoder runs through
CPU LibTorch, exported from the supplied KittenTTS Python package. No Python
interpreter or subprocess is used during synthesis. Only KITTEN2 checkpoints
are accepted.

## Build

Requires a C++17 compiler, CMake, and CPU LibTorch. A CPU PyTorch installation
can provide LibTorch's headers and libraries. Use matching torch/torchaudio
versions when preparing assets; export and runtime must use the same LibTorch
version. The normalizer decompresses its grammar data at build time using Python.
Alternatively, set `KITTEN_DATA_DIR` to an already prepared grammar directory.

```sh
git submodule update --init vendor/kitten-text-processing
python -m cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF \
  -DLLAMA_BUILD_KITTEN_TTS=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
python -m cmake --build build --target kitten-tts -j 8
```

The GitHub normalizer URL was unavailable during initial setup. This workspace's
submodule was populated from the clean sibling checkout, while `.gitmodules`
retains `https://github.com/KittenML/kitten-text-processing.git`. A fresh clone
needs access to that URL, or an equivalent checkout of the pinned commit.

## Download for inference

Users do not run the exporter. Run `build/bin/kitten-tts --text 'Hello.' --decoder student_w4 --output hello.wav`. The C++ runtime reads the model repository's `config.json` first and caches the selected native files. See the root README for repository, revision, cache, and offline options.

## Publish native assets

This section is for model publishers. Export each supported decoder using the instructions below, then assemble the exports into the same repository as the Python checkpoint:

```sh
python tools/kitten-tts/prepare_model_repo.py \
  --repo ../kitten-tts-2 --assets models/kitten2 \
  --student-w4 models/kitten2-student-w4 \
  --student-w8 models/kitten2-student-w8
```

This copies the shared GGUF into `cpp/model-tq2_1.gguf` and each decoder's TorchScript and matching prepared `voices.json` into `cpp/<decoder>/`. It preserves Python configuration fields and adds `cpp.version`, `cpp.gguf`, and `cpp.decoders`. Each file descriptor contains `file` (repository-relative path) and `size` (bytes). The runtime rejects unsupported manifest versions, unsafe paths, and size mismatches. Publish `cpp/` before the updated `config.json`, or upload both in one repository revision. Pin a commit with `--revision` for reproducible downloads. Existing Python decoder weights under `decoders/` remain separate from TorchScript exports.

## Export assets (publishers only)

Install `requirements-export.txt` in a conversion environment with a CPU
torch/torchaudio pair. The KittenTTS2 dependencies include
`s3tokenizer`, `diffusers`, `omegaconf`, `einops`, `transformers`, `safetensors`,
`librosa`, and `soundfile`. The exporter uses the normalizer submodule directly.
S3's weights must be cached or downloadable from
`ResembleAI/chatterbox-turbo/s3gen_meanflow.safetensors`.

```sh
python tools/kitten-tts/export.py \
  --repo ../kitten-tts-2 --python-package ../KittenTTS \
  --output models/kitten2
```

This writes `model-tq2_1.gguf`, `decoder.pt`, `voices.json`, `config.json`,
and a decoder parity report. All 38 built-in voices in the current checkpoint
are exported, including their speaker projections and S3 reference conditioning.
`--stage lm` and `--stage decoder` can be run independently.

The checkpoint contains folded ternary weights with 128-element scale groups. The default `--outtype tq2_1` preserves them with this fork's `TQ2_1` GGUF extension (tensor type 43, file type 42). Each 68-byte block stores 256 weights: 64 packed bytes followed by two FP16 scales. Bytes 0-31 encode the first 128 weights; bytes 32-63 encode the second 128. In each group, byte `m` stores values `m`, `m+32`, `m+64`, and `m+96` in consecutive two-bit fields. Codes 0, 1, and 2 mean -1, 0, and +1. Code 3 is invalid.

The exporter verifies exact reconstruction and rejects non-ternary matrices or scales that cannot be stored exactly in FP16. Embeddings and the tied output head remain FP16; one-dimensional tensors remain FP32. The model is 1.03 GB, versus 1.45 GB for the previous Q4_0 export and 3.47 GB for FP16. Packed transformer storage is 53% smaller than Q4_0; total model savings are 29% because other tensors are unchanged.

This extension is CPU-only and requires this fork. Standard `TQ2_0` uses one scale per 256 weights and cannot preserve the checkpoint's two scales. `TQ2_1` has an AVX2 dot-product kernel with optional VNNI instructions and a portable scalar fallback. The compact kernels use Q8_K activations, so lossless weight storage does not imply identical logits or sampled audio.

On supported Intel AMX builds, two-dimensional TQ2_1 matrices are expanded losslessly into the existing AMX Q4_0 layout at load time. This reuses the optimized prompt and single-token matrix kernels with Q8_0 activations. The original two FP16 scales are copied to their corresponding 32-weight blocks; no weight rounding is introduced. The file format and 1.03 GB GGUF remain unchanged. Runtime transformer-weight storage is about 756 MiB instead of 357 MiB, with temporary conversion buffers and mapped file pages also contributing to peak memory.

Pass `--no-repack` to disable extra CPU weight buffers and retain compact TQ2_1 inference. CPUs without AMX retain the AVX2/VNNI or scalar path. Batched weight tensors are excluded from the AMX TQ2_1 path because their stored strides describe the compact layout.

Use `--outtype ternary-q4_0` to retain the previous standard GGUF format, then select `model-ternary-q4_0.gguf` explicitly with `--model`.

For a full-precision reference:

```sh
python tools/kitten-tts/export.py \
  --repo ../kitten-tts-2 --python-package ../KittenTTS \
  --output models/kitten2 --stage lm --outtype f16
```

To export a student decoder, use a separate asset directory:

```sh
python tools/kitten-tts/export.py \
  --repo ../kitten-tts-2 --python-package ../KittenTTS \
  --output models/kitten2-student-w4 --stage decoder --decoder student_w4
```

Then pass that directory with `--assets` and reuse the main GGUF via `--model`.
Student exports preserve Python's shorter reference conditioning and one-step
flow. `student_w8` is also accepted. The exported student uses LibTorch FP32
operators over the reconstructed quantized weights; it is not an integer-only
vocoder.

## Synthesize

```sh
build/bin/kitten-tts \
  --assets models/kitten2 \
  --text 'Hello there. This is Kitten T T S running on the CPU.' \
  --voice Bruno --threads 8 --seed 1234 \
  --output hello.wav --report hello.json
```

The output is mono float32 WAV at 24 kHz. `--preset expressive`, expression
markup, reference prompts, chunk budgets, repetition/run penalties, silence
padding, and chunk joins follow `kittenml.kittentts2`. See `--help` for tuning
options. `--tokens-only` skips waveform decoding. Reports include normalized
text, prompt IDs, generated IDs, per-chunk termination status and stage timings.
A chunk that reaches its budget retains its audio and reports `terminated:false`,
as the Python implementation does.

`--repeat 3` keeps the model and decoder loaded for benchmarking. The report file
contains the final iteration, while stdout includes every iteration. Decoder JIT
warmup affects the first two iterations. RTF is synthesis time divided by audio
duration and excludes loading and initial text normalization. An RTF below 1
means synthesis is faster than playback. `--decoder-threads` tunes S3 separately.

## Prepare a cloned voice

Voice extraction is an offline Python preparation step. The C++ runtime consumes
the resulting voice exactly like a built-in voice. Supply the reference transcript;
automatic Whisper transcription and direct WAV ingestion are not implemented in
the C++ runtime.

```sh
python tools/kitten-tts/prepare_reference.py \
  --repo ../kitten-tts-2 --python-package ../KittenTTS --assets models/kitten2 \
  --reference reference.wav --transcript 'The words in the recording.' \
  --name MyVoice
build/bin/kitten-tts --assets models/kitten2 --voice MyVoice \
  --text 'This uses the prepared reference voice.' --output cloned.wav
```

## Verify parity

```sh
python tools/kitten-tts/check_parity.py \
  --binary build/bin/kitten-tts --python-package ../KittenTTS \
  --repo ../kitten-tts-2 --assets models/kitten2 \
  --model models/kitten2/model-tq2_1.gguf --lm --decoder
```

For exact weight reconstruction, also run:

```sh
python tools/kitten-tts/check_weights.py \
  --reference models/kitten2/model-f16.gguf \
  --packed models/kitten2/model-tq2_1.gguf
```

Checks include 228 frontend cases, all sampling stages in their Python order,
audio joins, exact prompt IDs, teacher-forced LM logits, and native C++ decoder
waveforms against `S3Codec.decode`. Decoder checks fix codec IDs and the RNG seed
so LM sampling differences cannot hide decoder errors. Use `--assets` pointing
to a student bundle to check that decoder.

Detailed measurements and source revisions are in [validation.json](validation.json).
Validated on Linux x86-64, Xeon Platinum 8462Y+, eight CPU threads:

- 228/228 frontend cases and 12/12 sampling cases passed.
- FP16 LM versus FP32 Python: logit RMSE 0.002-0.003, all tested argmaxes equal.
- AMX TQ2_1 LM versus FP32 Python: RMSE 0.022-0.029; cosine similarity above 0.999998.
- AMX TQ2_1 LM versus default BF16 Python: RMSE about 0.09. Some argmaxes differ.
- Default, student_w4 and student_w8 native decoders: sample-for-sample equal to Python in
  three cases each, across Bruno/Kiki and 7/51/113 generated codec tokens.
- Default decoder export: exact eager/traced agreement at 4/17/53/101 input tokens.
- AMX TQ2_1: 45.7 generated tokens/sec versus the compact path at 40.0 and Q4_0 at 44.3. The identical 256-token forced-sequence workload uses eight pinned cores, with the median of runs 2-4. Rates include prompt evaluation and sampling but exclude loading. Total time improved from 6.40 to 5.60 seconds; prompt evaluation improved from 1.09 to 0.58 seconds.
- Eight AMX matrix tests (including single-token and prompt batches) exactly match the lossless Q4_0 reference. Three full W4 S3 speech samples have identical tokens and waveforms to the previous Q4_0 runs.

This is behavioral and numerical parity, not bit-identical end-to-end generation.
BF16 arithmetic, GGML activation quantization, and C++/PyTorch sampling RNGs differ.
The same seed does not produce the same sampled tokens in both implementations.
FP16 is available when minimizing LM numerical differences matters more than speed.
All 310 packed GGUF tensors were verified to reconstruct the FP16 reference
exactly, including all 196 packed transformer matrices.
The runtime is a batch CLI, not a drop-in implementation of the Python streaming
API. The decoder currently depends on LibTorch rather than a GGML S3 implementation.

Additional format checks:

```sh
cmake --build build --target test-tq2_1 test-quantize-fns test-backend-ops -j 8
ctest --test-dir build -R 'test-(tq2_1|quantize-fns)$' --output-on-failure
python tools/kitten-tts/test_tq2.py
build/bin/test-backend-ops test -b CPU -o MUL_MAT,GET_ROWS -p tq2_1
```

Configure with `-DLLAMA_BUILD_TESTS=ON` first. The Python byte-layout check uses the Linux library in `build/bin`. The dedicated native test checks exact reconstruction and the dot product against dequantized Q8_K activations. Native VNNI, AVX2 without VNNI, and a build with AVX disabled passed locally.

The optional CTest frontend check can be enabled with
`-DKITTEN_REFERENCE_PACKAGE=/path/to/KittenTTS` and
`-DKITTEN_TEST_PYTHON=/path/to/conversion/python`, then run with
`ctest --test-dir build/tools/kitten-tts -R kitten.frontend-parity --output-on-failure`.

## Attribution

`text.cpp`, the math substitutions, prompt layout and sampling behavior are ports
of KittenTTS2. Its Apache-2.0 license is included in `LICENSE-KITTENTTS`.
Decoder exports come from the Python package's S3 implementation and preserve its
ResembleAI, Alibaba/CosyVoice and Matcha-TTS provenance. The normalizer retains
its own license and notices in the submodule. The underlying llama.cpp fork
retains its MIT license.
