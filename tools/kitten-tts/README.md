# KittenTTS2 CPU and CUDA inference

This is a local fork of DeepGrove's llama.cpp with a native C++ KittenTTS2 runner.
The speech LM runs in llama.cpp/GGML on CPU, or on an NVIDIA GPU in a CUDA build
(see [GPU (CUDA) build](#gpu-cuda-build)). Text normalization uses the pinned
`vendor/kitten-text-processing` submodule. The S3 waveform decoder runs through
LibTorch on the CPU, or on an NVIDIA GPU with a locally exported CUDA decoder
(see [CUDA waveform decoder](#cuda-waveform-decoder-optional-local-export)). It is
exported from the supplied KittenTTS Python package. No Python
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

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF \
  -DLLAMA_BUILD_KITTEN_TTS=ON \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

cmake --build build --target kitten-tts -j 8
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
and a decoder parity report. All built-in voices in the checkpoint are exported
(47 in the current published `voices.json`), including their speaker projections
and S3 reference conditioning.
`--stage lm` and `--stage decoder` can be run independently.

The checkpoint contains folded ternary weights with 128-element scale groups. The default `--outtype tq2_1` preserves them with this fork's `TQ2_1` GGUF extension (tensor type 43, file type 42). Each 68-byte block stores 256 weights: 64 packed bytes followed by two FP16 scales. Bytes 0-31 encode the first 128 weights; bytes 32-63 encode the second 128. In each group, byte `m` stores values `m`, `m+32`, `m+64`, and `m+96` in consecutive two-bit fields. Codes 0, 1, and 2 mean -1, 0, and +1. Code 3 is invalid.

The exporter verifies exact reconstruction and rejects non-ternary matrices or scales that cannot be stored exactly in FP16. Embeddings and the tied output head remain FP16; one-dimensional tensors remain FP32. The model is 1.03 GB, versus 1.45 GB for the previous Q4_0 export and 3.47 GB for FP16. Packed transformer storage is 53% smaller than Q4_0; total model savings are 29% because other tensors are unchanged.

This extension requires this fork, and only the CPU has TQ2_1 kernels; GPU builds repack it to Q2_0 (see [GPU (CUDA) build](#gpu-cuda-build)). Standard `TQ2_0` uses one scale per 256 weights and cannot preserve the checkpoint's two scales. `TQ2_1` has an AVX2 dot-product kernel with optional VNNI instructions and a portable scalar fallback. The compact kernels use Q8_K activations, so lossless weight storage does not imply identical logits or sampled audio.

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

## CUDA waveform decoder (optional, local export)

The published `decoder.pt` files are traced on the CPU and run only with `--decoder-device cpu`, the default. To run the S3 decoder on a GPU, build against a CUDA LibTorch, export a CUDA decoder locally, and pass `--decoder-device cuda`. CUDA decoders are not published.

LibTorch: torch 2.13.0 has no cu128 build, so use 2.13.0+cu129 (CUDA 12.9 runtime; kernels for sm_75 to sm_120; tested with driver 580.95). This one command builds the measured setup, the speech LM and the decoder both on the GPU (`--gpu-layers 99 --decoder-device cuda`): GGML's CUDA backend with the CUDA 12.8 toolkit, and the wheel's CUDA LibTorch.

```sh
python -m venv venv-cuda
venv-cuda/bin/pip install torch==2.13.0+cu129 --index-url https://download.pytorch.org/whl/cu129
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DGGML_CUDA=ON -DGGML_METAL=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES='75-real;86-real;89-real;120a-real' \
  -DLLAMA_BUILD_KITTEN_TTS=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_PREFIX_PATH="$(venv-cuda/bin/python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-cuda --target kitten-tts -j 8
```

Without `-DGGML_CUDA=ON` only the decoder runs on the GPU. A CPU LibTorch is enough when only the LM runs on the GPU (see [GPU (CUDA) build](#gpu-cuda-build)); `--decoder-device cuda` then fails with "needs a CUDA build of LibTorch".

The wheel's `libtorch_cuda.so` finds its CUDA 12.9 libraries (cudart, cuBLAS, cuDNN 9) through its RPATH. With `-DGGML_CUDA=ON` (CUDA 12.8 toolkit), the process still loads one `libcudart.so.12`, the 12.9 one, which GGML also uses (checked with `LD_DEBUG=libs`). The `libtorch-shared-with-deps-2.13.0+cu129.zip` download also configures, but it does not ship cuDNN, cuSPARSELt or NVSHMEM; those must be on the library path. A CUDA LibTorch build runs the CPU decoder unchanged.

Export with the same torch in a conversion environment that has a GPU. torchaudio has no 2.13 release; `torchaudio==2.11.0` installed with `--no-deps` imports correctly with torch 2.13.0.

```sh
python tools/kitten-tts/export.py \
  --repo ../kitten-tts-2 --python-package ../KittenTTS \
  --output models/kitten2-cuda --stage decoder --device cuda \
  --reuse-voices ../kitten-tts-2/cpp/default/voices.json
build-cuda/bin/kitten-tts --assets models/kitten2-cuda --model ../kitten-tts-2/cpp/model-tq2_1.gguf \
  --gpu-layers 99 --decoder-device cuda --text 'Hello.' --output hello.wav
```

- `--reuse-voices` keeps the published conditioning. The copy is byte-identical, and no LM shards or reference audio are needed.
- `--rng device` (default) draws the decoder noise on the GPU. A seed then gives a different, equally valid, noise sample than the CPU decoder. `--rng cpu` draws all noise from the CPU generator and copies it, so a seed gives the same noise as the CPU decoder. Use it for parity checks; it costs 13 to 28 ms per call. `--rng cpu` works only with `--decoder default`: with `student_w4` the export's trace self-check fails (maximum error 4.6e-4, limit 2e-4), and `student_w8` is not checked, so export.py rejects both.
- With either `--rng`, s3gen's SineGen draws its 9 phase offsets per call from the CPU generator and copies them to the GPU. The export creates SineGen's harmonic buffer (9 floats per output sample) directly on the GPU; the reference code fills it on the CPU and copies it on every call, and decoder.pt files exported before this change still do that. `lm.cpu_stages` in the report lists these steps.
- The export writes `kitten_decoder.json` (device, rng, sinegen, torch version) into the archive. The runtime rejects a decoder exported for the other device. The trace fixes `cuda:0`, so select the GPU with `CUDA_VISIBLE_DEVICES`.
- On CUDA the runtime turns TF32 off, skips the TorchScript profiling executor and graph passes, and prints caching-allocator statistics to stderr after each decode. `KITTEN_DECODER_EMPTY_CACHE=1` frees cached blocks after each decode.
- Loading a CUDA decoder takes about 6 s (5.6 to 6.3 s in three sessions), against 1.1 s for the CPU decoder. An Nsight Systems trace shows about 255,000 tensor comparisons on the GPU during the load, each followed by a synchronizing copy.

Measured on an RTX 4090 (LibTorch 2.13.0+cu129, voice Bruno, seed 7, decode only):

| Audio | CPU decoder, 8 threads, warm | CUDA first call | CUDA warm | CUDA peak allocated / reserved | process GPU memory (nvidia-smi) |
|---|---|---|---|---|---|
| 12.3 s (306 tokens) | 4.11 s | 0.73 s | 0.091 s | 721 / 782 MiB | 1252 MiB |
| 22.4 s (558 tokens) | 8.37 s | 0.71 s | 0.128 s | 935 / 1052 MiB | 1522 MiB |

The decoder weights take 514 MiB of GPU memory (309 MiB for `student_w4`, which decodes in 0.077 s and 0.094 s). Output is not bit-identical to the CPU decoder: with the same noise (`--rng cpu`), SNR against the CPU export is 88.3 dB (12.3 s) and 82.9 dB (22.4 s), with a maximum sample difference of 2.9e-4 and 9.3e-4. CPU and GPU kernels sum in different orders and use different transcendental functions, and the vocoder's phase is a cumulative sum over every sample, so small float32 differences add up over time. GPU runs are also not bit-identical to each other (about 123 dB SNR between runs).

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

## GPU (CUDA) build

The speech LM can run on an NVIDIA GPU through llama.cpp's CUDA backend. CPU
builds, and CUDA builds run with the default `--gpu-layers 0`, give the same
tokens, audio and logits as before. They use less memory, because each context
reserves one output row and is freed before the decoder runs. The report always
has an `lm` object with the LM placement (`cpu`, `gpu` or `hybrid`), and gains
`decoder_device` only when `--decoder-device` is given. Use one CUDA 12.8 toolkit
(tested with driver 580.95). With the configure command below, a CPU LibTorch is
enough: the LM runs on the GPU and the decoder on the CPU. To also run the decoder
on the GPU, use the CUDA LibTorch command in
[CUDA waveform decoder](#cuda-waveform-decoder-optional-local-export) instead.

```sh
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DGGML_CUDA=ON -DGGML_METAL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES='75-real;86-real;89-real;120a-real' \
  -DLLAMA_BUILD_KITTEN_TTS=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-cuda --target kitten-tts -j 8
build-cuda/bin/kitten-tts --gpu-layers 99 --timings \
  --text 'Hello from the GPU.' --output gpu.wav
```

The `-real` architectures are compiled ahead of time, so the first start needs no
PTX JIT compile: 75 is Turing (GTX 16xx, RTX 20xx), 86 is Ampere (RTX 30xx,
A2000), 89 is Ada (RTX 40xx) and 120a is Blackwell (RTX 50xx, RTX PRO 6000). No
PTX is embedded, so other cards, for example SM80 (A100) or SM90 (H100), stop with
"no kernel image is available for execution on the device". Add your card's
architecture (for example `80-real` or `90-real`) if it is missing.

GPU options:

- `--gpu-layers N|all` (default 0). llama.cpp offloads the output head first,
  then the last N-1 of the 28 transformer layers. 29, 99 or `all` put the whole LM
  on the GPU; the report writes `all` as `"gpu_layers": -1`. llama.cpp's load log
  then prints "offloading 27 repeating layers" next to "offloading output layer";
  the "offloaded 29/29 layers" line is the full count. With 0 the LM gets no GPU
  device at all, so a CUDA build does not offload prompt operations and gives the
  same tokens as a CPU build.
- `--main-gpu N` selects the GPU (default 0). With `--gpu-layers` above 0 the LM
  uses that one GPU only.
- `--flash-attn auto|on|off` (default `auto`).
- `--allow-hybrid`. No GPU backend has a TQ2_1 kernel, so at load time the runtime
  repacks each TQ2_1 matrix that goes to the GPU into Q2_0 and logs it. The repack
  is lossless: both formats store 2-bit codes with FP16 scales, and each 128-weight
  scale is copied onto two 64-weight Q2_0 blocks. If any TQ2_1 matrix still stays
  on the CPU (for example with `--gpu-layers` below 29), the runtime stops with an
  error. `--allow-hybrid` runs those matrices on the CPU instead, and the report
  marks the run `"placement": "hybrid"`. This is supported, but it is not the
  fastest setup: the layers left on the CPU run at CPU speed. On a GTX 1080 Ti
  with `--gpu-layers 14`, the LM took 16.4 s for a 5.2 s sentence, against 1.4 s
  with every layer on the GPU and 36 s on the CPU alone (8 threads, with the
  portable CPU code of the CI wheels, which is slower than a native build's). Use
  it only when the whole LM does not fit in GPU memory.

The CPU decoder next to a GPU LM (`--decoder-device cpu`, the default) is also
supported but not the fastest setup: the decoder then runs on the CPU and is the
slowest stage. On the RTX 4090 machine below, the decoder's RTF was 0.28 on the
CPU against 0.008 on the GPU, so the whole run's RTF was 0.41 instead of 0.15. On
a Tesla T4 (8 CPU threads), the decoder took 5.0 s on the CPU against 0.2 s on the
GPU for a 5.4 s sentence (RTF 1.23 against 0.31). It is the only choice when the
CUDA LibTorch has no kernels for the card, as for compute capability 6.1 with
PyTorch 2.13.0+cu129.

With `--gpu-layers 99`, these stages run on each device:

| Stage | Device |
|---|---|
| Text normalization, chunking, tokenization | CPU |
| Token embedding lookup (input layer, one row per token) | CPU |
| 28 transformer layers and the tied FP16 output head | GPU |
| Logits copy (158,241 floats per token) and sampling | CPU |
| S3 decoder (LibTorch) | CPU, or GPU with `--decoder-device cuda` and a CUDA-exported decoder (see above). The GPU decoder still draws SineGen's 9 phase offsets per chunk on the CPU, uploads the tokens and copies the waveform back; older exports also fill SineGen's harmonic buffer on the CPU |
| Chunk joins and WAV writing | CPU |

Precision differs from CPU builds:

- CUDA's quantized matrix kernels quantize activations to Q8_1, while the CPU
  TQ2_1 kernels use Q8_K (Q8_0 on the AMX path).
- The tied F16 output head uses FP16 products and partial sums on CUDA. This CLI
  reads one logits row per step, which runs as a matrix-vector product. A caller
  that asks for more than one logits row (for example llama-perplexity) gets cuBLAS
  FP16 compute for the head; `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` switches it to
  FP32. The CLI is not affected.
- Flash attention (`--flash-attn auto` resolves to on with CUDA). CUDA decode on
  SM89 and SM120 uses the vector kernel, which accumulates in FP32. Prefill, and
  decode on SM75 and SM86 (inferred from the kernel selection code, not run),
  use the tensor-core kernel: K*Q in FP32, softmax*V in FP16. The CPU accumulates
  softmax*V in FP16 (F16 V cache), so for prefill both backends sum it in FP16.
- With `--flash-attn off`, K*Q runs in FP32 on both backends. CUDA computes
  softmax*V with FP16 partial sums for decode and cuBLAS FP16 compute for prefill,
  where the CPU sums in FP32. This is inferred from code; logits were not compared
  with flash attention off.
- Norm, RoPE and softmax sums also run in a different order. Logits parity below
  was measured on SM89 only.

The weights are identical, but the logits differ slightly, so the same seed can
sample different tokens. Teacher-forced logits for the 10 test cases (the 5
sentences listed below x voices Bruno and Kiki; 1,723 positions of 158,241 logits
each, RTX 4090):

| Compared | Logit RMSE | Top-1 agreement |
|---|---|---|
| GPU (Q8_1 activations) vs FP32 reference | 0.022 | 98.8% |
| CPU TQ2_1 (Q8_K activations) vs FP32 reference | 0.042 | 97.2% |
| GPU vs CPU TQ2_1 | 0.046 | 97.2% |
| CPU vs CPU, same build on two CPU models (AVX-512 and AVX2) | 0.041 | 97.6% |

The FP32 reference is the same GGUF dequantized with
`llama-quantize --allow-requantize model-tq2_1.gguf model-f32.gguf F32` and run on
the CPU, so no activation quantization is involved. The GPU path is closer to it
than the CPU path is, and the GPU-to-CPU difference is about as large as the
difference between two CPUs.

The 5 test sentences are: "One day, a little girl named Lily found a needle in her
room."; "Hello there. This is Kitten T T S running on the CPU."; "On March 3rd,
2025, the meeting moved to 10:30 a.m., so please bring the $250 invoice and two
printed copies of the report."; "Could you tell me how long the train ride takes
from the airport to the city center late at night?"; and "Scientists have found
that the deep ocean holds far more life than anyone expected, from glowing
jellyfish to tiny shrimp that drift through the darkness, feeding on whatever
falls from the surface far above."

The Q2_0 CUDA kernels are validated on SM89 (RTX 4090) only: test-backend-ops
cases at the model's matrix shapes, plus the end-to-end logits above. Builds of
this source also ran full generations on an RTX PRO 6000 (12.0, where
`test-tq2_1-load` passed too), a Tesla T4 (7.5) and a GTX 1080 Ti (6.1, built with
`61-real` added), but their logits were not compared with the reference. Ampere
(86) builds compile but were not run.

GPU memory on an RTX 4090, from llama.cpp's buffer sizes and from device memory
sampled about 100 times per second, for the 10 test cases (3.5 to 12.2 s of audio,
`--repeat 3`), one 367-character chunk (19.0 and 20.9 s of audio) and one 4-chunk
text. Each chunk's context holds the prompt plus its token budget (1,536 positions
for these sentences) and is freed before the decoder runs, so the LM's share does
not grow with the number of chunks or requests. The CUDA decoder's share does:
LibTorch keeps the blocks it cached for one decode while the next LM context is
allocated, for the next chunk or the next request in the same process. The first
request of each test case peaked at 2,219 to 2,265 MiB (stage-end samples, which
see the decoder's high-water mark because LibTorch keeps its blocks). The second and third
requests peaked at 2,335 to 2,495 MiB, more for longer audio. The 20.9 s chunk
peaked at about 2,496 MiB in its first request, and the 4-chunk text (longest
chunk 327 characters, 19.6 s) at about 2,876 MiB. `KITTEN_DECODER_EMPTY_CACHE=1`
frees the cached blocks after each decode: the 4-chunk peak was then about
2,462 MiB, with the same RTF (0.154 against 0.156). Other cards are not measured
yet.

| Item | Size |
|---|---|
| Q2_0 transformer weights, FP16 output head (copy of the tied embedding) and norms | 996.6 MiB |
| F16 KV cache (112 KiB per position) | 168.0 MiB |
| GPU compute buffer | 49.5 MiB, plus 9.5 MiB pinned host memory |
| CUDA context | about 395 MiB |
| Device memory in use at the peak, including the CUDA context | about 1,670 MiB (every test sentence) |
| The same with the CUDA decoder (`--decoder-device cuda`), one chunk | 2,219 to 2,265 MiB for the first request; up to about 2,495 MiB for later requests in the same process and for a 20.9 s chunk (decoder: 514 MiB weights; LibTorch at most 721 MiB allocated and 782 MiB reserved at 12.2 s of audio) |
| The same, 4 chunks | about 2,876 MiB (LibTorch up to 1,218 MiB reserved); about 2,462 MiB with `KITTEN_DECODER_EMPTY_CACHE=1` |

Each context sizes its compute graph for one output row (`n_outputs_max = 1`)
instead of 512, because the runtime reads only the last position's logits. On the
CPU this shrinks the compute buffer from 317.1 to 53.5 MiB per context, with
bit-identical logits; the GPU compute buffer is 49.5 MiB with this setting (the
512-row GPU size was not measured). The runtime also frees each chunk's KV cache
and compute buffers before the decoder runs. Host RAM still maps the GGUF, whose
FP16 token embedding serves the CPU lookup. Peak RSS was about 2.1 GiB in every
configuration; how much of it is the mapped GGUF, load-time page touches or the
shared CUDA and LibTorch libraries is not measured.

Measured on one shared machine (RTX 4090, AMD EPYC 7713, `--threads 8` pinned to
the 8 cores of one CCD, which share one L3 cache) for the 10 test cases with seed
1234 and `--repeat 3`. Values are medians of the third iteration. RTF is LM plus
decoder time over audio time; `rtf_total` in the report also counts tokenization,
context creation and release, and WAV writing. Neither includes loading. LM
forward passes per second count only the forward passes; generated tokens per
second also count the CPU sampling between them.

| Setup | RTF | `rtf_total` | RTF, first iteration | LM forward passes | Generated tokens (forward passes and sampling) | Sampling per token (share of the generation loop) | Decoder RTF | Peak GPU memory |
|---|---|---|---|---|---|---|---|---|
| CPU only, 8 threads on one CCD | 1.28 | 1.31 | 1.56 | 37/s | 33/s | 3.6 ms (12%) | 0.29 | none |
| `--gpu-layers 99`, CPU decoder | 0.41 | 0.41 | 0.65 | 455/s | 184/s | 3.2 ms (59%) | 0.28 | about 1,670 MiB |
| `--gpu-layers 99 --decoder-device cuda` | 0.15 | 0.16 | 0.26 | 454/s | 177/s | 3.6 ms (62%) | 0.008 | 2,219 to 2,265 MiB for the first request, up to about 2,495 MiB for later ones |

The CPU-only row is limited by memory bandwidth, so the speedups against it are
larger than against a better-placed CPU run. On an AMD EPYC 9554 machine, the same
8 threads placed one per CCD read memory 4.5 times faster and ran 2.1 times more
LM forward passes per second (58.5 against 27.8), with RTF 1.01 against 1.35. On
the machine above, that placement ran 49 to 54 forward passes per second in 7 of
9 iterations but was not stable (1.5 and 3.4 per second in the other two, and the
CPU decoder up to 12 times slower), so no faster stable CPU-only row was measured.
The machine is shared with other users: full-GPU RTF over four sessions of
effectively the same code was 0.14 to 0.16, so only paired measurements in one
session are used to compare code changes.

On the GPU one forward pass takes 2.2 ms, less than the CPU sampler, which makes
several passes over all 158,241 logits per token (top-k selection about 1.7 ms,
top-p about 1.25 ms and the final draw about 0.64 ms on this CPU). Sampling
therefore limits LM speed on the GPU. Loading takes about 0.6 s for the GPU LM,
including a 0.2 s repack, against 0.3 s on the CPU. A GPU process also spends about
1.0 s outside the timed stages (0.5 s for a CPU process), and the CUDA decoder takes
about 6 s to load (5.6 to 6.3 s). A one-request full-GPU process therefore spends about 8.5 s
outside synthesis (0.6 s voices, 0.6 s LM, about 6 s decoder, 1.0 s untimed);
`process_seconds` in the `--timings` report gives the wall time since the engine
was created (for the CLI, about the process start; for Python, the model load). The
CPU-only row was measured on the build before these timing fields were added; its
code path is unchanged apart from parsing the voice conditioning once.

`--timings` prints per-stage times for each request: normalization, voice
preparation (first use of a voice only), tokenization, context
creation and release, LM prefill (positions and tok/s), LM decode (forward passes
and tok/s), sampling (time and share of the generation loop), decoder, output,
total, audio seconds and RTF, the wall time since the engine was created, plus
one-time startup costs (model load, placement check, decoder load, normalizer
initialization) for the first request. With
a GPU (the LM's, or device 0 for `--decoder-device cuda`) it also prints peak
device memory, which is the device-wide used memory sampled at the end of each
stage. This needs GGML's CUDA backend: a `-DGGML_CUDA=OFF` build with a CUDA
LibTorch prints no GPU memory. The runtime queries every visible GPU at start,
which creates a CUDA context on each; `CUDA_VISIBLE_DEVICES` limits this. The
value includes the CUDA context, and it is only valid while no other process uses
that GPU; the report flags runs where another process freed memory. Because it is
sampled only at stage ends, it can miss short peaks inside a stage. In the
measurements above it agreed with 100 Hz sampling within 1 MiB, except with
`KITTEN_DECODER_EMPTY_CACHE=1`, which frees the decoder's blocks before the stage
ends: for the 20.9 s chunk it printed 2,219 MiB where 100 Hz sampling saw about
2,496 MiB.
`llama_buffers_mib` in the report lists llama.cpp's own model, KV and compute
buffers per buffer type, which other processes do not affect.
With `--timings` the report gains a `timings` object. The `lm` object lists
the placement, the weights per buffer, the devices, the stages that
stay on the CPU, the precision note, `flash_attn` (the resolved setting) next to
`flash_attn_requested`, and `graph_splits` (backend splits of the prefill and
decode graphs). With the whole LM on the GPU a graph has 2 splits: the token
embedding lookup on the CPU, then everything else on the GPU. More splits mean
an operation fell back to the CPU, and the runtime prints a warning. Two splits
are necessary but not sufficient: an unsupported operation right after the lookup
could join the CPU split without adding one, and the speaker-embedding step's graph
is not counted. `GGML_SCHED_DEBUG=2` lists the operations of each split; on an
RTX 4090 the CPU split holds only the lookup (GET_ROWS).

Streaming and runtime voice cloning are not supported on either device.

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
The CLI runs batch synthesis. The Python binding supports sentence-chunk streaming through the same native engine. The decoder currently depends on LibTorch rather than a GGML S3 implementation.

Additional format checks:

```sh
cmake --build build --target test-tq2_1 test-quantize-fns test-backend-ops -j 8
ctest --test-dir build -R 'test-(tq2_1|quantize-fns)$' --output-on-failure
python tools/kitten-tts/test_tq2.py
build/bin/test-backend-ops test -b CPU -o MUL_MAT,GET_ROWS -p tq2_1
```

Configure with `-DLLAMA_BUILD_TESTS=ON` first. The Python byte-layout check uses the Linux library in `build/bin`. The dedicated native test checks exact reconstruction and the dot product against dequantized Q8_K activations. It also checks the TQ2_1 to Q2_0 repack against a per-element reference at row lengths 256 to 6,144, including all 65,536 FP16 scale bit patterns. Native VNNI, AVX2 without VNNI, and a build with AVX disabled passed locally.

For a CUDA build (`build-cuda`, configured with `-DLLAMA_BUILD_TESTS=ON`):

```sh
cmake --build build-cuda --target test-tq2_1 test-tq2_1-load test-backend-ops -j 8
ctest --test-dir build-cuda -R 'test-tq2_1$' --output-on-failure
build-cuda/bin/test-backend-ops test -b CUDA0 -o MUL_MAT,GET_ROWS -p q2_0
build-cuda/bin/test-tq2_1-load ../kitten-tts-2/cpp/model-tq2_1.gguf
```

`test-tq2_1-load` needs the model file, so ctest does not run it. It loads the model with every layer on the GPU, with and without mmap, and checks that each TQ2_1 weight is on the GPU as Q2_0 and dequantizes to the same values as the file. With the repack turned off it checks that these weights stay on the CPU as TQ2_1. On an RTX 4090, test-backend-ops lists the Q2_0 cases with F16 activations as "not supported"; none fail.

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

## Python binding development

The package at `python/kitten_tts_cpp` wraps `kitten::engine` through a pybind11 extension. The CLI and binding share model loading, token generation, waveform decoding, and audio joins. `setup.py` builds a CPU extension using the pybind11 headers shipped with PyTorch, links the project libraries statically, and bundles the normalizer grammars. LibTorch remains a dependency supplied by the installed PyTorch package.

```sh
python setup.py build_ext --inplace
PYTHONPATH=python \
  KITTEN_TEST_ASSETS=models/kitten2-student-w4 \
  KITTEN_TEST_GGUF=models/kitten2/model-tq2_1.gguf \
  python tools/kitten-tts/test_python.py
```

The integration tests cover actual native inference, CLI token/audio parity, deterministic repeated calls, file output, input validation, and streaming cancellation. Run `test_assets.py` and `check_parity.py` as well after modifying the shared engine. Source distributions include the normalizer source/data, so building from a source archive does not require Git submodule access.
