/*
 * kt2n engine C ABI (G0 interface freeze, 10 Oct 2026).
 *
 * One native library, libkt2n_engine.so, serves Kitten TTS 2 end to end:
 * text normalisation and tokens -> LM -> conformer encoder -> 2-step MeanFlow
 * estimator -> HiFT -> postprocessing -> in-process encoders, plus the clone
 * voice build. The same ABI drives the Python adapter
 * (kitten_tts_09/native_engine_adapter.py, through a nanobind binding that
 * releases the GIL on every call that can wait), the kt2n-say CLI and the
 * bench driver.
 *
 * Plain C (C11 / C++17 and later both compile it). Every struct passed IN
 * starts with `struct_size` so fields can be appended without breaking older
 * callers; the library reads only the bytes the caller declared. Handles are
 * 64-bit generation-checked ids, never pointers: a stale or foreign id fails
 * with KT2N_ERR_INVALID_HANDLE instead of touching freed memory.
 *
 * Threading. Every function is thread-safe. Only kt2n_engine_create,
 * kt2n_engine_destroy and kt2n_poll (with timeout_ms != 0) may wait; every
 * other call returns in microseconds and never waits for the GPU. Callers
 * own the engine pointer's lifetime: stop new calls and finish non-poll calls
 * before destroy, and return all previously borrowed events. Already blocked
 * polls may overlap destroy; they wake with zero events before it frees the
 * engine. No call may begin once destroy starts. No call
 * allocates device memory after create (one device arena + one pinned arena
 * are allocated there); arena exhaustion is admission back-pressure.
 *
 * Clock. Every time is CLOCK_MONOTONIC in nanoseconds. On Linux CPython's
 * time.perf_counter() reads the same clock, so the adapter turns an event's
 * at_ns into production's timeline `at_ms` exactly as GenerationControl does:
 * at_ms = (at_ns / 1e9 - control.timeline_started) * 1000.
 *
 * Production reference for every behaviour named here: kittenml-api main
 * ebcdd3f (paths below are relative to that repository).
 */
#ifndef KT2N_ENGINE_H_
#define KT2N_ENGINE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define KT2N_API __attribute__((visibility("default")))
#else
#define KT2N_API
#endif

/* Bumped on any incompatible change (a field removed, moved or re-typed, an
 * enum value renumbered). Appending fields to a struct_size struct or adding
 * enum values does not bump it. */
#define KT2N_ABI_VERSION 2u

/* ------------------------------------------------------------------------- */
/* Handles                                                                   */
/* ------------------------------------------------------------------------- */
typedef struct kt2n_engine kt2n_engine; /* opaque */
typedef uint64_t kt2n_request_id;       /* one synthesis request */
typedef uint64_t kt2n_session_id;       /* one incremental text session */
typedef uint64_t kt2n_voice_id;         /* a loaded voice (built-in, clone, saved) */
typedef uint64_t kt2n_job_id;           /* an asynchronous voice build */
#define KT2N_INVALID_ID ((uint64_t)0)

/* ------------------------------------------------------------------------- */
/* Status codes and production's public error codes                          */
/* ------------------------------------------------------------------------- */
/* Every function returns a kt2n_status (0 = OK). Codes >= 100 are public
 * errors: kt2n_error_lookup() gives the exact `error.code` string, HTTP status
 * and `error.type` production answers with (kitten_tts/api.py CustomVoiceError,
 * _HTTP_STATUS_ERROR_CODES at :1916, _openai_error_type at :1927;
 * user_facing_website/api_router/tts09_direct.py error_envelope). Codes below
 * 100 are internal: the adapter never shows them to a client as such. */
typedef enum kt2n_status {
  KT2N_OK = 0,
  /* internal */
  KT2N_ERR_INVALID_ARGUMENT = 1, /* caller bug (NULL, bad struct_size); public: internal_error */
  KT2N_ERR_INVALID_HANDLE = 2,   /* unknown or released id */
  KT2N_ERR_ABI_MISMATCH = 3,     /* config.abi_version != KT2N_ABI_VERSION */
  KT2N_ERR_AGAIN = 4,            /* admission full right now; wait on the event fd and retry */
  KT2N_ERR_CANCELLED = 5,        /* cancelled (client gone / shutdown): no public body */
  KT2N_ERR_NOT_READY = 6,        /* create still warming / destroy in progress */
  KT2N_ERR_UNSUPPORTED = 7,      /* a lever value or path the native engine does not implement:
                                    create() fails closed instead of serving it differently */
  KT2N_ERR_TOO_SMALL = 8,        /* output buffer too small (size returned) */

  /* public: request validation (HTTP 4xx) */
  KT2N_ERR_INVALID_REQUEST = 100,         /* invalid_request_error 400; `param` names the field
                                             (sampler fields: kitten_tts/sampler_params.py) */
  KT2N_ERR_INVALID_INPUT = 101,           /* invalid_input 400 ("input must not be blank", api.py:7043) */
  KT2N_ERR_INVALID_VOICE = 102,           /* invalid_voice 400 (api.py:7063) */
  KT2N_ERR_INVALID_RESPONSE_FORMAT = 103, /* invalid_response_format 400 (api.py:7097) */
  KT2N_ERR_INVALID_STREAM_FORMAT = 104,   /* invalid_stream_format 400 (api.py:7104) */
  KT2N_ERR_REQUEST_TOO_LARGE = 105,       /* request_too_large 413 (input > 40,000 chars,
                                             reference audio > 10 MiB decoded) */
  KT2N_ERR_CUSTOM_VOICE_REALTIME_NOT_SUPPORTED = 106, /* custom_voice_realtime_not_supported 400 */
  KT2N_ERR_INPUT_ALREADY_CLOSED = 107,    /* input_already_closed 400; WS close 1008 (api.py:6915) */
  KT2N_ERR_INVALID_SESSION_CONFIGURATION = 108, /* invalid_session_configuration 400 (api.py:5843) */
  KT2N_ERR_MODEL_NOT_FOUND = 109,         /* model_not_found 404 */
  KT2N_ERR_VOICE_NOT_FOUND = 110,         /* voice_not_found 404 */

  /* public: voice build / clone paths (kitten_tts_09/runtime.py 11029-12960) */
  KT2N_ERR_INVALID_AUDIO = 200,                 /* invalid_audio 400 (no decodable audio) */
  KT2N_ERR_INVALID_AUDIO_DURATION = 201,        /* invalid_audio_duration 400 (2-30 s) */
  KT2N_ERR_INVALID_AUDIO_SIGNAL = 202,          /* invalid_audio_signal 400 (silent / too quiet) */
  KT2N_ERR_INVALID_REFERENCE_AUDIO = 203,       /* invalid_reference_audio 400 */
  KT2N_ERR_REFERENCE_ALIGNMENT_FAILED = 204,    /* invalid_reference_audio 422 (runtime.py:8400) */
  KT2N_ERR_SPEAKER_EMBEDDING_FAILED = 205,      /* speaker_embedding_failed 422 */
  KT2N_ERR_SPEAKER_VERIFICATION_FAILED = 206,   /* speaker_verification_failed 422 */
  KT2N_ERR_CONSENT_PHRASE_MISMATCH = 207,       /* consent_phrase_mismatch 400 */
  KT2N_ERR_UNSUPPORTED_CONSENT_LANGUAGE = 208,  /* unsupported_consent_language 400 */
  KT2N_ERR_CONSENT_REQUIRES_SPEAKER_VERIFICATION = 209, /* consent_requires_speaker_verification 409 */
  KT2N_ERR_VOICE_ARTIFACT_VERSION_MISMATCH = 210, /* voice_artifact_version_mismatch 409 */
  KT2N_ERR_VOICE_ARTIFACT_REBUILDING = 211,     /* voice_artifact_rebuilding 503 */
  KT2N_ERR_CUSTOM_VOICE_CREATION_BUSY = 212,    /* custom_voice_creation_busy 429 (4 slots / queue 16) */
  KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR = 213,    /* custom_voice_service_error 503 */
  KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE = 214, /* custom_voice_service_unavailable 503 */
  KT2N_ERR_INVALID_REFERENCE_TRANSCRIPT = 215,  /* invalid_reference_transcript 400, param reference_transcript
                                                   (missing, > 4096 chars / 16 KiB, control characters:
                                                   runtime.py:11029 _custom_transcript) */
  KT2N_ERR_INVALID_CONSENT_TRANSCRIPT = 216,    /* invalid_consent_transcript 400, param consent_transcript
                                                   (same rule, the consent path) */

  /* public: server (HTTP 5xx) */
  KT2N_ERR_SPEECH_GENERATION_FAILED = 300, /* speech_generation_failed 503 before any audio (api.py:7906);
                                              after audio: the transfer is aborted (SpeechStreamAborted);
                                              WS: error event + close 1011 */
  KT2N_ERR_DEADLINE_EXCEEDED = 301,        /* public code speech_generation_failed (production raises
                                              TimeoutError -> same envelope); kept apart for stats */
  KT2N_ERR_SERVICE_UNAVAILABLE = 302,      /* service_unavailable 503 (engine owner failed) */
  KT2N_ERR_INTERNAL = 303                  /* internal_error 500 */
} kt2n_status;

typedef struct kt2n_error_info {
  int32_t status;      /* kt2n_status */
  int32_t http_status; /* 400, 404, 409, 413, 422, 429, 500, 503; 0 for internal codes */
  const char* code;    /* production `error.code`, e.g. "invalid_request_error" (static storage) */
  const char* type;    /* production `error.type`: invalid_request_error | authentication_error |
                          rate_limit_error | server_error (static storage) */
} kt2n_error_info;

/* Static table lookup; never NULL (unknown status -> internal_error 500). */
KT2N_API const kt2n_error_info* kt2n_error_lookup(int32_t status);

/* Detail of one failure. Optional out-parameter of most calls (NULL = not wanted). */
typedef struct kt2n_error {
  int32_t status;      /* kt2n_status */
  int32_t http_status; /* as kt2n_error_lookup(status)->http_status */
  char code[48];       /* public error.code, NUL-terminated */
  char param[48];      /* public error.param ("" = null), e.g. "temperature", "voice", "input" */
  char message[400];   /* public error.message, production's wording where it exists */
} kt2n_error;

/* ------------------------------------------------------------------------- */
/* Engine                                                                    */
/* ------------------------------------------------------------------------- */
enum kt2n_config_flags {
  KT2N_CFG_TIMELINE_ALLOWED = 1u << 0, /* lab: requests may enable the benchmark timeline */
  KT2N_CFG_X_SCHED = 1u << 1,          /* experiment arm X-SCHED (deadline / urgent-class dispatch).
                                          NEVER in the RC: the RC ports production's timing rules */
  KT2N_CFG_NO_GRAPHS = 1u << 2,        /* debug: no CUDA graphs (eager launches) */
  KT2N_CFG_SPIN_KEEPWARM = 1u << 3,    /* keep the engine awake between requests */
  KT2N_CFG_INFERENCE_ONLY = 1u << 4   /* reserve only configured voices; disable clone builds */
};

typedef struct kt2n_engine_config {
  uint32_t struct_size;  /* sizeof(kt2n_engine_config) */
  uint32_t abi_version;  /* KT2N_ABI_VERSION */
  uint32_t flags;        /* kt2n_config_flags */
  int32_t device;        /* CUDA ordinal */

  /* Files (the container format in tools/kt2n_assets/format.py). */
  const char* lm_file;        /* model.kt2n: packed language model */
  const char* s3gen_file;     /* s3gen.kt2n: FP32 encoder, BF16 estimator, folded FP32 HiFT */
  const char* voices_file;    /* voices.kt2n: prepared voice conditioning */
  const char* tokenizer_json; /* tokenizer matching the language model */
  const char* text_data_dir;  /* prepared normalization grammars and regex tables */

  /* Optional inference policy overrides: {"KITTENTTS_09_X": "value", ...}.
   * NULL uses the built-in defaults. Unsupported values fail with
   * KT2N_ERR_UNSUPPORTED and identify the setting in error.param. */
  const char* levers_json;
  const char* app; /* NULL or "default"; legacy "us", "ap", "fq" profiles are accepted */

  /* Capacity. 0 = default. */
  int32_t max_streams;          /* concurrent requests/phrases; inference-only default is 1 */
  int32_t max_lm_rows;          /* LM decode rows (step graphs for buckets 1-16, 20, 24, 32, 40, 48, 64) */
  int32_t render_pages;         /* 64-frame K/V pages; 0 = voice prefixes plus 40 pages per stream */
  int32_t inline_voice_lru;     /* inline clone LRU entries (production: 32) */
  int32_t clone_slots;          /* concurrent voice builds (production: 4) */
  int32_t clone_queue;          /* queued voice builds before custom_voice_creation_busy (production: 16) */
  int32_t clone_timeout_ms;     /* production: 120,000 */
  uint64_t device_arena_bytes;  /* initialization GPU allocation budget; 0 = available VRAM */
  uint64_t pinned_arena_bytes;  /* reserved; must be 0 */

  /* CPU threads (budget: production's 4 physical cores / 8 vCPU). The conductor
   * and the render dispatcher are always 1 thread each. 0 = default. */
  int32_t threads_text;  /* 0 = default 4 */
  int32_t threads_post;  /* default 2 (post + encoders) */
  int32_t threads_voice; /* default 1 */
  const char* cpuset;    /* optional "0-7" style list the engine pins its threads to; NULL = inherit */

  /* Appended (L8, 10 Oct; append-only, no ABI bump): the clone-build networks until s3gen.kt2n carries
   * them. A directory holding s3gen.safetensors (the S3Gen weights: speech tokenizer + speaker encoder),
   * voicebuild_consts.safetensors and voicebuild_pyannote.safetensors (L11 converters). NULL = clone
   * builds unavailable (kt2n_voice_build -> custom_voice_service_unavailable). */
  const char* voice_build_dir;

  /* Optional custom-voice capacity. 0 preserves the production cache reserve;
   * a positive value reserves this many additional LM/render voice slots.
   * Prefix storage uses the production allowance of 512 tokens per slot;
   * unusually long references can exhaust that pool before the slot count.
   * Ignored with KT2N_CFG_INFERENCE_ONLY. */
  int32_t max_custom_voices;
} kt2n_engine_config;

KT2N_API uint32_t kt2n_abi_version(void);
/* Static string: git sha, build time, CUDA version, compiled archs. */
KT2N_API const char* kt2n_build_info(void);
/* Capability tokens an engine includes (as plain substrings) in kt2n_build_info().
 * KT2N_CAP_AUDIO_OPENING_SKIP: every AUDIO event carries kt2n_event.opening_skipped_tokens. */
#define KT2N_CAP_AUDIO_OPENING_SKIP "cap:audio.opening_skipped_tokens"
#define KT2N_CAP_VOICE_CLONING "cap:voice.cloning"

/* Loads every file, allocates both arenas, builds the voice prefixes, captures
 * the graphs and warms up. Blocks (seconds). */
KT2N_API int32_t kt2n_engine_create(const kt2n_engine_config* config, kt2n_engine** out, kt2n_error* err);
/* Cancels everything (KT2N_ERR_CANCELLED to each owner), waits for the device
 * to go idle, wakes and drains already blocked polls, then frees everything.
 * The pointer lifetime rules above apply. The event fd is closed. Safe on NULL. */
KT2N_API void kt2n_engine_destroy(kt2n_engine* engine);

/* ------------------------------------------------------------------------- */
/* Voices                                                                    */
/* ------------------------------------------------------------------------- */
typedef enum kt2n_voice_kind {
  KT2N_VOICE_BUILTIN = 0, /* voices.kt2n; immutable, never evicted */
  KT2N_VOICE_INLINE = 1,  /* one-shot clone from a request (LRU of inline_voice_lru) */
  KT2N_VOICE_SAVED = 2    /* a stored custom voice loaded from its artifact */
} kt2n_voice_kind;

typedef struct kt2n_voice_info {
  uint32_t struct_size;
  int32_t kind;            /* kt2n_voice_kind */
  char name[96];           /* internal voice id, e.g. "03_radio_dj_smooth_male"; clones: the build's key */
  char language[16];       /* e.g. "en" (catalogue language_code, lower-case primary subtag) */
  int32_t prompt_tokens;   /* decoder prompt tokens Tp (built-ins: 201..250) */
  int32_t prompt_frames;   /* 2 * Tp */
  int32_t lm_prefix_ids;   /* LM reference prompt length */
  int32_t hidden;          /* 1 = the 48th voice: nearest-flagship projection only, not public */
  float peak_gain_db;      /* streamed peak-norm start gain (custom: the catalogue median) */
  int32_t refs;            /* in-flight requests holding it */
} kt2n_voice_info;

/* Built-in id or public alias (production's catalogue + LEGACY_VOICE_NAMES,
 * runtime.py:2957) -> id. Unknown: KT2N_ERR_INVALID_VOICE. */
KT2N_API int32_t kt2n_voice_lookup(kt2n_engine* engine, const char* name, kt2n_voice_id* out, kt2n_error* err);
KT2N_API int32_t kt2n_voice_info_get(kt2n_engine* engine, kt2n_voice_id voice, kt2n_voice_info* out);
/* The voice's 512-d speaker embedding (pyannote; consent verification and
 * nearest-flagship use it). */
KT2N_API int32_t kt2n_voice_speaker(kt2n_engine* engine, kt2n_voice_id voice, float out[512]);

enum kt2n_voice_build_flags {
  KT2N_VB_INLINE = 1u << 0,       /* keep in the inline LRU under identity_key */
  KT2N_VB_SAVED = 1u << 1,        /* the result will be exported and stored (kt2n_voice_export) */
  KT2N_VB_DECODED_PCM = 1u << 2,  /* input is float PCM (pcm/pcm_samples/pcm_rate), not encoded bytes */
  KT2N_VB_CONSENT = 1u << 3,      /* this audio is a consent recording: validate its transcript/language,
                                     then decode + pyannote only (runtime.py:12277 create_voice_consent).
                                     The VOICE_BUILT voice is speaker-only: read kt2n_voice_speaker()
                                     for the store, then kt2n_voice_evict(); submit with it fails
                                     (KT2N_ERR_INVALID_VOICE) */
  KT2N_VB_VERIFY_SPEAKER = 1u << 4 /* compare against consent_speaker (speaker_verification_failed) */
};

/* Clone build inputs. Every documented clone path (consent, no consent, admin,
 * inline, saved) maps onto these flags; the API layer keeps auth/ownership. */
typedef struct kt2n_voice_build_params {
  uint32_t struct_size;
  uint32_t flags;                   /* kt2n_voice_build_flags */
  const void* audio;                /* encoded reference audio (<= 10 MiB decoded) */
  size_t audio_len;
  const char* audio_format;         /* "mp3"|"wav"|"ogg"|"aac"|"flac"|"webm"|"mp4" (api.py InlineReferenceAudio) */
  const float* pcm;                 /* with KT2N_VB_DECODED_PCM: mono float PCM */
  size_t pcm_samples;
  int32_t pcm_rate;
  int32_t consent_speaker_dim;      /* 512 with KT2N_VB_VERIFY_SPEAKER, else 0 */
  const float* consent_speaker;     /* the consent's stored speaker embedding */
  const char* reference_transcript; /* REQUIRED, 1..4096 chars and <= 16 KiB, no control characters
                                       (runtime.py:11029). The relay transcribes the reference with its
                                       ASR when the client sent none (tts09_direct.py :2302,
                                       reference_transcription_failed), so the engine always gets one.
                                       With KT2N_VB_CONSENT it is the consent transcript, checked against
                                       the consent phrase (runtime.py:11054 consent_phrase_mismatch) */
  const char* language;             /* BCP 47 tag; consent: production supports English only */
  const char* identity_key;         /* inline LRU key (production: request-bound lease identity) */
  int64_t deadline_ns;              /* absolute; 0 = clone_timeout_ms from now */
  uint64_t user_data;               /* echoed in the KT2N_EVENT_VOICE_BUILT event */
} kt2n_voice_build_params;

/* Queues a build (admission: clone_slots running + clone_queue waiting, else
 * KT2N_ERR_CUSTOM_VOICE_CREATION_BUSY). The result arrives as one
 * KT2N_EVENT_VOICE_BUILT event (voice id, or status + error). An inline key
 * already in the LRU completes immediately with the cached voice. */
KT2N_API int32_t kt2n_voice_build(kt2n_engine* engine, const kt2n_voice_build_params* params, kt2n_job_id* out,
                                  kt2n_error* err);
/* Optional additive ABI: atomically cancel a build only while it remains queued.
 * *cancelled is 1 if marked cancelled under the dequeue mutex, otherwise 0.
 * An admitted, terminal or unknown job is unchanged. Repeated calls are safe.
 * Older libraries may omit this symbol; bindings must then report false. */
KT2N_API int32_t kt2n_voice_build_cancel_queued(kt2n_engine* engine, kt2n_job_id job, int32_t* cancelled);
/* Saved voices: the artifact is a one-voice voices.kt2n blob (the container format in tools/kt2n_assets/format.py).
 * Production-format artifacts in the voice store are converted offline or
 * rebuilt (production's voice_artifact_rebuilding path); load never guesses. */
KT2N_API int32_t kt2n_voice_load(kt2n_engine* engine, const void* artifact, size_t len, kt2n_voice_id* out,
                                 kt2n_error* err);
/* Writes the one-voice artifact. buf == NULL or too small: *len = bytes needed,
 * returns KT2N_ERR_TOO_SMALL. */
KT2N_API int32_t kt2n_voice_export(kt2n_engine* engine, kt2n_voice_id voice, void* buf, size_t cap, size_t* len);
/* Drops the engine's reference; requests already holding the voice finish
 * with it. Built-ins cannot be evicted (KT2N_ERR_INVALID_ARGUMENT). */
KT2N_API int32_t kt2n_voice_evict(kt2n_engine* engine, kt2n_voice_id voice);

/* ------------------------------------------------------------------------- */
/* Requests                                                                  */
/* ------------------------------------------------------------------------- */
typedef enum kt2n_audio_format {
  KT2N_FORMAT_PCM = 0, /* raw s16le, 24 kHz mono */
  KT2N_FORMAT_WAV = 1, /* streaming WAV header (api.py streaming_wav_header) or complete header */
  KT2N_FORMAT_MP3 = 2, /* libmp3lame 128k, -write_xing 0 (api.py:3149) */
  KT2N_FORMAT_OPUS = 3,/* libopus 96k (api.py:3151) */
  KT2N_FORMAT_AAC = 4,
  KT2N_FORMAT_FLAC = 5,
  /* Appended 10 Oct (L8b adapter; an appended enum value, no ABI bump). Host float32 LE,
   * 24 kHz mono, at production's engine -> transport boundary: what
   * KittenTTS09Engine.generate_stream yields (after the engine's trims, speed stretch and
   * startup hold, BEFORE kitten_tts/api.py's normalize_audio + peak norm + PCM16 / encoder).
   * The Python adapter asks for it so api.py's routes run unchanged (levelling and encoding
   * stay exactly production's). The in-process L10 post/encoder path is the bytes formats
   * above. */
  KT2N_FORMAT_F32 = 6
} kt2n_audio_format;

typedef enum kt2n_output_mode {
  KT2N_OUTPUT_STREAM = 0,      /* progressive chunks (HTTP audio / SSE / binary transports) */
  KT2N_OUTPUT_STREAM_FILE = 1, /* the buffered complete WAV of stream_format=audio + wav: the streaming
                                  path, then the demo peak formula over the whole file
                                  (api.py peak_norm_full_file :2805) */
  KT2N_OUTPUT_COMPLETE = 2     /* `stream: false`: one complete file in any format, DEMO path
                                  (speech_mode.py COMPLETE_FILE_HEADER; whole render + 3 x 4299) */
} kt2n_output_mode;

typedef enum kt2n_speech_mode {
  KT2N_SPEECH_MODE_DEFAULT = 0, /* what the levers resolve (live: SPEECH_MODES=0, so QUALITY-less base) */
  KT2N_SPEECH_MODE_FAST = 1,
  KT2N_SPEECH_MODE_QUALITY = 2,
  KT2N_SPEECH_MODE_DEMO = 3
} kt2n_speech_mode;

/* Public sampler fields (kitten_tts/sampler_params.py). Only the fields whose
 * bit is set in `present` were sent; the engine validates them (RANGES there)
 * and resolves the preset exactly like sampler_params.resolve with the levers'
 * KITTENTTS_09_SAMPLER_DEFAULT. "Advanced" in the demo = explicit fields. */
typedef enum kt2n_sampler_mode {
  KT2N_SAMPLER_UNSET = 0,
  KT2N_SAMPLER_STABLE = 1,    /* temperature 0.8, top_p 0.8, top_k 50, min_p 0 */
  KT2N_SAMPLER_EXPRESSIVE = 2 /* temperature 0.9, top_p 0.9, top_k 50, min_p 0 */
} kt2n_sampler_mode;

enum kt2n_sampler_present {
  KT2N_SAMPLER_HAS_MODE = 1u << 0,
  KT2N_SAMPLER_HAS_TEMPERATURE = 1u << 1, /* 0.3 .. 2.0 */
  KT2N_SAMPLER_HAS_TOP_P = 1u << 2,       /* 0.5 .. 1.0 */
  KT2N_SAMPLER_HAS_TOP_K = 1u << 3,       /* integer 0 .. 200 (0 = no top-k) */
  KT2N_SAMPLER_HAS_MIN_P = 1u << 4,       /* 0.0 .. 0.5 */
  KT2N_SAMPLER_HAS_MAX_NEW_TOKENS = 1u << 5 /* integer 500 .. 12000 */
};

typedef struct kt2n_sampler {
  uint32_t present; /* kt2n_sampler_present */
  int32_t mode;     /* kt2n_sampler_mode */
  double temperature;
  double top_p;
  double min_p;
  int64_t top_k;
  int64_t max_new_tokens;
} kt2n_sampler;

enum kt2n_request_flags {
  KT2N_REQ_TIMELINE = 1u << 0,   /* emit KT2N_EVENT_TIMELINE (benchmark trace; needs KT2N_CFG_TIMELINE_ALLOWED) */
  KT2N_REQ_SEED = 1u << 1,       /* use `seed` as the segment-seed base instead of the request-id salt (lab pairing) */
  KT2N_REQ_DEMO_CHECK = 1u << 2, /* DEMO+CHECK retries (speech_mode.py DEMO_CHECK_HEADER) */
  KT2N_REQ_NORMALIZE_OFF = 1u << 4 /* raw text for requests / every phrase of a session */
};

typedef struct kt2n_request_params {
  uint32_t struct_size;
  uint32_t flags;          /* kt2n_request_flags */
  const char* text;        /* UTF-8 input as the engine transport receives it (before normalisation) */
  size_t text_len;         /* UTF-8 bytes */
  kt2n_voice_id voice;
  const char* request_id;  /* production request id: segment seed = salt + ordinal, salt =
                              (BE32(sha256(request_id)[0:4]) & 0x3FFFFFFF) * 1024 (runtime.py:2946) */
  uint64_t seed;           /* with KT2N_REQ_SEED only */
  kt2n_sampler sampler;
  double speed;            /* 0.25 .. 4.0 (api.py:413); 1.0 = no stretch */
  int32_t format;          /* kt2n_audio_format */
  int32_t output;          /* kt2n_output_mode */
  int32_t speech_mode;     /* kt2n_speech_mode */
  int32_t reserved0;
  int64_t deadline_ns;     /* absolute CLOCK_MONOTONIC; 0 = none */
  uint64_t user_data;      /* echoed in every event of this request */
  /* Appended 10 Oct (L8b adapter; read only when struct_size covers it, NULL = none).
   * The request's production GenerationControl policy fields (kitten_tts/api.py :2012) as one
   * JSON object, production's names, only the fields the transport set (not null). The Python
   * adapter keeps api.py's WS phrase planner, so each realtime phrase arrives as one submit
   * carrying the overrides api.py gives a phrase (:6089-6091): "startup_hold_seconds": 0,
   * "initial_audio_buffer_seconds": 0, "concurrent_initial_audio_buffer_seconds": 0, and the
   * turn's EMO decision "emotion" (bool) / "emotion_carry_tag" (live: EMO_REQUEST_LEVEL=1).
   * Also: "skip_opening_silence", "q_skip_limiter" (DEMO, speech_mode.apply_to_control),
   * "demo_check", "phrase" (true for a WS phrase), and the traced lab overrides
   * ("startup_hold_override", "delivery_override", "preview_finalize", "preview_flow_steps",
   * "n_*", "normalize_fast_path", "incremental_normalization", "q_overrides", "lab_force_ids",
   * "lab_demo_chunks"). The engine applies every key it knows exactly as runtime.py does; a key
   * it does not implement fails the submit with KT2N_ERR_UNSUPPORTED, param = the key, so a
   * request is never served under a different policy. */
  const char* control_json;
} kt2n_request_params;

/* Validates (public 400s come back here, synchronously, before any work),
 * then queues. Never waits for the GPU. */
KT2N_API int32_t kt2n_submit(kt2n_engine* engine, const kt2n_request_params* params, kt2n_request_id* out,
                             kt2n_error* err);

typedef enum kt2n_cancel_reason {
  KT2N_CANCEL_REQUEST = 0,  /* production "request_cancelled" (client closed) */
  KT2N_CANCEL_DEADLINE = 1, /* "deadline_exceeded" */
  KT2N_CANCEL_SHUTDOWN = 2
} kt2n_cancel_reason;

/* Cancels a request, a session (all its phrases) or a voice build. Idempotent.
 * The owner still receives a terminal event (KT2N_EVENT_DONE with status
 * KT2N_ERR_CANCELLED / KT2N_ERR_DEADLINE_EXCEEDED). */
KT2N_API int32_t kt2n_cancel(kt2n_engine* engine, uint64_t id, int32_t reason);

/* Frees a finished request / session / build id (after its terminal event).
 * Until then the id stays valid for kt2n_cancel. */
KT2N_API int32_t kt2n_release(kt2n_engine* engine, uint64_t id);

/* ------------------------------------------------------------------------- */
/* Realtime WebSocket sessions                                               */
/* ------------------------------------------------------------------------- */
/* The engine owns the phrase planner (kitten_tts/api.py realtime_phrase_cut :353,
 * _RealtimeEmotionTurn :2233, _RealtimeTtsPhrase :2355): every phrase becomes an
 * internal generation with its own phrase_index; events carry it. */
typedef struct kt2n_session_params {
  uint32_t struct_size;
  uint32_t flags;          /* kt2n_request_flags */
  kt2n_voice_id voice;     /* clones: KT2N_ERR_CUSTOM_VOICE_REALTIME_NOT_SUPPORTED, as production */
  const char* request_id;  /* session id for seeds/logs (phrases derive their own as production does) */
  uint64_t seed;
  kt2n_sampler sampler;    /* from the socket query (from_text spellings validated by the adapter) */
  double speed;
  int32_t format;          /* KT2N_FORMAT_PCM or KT2N_FORMAT_F32 */
  int32_t speech_mode;
  int64_t deadline_ns;
  uint64_t user_data;
} kt2n_session_params;

KT2N_API int32_t kt2n_session_open(kt2n_engine* engine, const kt2n_session_params* params, kt2n_session_id* out,
                                   kt2n_error* err);
/* Appends client text (a text frame). After kt2n_session_done:
 * KT2N_ERR_INPUT_ALREADY_CLOSED. */
KT2N_API int32_t kt2n_session_text_append(kt2n_engine* engine, kt2n_session_id session, const char* text,
                                          size_t len, kt2n_error* err);
/* The client's commit / flush: everything appended so far is spoken without
 * waiting for more text (production records the turn end). */
KT2N_API int32_t kt2n_session_commit(kt2n_engine* engine, kt2n_session_id session, kt2n_error* err);
/* Input closed: flush, finish every phrase, then KT2N_EVENT_DONE for the session. */
KT2N_API int32_t kt2n_session_done(kt2n_engine* engine, kt2n_session_id session, kt2n_error* err);

/* ------------------------------------------------------------------------- */
/* Events                                                                    */
/* ------------------------------------------------------------------------- */
typedef enum kt2n_event_kind {
  KT2N_EVENT_AUDIO = 1,        /* encoded audio bytes in the request's format */
  KT2N_EVENT_TIMELINE = 2,     /* one production timeline point (name + JSON fields) */
  KT2N_EVENT_METRICS = 4,      /* end-of-generation engine metrics (production control.metrics_snapshot) */
  KT2N_EVENT_PHRASE_DONE = 5,  /* WS: one phrase finished (phrase_index) */
  KT2N_EVENT_DONE = 6,         /* terminal: request / session finished; status = OK or the failure */
  KT2N_EVENT_VOICE_BUILT = 7,  /* terminal for a build job: voice, or status + error */
  KT2N_EVENT_HEADERS = 8       /* response metadata before the first audio (fields_json: speech_mode,
                                  sample_rate, complete-file byte length when known) */
} kt2n_event_kind;

/* Every pointer member points into engine-owned memory that stays valid until
 * kt2n_events_release() for that event. Unused members are 0 / NULL / -1. */
typedef struct kt2n_event {
  int32_t kind;            /* kt2n_event_kind */
  int32_t status;          /* DONE / VOICE_BUILT / mid-stream failure: kt2n_status; else 0 */
  uint64_t id;             /* request, session or job id */
  uint64_t user_data;      /* from the submit / open / build params */
  uint64_t seq;            /* per-id sequence number from 0, no gaps */
  int64_t at_ns;           /* CLOCK_MONOTONIC when it happened (not when it was polled) */
  int32_t segment_index;   /* production segment_index, -1 = none */
  int32_t phrase_index;    /* WS phrase, -1 = none */

  /* KT2N_EVENT_AUDIO */
  const uint8_t* data;     /* encoded bytes (PCM16 LE for KT2N_FORMAT_PCM) */
  size_t data_len;
  int64_t sample_begin;    /* first 24 kHz sample of this chunk in the request's (post-speed) output */
  int32_t samples;         /* samples this chunk carries */
  int32_t engine_chunk_index; /* production's engine_chunk_index, from 0 */
  int32_t is_final_chunk;  /* 1 on the last audio event of the request / phrase */
  /* (L8b, 10 Oct; was reserved0: same offset, no ABI bump) production's control.n_opening_skipped_tokens
   * for this request / WS phrase: the opening-silence ids the skip dropped on segment 0 (runtime.py:15545
   * @ebcdd3f, "final when the opening closes, before the first publication"). Set on EVERY audio event,
   * so the first one already carries the final count. api.py's realtime reserve reads it before it
   * releases a phrase's first audio (kitten_tts/api.py:1631, the skip-aware per-listener term,
   * KITTENTTS_09_SKIP_HOLD_*). An engine that fills it says KT2N_CAP_AUDIO_OPENING_SKIP in
   * kt2n_build_info(); older builds leave 0 here, and the adapter then refuses to serve while the
   * skip-aware hold is configured (a silent timing-policy change otherwise). */
  int32_t opening_skipped_tokens;

  /* KT2N_EVENT_TIMELINE / METRICS / HEADERS */
  const char* name;        /* production event name (KT2N_TL_* below; static storage) */
  const char* fields_json; /* JSON object: production's field names and units for that event */

  /* DONE / VOICE_BUILT with an error */
  const kt2n_error* error;

  /* KT2N_EVENT_VOICE_BUILT */
  kt2n_voice_id voice;
} kt2n_event;

/* An eventfd (EFD_NONBLOCK | EFD_CLOEXEC) owned by the engine: readable while
 * events are queued. read() it to re-arm, then kt2n_poll until it returns 0
 * events. One fd for the whole engine; demultiplex by id / user_data. */
KT2N_API int kt2n_event_fd(kt2n_engine* engine);
/* Moves up to `max` queued events into `out` (*n = count). timeout_ms: 0 =
 * never wait, < 0 = wait until at least one event, > 0 = wait at most that
 * long. Shutdown wakes blocked polls with zero events. The binding releases
 * the GIL around it and retains the engine until copied buffers are returned. */
KT2N_API int32_t kt2n_poll(kt2n_engine* engine, kt2n_event* out, int32_t max, int32_t timeout_ms, int32_t* n);
/* Returns the events' buffers to the engine (pinned / post-pool memory). */
KT2N_API void kt2n_events_release(kt2n_engine* engine, const kt2n_event* events, int32_t n);

/* ------------------------------------------------------------------------- */
/* Stats                                                                     */
/* ------------------------------------------------------------------------- */
typedef struct kt2n_stats {
  uint32_t struct_size;
  int32_t active_requests;   /* holding a seat */
  int32_t queued_requests;   /* admitted to the engine, waiting for a seat */
  int32_t active_sessions;
  int32_t lm_rows;           /* rows in the current decode step */
  int32_t render_queue_depth;/* render batches queued (all render streams) */
  int32_t render_inflight;
  int32_t pages_total;       /* 64-frame K/V pages */
  int32_t pages_free;
  int32_t voices_loaded;
  int32_t voice_builds_active;
  int32_t voice_builds_queued;
  uint64_t device_arena_bytes;
  uint64_t device_arena_used;
  uint64_t pinned_arena_bytes;
  uint64_t pinned_arena_used;
  uint64_t requests_done;
  uint64_t requests_failed;
  uint64_t requests_cancelled;
  uint64_t lm_steps;
  uint64_t renders;          /* render calls */
  uint64_t render_rows;      /* rows over all render calls */
  uint64_t underrun_risks;   /* renders that published after their underrun deadline (engine's own view) */
} kt2n_stats;

KT2N_API int32_t kt2n_stats_get(kt2n_engine* engine, kt2n_stats* out);
/* Health snapshot as JSON (production /health engine block shape where one exists).
 * buf too small: *len = bytes needed (incl. NUL), KT2N_ERR_TOO_SMALL. */
KT2N_API int32_t kt2n_stats_json(kt2n_engine* engine, char* buf, size_t cap, size_t* len);

/* ------------------------------------------------------------------------- */
/* Timeline event names (KT2N_EVENT_TIMELINE `name`)                         */
/* ------------------------------------------------------------------------- */
/* Exactly production's names (GenerationControl.mark_timeline, kitten_tts/api.py
 * :2202), so eval/analyse/fa_breakdown.py and eval/analyse/underruns.py run
 * unchanged, and customer_origin.py's ENGINE_STAGE_EVENTS stamp the same origin
 * stages (normalized, admitted, prefill, first_commit, hold_released,
 * engine_chunk). "Required fields" are the ones those scripts read; the native
 * engine must emit them with the same names and units (ms floats, counts ints).
 * Other production fields are emitted when the native engine has the value.
 * Production file:line of the emitter (ebcdd3f) in brackets. */

/* Text and admission */
#define KT2N_TL_GENERATE_STREAM_ENTERED "gpu.generate_stream_entered" /* [runtime.py:20499] */
#define KT2N_TL_EMOTION_DECIDED "gpu.emotion_decided"       /* emo_* [runtime.py:20520] */
#define KT2N_TL_TEXT_SEGMENTED "gpu.text_segmented"         /* characters, segments, wfst [runtime.py:20557] */
#define KT2N_TL_TEXT_NORMALIZED "gpu.text_normalized"       /* stage `normalized`; characters, fast_path,
                                                               incremental, normalized_segments,
                                                               pending_segments [runtime.py:20596] */
#define KT2N_TL_SEGMENT_NORMALIZED "gpu.segment_normalized" /* stage `normalized`; characters, chunks,
                                                               normalize_ms, segment_index [runtime.py:20272] */
#define KT2N_TL_LM_STARTED "cpu.lm_started"                 /* [runtime.py:21314] */
#define KT2N_TL_SCHEDULER_ENQUEUED "gpu.scheduler_enqueued" /* [runtime.py:19772] */
#define KT2N_TL_SCHEDULER_REGISTERED "gpu.scheduler_registered" /* [runtime.py:19781] */
#define KT2N_TL_SCHEDULER_EXTENDED "gpu.scheduler_extended" /* [runtime.py:19813] */
#define KT2N_TL_SCHEDULER_ADMITTED "gpu.scheduler_admitted" /* stage `admitted`. Required: admission_batch,
                                                               active_before, queue_ms, segment_index
                                                               [runtime.py:19009] */
#define KT2N_TL_SCHEDULER_REST "gpu.scheduler_rest"         /* [runtime.py:18903] */
#define KT2N_TL_SCHEDULER_RESUME "gpu.scheduler_resume"     /* [runtime.py:18938] */
#define KT2N_TL_SCHEDULER_FINISHED "gpu.scheduler_finished" /* [runtime.py:13452] */
#define KT2N_TL_SEGMENT_PREFETCH_RELEASED "gpu.segment_prefetch_released" /* [runtime.py:20386] */
#define KT2N_TL_DEMO_PREFETCH_RELEASED "gpu.demo_prefetch_released" /* [runtime.py:14835] */

/* LM */
#define KT2N_TL_LM_PREFILL_FINISHED "gpu.lm_prefill_finished" /* stage `prefill`. Required: batch_size,
                                                                 segment_index; also lm_share_ms,
                                                                 lm_wall_ms [runtime.py:19032] */
#define KT2N_TL_NATIVE_PREFILL "gpu.kt2n_native_prefill"      /* lab native-prefill marker (underruns.py
                                                                 OWNER_EVENTS) */
#define KT2N_TL_LM_DECODE_STEP "gpu.lm_decode_step"           /* batch_size, cuda_ms, generated_tokens,
                                                                 segment_index, wall_ms [runtime.py:14488] */
#define KT2N_TL_N_OPENING_SKIP "gpu.n_opening_skip"           /* Required: skipped (list of ids); cap, closing,
                                                                 lm_head, pending, skipped_in_set
                                                                 [runtime.py:6889] */
#define KT2N_TL_LM_TOKENS_FINALIZED "gpu.lm_tokens_finalized" /* audio_token_count, audio_token_sha256,
                                                                 segment_index [runtime.py:16525] */
#define KT2N_TL_LM_SEGMENT_FINISHED "gpu.lm_segment_finished" /* Required: lm_steps, segment_index; budget,
                                                                 finish_reason, audio_tokens, silent_tokens,
                                                                 longest_silent_run_* [runtime.py:14758] */
#define KT2N_TL_Q_PRE_EOS "gpu.q_pre_eos"                     /* [runtime.py:16606] */

/* Commit planning and renders */
#define KT2N_TL_Q_COMMIT_PLAN "gpu.q_commit_plan"   /* [runtime.py:20025] */
#define KT2N_TL_D_COMMIT_PLAN "gpu.d_commit_plan"   /* buffer_ms, commit_index, next_commit_tokens, render_ms,
                                                       segment_index, token_ms [runtime.py:17238] */
#define KT2N_TL_D_EMERGENCY_COMMIT "gpu.d_emergency_commit" /* [runtime.py:17167] */
#define KT2N_TL_D_JOIN_CAP "gpu.d_join_cap"         /* [runtime.py:17953] */
#define KT2N_TL_Q_ALIGN_COMMIT "gpu.q_align_commit" /* [runtime.py:17375] */
#define KT2N_TL_Q_PAUSE_FLUSH "gpu.q_pause_flush"   /* [runtime.py:17465] */
#define KT2N_TL_Q_PAUSE_TAIL "gpu.q_pause_tail"     /* [runtime.py:17575] */
#define KT2N_TL_Q_PAUSE "gpu.q_pause"               /* [runtime.py:17586] */
#define KT2N_TL_Q_SEAM_SNAP "gpu.q_seam_snap"       /* [runtime.py:17644] */
#define KT2N_TL_Q_FAST_SEAM_SNAP "gpu.q_fast_seam_snap" /* [runtime.py:17730] */
#define KT2N_TL_T9_DEFERRAL_ENDED "gpu.t9_acoustic_deferral_ended" /* Required: deferred_turns, reservoir_s,
                                                                      segment_index [runtime.py:18712] */
#define KT2N_TL_DW_QUEUED "gpu.dw_queued"           /* [runtime.py:16361] */
#define KT2N_TL_DW_RENDER "gpu.dw_render"           /* [runtime.py:16455] */
#define KT2N_TL_RG_FIRST_HANDOFF "gpu.rg_first_handoff" /* [runtime.py:16999] */
#define KT2N_TL_S3GEN_STARTED "gpu.s3gen_started"   /* render launched. Required: commit_tokens, prefix_tokens,
                                                       segment_index; also batch_size, async_launch,
                                                       cached_encoder, cached_flow [runtime.py:16069] */
#define KT2N_TL_PREVIEW_PUBLICATION "gpu.preview_publication" /* stage `first_commit` (1-step preview);
                                                                 audio_ms, audio_samples, batch_size,
                                                                 flow_steps, preview_share_ms,
                                                                 preview_tokens, preview_wall_ms,
                                                                 segment_index [runtime.py:15941] */
#define KT2N_TL_S3GEN_PUBLICATION "gpu.s3gen_publication" /* stage `first_commit` (first one). Required:
                                                             s3gen_wall_ms, s3gen_gpu_ms, s3gen_launch_ms,
                                                             s3gen_queue_depth, batch_size, commit_tokens,
                                                             audio_ms, segment_index; also audio_samples,
                                                             flow_cuda_ms, hift_cuda_ms, lm_ms_total,
                                                             s3gen_share_ms [runtime.py:16133, :18424] */
#define KT2N_TL_S3GEN_FINAL_PUBLICATION "gpu.s3gen_final_publication" /* the segment's final flush; same
                                                                         fields as s3gen_publication
                                                                         [runtime.py:16683] */
#define KT2N_TL_CPU_S3GEN_FINAL_PUBLICATION "cpu.s3gen_final_publication" /* [runtime.py:14150] */
#define KT2N_TL_CPU_LM_TOKENS_FINALIZED "cpu.lm_tokens_finalized" /* [runtime.py:13782] */

/* Post-processing inside the engine (production emits these from runtime.py
 * despite the "api." prefix) */
#define KT2N_TL_STARTUP_HOLD_RELEASED "api.startup_hold_released" /* stage `hold_released` (HTTP/SSE).
                                                                     Required: target_ms, held_ms, sharing,
                                                                     mode; chunks, reached [runtime.py:1404] */
#define KT2N_TL_INITIAL_RESERVOIR_SELECTED "api.initial_reservoir_selected" /* [runtime.py:20993] */
#define KT2N_TL_LEADING_SILENCE_TRIMMED "api.leading_silence_trimmed" /* [runtime.py:19988] */
#define KT2N_TL_SEAM_TRIMMED "api.seam_trimmed"       /* [runtime.py:19667] */
#define KT2N_TL_SPEED_STRETCH "api.speed_stretch"     /* [runtime.py:7038] */
#define KT2N_TL_SILENCE_CAPPED "api.silence_capped"   /* [runtime.py:6723] */
#define KT2N_TL_Q_HOST_AUDIO "api.q_host_audio"       /* [runtime.py:21273] */
#define KT2N_TL_Q_DCG_VERDICT "api.q_dcg_verdict"     /* [runtime.py:19520] */
#define KT2N_TL_Q_DCG_FOLLOWUP "api.q_dcg_followup"   /* [runtime.py:19603] */
#define KT2N_TL_LB_HEAD_BUFFER "lb.head_buffer"       /* [runtime.py:15322] */
#define KT2N_TL_LB_PHRASE_PLAN "lb.phrase_plan"       /* [runtime.py:15391] */
#define KT2N_TL_LB_VERIFY "lb.verify"                 /* [runtime.py:15215] */

/* Transport events: emitted by the Python adapter / api.py around the engine
 * (they need the socket). The engine never emits them; the adapter derives
 * engine_publication_ready from KT2N_EVENT_AUDIO (engine_chunk_index, samples,
 * data_len). Listed so every name the analysis scripts read has one home. */
#define KT2N_TL_API_REQUEST_READY "api.request_ready"                   /* stage `engine_ready` */
#define KT2N_TL_API_WEBSOCKET_CONNECTED "api.websocket_connected"       /* stage `engine_ready` (WS) */
#define KT2N_TL_API_STREAM_STARTED "api.stream_started"
#define KT2N_TL_API_ENGINE_PUBLICATION_READY "api.engine_publication_ready" /* stage `engine_chunk`. Required:
                                                                              engine_chunk_index */
#define KT2N_TL_API_SSE_PUBLICATION "api.sse_publication"               /* Required: pcm_bytes, engine_chunk_index */
#define KT2N_TL_API_WEBSOCKET_PUBLICATION "api.websocket_publication"   /* stage `engine_chunk` (WS). Required:
                                                                           pcm_bytes, engine_chunk_index */
#define KT2N_TL_API_WEBSOCKET_RESERVE_RELEASED "api.websocket_reserve_released" /* WS hold. Required:
                                                                                  target_ms, held_ms,
                                                                                  sharing, mode */
#define KT2N_TL_API_WEBSOCKET_RESERVE_REARMED "api.websocket_reserve_rearmed"
#define KT2N_TL_API_PHRASE_GENERATION_STARTED "api.phrase_generation_started"
#define KT2N_TL_API_R_WS_CUT_SETTLED "api.r_ws_cut_settled"
#define KT2N_TL_API_COMPLETE_BOUNDARY "api.complete_boundary"
#define KT2N_TL_API_COMPLETE_SLOT_WAIT "api.complete_slot_wait"
#define KT2N_TL_API_COMPLETE_SLOT_ACQUIRED "api.complete_slot_acquired"

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* KT2N_ENGINE_H_ */
