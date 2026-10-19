/* kt2n text normalisation C ABI (L9a).
 *
 * Production's written-to-spoken normaliser (kittenml-api ebcdd3f
 * kitten_tts_09/normalization.py + NeMo text_processing 1.2.0 English), native.
 * Thread-safe: calls run on `grammar_instances` independent grammar copies
 * (a caller waits for a free one); the LRU cache is shared.
 *
 * Strings are UTF-8 (not NUL-terminated on input: pass the byte length).
 * Outputs are malloc'd; release with kt2n_text_free.
 * Return codes: 0 ok, 1 the grammar raised (only kt2n_text_nemo; the full
 * normaliser fails open like production and still returns 0), 2 bad input
 * (invalid UTF-8) or internal error.
 */
#ifndef KT2N_TEXT_H_
#define KT2N_TEXT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kt2n_text kt2n_text;

typedef struct kt2n_text_config {
  const char* grammar_dir;   /* tagger.fst, verbalizer.fst, post_processor.fst, *_words.txt */
  int32_t grammar_instances; /* parallel grammar copies (>= 1) */
  int32_t cache_entries;     /* KITTENTTS_09_N_CACHE (live 4096; 0 = off) */
  int32_t fast_path;         /* KITTENTTS_09_NORMALIZE_FAST_PATH (live 1) */
  int32_t demo_formatting;   /* KITTENTTS_09_NORMALIZE_DEMO_FORMATTING (live 1) */
  int32_t emo;               /* KITTENTTS_09_EMO_TAGS / EMO_REQUEST_LEVEL (live 1) */
  int32_t unit_words;        /* KITTENTTS_09_N_UNIT_WORDS (live 1) */
  int32_t version_words;     /* KITTENTTS_09_N_VERSION_WORDS (live 1) */
  int32_t language_mode;     /* KITTENTTS_09_NORMALIZE_LANGUAGE: 0 voice, 1 text (live), 2 all */
} kt2n_text_config;

/* Fills `cfg` with the live production settings (grammar_dir left NULL). */
void kt2n_text_config_default(kt2n_text_config* cfg);
/* NULL on failure; `error` (optional) receives a malloc'd message. */
kt2n_text* kt2n_text_create(const kt2n_text_config* cfg, char** error);
void kt2n_text_destroy(kt2n_text* t);

/* SpeechTextNormalizer.normalize(text, language=language). */
int kt2n_text_normalize(kt2n_text* t, const char* text, int64_t len, const char* language, char** out,
                        int64_t* out_len);
/* Request-local switch; 0 preserves input bytes, 1 applies normalization. */
int kt2n_text_normalize_ex(kt2n_text* t, const char* text, int64_t len, const char* language,
                         int normalize, char** out, int64_t* out_len);
/* The raw grammar call: Normalizer.normalize(text, punct_post_process=True). */
int kt2n_text_nemo(kt2n_text* t, const char* text, int64_t len, char** out, int64_t* out_len);
void kt2n_text_free(char* p);

#ifdef __cplusplus
}
#endif

#endif /* KT2N_TEXT_H_ */
