/* Usage: stream model.kt2n s3gen.kt2n voices.kt2n tokenizer.json TEXT_DATA VOICE TEXT > audio.pcm */
#include "kt2n/engine.h"
#include <signal.h>
#include <stdio.h>
#include <string.h>

static volatile sig_atomic_t interrupted;
static void interrupt_handler(int signal_number) { (void) signal_number; interrupted = 1; }

int main(int argc, char ** argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s MODEL ACOUSTIC VOICES TOKENIZER TEXT_DATA VOICE TEXT > audio.pcm\n", argv[0]);
        return 1;
    }
    kt2n_engine_config config = {0};
    config.struct_size = sizeof(config); config.abi_version = KT2N_ABI_VERSION;
    config.flags = KT2N_CFG_INFERENCE_ONLY; config.max_streams = 1; config.app = "us";
    config.lm_file = argv[1]; config.s3gen_file = argv[2]; config.voices_file = argv[3];
    config.tokenizer_json = argv[4]; config.text_data_dir = argv[5];
    kt2n_engine * engine = NULL;
    kt2n_error error = {0};
    int status = kt2n_engine_create(&config, &engine, &error);
    if (status) { fprintf(stderr, "%s\n", error.message); return 1; }
    kt2n_voice_id voice = 0;
    status = kt2n_voice_lookup(engine, argv[6], &voice, &error);
    if (status) goto finish;
    kt2n_request_params params = {0};
    params.struct_size = sizeof(params); params.text = argv[7]; params.text_len = strlen(argv[7]);
    params.voice = voice; params.speed = 1.0; params.format = KT2N_FORMAT_PCM; params.output = KT2N_OUTPUT_STREAM;
    kt2n_request_id request = 0;
    status = kt2n_submit(engine, &params, &request, &error);
    if (status) goto finish;
    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, interrupt_handler);
    signal(SIGPIPE, SIG_IGN);
    int done = 0, cancelled = 0, output_failed = 0;
    while (!done) {
        if (interrupted && !cancelled) { kt2n_cancel(engine, request, KT2N_CANCEL_REQUEST); cancelled = 1; }
        kt2n_event events[16]; int32_t count = 0;
        status = kt2n_poll(engine, events, 16, 100, &count);
        if (status) break;
        for (int i = 0; i < count; ++i) {
            if (events[i].kind == KT2N_EVENT_AUDIO && !cancelled) {
                if (fwrite(events[i].data, 1, events[i].data_len, stdout) != events[i].data_len || fflush(stdout)) {
                    kt2n_cancel(engine, request, KT2N_CANCEL_REQUEST); cancelled = 1; output_failed = 1;
                }
            } else if (events[i].kind == KT2N_EVENT_DONE) {
                done = 1; status = events[i].status;
                if (events[i].error) error = *events[i].error;
            }
        }
        kt2n_events_release(engine, events, count);
        if (done) kt2n_release(engine, request);
    }
    if (output_failed && !status) status = KT2N_ERR_CANCELLED;
finish:
    if (status) fprintf(stderr, "native error %d: %s\n", status, error.message);
    kt2n_engine_destroy(engine);
    return status ? 1 : 0;
}
