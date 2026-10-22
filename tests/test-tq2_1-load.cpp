// Loads a TQ2_1 GGUF with all layers on the GPU and checks that every TQ2_1 weight is on the GPU as Q2_0
// and dequantizes to the same values as the file. Needs a GPU build and a model file, so it is not run by ctest.
// usage: test-tq2_1-load model-tq2_1.gguf

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "../src/llama-ext.h"

#include <cstdio>
#include <cstring>
#include <vector>

// GPU memory, not the GPU's pinned host buffer. Do not use is_host: CPU extra buffers (AMX, CPU_REPACK) do not set it.
static bool on_gpu(const ggml_tensor * t) {
    if (!t->buffer) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
    ggml_backend_dev_t         dev  = ggml_backend_buft_get_device(buft);
    if (!dev) {
        return false;
    }
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
    return (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) && buft != ggml_backend_dev_host_buffer_type(dev);
}

static int check(const char * path, ggml_context * fctx, llama_load_mode load_mode, bool gpu_repack) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.load_mode    = load_mode;
    mp.gpu_repack   = gpu_repack;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        std::fprintf(stderr, "load failed (%s, gpu_repack %d)\n", llama_load_mode_name(load_mode), gpu_repack);
        return 1;
    }

    int n_tq2_1 = 0, n_ok = 0, n_bad = 0;
    for (int32_t i = 0; i < llama_model_n_tensors(model); ++i) {
        const ggml_tensor * t  = llama_model_get_tensor_by_index(model, i);
        const ggml_tensor * ft = ggml_get_tensor(fctx, ggml_get_name(t));
        if (!ft || ft->type != GGML_TYPE_TQ2_1) {
            continue;
        }
        n_tq2_1++;
        const bool gpu = on_gpu(t);
        if (!gpu_repack) {
            // without the repack, no GPU backend can run TQ2_1, so the weight must stay on the CPU unchanged
            if (t->type == GGML_TYPE_TQ2_1 && !gpu) { n_ok++; } else { n_bad++; std::printf("NOT ON CPU AS TQ2_1: %s\n", ggml_get_name(t)); }
            continue;
        }
        if (t->type != GGML_TYPE_Q2_0 || !gpu) {
            n_bad++;
            std::printf("NOT REPACKED TO A GPU BUFFER: %s (%s @ %s)\n", ggml_get_name(t), ggml_type_name(t->type), ggml_backend_buffer_name(t->buffer));
            continue;
        }
        const int64_t n = ggml_nelements(t);
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
        std::vector<float> got(n), want(n);
        ggml_get_type_traits(GGML_TYPE_Q2_0)->to_float(bytes.data(), got.data(), n);
        ggml_get_type_traits(GGML_TYPE_TQ2_1)->to_float(ft->data, want.data(), n);
        if (std::memcmp(got.data(), want.data(), n*sizeof(float)) == 0) { n_ok++; } else { n_bad++; std::printf("MISMATCH: %s\n", ggml_get_name(t)); }
    }
    llama_model_free(model);

    std::printf("%s, gpu_repack %d: %d TQ2_1 weights, %d ok, %d failed\n", llama_load_mode_name(load_mode), gpu_repack, n_tq2_1, n_ok, n_bad);
    return n_tq2_1 > 0 && n_bad == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s model-tq2_1.gguf\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    if (!llama_supports_gpu_offload()) {
        std::fprintf(stderr, "no GPU backend: skipped\n");
        llama_backend_free();
        return 0;
    }

    ggml_context * fctx = nullptr;
    gguf_init_params gp = { /*no_alloc =*/ false, /*ctx =*/ &fctx };
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    if (!g) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }

    int failed = 0;
    failed += check(argv[1], fctx, LLAMA_LOAD_MODE_MMAP, true);
    failed += check(argv[1], fctx, LLAMA_LOAD_MODE_NONE, true);
    failed += check(argv[1], fctx, LLAMA_LOAD_MODE_MMAP, false);

    gguf_free(g);
    ggml_free(fctx);
    llama_backend_free();
    std::printf("%s\n", failed == 0 ? "OK" : "FAILED");
    return failed == 0 ? 0 : 1;
}
