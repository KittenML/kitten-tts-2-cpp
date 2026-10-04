#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static void test_amx() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 4);
    auto dev = ggml_backend_get_device(backend);
    auto reg = ggml_backend_dev_backend_reg(dev);
    auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
    ggml_backend_buffer_type_t amx = nullptr;
    if (get_extra) {
        for (auto * buft = get_extra(dev); buft && *buft; ++buft) {
            if (std::strcmp(ggml_backend_buft_name(*buft), "AMX") == 0) {
                amx = *buft;
            }
        }
    }
    if (!amx) {
        std::puts("TQ2_1 AMX: skipped (not available)");
        ggml_backend_free(backend);
        return;
    }
    for (int k : {256, 768}) {
        const int rows = 64;
        auto wc = ggml_init({1024*1024, nullptr, true});
        auto wt = ggml_new_tensor_2d(wc, GGML_TYPE_TQ2_1, k, rows);
        auto wq = ggml_new_tensor_2d(wc, GGML_TYPE_Q4_0, k, rows);
        auto wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, amx);
        assert(wb);
        std::vector<float> weights(k*rows);
        std::vector<block_tq2_1> tq(k*rows/256);
        std::vector<block_q4_0> q4(k*rows/32);
        for (int i = 0; i < k*rows; ++i) {
            const float scale = (i/128)%3 == 0 ? 0.0f : (i/128)%3 == 1 ? 0.125f : 2.0f;
            weights[i] = (i%3-1)*scale;
        }
        quantize_row_tq2_1_ref(weights.data(), tq.data(), weights.size());
        for (size_t b = 0; b < q4.size(); ++b) {
            float scale = 0.0f;
            for (int j = 0; j < 32; ++j) {
                scale = std::max(scale, std::abs(weights[b*32+j]));
            }
            q4[b].d = ggml_fp32_to_fp16(scale);
            for (int j = 0; j < 16; ++j) {
                const int lo = scale ? int(weights[b*32+j]/scale)+8 : 8;
                const int hi = scale ? int(weights[b*32+j+16]/scale)+8 : 8;
                q4[b].qs[j] = lo | (hi << 4);
            }
        }
        ggml_backend_tensor_set(wt, tq.data(), 0, ggml_nbytes(wt));
        ggml_backend_tensor_set(wq, q4.data(), 0, ggml_nbytes(wq));
        for (int m : {1, 3, 16, 33}) {
            auto ctx = ggml_init({4*1024*1024, nullptr, true});
            auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
            auto actual = ggml_mul_mat(ctx, wt, input);
            auto expected = ggml_mul_mat(ctx, wq, input);
            auto graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, actual);
            ggml_build_forward_expand(graph, expected);
            assert(ggml_backend_dev_supports_op(dev, actual));
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            assert(buffer);
            std::vector<float> a(k*m), out(rows*m), ref(rows*m);
            for (int i = 0; i < k*m; ++i) {
                a[i] = std::sin(i*0.17f);
            }
            ggml_backend_tensor_set(input, a.data(), 0, ggml_nbytes(input));
            assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(actual, out.data(), 0, ggml_nbytes(actual));
            ggml_backend_tensor_get(expected, ref.data(), 0, ggml_nbytes(expected));
            assert(out == ref);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
        ggml_backend_buffer_free(wb);
        ggml_free(wc);
    }
    ggml_backend_free(backend);
    std::puts("TQ2_1 AMX: 8 matmul cases exactly match lossless Q4_0");
}

int main() {
    ggml_cpu_init();
    ggml_context * ctx = ggml_init({1024*1024, nullptr, true});
    const auto * traits = ggml_get_type_traits(GGML_TYPE_TQ2_1);
    const auto * cpu = ggml_get_type_traits_cpu(GGML_TYPE_TQ2_1);
    const auto * activation_cpu = ggml_get_type_traits_cpu(cpu->vec_dot_type);
    assert(ggml_blck_size(GGML_TYPE_TQ2_1) == 256);
    assert(ggml_type_size(GGML_TYPE_TQ2_1) == 68);
    const float scales[] = {0.0f, 0.03125f, 0.5f, 2.0f, 0.00006103515625f, 16.0f};
    for (int n : {256, 512, 2048}) {
        std::vector<float> weights(n), restored(n), input(n), input_q(n);
        std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_TQ2_1, n));
        std::vector<uint8_t> reference(packed.size());
        std::vector<uint8_t> quant_input(ggml_row_size(cpu->vec_dot_type, n));
        for (int trial = 0; trial < 18; ++trial) {
            for (int i = 0; i < n; ++i) {
                weights[i] = ((i + trial)%3 - 1)*scales[(i/128 + trial)%6];
                input[i] = std::sin(i*0.17f + trial)*3.0f;
            }
            cpu->from_float(weights.data(), packed.data(), n);
            traits->from_float_ref(weights.data(), reference.data(), n);
            assert(reference == packed);
            assert(ggml_validate_row_data(GGML_TYPE_TQ2_1, packed.data(), packed.size()));
            traits->to_float(packed.data(), restored.data(), n);
            assert(weights == restored);
            activation_cpu->from_float(input.data(), quant_input.data(), n);
            dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(quant_input.data()), input_q.data(), n);
            double expected = 0.0, magnitude = 0.0;
            for (int i = 0; i < n; ++i) {
                expected += double(weights[i])*input_q[i];
                magnitude += std::abs(double(weights[i])*input_q[i]);
            }
            float actual;
            cpu->vec_dot(n, &actual, 0, packed.data(), 0, quant_input.data(), 0, 1);
            if (std::abs(actual - expected) > 2e-6*magnitude + 1e-5) {
                std::fprintf(stderr, "n=%d trial=%d actual=%.9g expected=%.9g magnitude=%.9g\n", n, trial, actual, expected, magnitude);
            }
            assert(std::abs(actual - expected) <= 2e-6*magnitude + 1e-5);
        }
        packed[0] |= 3;
        assert(!ggml_validate_row_data(GGML_TYPE_TQ2_1, packed.data(), packed.size()));
    }
    ggml_free(ctx);
    test_amx();
    std::puts("TQ2_1: exact weights, two scales, zero groups, CPU dot product and invalid codes passed");
}
