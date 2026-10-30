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

// element-wise reference: TQ2_1 element g*128 + l*32 + m is byte g*32 + m, bits 2*l; Q2_0 element j is byte j/4, bits 2*(j%4)
static void repack_naive(const block_tq2_1 * x, block_q2_0 * y, int64_t nb) {
    std::memset(y, 0, 4*nb*sizeof(block_q2_0));
    for (int64_t i = 0; i < nb; ++i) {
        for (int e = 0; e < QK_K; ++e) {
            const int g = e/128, l = (e%128)/32, m = e%32;
            const int code = (x[i].qs[g*32 + m] >> (2*l)) & 3;
            block_q2_0 & out = y[4*i + e/QK2_0];
            const int j = e%QK2_0;
            out.d = x[i].d[g];
            out.qs[j/4] |= code << (2*(j%4));
        }
    }
}

static uint32_t rng_state = 12345;
static uint32_t rng() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

// fills codes in {0,1,2}; mode 0: random, 1: all 0, 2: all 1, 3: all 2, 4: code (k+trial)%3
static void fill_codes(block_tq2_1 * x, int64_t nb, int mode, int trial) {
    for (int64_t i = 0; i < nb; ++i) {
        for (int k = 0; k < QK_K/4; ++k) {
            uint8_t b = 0;
            for (int l = 0; l < 4; ++l) {
                int c = 0;
                switch (mode) {
                    case 0: c = rng()%3; break;
                    case 1: c = 0; break;
                    case 2: c = 1; break;
                    case 3: c = 2; break;
                    default: c = (k*4 + l + trial)%3; break;
                }
                b |= c << (2*l);
            }
            x[i].qs[k] = b;
        }
    }
}

static bool fp16_is_finite(uint16_t h) {
    return (h & 0x7c00) != 0x7c00;
}

static void test_repack() {
    const int64_t q2_0_row = ggml_row_size(GGML_TYPE_Q2_0, 256);
    assert(q2_0_row == 4*(int64_t) sizeof(block_q2_0));
    assert(ggml_row_size(GGML_TYPE_TQ2_1, 256) == sizeof(block_tq2_1));

    // edge scales: +-0, smallest subnormal, largest subnormal, smallest normal, 1, -1, largest finite, -largest finite
    const uint16_t edge_scales[] = {0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x2e66};
    const int n_edge = sizeof(edge_scales)/sizeof(edge_scales[0]);

    int n_cases = 0;
    for (int64_t n_per_row : {256, 512, 2048, 6144}) {
        for (int64_t nrows : {1, 3, 8}) {
            const int64_t nb = nrows*n_per_row/QK_K;
            const int64_t n = nrows*n_per_row;
            std::vector<block_tq2_1> x(nb);
            std::vector<block_q2_0> y(4*nb), y_ref(4*nb);
            std::vector<float> f_tq(n), f_q2(n);
            for (int mode = 0; mode < 5; ++mode) {
                for (int trial = 0; trial < 3; ++trial) {
                    fill_codes(x.data(), nb, mode, trial);
                    for (int64_t i = 0; i < nb; ++i) {
                        for (int g = 0; g < 2; ++g) {
                            uint16_t h;
                            if (trial == 0) {
                                h = edge_scales[(i*2 + g) % n_edge];
                            } else {
                                do { h = rng() & 0xffff; } while (!fp16_is_finite(h));
                            }
                            std::memcpy(&x[i].d[g], &h, sizeof(h));
                        }
                    }
                    assert(ggml_validate_row_data(GGML_TYPE_TQ2_1, x.data(), nb*sizeof(block_tq2_1)));
                    std::memset(y.data(), 0xa5, y.size()*sizeof(block_q2_0));
                    assert(ggml_repack_tq2_1_to_q2_0(x.data(), y.data(), nrows, n_per_row));
                    repack_naive(x.data(), y_ref.data(), nb);
                    assert(std::memcmp(y.data(), y_ref.data(), y.size()*sizeof(block_q2_0)) == 0);
                    assert(ggml_validate_row_data(GGML_TYPE_Q2_0, y.data(), y.size()*sizeof(block_q2_0)));
                    dequantize_row_tq2_1(x.data(), f_tq.data(), n);
                    dequantize_row_q2_0(y.data(), f_q2.data(), n);
                    assert(std::memcmp(f_tq.data(), f_q2.data(), n*sizeof(float)) == 0);
                    ++n_cases;
                }
            }
        }
    }

    // each element on its own: one +1 code in a zero block, distinct scales per group
    {
        block_tq2_1 x;
        block_q2_0 y[4];
        float f_tq[QK_K], f_q2[QK_K];
        x.d[0] = ggml_fp32_to_fp16(0.25f);
        x.d[1] = ggml_fp32_to_fp16(-3.0f);
        for (int e = 0; e < QK_K; ++e) {
            std::memset(x.qs, 0x55, sizeof(x.qs)); // all codes 1 (value 0)
            const int g = e/128, l = (e%128)/32, m = e%32;
            x.qs[g*32 + m] ^= 3 << (2*l); // code 1 -> 2 (value +d)
            assert(ggml_repack_tq2_1_to_q2_0(&x, y, 1, QK_K));
            dequantize_row_tq2_1(&x, f_tq, QK_K);
            dequantize_row_q2_0(y, f_q2, QK_K);
            assert(std::memcmp(f_tq, f_q2, sizeof(f_tq)) == 0);
            for (int k = 0; k < QK_K; ++k) {
                assert(f_q2[k] == (k == e ? (g == 0 ? 0.25f : -3.0f) : 0.0f));
            }
            ++n_cases;
        }
    }

    // scale bits are copied unchanged for all 65536 fp16 patterns, also inf and nan
    {
        block_tq2_1 x;
        block_q2_0 y[4];
        fill_codes(&x, 1, 0, 0);
        for (uint32_t h = 0; h < 65536; ++h) {
            const uint16_t h0 = (uint16_t) h, h1 = (uint16_t) (h ^ 0x8001);
            std::memcpy(&x.d[0], &h0, 2);
            std::memcpy(&x.d[1], &h1, 2);
            assert(ggml_repack_tq2_1_to_q2_0(&x, y, 1, QK_K));
            assert(std::memcmp(&y[0].d, &h0, 2) == 0 && std::memcmp(&y[1].d, &h0, 2) == 0);
            assert(std::memcmp(&y[2].d, &h1, 2) == 0 && std::memcmp(&y[3].d, &h1, 2) == 0);
        }
    }

    // code 3 at any position in any block is rejected
    {
        const int64_t nrows = 2, n_per_row = 512, nb = nrows*n_per_row/QK_K;
        std::vector<block_tq2_1> x(nb);
        std::vector<block_q2_0> y(4*nb);
        int n_rejected = 0;
        for (int64_t i = 0; i < nb; ++i) {
            for (int e = 0; e < QK_K; ++e) {
                fill_codes(x.data(), nb, 0, 0);
                x[i].d[0] = x[i].d[1] = ggml_fp32_to_fp16(1.0f);
                const int g = e/128, l = (e%128)/32, m = e%32;
                x[i].qs[g*32 + m] |= 3 << (2*l);
                assert(!ggml_validate_row_data(GGML_TYPE_TQ2_1, x.data(), nb*sizeof(block_tq2_1)));
                assert(!ggml_repack_tq2_1_to_q2_0(x.data(), y.data(), nrows, n_per_row));
                ++n_rejected;
            }
        }
        assert(n_rejected == nb*QK_K);
    }

    // rows that are not a multiple of 256 are rejected
    {
        block_tq2_1 x[2] = {};
        block_q2_0 y[8];
        assert(!ggml_repack_tq2_1_to_q2_0(x, y, 1, 128));
        assert(!ggml_repack_tq2_1_to_q2_0(x, y, 1, 320));
        assert(ggml_repack_tq2_1_to_q2_0(x, y, 0, 256));
    }

    std::printf("TQ2_1 -> Q2_0 repack: %d cases, dequantized output identical (memcmp), scale bits identical for all 65536 fp16 patterns, code 3 rejected at every position\n", n_cases);
}

int main() {
    ggml_cpu_init();
    test_repack();
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
