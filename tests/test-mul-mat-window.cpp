// test-mul-mat-window.cpp: GGML_OP_MUL_MAT on the CPU with an f32 src1 whose rows overlap in memory.
//
// A 1-D convolution can read its input in place: column t of the window matrix is x[t*step .. t*step + k), a view
// with row stride step < k (overlapping rows). ggml-cpu runs such a src1 through llamafile's sgemm (B only read,
// ldb < k). The test compares it with the same product on a contiguous copy of the window (ggml_cont), which takes
// the ordinary contiguous path, for several shapes and thread counts. test-backend-ops cannot check this: it does
// not compare the CPU backend with itself.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <vector>

struct window_case {
    int64_t m;      // output rows (weights)
    int64_t n;      // windows
    int64_t k;      // window length
    int64_t step;   // distance between windows (< k: overlapping rows; 0: every row is the same row)
};

static const window_case CASES[] = {
    {  64,  50,   96,   1 },
    {  64,  50,   96,  32 },
    {  17, 300,  384,  96 },   // k = 3 * 128 channels: a kernel-3 convolution read as overlapping rows
    { 128, 257,  640, 128 },
    {   3,   7,   16,  15 },
    {  64,  50,   96,  96 },   // contiguous rows
    {  64,  50,   96,   0 },   // all rows alias one row (row stride 0): not a window, stays on the vec_dot path
};

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static float frand() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (float) ((rng_state >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f;
}

int main() {
    int fails = 0;
    for (const window_case & c : CASES) {
        for (int n_threads : {1, 4}) {
            ggml_init_params params = { 64ull * 1024 * 1024, nullptr, false };
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.k, c.m);
            ggml_tensor * buf = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.k + (c.n - 1) * c.step);
            for (int64_t i = 0; i < ggml_nelements(a); i++) ((float *) a->data)[i] = frand();
            for (int64_t i = 0; i < ggml_nelements(buf); i++) ((float *) buf->data)[i] = frand();

            // ggml_view_2d checks the size as if the view were contiguous: make a small view and widen it
            ggml_tensor * win = ggml_view_1d(ctx, buf, 1, 0);
            win->ne[0] = c.k; win->ne[1] = c.n;
            win->nb[1] = c.step * sizeof(float); win->nb[2] = win->nb[3] = std::max<size_t>(win->nb[1], sizeof(float)) * c.n;
            GGML_ASSERT(ggml_nbytes(win) <= ggml_nbytes(buf));

            ggml_tensor * y_win = ggml_mul_mat(ctx, a, win);
            ggml_tensor * y_ref = ggml_mul_mat(ctx, a, ggml_cont(ctx, win));
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y_win);
            ggml_build_forward_expand(gf, y_ref);
            ggml_graph_compute_with_ctx(ctx, gf, n_threads);

            // against the contiguous copy (same kernel family) and against a double-precision reference
            const float * A = (const float *) a->data, * B = (const float *) buf->data;
            double err_cont = 0.0, err_dbl = 0.0, ref = 0.0;
            for (int64_t j = 0; j < c.n; j++) {
                for (int64_t i = 0; i < c.m; i++) {
                    double r = 0.0;
                    for (int64_t l = 0; l < c.k; l++) r += (double) A[i * c.k + l] * B[j * c.step + l];
                    const double w = ((const float *) y_win->data)[j * c.m + i], cc = ((const float *) y_ref->data)[j * c.m + i];
                    err_cont += (w - cc) * (w - cc);
                    err_dbl += (w - r) * (w - r);
                    ref += r * r;
                }
            }
            const double nmse_cont = err_cont / ref, nmse_dbl = err_dbl / ref;
            const bool ok = nmse_cont <= 1e-12 && nmse_dbl <= 1e-10;
            fails += !ok;
            printf("mul_mat window m=%3d n=%3d k=%3d step=%3d threads=%d: nmse vs contiguous %.2e, vs double %.2e %s\n", (int) c.m,
                   (int) c.n, (int) c.k, (int) c.step, n_threads, nmse_cont, nmse_dbl, ok ? "OK" : "FAIL");
            ggml_free(ctx);
        }
    }
    printf(fails == 0 ? "all mul_mat window checks passed\n" : "%d mul_mat window checks FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
