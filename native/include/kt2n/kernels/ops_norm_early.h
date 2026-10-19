/* kt2n prof-step: residual add + RMSNorm with the PDL trigger at kernel start (same bits as k2's kernel).
 *
 * kt2n_ops_residual_rmsnorm_early takes exactly kt2n_ops_residual_rmsnorm's arguments (include/kt2n/kernels/ops.h)
 * and computes the same values operation for operation (src/cuda/kernels/ops_norm_early.cu holds a verbatim copy
 * of k2's residual_rmsnorm_kernel).  The only difference: with pdl != 0 the kernel executes
 * griddepcontrol.launch_dependents as its FIRST instruction instead of after its post-wait loads.
 *
 * Nsight Systems observation: k2's kernel triggers its dependents only after its own
 * griddepcontrol.wait, i.e. after the preceding o / down projection has completed.  The projection that follows
 * the norm (qkv, gate_up) could therefore not launch and start streaming its weights until then: it started only
 * 0.1-0.5 us before the norm ended, so its whole weight read sat on the step's critical path.  Triggering at
 * start lets it launch while the o / down projection is still running.
 *
 * Contract: unchanged PDL rules.  A dependent launched with programmatic stream serialization must execute
 * griddepcontrol.wait before it reads `out` / `h_out` or writes anything this kernel reads (every kt2n step
 * kernel does); its wait still returns only after this grid has completed, and this grid completes only after
 * its own wait on its predecessor, so the ordering of the whole chain is unchanged.  Never changes results.
 */
#ifndef KT2N_KERNELS_OPS_NORM_EARLY_H
#define KT2N_KERNELS_OPS_NORM_EARLY_H

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t kt2n_ops_residual_rmsnorm_early(const __nv_bfloat16* residual, int64_t residual_stride,
                                            const __nv_bfloat16* delta, int64_t delta_stride,
                                            const __nv_bfloat16* w, float eps,
                                            __nv_bfloat16* h_out, int64_t h_out_stride,
                                            __nv_bfloat16* out, int64_t out_stride,
                                            int rows, int n, int pdl, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* KT2N_KERNELS_OPS_NORM_EARLY_H */
