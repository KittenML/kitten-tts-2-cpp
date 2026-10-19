/* Host dispatch policy: native prefill keeps its fixed arithmetic across bursts. */
#ifndef KT2N_KERNELS_TERN_POLICY_H
#define KT2N_KERNELS_TERN_POLICY_H
#define KT2N_TERN_FLAG_PREFILL 2u
static inline int kt2n_tern_uses_production_decode(unsigned flags) {
  return (flags & KT2N_TERN_FLAG_PREFILL) == 0;
}
#endif
