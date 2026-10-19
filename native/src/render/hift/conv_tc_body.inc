// Body of the tensor-core conv (v21), shared by conv1d_tc_k (one conv per launch; KT2N_TC_BZ / KT2N_TC_NZ =
// blockIdx.z / gridDim.z, so its code is exactly v21's) and conv_tc_body (v26 grouped launches: the z index and
// extent of one branch). Included inside a function body with p, wtc, ks, part, ctr and BN / K / TG in scope.
  constexpr int NT = BN / 32;            // n8 tiles per warp (warp grid 2 x 4)
  constexpr int NG = (K + TG - 1) / TG;  // pipeline steps (tap groups) per chunk of 8 input channels
  constexpr int XC = tc_xc(K);           // 8-channel chunks per x stage / weight step (> 1 only with TG == K)
  static_assert(XC == 1 || (TG == K && KT2N_TC_ONEBAR), "XC > 1 needs whole-chunk steps and the one-barrier loop");
  extern __shared__ __align__(16) uint32_t tsm[];
  __shared__ int s_last;
  const int xw = BN + (K - 1) * p.dil;
  const int xw_pad = (xw + 3) & ~3;
  const int wsz = K * 4 * 32 * 4;        // words per chunk of weights (all taps)
  constexpr int WST = XC * TG * 4 * 32 * 4;   // words per weight stage (one tap group, or XC whole chunks)
  auto ws = [&](int st) { return tsm + st * WST; };
  auto xs = [&](int st) { return tsm + KT2N_TC_WSN * WST + st * XC * xw_pad * 8; };
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, tig = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  const int t0 = blockIdx.x * BN, cb = blockIdx.y;
  const int co0 = cb * TC_BM;
  const int nph = p.out_stride > 0 ? p.out_stride : 1;
  const int B = KT2N_TC_NZ / (ks * nph);
  const int split = KT2N_TC_BZ % ks;
  const int phase = (KT2N_TC_BZ / ks) % nph;
  const int b = KT2N_TC_BZ / (ks * nph);
  const int nchunks = p.cin / 8;
  const int cps = nchunks / ks;
  const int q0 = split * cps, q1 = q0 + cps;
  const uint32_t* wbase = wtc + (size_t)phase * p.phase_words + (size_t)cb * nchunks * wsz;
  const size_t xb = (size_t)b * p.cin * p.lin;
  float acc[2][NT][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < NT; ++ni)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[mi][ni][r] = 0.f;
  auto load_w = [&](int step, int st) {   // weights of pipeline step `step` (chunk q0 + step / NG, group step % NG)
    const int gi = step % NG;
    const int ntap = (gi + 1) * TG <= K ? TG : K - gi * TG;
    const uint32_t* src = wbase + (size_t)(q0 + (step / NG) * XC) * wsz + (size_t)gi * TG * 512;
    const int n16 = (XC > 1 ? XC * K : ntap) * 128;  // XC > 1: XC whole chunks, contiguous in the layout
    for (int i = threadIdx.x; i < n16; i += 256) cp_async16(ws(st) + 4 * i, src + 4 * i);
    cp_async_commit();
  };
  // x elements per thread per stage. XC = 1: v21's sizing (halo <= 50), so the default build's kernels are v21's;
  // XC > 1: halo (K - 1) x dilation with dilation <= TC_MAX_DIL (launch_conv checks).
  constexpr int XR = XC == 1 ? (8 * (BN + 50) + 255) / 256 : (8 * XC * (BN + (K - 1) * TC_MAX_DIL) + 255) / 256;
  // load_x only issues the global loads (raw x, and Snake's alpha / 1/alpha); the activation and the
  // tf32 conversion happen in store_x, after the chunk's MMAs, so the loads really overlap the math
  // (ncu, first version: 50% of stall cycles were L1TEX waits inside load_x, 77% of cycles no eligible warp).
  struct XStage { float r[XR], a[XR], i[XR]; };
  auto load_x = [&](int q, XStage& xsg) {
#pragma unroll
    for (int k = 0; k < XR; ++k) {
      const int e = threadIdx.x + 256 * k;
      float v = 0.f, a = 0.f, iv = 0.f;
      if (e < 8 * XC * xw) {
        const int c = e / xw, tl = e % xw;   // c < 8 XC: channel within the stage
        const int ci = q * 8 + c;
        const int src = t0 - p.pad + tl;
        if (src >= 0 && src < p.lin) {
          if (p.pre == PRE_SNAKE) { a = __ldg(&p.alpha[ci]); iv = __ldg(&p.inv[ci]); }
          v = in_load(p.x, xb + (size_t)ci * p.lin + src);
        }
      }
      xsg.r[k] = v; xsg.a[k] = a; xsg.i[k] = iv;
    }
  };
  auto store_x = [&](int st, const XStage& xsg) {
#pragma unroll
    for (int k = 0; k < XR; ++k) {
      const int e = threadIdx.x + 256 * k;
      if (e < 8 * XC * xw) {
        const int c = (e / xw) & 7, tl = e % xw, sub = XC > 1 ? (e / xw) >> 3 : 0;
        // act(0) == 0 for every pre-activation, so padding stays exactly zero
        // pair slot swizzled by (tl >> 2) & 3: consecutive-t stores hit 16 banks instead of 4
        xs(st)[sub * xw_pad * 8 + tl * 8 + 2 * ((c & 3) ^ (KT2N_TC_SWIZZLE ? ((tl >> 2) & 3) : 0)) + (c >> 2)] =
            to_tf32(act(p.pre, xsg.r[k], p.slope, xsg.a[k], xsg.i[k]));
      }
    }
  };
  auto compute = [&](int wst, int xst, int gi) {
#pragma unroll
   for (int sub = 0; sub < XC; ++sub) {   // XC > 1: chunk q's taps, then chunk q + 1's (the XC = 1 order)
    const uint32_t* wsc = ws(wst) + sub * TG * 512;
    const uint32_t* xsc = xs(xst) + sub * xw_pad * 8;
#pragma unroll
    for (int jj = 0; jj < TG; ++jj) {
      const int j = gi * TG + jj;
      if (NG > 1 && j >= K) break;
      uint4 a[2];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
        a[mi] = *reinterpret_cast<const uint4*>(&wsc[((jj * 4 + wm * 2 + mi) * 32 + lane) * 4]);
#if KT2N_TC_BPRE
      // v27 experiment: all NT B fragments of the tap loaded before its MMAs (v21 reloads one register pair per
      // n-tile, so every MMA pair waits on the shared-memory load before it). Same MMAs in the same order.
      uint2 bb[NT];
#pragma unroll
      for (int ni = 0; ni < NT; ++ni) {
        const int tl = wn * (BN / 4) + ni * 8 + g + j * p.dil;
        bb[ni] = *reinterpret_cast<const uint2*>(&xsc[tl * 8 + 2 * (tig ^ (KT2N_TC_SWIZZLE ? ((tl >> 2) & 3) : 0))]);
      }
#pragma unroll
      for (int ni = 0; ni < NT; ++ni)
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) mma_tf32u(acc[mi][ni], a[mi], bb[ni].x, bb[ni].y);
#else
#pragma unroll
      for (int ni = 0; ni < NT; ++ni) {
        const int tl = wn * (BN / 4) + ni * 8 + g + j * p.dil;
        const uint2 bb =
            *reinterpret_cast<const uint2*>(&xsc[tl * 8 + 2 * (tig ^ (KT2N_TC_SWIZZLE ? ((tl >> 2) & 3) : 0))]);
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) mma_tf32u(acc[mi][ni], a[mi], bb.x, bb.y);
      }
#endif
    }
   }
  };
  XStage sa;
#if KT2N_TC_ONEBAR
  {
    // Ring of WSN weight stages, one commit group per step (empty groups at the end keep the count uniform), so
    // `cp.async.wait_group WSN-2` always means "this step's weights have landed" (for this thread; the barrier
    // then makes every thread's copies visible). After the barrier every warp is done with step - 1, so step
    // + WSN - 1 may overwrite stage (step - 1) % WSN, and the next chunk's x may overwrite the x buffer of the
    // previous chunk (read last at step - 1 or earlier). The next barrier publishes both.
    constexpr int WSN = KT2N_TC_WSN;
    const int nsteps = ((q1 - q0) / XC) * NG;   // launch_conv keeps (q1 - q0) % XC == 0
#pragma unroll
    for (int s = 0; s < WSN - 1; ++s) {
      if (s < nsteps) load_w(s, s);
      else cp_async_commit();
    }
    grid_dep_wait();
    if (nsteps > 0) { load_x(q0, sa); store_x(0, sa); }
    int wst = 0, wnext = WSN - 1;
    for (int step = 0; step < nsteps; ++step) {
      const bool next_chunk = step + 1 < nsteps && (step + 1) % NG == 0;
      if (next_chunk) load_x(q0 + ((step + 1) / NG) * XC, sa);
      asm volatile("cp.async.wait_group %0;\n" ::"n"(WSN - 2) : "memory");
      __syncthreads();
      if (step + WSN - 1 < nsteps) load_w(step + WSN - 1, wnext);
      else cp_async_commit();
      compute(wst, (step / NG) & 1, step % NG);
      if (next_chunk) store_x(((step + 1) / NG) & 1, sa);
      wst = wst + 1 == WSN ? 0 : wst + 1;
      wnext = wnext + 1 == WSN ? 0 : wnext + 1;
    }
  }
#else
  if constexpr (NG == 1) {  // all taps per step: the v10-v18 loop, unchanged (same registers)
    int st = 0;
    if (q0 < q1) load_w(0, 0);
    grid_dep_wait();
    if (q0 < q1) { load_x(q0, sa); store_x(0, sa); }
    for (int q = q0; q < q1; ++q) {
      const bool more = q + 1 < q1;
      if (more) { load_w(q + 1 - q0, st ^ 1); load_x(q + 1, sa); }
      if (more) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
      else cp_async_wait_all();
      __syncthreads();
      compute(st, st, 0);
      if (more) store_x(st ^ 1, sa);
      __syncthreads();
      st ^= 1;
    }
  } else {
    const int nsteps = (q1 - q0) * NG;
    if (nsteps > 0) load_w(0, 0);
    grid_dep_wait();
    if (nsteps > 0) { load_x(q0, sa); store_x(0, sa); }
    for (int step = 0; step < nsteps; ++step) {
      const bool more = step + 1 < nsteps;
      const bool next_chunk = more && (step + 1) % NG == 0;
      if (more) load_w(step + 1, (step + 1) & 1);
      if (next_chunk) load_x(q0 + (step + 1) / NG, sa);
      if (more) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
      else cp_async_wait_all();
      __syncthreads();
      compute(step & 1, (step / NG) & 1, step % NG);
      if (next_chunk) store_x(((step + 1) / NG) & 1, sa);
      __syncthreads();
    }
  }
#endif
  grid_dep_launch();
  if (p.out_stride > 0) {  // polyphase ConvTranspose (always ks == 1)
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < NT; ++ni)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int co = co0 + (wm * 2 + mi) * 16 + g + ((r >> 1) << 3);
          const int u = t0 + wn * (BN / 4) + ni * 8 + 2 * tig + (r & 1);
          const int t = u * p.out_stride + phase + p.out_shift;
          if (u >= p.lout || t < 0 || t >= p.out_len) continue;
          const float v = __fadd_rn(acc[mi][ni][r], p.bias[co]);
          const int L = p.out_len + p.reflect;
          float* yr = p.y + ((size_t)b * p.cout + co) * L;
          yr[t + p.reflect] = v;
          if (p.reflect && t == 1) yr[0] = v;
        }
    return;
  }
  if (ks == 1) {
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < NT; ++ni)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int co = co0 + (wm * 2 + mi) * 16 + g + ((r >> 1) << 3);
          const int t = t0 + wn * (BN / 4) + ni * 8 + 2 * tig + (r & 1);
          if (t < p.lout) conv_store(p, b, co, t, acc[mi][ni][r]);
        }
    return;
  }
  const size_t plane = (size_t)B * p.cout * p.lout;
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < NT; ++ni)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int co = co0 + (wm * 2 + mi) * 16 + g + ((r >> 1) << 3);
        const int t = t0 + wn * (BN / 4) + ni * 8 + 2 * tig + (r & 1);
        if (t < p.lout) part[(size_t)split * plane + ((size_t)b * p.cout + co) * p.lout + t] = acc[mi][ni][r];
      }
  __threadfence();
  __syncthreads();
  const int tile = (b * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0) s_last = (atomicAdd(&ctr[tile], 1u) == (unsigned)(ks - 1));
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int o = threadIdx.x; o < TC_BM * BN; o += 256) {
    const int c = o / BN, tt = o % BN;
    const int co = co0 + c, t = t0 + tt;
    if (t >= p.lout) continue;
    const size_t i = ((size_t)b * p.cout + co) * p.lout + t;
    float s = __ldcg(&part[i]);
    for (int k = 1; k < ks; ++k) s = __fadd_rn(s, __ldcg(&part[(size_t)k * plane + i]));
    conv_store(p, b, co, t, s);
  }
  if (threadIdx.x == 0) ctr[tile] = 0u;
