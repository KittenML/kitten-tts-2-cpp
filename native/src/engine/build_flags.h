// L8: build-feature macros (set by cmake/Kt2nEngine.cmake when the lane library is linked). Defaults here so
// that every `#if` sees a defined value (-Wundef / KT2N_WERROR builds).
#pragma once

#ifndef KT2N_ENGINE_WITH_TEXT
#define KT2N_ENGINE_WITH_TEXT 0  // L9a native normaliser (kt2n_text)
#endif
#ifndef KT2N_ENGINE_WITH_L7
#define KT2N_ENGINE_WITH_L7 0  // L7 render service (kt2n_render_cuda)
#endif
#ifndef KT2N_ENGINE_WITH_RENDERER
#define KT2N_ENGINE_WITH_RENDERER 0  // L7 Renderer = L4 encoder + L5 estimator + L6 HiFT (kt2n_render_flow)
#endif
#ifndef KT2N_ENGINE_WITH_VOICE
#define KT2N_ENGINE_WITH_VOICE 0  // L11 clone builds (kt2n_voice_service)
#endif
#ifndef KT2N_ENGINE_WITH_AV
#define KT2N_ENGINE_WITH_AV 0  // L10 in-process encoders (kt2n_encode, FFmpeg)
#endif
#ifndef KT2N_TSAN_COMPAT
#define KT2N_TSAN_COMPAT 0  // TSan build: wait on the system clock (GCC 11 TSan lacks pthread_cond_clockwait)
#endif
