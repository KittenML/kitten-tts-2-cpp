/* engine.h must compile as plain C (C11) and give C the same struct layouts as
 * C++: test_headers.cc compares these sizes / offsets with its own. */
#include <stddef.h>

#include "kt2n/engine.h"

size_t kt2n_c_layout_probe(int which);

size_t kt2n_c_layout_probe(int which) {
  switch (which) {
    case 0: return sizeof(kt2n_event);
    case 1: return sizeof(kt2n_request_params);
    case 2: return sizeof(kt2n_engine_config);
    case 3: return sizeof(kt2n_error);
    case 4: return sizeof(kt2n_stats);
    case 5: return sizeof(kt2n_session_params);
    case 6: return sizeof(kt2n_voice_build_params);
    case 7: return sizeof(kt2n_voice_info);
    case 8: return sizeof(kt2n_sampler);
    case 9: return offsetof(kt2n_event, data);
    case 10: return offsetof(kt2n_event, name);
    case 11: return offsetof(kt2n_event, voice);
    case 12: return offsetof(kt2n_request_params, user_data);
    case 13: return offsetof(kt2n_request_params, control_json);
    default: return 0;
  }
}
