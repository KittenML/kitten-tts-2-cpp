# L9b: the native text tail (CPU only, no third-party dependencies) -- demo
# segmentation, emotion tags, LM token budgets, the Qwen2 byte-level BPE
# tokenizer, the prompt builder and the realtime (WebSocket) phrase planner.
# include/kt2n/textplan.h; replay tests in tests/textplan/README.md.
# Kept in its own file so it merges cleanly with the other lanes' CMake edits.
# kt2n_core is defined later in CMakeLists.txt; target names resolve at
# generate time.

add_library(kt2n_textplan STATIC
  src/textplan/unicode.cc
  src/textplan/spans.cc
  src/textplan/segment.cc
  src/textplan/tokenizer.cc
  src/textplan/request.cc
  src/textplan/ws.cc)
add_library(kt2n::textplan ALIAS kt2n_textplan)
target_include_directories(kt2n_textplan
  PUBLIC "${PROJECT_SOURCE_DIR}/include"
  PRIVATE "${PROJECT_SOURCE_DIR}/src/core" "${PROJECT_SOURCE_DIR}/src/textplan")
target_compile_features(kt2n_textplan PUBLIC cxx_std_20)
target_link_libraries(kt2n_textplan PUBLIC kt2n_core)
kt2n_set_warnings(kt2n_textplan)

# kt2n_textplan: replay the production goldens, bench, encode.
