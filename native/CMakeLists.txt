# Native C++20 / CUDA engine for Kitten TTS 2.
cmake_minimum_required(VERSION 3.28)

project(kt2n
  VERSION 0.1.0
  DESCRIPTION "Kitten TTS 2 native inference engine"
  LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Kt2nOptions)
include(Kt2nWarnings)
include(Kt2nCuda)
include(Kt2nDeps)
include(Kt2nTextPlan)  # L9b native text tail (cmake/Kt2nTextPlan.cmake)

# ---------------------------------------------------------------------------
# kt2n_core (CPU)
# ---------------------------------------------------------------------------
add_library(kt2n_core STATIC
  src/core/engine_file.cc
  src/core/json.cc
  src/core/ref_kernels.cc
  src/core/sha256.cc
  src/core/verify.cc)
add_library(kt2n::core ALIAS kt2n_core)
target_include_directories(kt2n_core
  PUBLIC "${PROJECT_SOURCE_DIR}/include"
  PRIVATE "${PROJECT_SOURCE_DIR}/src/core")
target_compile_features(kt2n_core PUBLIC cxx_std_20)
target_compile_definitions(kt2n_core PUBLIC KT2N_VERSION_STRING="${PROJECT_VERSION}")
find_package(Threads REQUIRED)
target_link_libraries(kt2n_core PUBLIC Threads::Threads)
kt2n_exact_numerics(kt2n_core)
kt2n_set_warnings(kt2n_core)

# ---------------------------------------------------------------------------
# kt2n_post (CPU): host-audio post-processing ports (L10, include/kt2n/post/).
# ---------------------------------------------------------------------------
add_library(kt2n_post STATIC
  src/post/post.cc
  src/post/atempo.cc
  src/post/response.cc)
add_library(kt2n::post ALIAS kt2n_post)
target_include_directories(kt2n_post
  PUBLIC "${PROJECT_SOURCE_DIR}/include"
  PRIVATE "${PROJECT_SOURCE_DIR}/src/core")
target_compile_features(kt2n_post PUBLIC cxx_std_20)
target_link_libraries(kt2n_post PUBLIC kt2n_core)
kt2n_exact_numerics(kt2n_post)
kt2n_set_warnings(kt2n_post)

# ---------------------------------------------------------------------------
# kt2n_encode (CPU, optional): in-process mp3/aac/opus/flac encoders over the
# FFmpeg libraries production's ffmpeg CLI uses (L10, include/kt2n/post/encoder.h).
# KT2N_AV=AUTO builds it when pkg-config finds libavcodec/libavformat/libavutil/
# libswresample dev files; ON makes them required. Link the SAME FFmpeg as the
# production image to reproduce its bytes (the encoders' output differs between
# FFmpeg versions).
# ---------------------------------------------------------------------------
set(KT2N_AV "AUTO" CACHE STRING "In-process encoders over FFmpeg libraries: AUTO|ON|OFF")
set(KT2N_HAVE_AV OFF)
if(NOT KT2N_AV STREQUAL "OFF")
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(KT2N_AVLIBS QUIET IMPORTED_TARGET libavcodec libavformat libavutil libswresample)
  endif()
  if(KT2N_AVLIBS_FOUND)
    set(KT2N_HAVE_AV ON)
    add_library(kt2n_encode STATIC src/post/encoder.cc)
    add_library(kt2n::encode ALIAS kt2n_encode)
    target_link_libraries(kt2n_encode PUBLIC kt2n_post PRIVATE PkgConfig::KT2N_AVLIBS)
    kt2n_set_warnings(kt2n_encode)
    message(STATUS "kt2n: in-process encoders ON (libavcodec ${KT2N_AVLIBS_libavcodec_VERSION})")
  elseif(KT2N_AV STREQUAL "ON")
    message(FATAL_ERROR "KT2N_AV=ON but the FFmpeg dev files (libavcodec, libavformat, libavutil, libswresample) were not found")
  else()
    message(STATUS "kt2n: in-process encoders OFF (FFmpeg dev files not found)")
  endif()
endif()

# ---------------------------------------------------------------------------
# kt2n_cuda (GPU)
# ---------------------------------------------------------------------------
if(KT2N_HAVE_CUDA)
  # Kernel lanes add src/cuda/kernels/*.cu with extern "C" launchers declared in
  # include/kt2n/kernels/*.h. Re-run cmake (or rely on CONFIGURE_DEPENDS) after
  # adding files.
  file(GLOB KT2N_CUDA_KERNEL_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/cuda/kernels/*.cu")
  list(SORT KT2N_CUDA_KERNEL_SOURCES)
  # Native prefill (M2, include/kt2n/prefill.h): model loader + run context over
  # the kernel lanes' launchers; host code plus one small kernel per file.
  file(GLOB KT2N_PREFILL_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/prefill/*.cu")
  list(SORT KT2N_PREFILL_SOURCES)
  # Torch-free native LM (L3, include/kt2n/lm.h): engine, stop rules, run-ahead ring.
  file(GLOB KT2N_LM_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/lm/*.cu")
  list(SORT KT2N_LM_SOURCES)

  # Share compiled CUDA objects between the engine and C++ tests.
  add_library(kt2n_cuda_objs OBJECT src/cuda/smoke.cu ${KT2N_CUDA_KERNEL_SOURCES} ${KT2N_PREFILL_SOURCES}
    ${KT2N_LM_SOURCES})
  target_include_directories(kt2n_cuda_objs PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    PRIVATE "${PROJECT_SOURCE_DIR}/src/cuda/kernels" "${PROJECT_SOURCE_DIR}/src/prefill"
    "${PROJECT_SOURCE_DIR}/src/lm")
  target_compile_definitions(kt2n_cuda_objs PUBLIC KT2N_HAVE_CUDA=1)
  if(KT2N_TERN_PRODUCTION_ORDER)
    target_compile_definitions(kt2n_cuda_objs PUBLIC KT2N_TERN_PRODUCTION_ORDER=1)
  endif()
  if(KT2N_HEAD_PRODUCTION_ORDER)
    target_compile_definitions(kt2n_cuda_objs PRIVATE KT2N_HEAD_PRODUCTION_ORDER=1)
  endif()
  if(KT2N_ATTN_PRODUCTION_ORDER)
    # Keep the attention launcher, optimized decode dispatch and prefill shortcuts
    # on the same arithmetic contract in every CUDA translation unit.
    target_compile_definitions(kt2n_cuda_objs PRIVATE KT2N_ATTN_PRODUCTION_ORDER=1)
  endif()
  target_link_libraries(kt2n_cuda_objs PUBLIC kt2n_core)
  set_target_properties(kt2n_cuda_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
  kt2n_cuda_target(kt2n_cuda_objs)
  kt2n_set_warnings(kt2n_cuda_objs)

  add_library(kt2n_cuda STATIC $<TARGET_OBJECTS:kt2n_cuda_objs>)
  add_library(kt2n::cuda ALIAS kt2n_cuda)
  target_include_directories(kt2n_cuda PUBLIC "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(kt2n_cuda PUBLIC kt2n_core CUDA::cudart_static)
  if(KT2N_HEAD_PRODUCTION_ORDER)
    target_link_libraries(kt2n_cuda PUBLIC CUDA::cublas)
  endif()
  target_compile_definitions(kt2n_cuda PUBLIC KT2N_HAVE_CUDA=1)
  set_target_properties(kt2n_cuda PROPERTIES LINKER_LANGUAGE CUDA)
  kt2n_cuda_target(kt2n_cuda)

endif()

# L5 estimator (src/render/estimator): static library + est_check tool.
if(KT2N_HAVE_CUDA)
  add_subdirectory(src/render/estimator)
endif()

# ---------------------------------------------------------------------------
# L11 voice build (native clone build): src/voice/CMakeLists.txt
# ---------------------------------------------------------------------------
add_subdirectory(src/voice)

# ---------------------------------------------------------------------------
# kt2n_render (L7): render state, RNG ports, descriptors, render calls.
# ---------------------------------------------------------------------------
add_subdirectory(src/render)

# L9a native text normaliser (opt-in: KT2N_BUILD_TEXT, KT2N_TEXT_DEPS).
include(Kt2nText)

# L8 native engine runtime (cmake/Kt2nEngine.cmake).
include(Kt2nEngine)

# ---------------------------------------------------------------------------
# Tests (ctest labels: cpu, gpu)
# ---------------------------------------------------------------------------
if(KT2N_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
  add_subdirectory(tests/render)
  if(KT2N_HAVE_CUDA)
    add_executable(test_gpu_portability tests/gpu/portability.cu)
    target_include_directories(test_gpu_portability PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(test_gpu_portability PRIVATE CUDA::cudart_static)
    kt2n_cuda_target(test_gpu_portability)
    add_test(NAME gpu.portability COMMAND test_gpu_portability)
    set_tests_properties(gpu.portability PROPERTIES LABELS gpu TIMEOUT 120)
  endif()
endif()

kt2n_print_summary()
