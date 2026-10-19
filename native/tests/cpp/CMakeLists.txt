# C++ tests. Labels: "cpu" (run anywhere) and "gpu" (need a CUDA device; they
# SKIP without one unless KT2N_REQUIRE_GPU=1, which turns a missing GPU into a
# failure -- set it on the GPU box).
#
#   ctest --test-dir build -L cpu --output-on-failure
#   KT2N_REQUIRE_GPU=1 ctest --test-dir build -L gpu --output-on-failure
#
# Kernel lanes: drop GoogleTest files named tests/kernels/<name>_test.cc or
# <name>_test.cu. Each becomes its own executable (linked with kt2n_cuda,
# kt2n_core and gtest_main), labelled "gpu", with the tiny engine-file fixture
# available via KT2N_TEST_DATA_DIR and tests/cpp/test_util.h on the include path.

set(KT2N_TEST_DATA_DIR "${CMAKE_BINARY_DIR}/test-data")

if(NOT Python_EXECUTABLE)
  message(FATAL_ERROR "tests need a Python interpreter with numpy (for make_tiny_kt2n.py)")
endif()

# Fixture: tiny engine files written by our own numpy script.
add_test(NAME kt2n_make_tiny
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/make_tiny_kt2n.py"
          --out-dir "${KT2N_TEST_DATA_DIR}")
set_tests_properties(kt2n_make_tiny PROPERTIES
  FIXTURES_SETUP kt2n_tiny
  LABELS "cpu;gpu")

set(_kt2n_test_env "KT2N_TEST_DATA_DIR=${KT2N_TEST_DATA_DIR}")

# kt2n_add_gtest(<name> LABEL <cpu|gpu> SOURCES ... [LIBS ...])
function(kt2n_add_gtest name)
  cmake_parse_arguments(ARG "" "LABEL" "SOURCES;LIBS" ${ARGN})
  add_executable(${name} ${ARG_SOURCES})
  target_link_libraries(${name} PRIVATE kt2n_core GTest::gtest_main ${ARG_LIBS})
  target_include_directories(${name} PRIVATE
    "${PROJECT_SOURCE_DIR}/tests/cpp"
    "${PROJECT_SOURCE_DIR}/src/core")
  target_compile_definitions(${name} PRIVATE KT2N_TEST_DATA_DIR_DEFAULT="${KT2N_TEST_DATA_DIR}")
  kt2n_exact_numerics(${name})
  kt2n_set_warnings(${name})
  # PRE_TEST: list tests at ctest time, so a build on a machine without a GPU
  # (CI's cuda-compile job) never has to run a GPU binary.
  gtest_discover_tests(${name}
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS "${ARG_LABEL}" FIXTURES_REQUIRED kt2n_tiny ENVIRONMENT "${_kt2n_test_env}")
endfunction()

kt2n_add_gtest(kt2n_core_tests LABEL cpu SOURCES
  sha256_test.cc
  bf16_test.cc
  ref_kernels_test.cc
  json_test.cc
  engine_file_test.cc)

# G0 interface headers: include/kt2n/engine.h (C ABI; also compiled as C11 here so
# the test can compare C and C++ layouts) and include/kt2n/render/ops.h.
add_library(kt2n_engine_h_c OBJECT test_engine_h.c)
target_include_directories(kt2n_engine_h_c PRIVATE "${PROJECT_SOURCE_DIR}/include")
set_target_properties(kt2n_engine_h_c PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(kt2n_engine_h_c PRIVATE -Wall -Wextra -Wpedantic $<$<BOOL:${KT2N_WERROR}>:-Werror>)
endif()
kt2n_add_gtest(kt2n_header_tests LABEL cpu SOURCES test_headers.cc $<TARGET_OBJECTS:kt2n_engine_h_c>)

# L10 post-processing ports (CPU).
kt2n_add_gtest(kt2n_post_tests LABEL cpu SOURCES post_test.cc LIBS kt2n_post)
if(TARGET kt2n_encode)
  kt2n_add_gtest(kt2n_encode_tests LABEL cpu SOURCES encoder_test.cc LIBS kt2n_encode)
endif()
# The same SHA-256 tests on the portable path (the default picks SHA-NI when the CPU has it).
add_test(NAME kt2n_core_sha256_portable COMMAND kt2n_core_tests --gtest_filter=Sha256.*)
set_tests_properties(kt2n_core_sha256_portable PROPERTIES LABELS cpu ENVIRONMENT "KT2N_SHA256_PORTABLE=1")

# GPU tests.
if(KT2N_HAVE_CUDA)
  kt2n_add_gtest(kt2n_gpu_smoke_tests LABEL gpu SOURCES gpu/smoke_test.cc LIBS kt2n_cuda)

  file(GLOB _kt2n_kernel_tests CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/tests/kernels/*_test.cc"
    "${PROJECT_SOURCE_DIR}/tests/kernels/*_test.cu")
  list(SORT _kt2n_kernel_tests)
  foreach(_src IN LISTS _kt2n_kernel_tests)
    get_filename_component(_stem "${_src}" NAME_WE)
    kt2n_add_gtest(kt2n_kernel_${_stem} LABEL gpu SOURCES "${_src}" LIBS kt2n_cuda)
    if(_src MATCHES "\\.cu$")
      kt2n_cuda_target(kt2n_kernel_${_stem})
    endif()
  endforeach()


endif()

# Text planning and tokenizer tests.
kt2n_add_gtest(kt2n_textplan_tests LABEL cpu SOURCES textplan_test.cc LIBS kt2n_textplan)
