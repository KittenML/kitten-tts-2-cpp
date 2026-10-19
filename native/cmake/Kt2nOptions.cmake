# Build options, language standard and compiler floor.

option(KT2N_WERROR "Treat compiler warnings as errors (CI turns this on)" OFF)
option(KT2N_TERN_PRODUCTION_ORDER "Use experimental production scalar/Tensor Core projection order for decode rows 1..16" OFF)
option(KT2N_BUILD_TESTS "Build the C++ tests (GoogleTest)" ${PROJECT_IS_TOP_LEVEL})
option(KT2N_BUILD_CLONING "Build on-demand voice cloning networks" OFF)
option(KT2N_ATTN_PRODUCTION_ORDER "Experimental production-order attention arithmetic; requires model and serving qualification" OFF)
option(KT2N_HEAD_PRODUCTION_ORDER "Experimental rowwise cuBLAS output head; requires model and serving qualification" OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Static libraries link into the shared engine library.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Floor: gcc 11 (Ubuntu 22.04, the CUDA 12.8 container and the GPU box).
# Code must not use std::format / std::expected / other post-gcc-11 library parts.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
  message(FATAL_ERROR "kt2n needs gcc >= 11 (found ${CMAKE_CXX_COMPILER_VERSION})")
endif()

# Exact numerics: refuse flags that change floating-point results.
foreach(_var IN ITEMS CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELWITHDEBINFO
                      CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_MINSIZEREL)
  if("${${_var}}" MATCHES "(-ffast-math|-Ofast|-funsafe-math-optimizations|-ffinite-math-only)")
    message(FATAL_ERROR "${_var} contains '${CMAKE_MATCH_1}'; kt2n needs exact IEEE numerics")
  endif()
endforeach()

# kt2n_exact_numerics(<target>): no FP contraction on the CPU side, so the CPU
# reference kernels give the same bits with gcc and clang at any -O level.
function(kt2n_exact_numerics target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()
