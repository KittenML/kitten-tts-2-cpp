# CUDA selection.
#
#   KT2N_CUDA=AUTO  use CUDA when an nvcc >= 12.8 is found, otherwise build CPU only
#   KT2N_CUDA=ON    require nvcc >= 12.8 (configure fails otherwise)
#   KT2N_CUDA=OFF   CPU only
#
#   KT2N_CUDA_ARCHS default "80;86;89;90;100;120" (Ampere, Ada, Hopper and Blackwell).
#   Each entry builds a cubin (sm_XX) plus PTX for that arch.
#
# Point CMake at a specific toolkit with -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc.
#
# Numerics: never --use_fast_math, never -ftz=true (configure refuses them).
# The CUDA runtime is linked statically (the serving image has no nvcc and we
# do not want to depend on a particular libcudart.so).

set(KT2N_CUDA "AUTO" CACHE STRING "Build CUDA targets: AUTO, ON or OFF")
set_property(CACHE KT2N_CUDA PROPERTY STRINGS AUTO ON OFF)
set(KT2N_CUDA_ARCHS "80;86;89;90;100;120" CACHE STRING "CUDA architectures for kt2n_cuda and GPU tests")
set(KT2N_CUDA_MIN_VERSION "12.8")

set(KT2N_HAVE_CUDA OFF)
set(KT2N_NVCC_VERSION "")
string(TOUPPER "${KT2N_CUDA}" _kt2n_cuda_mode)

if(NOT _kt2n_cuda_mode MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "KT2N_CUDA must be AUTO, ON or OFF (got '${KT2N_CUDA}')")
endif()

if(NOT _kt2n_cuda_mode STREQUAL "OFF")
  include(CheckLanguage)
  if(NOT CMAKE_CUDA_COMPILER)
    check_language(CUDA)
  endif()
  if(CMAKE_CUDA_COMPILER)
    execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" --version
      OUTPUT_VARIABLE _kt2n_nvcc_out ERROR_QUIET RESULT_VARIABLE _kt2n_nvcc_rc)
    string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" _kt2n_match "${_kt2n_nvcc_out}")
    set(KT2N_NVCC_VERSION "${CMAKE_MATCH_1}")
    if(NOT _kt2n_nvcc_rc EQUAL 0 OR KT2N_NVCC_VERSION STREQUAL "")
      set(_kt2n_why "could not run '${CMAKE_CUDA_COMPILER} --version'")
    elseif(KT2N_NVCC_VERSION VERSION_LESS KT2N_CUDA_MIN_VERSION)
      set(_kt2n_why "nvcc ${KT2N_NVCC_VERSION} < ${KT2N_CUDA_MIN_VERSION} (sm_100/sm_120 need 12.8)")
    else()
      set(KT2N_HAVE_CUDA ON)
    endif()
  else()
    set(_kt2n_why "no CUDA compiler found")
  endif()

  if(NOT KT2N_HAVE_CUDA)
    if(_kt2n_cuda_mode STREQUAL "ON")
      message(FATAL_ERROR "KT2N_CUDA=ON but ${_kt2n_why}. "
        "Pass -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc or use KT2N_CUDA=AUTO/OFF.")
    else()
      message(STATUS "kt2n: CUDA disabled (${_kt2n_why})")
    endif()
  endif()
endif()

if(KT2N_HAVE_CUDA)
  # Must be set before enable_language(CUDA), otherwise CMake probes a default arch.
  set(CMAKE_CUDA_ARCHITECTURES "${KT2N_CUDA_ARCHS}")
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  foreach(_var IN ITEMS CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_RELEASE CMAKE_CUDA_FLAGS_RELWITHDEBINFO
                        CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_MINSIZEREL)
    if("${${_var}}" MATCHES "(use_fast_math|-ftz=true|--ftz=true|-ftz true|prec-div=false|prec-sqrt=false)")
      message(FATAL_ERROR "${_var} contains '${CMAKE_MATCH_1}'; kt2n needs exact numerics")
    endif()
  endforeach()
endif()

# kt2n_cuda_target(<target>): common CUDA settings for a target with .cu sources.
function(kt2n_cuda_target target)
  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES "${KT2N_CUDA_ARCHS}"
    CUDA_RUNTIME_LIBRARY Static
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON)
  # -lineinfo for profilers; explicit exact-math flags so nothing can sneak in.
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --ftz=false --prec-div=true --prec-sqrt=true>)
endfunction()

function(kt2n_print_summary)
  message(STATUS "kt2n ${PROJECT_VERSION}: build type ${CMAKE_BUILD_TYPE}, "
    "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  if(KT2N_HAVE_CUDA)
    message(STATUS "kt2n: CUDA ON (nvcc ${KT2N_NVCC_VERSION}, archs ${KT2N_CUDA_ARCHS}, static cudart)")
  else()
    message(STATUS "kt2n: CUDA OFF (KT2N_CUDA=${KT2N_CUDA})")
  endif()
  message(STATUS "kt2n: tests=${KT2N_BUILD_TESTS} werror=${KT2N_WERROR}")
endfunction()
