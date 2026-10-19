# kt2n_set_warnings(<target> [THIRD_PARTY_HEADERS]): warning set for our own
# targets only (never for fetched dependencies). KT2N_WERROR=ON turns warnings
# into errors. THIRD_PARTY_HEADERS drops -Wshadow / -Wcast-qual, which nanobind's
# headers trigger (used for the binding module only).

function(kt2n_set_warnings target)
  cmake_parse_arguments(ARG "THIRD_PARTY_HEADERS" "" "" ${ARGN})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wnon-virtual-dtor
                                -Woverloaded-virtual -Wformat=2 -Wimplicit-fallthrough -Wundef>)
    if(NOT ARG_THIRD_PARTY_HEADERS)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wshadow -Wcast-qual>)
    endif()
    if(KT2N_WERROR)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif()
  endif()
  if(KT2N_HAVE_CUDA)
    # -Wno-unknown-pragmas: `#pragma unroll` in __host__ __device__ code is
    # valid CUDA but unknown to the host compiler.
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-unknown-pragmas>)
    if(KT2N_WERROR)
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings -Xcompiler=-Werror>)
    endif()
  endif()
endfunction()
