# Native MeanFlow estimator.
# Included from the top-level CMakeLists.txt when CUDA is enabled.
add_library(kt2n_render_est STATIC estimator.cu)
add_library(kt2n::render_est ALIAS kt2n_render_est)
target_include_directories(kt2n_render_est
  PUBLIC "${PROJECT_SOURCE_DIR}/include"
  PRIVATE "${PROJECT_SOURCE_DIR}/src/core" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(kt2n_render_est PUBLIC kt2n_core CUDA::cudart_static)
set_target_properties(kt2n_render_est PROPERTIES POSITION_INDEPENDENT_CODE ON)
kt2n_cuda_target(kt2n_render_est)
# Debug-only in-kernel phase trace (Estimator::debug_trace). Off in every normal build.
option(KT2N_EST_TRACE "L5 estimator: compile the in-kernel %globaltimer phase trace" OFF)
if(KT2N_EST_TRACE)
  target_compile_definitions(kt2n_render_est PRIVATE KT2N_EST_TRACE=1)
endif()
