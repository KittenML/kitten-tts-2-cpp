# Kitten TTS 2 mini / nano / pico (KITTEN2_TINY) on ggml. Needs only ggml (no LibTorch, no llama).
option(KITTEN_TTS_TINY_DEBUG_ENV "kitten-tts-tiny: read debugging environment variables (KT_TINY_DUMP, ...)" OFF)

set(kt_tiny_sources kitten_tts_tiny.cpp kitten_tts_tiny.h)
if(GGML_CUDA)
    # the LSTM recurrence and the harmonic source as CUDA kernels (ggml has no GPU op for them)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61-virtual 70-virtual 75-virtual 80-virtual 86-real 89-real)   # plain C++ kernels: PTX JIT covers newer GPUs
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND kt_tiny_sources kitten_tts_tiny_cuda.cu kitten_tts_tiny_cuda.h)
endif()
add_library(kitten-tts-tiny-lib SHARED ${kt_tiny_sources})
set_target_properties(kitten-tts-tiny-lib PROPERTIES OUTPUT_NAME kitten_tts_tiny)
target_include_directories(kitten-tts-tiny-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                               PRIVATE ${PROJECT_SOURCE_DIR}/vendor/nlohmann)
target_compile_definitions(kitten-tts-tiny-lib PRIVATE KT_TINY_BUILD _USE_MATH_DEFINES)
target_compile_features(kitten-tts-tiny-lib PRIVATE cxx_std_17)
target_link_libraries(kitten-tts-tiny-lib PRIVATE ggml)
if(GGML_CUDA)
    target_compile_definitions(kitten-tts-tiny-lib PRIVATE KT_TINY_CUDA)
    if(GGML_STATIC)
        target_link_libraries(kitten-tts-tiny-lib PRIVATE CUDA::cudart_static)
    else()
        target_link_libraries(kitten-tts-tiny-lib PRIVATE CUDA::cudart)
    endif()
    # exact float order in the kernels: no fused multiply-adds unless written
    set_source_files_properties(kitten_tts_tiny_cuda.cu PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>")
endif()
if(KITTEN_TTS_TINY_DEBUG_ENV)
    target_compile_definitions(kitten-tts-tiny-lib PRIVATE KT_TINY_DEBUG_ENV)
endif()
if(NOT MSVC)
    # the harmonic source repeats PyTorch's float operations one by one: no fused multiply-adds
    set(kt_tiny_flags -ffp-contract=off)
    if(GGML_NATIVE)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-march=native KT_TINY_MARCH_NATIVE)
        if(KT_TINY_MARCH_NATIVE)
            list(APPEND kt_tiny_flags -march=native)   # the CPU-only loops (LSTM, harmonic source) vectorise wider
        endif()
    endif()
    set_source_files_properties(kitten_tts_tiny.cpp PROPERTIES COMPILE_OPTIONS "${kt_tiny_flags}")
endif()

add_executable(kitten-tts-tiny main.cpp)
target_link_libraries(kitten-tts-tiny PRIVATE kitten-tts-tiny-lib)
target_compile_features(kitten-tts-tiny PRIVATE cxx_std_17)
install(TARGETS kitten-tts-tiny kitten-tts-tiny-lib RUNTIME LIBRARY)

if(KITTEN_BUILD_PYTHON AND UNIX AND NOT APPLE AND GGML_CUDA)
    # inside the Python wheel: the CUDA runtime and cuBLAS of the NVIDIA wheels that a CUDA build of torch installs
    set_target_properties(kitten-tts-tiny-lib PROPERTIES BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "$ORIGIN/../nvidia/cuda_runtime/lib;$ORIGIN/../nvidia/cublas/lib")
endif()
