# Native normalization uses the dependencies built by tools/build_text_deps.py.
option(KT2N_BUILD_TEXT "Build the native text normaliser (needs OpenFst + PCRE2 under KT2N_TEXT_DEPS)" OFF)
option(KT2N_BUILD_VOICE_NORMALIZATION "Build frozen multilingual voice normalization (ICU >=72)" OFF)
set(KT2N_TEXT_DEPS "" CACHE PATH "Prefix with include/fst, include/pcre2.h, lib/libfst.a, lib/libpcre2-8.a")

if(KT2N_BUILD_TEXT)
  if(NOT EXISTS "${KT2N_TEXT_DEPS}/lib/libfst.a" OR NOT EXISTS "${KT2N_TEXT_DEPS}/lib/libpcre2-8.a")
    message(FATAL_ERROR "Run python3 native/tools/build_text_deps.py --prefix <prefix>, then configure with -DKT2N_TEXT_DEPS=<prefix>")
  endif()
  set(_kt2n_text_sources
    src/text/unicode.cc src/text/pyre.cc src/text/nemo.cc src/text/normalizer.cc
    src/text/incremental.cc src/text/fast_tagger.cc src/text/c_api.cc)
  add_library(kt2n_text STATIC ${_kt2n_text_sources})
  target_include_directories(kt2n_text
    PUBLIC "${PROJECT_SOURCE_DIR}/include"
    PRIVATE "${PROJECT_SOURCE_DIR}/src/text" "${KT2N_TEXT_DEPS}/include")
  target_compile_options(kt2n_text PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
  target_link_libraries(kt2n_text PUBLIC "${KT2N_TEXT_DEPS}/lib/libfst.a" "${KT2N_TEXT_DEPS}/lib/libpcre2-8.a"
    ${CMAKE_DL_LIBS} Threads::Threads)

  # libkt2n_text.so: the C ABI (include/kt2n/text.h) with OpenFst/PCRE2 hidden.
  add_library(kt2n_text_shared SHARED ${_kt2n_text_sources})
  set_target_properties(kt2n_text_shared PROPERTIES OUTPUT_NAME kt2n_text
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  target_include_directories(kt2n_text_shared
    PUBLIC "${PROJECT_SOURCE_DIR}/include"
    PRIVATE "${PROJECT_SOURCE_DIR}/src/text" "${KT2N_TEXT_DEPS}/include")
  target_compile_options(kt2n_text_shared PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
  target_link_libraries(kt2n_text_shared PRIVATE "${KT2N_TEXT_DEPS}/lib/libfst.a" "${KT2N_TEXT_DEPS}/lib/libpcre2-8.a"
    ${CMAKE_DL_LIBS} Threads::Threads)
  target_link_options(kt2n_text_shared PRIVATE -Wl,--exclude-libs,ALL -Wl,-Bsymbolic)
  if(KT2N_BUILD_VOICE_NORMALIZATION)
    find_package(ICU 72 REQUIRED COMPONENTS i18n uc)
    foreach(_target kt2n_text kt2n_text_shared)
      target_sources(${_target} PRIVATE src/text/voice/voice_normalizer.cpp)
      target_link_libraries(${_target} PUBLIC ICU::i18n ICU::uc)
      target_compile_definitions(${_target} PUBLIC KT2N_TEXT_WITH_VOICE=1)
    endforeach()
  endif()

  add_executable(kt2n_prepare_text_data tools/prepare_text_data.cc)
  target_include_directories(kt2n_prepare_text_data PRIVATE "${KT2N_TEXT_DEPS}/include")
  target_link_libraries(kt2n_prepare_text_data PRIVATE "${KT2N_TEXT_DEPS}/lib/libfst.a" ${CMAKE_DL_LIBS} Threads::Threads)
  install(TARGETS kt2n_prepare_text_data RUNTIME DESTINATION bin)
endif()
