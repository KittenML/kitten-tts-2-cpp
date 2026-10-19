# Third-party dependencies, pinned by URL + SHA-256 (no git fetches, so CI and
# the GPU box need only plain HTTPS). For an offline build, unpack the same
# tarball and pass -DFETCHCONTENT_SOURCE_DIR_<NAME>=<dir>.

include(FetchContent)
find_package(Python 3.10 COMPONENTS Interpreter)

# GoogleTest v1.17.0 release tarball.
if(KT2N_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)
endif()
