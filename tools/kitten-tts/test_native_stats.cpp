#include "native_stats.h"

#include <cstring>
#include <functional>
#include <iostream>
#include <limits>

namespace {
std::function<int32_t(char *, size_t, size_t *)> snapshot;
void require(bool condition) { if (!condition) throw std::runtime_error("stats regression assertion failed"); }
void rejects(const std::string & message) {
    try { kitten::detail::read_native_runtime_stats(nullptr); }
    catch (const std::runtime_error & error) {
        require(std::string(error.what()).find(message) != std::string::npos);
        return;
    }
    throw std::runtime_error("invalid stats snapshot was accepted");
}
int32_t copy_snapshot(const std::string & value, char * buffer, size_t capacity, size_t * needed) {
    *needed = value.size() + 1;
    if (!buffer || capacity < *needed) return KT2N_ERR_TOO_SMALL;
    std::memcpy(buffer, value.c_str(), *needed);
    return KT2N_OK;
}
}

extern "C" int32_t kt2n_stats_json(kt2n_engine *, char * buffer, size_t capacity, size_t * needed) {
    return snapshot(buffer, capacity, needed);
}

int main() {
    // The real poll counter can gain a digit between sizing and the read.
    int calls = 0;
    const std::string final_snapshot = R"({"lm":"l3","render":"l7","normalizer":"native","tokenizer":"native","poll_idle":100})";
    snapshot = [&](char * buffer, size_t capacity, size_t * needed) {
        ++calls;
        const auto value = calls == 1 ? R"({"lm":"l3","render":"l7","normalizer":"native","tokenizer":"native","poll_idle":99})" : final_snapshot;
        return copy_snapshot(value, buffer, capacity, needed);
    };
    require(kitten::detail::read_native_runtime_stats(nullptr) == final_snapshot);
    require(calls == 3); // The exact-sized second call fails; the retry succeeds.

    // Growth may happen again while retrying, not only once.
    calls = 0;
    snapshot = [&](char * buffer, size_t capacity, size_t * needed) {
        const size_t lengths[] = {20, 80, 320, 320};
        require(calls < 4);
        return copy_snapshot(std::string(lengths[calls++], 'x'), buffer, capacity, needed);
    };
    require(kitten::detail::read_native_runtime_stats(nullptr) == std::string(320, 'x'));
    require(calls == 4);

    // An endlessly changing or broken producer cannot cause unbounded allocation.
    calls = 0;
    snapshot = [&](char *, size_t capacity, size_t * needed) {
        ++calls;
        require(capacity <= 1024 * 1024);
        *needed = capacity + 1;
        return KT2N_ERR_TOO_SMALL;
    };
    rejects("stats limit");
    require(calls < 24);

    snapshot = [](char *, size_t, size_t * needed) {
        *needed = std::numeric_limits<size_t>::max();
        return KT2N_ERR_TOO_SMALL;
    };
    rejects("stats limit");

    snapshot = [](char *, size_t, size_t *) { return KT2N_ERR_INVALID_ARGUMENT; };
    rejects("status " + std::to_string(KT2N_ERR_INVALID_ARGUMENT));

    snapshot = [](char *, size_t, size_t * needed) { *needed = 0; return KT2N_ERR_TOO_SMALL; };
    rejects("no runtime identity");

    snapshot = [](char * buffer, size_t, size_t * needed) {
        *needed = 2;
        if (!buffer) return KT2N_ERR_TOO_SMALL;
        buffer[0] = 'x'; buffer[1] = 'x';
        return KT2N_OK;
    };
    rejects("invalid runtime identity");
    std::cout << "native runtime stats: 7 regression cases passed\n";
}
