#pragma once
#include "kt2n/engine.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace kitten {
namespace detail {
inline std::string read_native_runtime_stats(kt2n_engine * engine) {
    constexpr size_t max_bytes = 1024 * 1024;
    size_t needed = 0;
    int status = kt2n_stats_json(engine, nullptr, 0, &needed);
    std::vector<char> buffer;
    for (;;) {
        if (status != KT2N_OK && status != KT2N_ERR_TOO_SMALL)
            throw std::runtime_error("cannot read native runtime identity (status " + std::to_string(status) + ")");
        if (!needed) throw std::runtime_error("native engine returned no runtime identity");
        if (needed > max_bytes || buffer.size() == max_bytes)
            throw std::runtime_error("native runtime identity exceeds the 1 MiB stats limit");
        // Each call takes a fresh snapshot. Counters may grow after sizing.
        const size_t capacity = std::max(needed, std::min(max_bytes, buffer.size() * 2));
        buffer.resize(capacity);
        status = kt2n_stats_json(engine, buffer.data(), buffer.size(), &needed);
        if (status != KT2N_OK) continue;
        if (!needed || needed > buffer.size() || buffer[needed - 1] != '\0')
            throw std::runtime_error("native engine returned an invalid runtime identity");
        return std::string(buffer.data(), needed - 1);
    }
}
} // namespace detail
} // namespace kitten
