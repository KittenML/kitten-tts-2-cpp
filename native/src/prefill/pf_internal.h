// kt2n p2-prefill internals: error plumbing shared by the model loader and the run context.
#pragma once

#include <exception>
#include <new>
#include <stdexcept>
#include <string>

#include "kt2n/prefill.h"

namespace kt2n_pf {

class PfError : public std::runtime_error {
 public:
  PfError(int code, const std::string& msg) : std::runtime_error(msg), code_(code) {}
  int code() const noexcept { return code_; }

 private:
  int code_;
};

void set_last_error(const std::string& msg) noexcept;

// Run `fn`, translating exceptions into a KT2N_PF_* code + the thread-local message.
template <class F>
int guarded(F&& fn) noexcept {
  try {
    set_last_error("");
    fn();
    return KT2N_PF_OK;
  } catch (const PfError& e) {
    set_last_error(e.what());
    return e.code();
  } catch (const std::bad_alloc&) {
    set_last_error("host allocation failed");
    return KT2N_PF_ECUDA;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return KT2N_PF_EFILE;
  } catch (...) {
    set_last_error("unknown error");
    return KT2N_PF_EINVAL;
  }
}

}  // namespace kt2n_pf
