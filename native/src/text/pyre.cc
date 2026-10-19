#define PCRE2_CODE_UNIT_WIDTH 8
#include "pyre.h"

#include <pcre2.h>

#include <memory>
#include <stdexcept>

namespace kt2n::text {
namespace {

struct MatchDataDeleter {
  void operator()(pcre2_match_data* d) const { pcre2_match_data_free(d); }
};

// One JIT stack and match context per thread (JIT needs more than the default
// 32 KiB stack on long inputs with \S* scans).
struct ThreadCtx {
  pcre2_jit_stack* stack = nullptr;
  pcre2_match_context* ctx = nullptr;
  ThreadCtx() {
    stack = pcre2_jit_stack_create(64 * 1024, 4 * 1024 * 1024, nullptr);
    ctx = pcre2_match_context_create(nullptr);
    if (stack && ctx) pcre2_jit_stack_assign(ctx, nullptr, stack);
  }
  ~ThreadCtx() {
    if (ctx) pcre2_match_context_free(ctx);
    if (stack) pcre2_jit_stack_free(stack);
  }
};

pcre2_match_context* thread_ctx() {
  thread_local ThreadCtx t;
  return t.ctx;
}

}  // namespace

std::string py_pattern(std::string_view p) {
  std::string out;
  out.reserve(p.size() + 64);
  int depth = 0;  // inside [...]
  for (size_t i = 0; i < p.size(); ++i) {
    const char c = p[i];
    if (c == '\\' && i + 1 < p.size()) {
      const char n = p[i + 1];
      if (n == 's' || n == 'S') {
        if (depth == 0) {
          out += (n == 's') ? "[" : "[^";
          out += kPySClass;
          out += "]";
        } else {
          if (n == 'S') throw std::runtime_error("py_pattern: \\S inside a class is not supported");
          out += kPySClass;
        }
      } else {
        out.push_back(c);
        out.push_back(n);
      }
      ++i;
      continue;
    }
    if (c == '[' && depth == 0) {
      depth = 1;
      out.push_back(c);
      // A leading ']' or '^]' is literal.
      if (i + 1 < p.size() && p[i + 1] == '^') {
        out.push_back('^');
        ++i;
      }
      if (i + 1 < p.size() && p[i + 1] == ']') {
        out.push_back(']');
        ++i;
      }
      continue;
    }
    if (c == ']' && depth == 1) depth = 0;
    out.push_back(c);
  }
  return out;
}

Regex::Regex(std::string_view pattern, uint32_t options) {
  int error = 0;
  PCRE2_SIZE offset = 0;
  code_ = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(),
                        PCRE2_UTF | PCRE2_UCP | PCRE2_NO_UTF_CHECK | options, &error, &offset, nullptr);
  if (code_ == nullptr) {
    PCRE2_UCHAR buffer[256];
    pcre2_get_error_message(error, buffer, sizeof(buffer));
    throw std::runtime_error("pcre2_compile: " + std::string(reinterpret_cast<char*>(buffer)) + " at " +
                             std::to_string(offset) + " in " + std::string(pattern));
  }
  jit_ = pcre2_jit_compile(code_, PCRE2_JIT_COMPLETE) == 0;
}

Regex::~Regex() {
  if (code_) pcre2_code_free(code_);
}

static bool run(pcre2_code* code, std::string_view subject, size_t pos, uint32_t opts, Match* m) {
  std::unique_ptr<pcre2_match_data, MatchDataDeleter> data(pcre2_match_data_create_from_pattern(code, nullptr));
  int rc = pcre2_match(code, reinterpret_cast<PCRE2_SPTR>(subject.data()), subject.size(), pos,
                       opts | PCRE2_NO_UTF_CHECK, data.get(), thread_ctx());
  if (rc == PCRE2_ERROR_NOMATCH) return false;
  if (rc == PCRE2_ERROR_JIT_STACKLIMIT || rc == PCRE2_ERROR_MATCHLIMIT || rc == PCRE2_ERROR_DEPTHLIMIT ||
      rc == PCRE2_ERROR_HEAPLIMIT) {
    // Python has no such limits: retry with the interpreter and no limits.
    pcre2_match_context* ctx = pcre2_match_context_create(nullptr);
    pcre2_set_match_limit(ctx, 0xFFFFFFFFu);
    pcre2_set_depth_limit(ctx, 0xFFFFFFFFu);
    pcre2_set_heap_limit(ctx, 0xFFFFFFFFu);
    rc = pcre2_match(code, reinterpret_cast<PCRE2_SPTR>(subject.data()), subject.size(), pos,
                     opts | PCRE2_NO_UTF_CHECK | PCRE2_NO_JIT, data.get(), ctx);
    pcre2_match_context_free(ctx);
    if (rc == PCRE2_ERROR_NOMATCH) return false;
  }
  if (rc < 0) throw std::runtime_error("pcre2_match failed: " + std::to_string(rc));
  if (m) {
    uint32_t pairs = pcre2_get_ovector_count(data.get());
    const PCRE2_SIZE* ov = pcre2_get_ovector_pointer(data.get());
    uint32_t groups = 0;
    pcre2_pattern_info(code, PCRE2_INFO_CAPTURECOUNT, &groups);
    m->subject = subject;
    m->starts.assign(groups + 1, -1);
    m->ends.assign(groups + 1, -1);
    for (uint32_t g = 0; g <= groups && g < pairs; ++g) {
      if (g < static_cast<uint32_t>(rc) || g == 0) {
        if (ov[2 * g] != PCRE2_UNSET) {
          m->starts[g] = static_cast<ptrdiff_t>(ov[2 * g]);
          m->ends[g] = static_cast<ptrdiff_t>(ov[2 * g + 1]);
        }
      }
    }
  }
  return true;
}

bool Regex::search(std::string_view subject, size_t pos, Match* m, uint32_t match_options) const {
  return run(code_, subject, pos, match_options, m);
}

bool Regex::match(std::string_view subject, size_t pos, Match* m) const {
  return run(code_, subject, pos, PCRE2_ANCHORED, m);
}

bool Regex::fullmatch(std::string_view subject, Match* m) const {
  return run(code_, subject, 0, PCRE2_ANCHORED | PCRE2_ENDANCHORED, m);
}

std::vector<Match> Regex::finditer(std::string_view subject) const {
  std::vector<Match> out;
  size_t pos = 0;
  bool last_empty = false;
  while (pos <= subject.size()) {
    Match m;
    if (!run(code_, subject, pos, last_empty ? PCRE2_NOTEMPTY_ATSTART : 0, &m)) break;
    last_empty = m.end() == m.start();
    pos = m.end();
    out.push_back(std::move(m));
  }
  return out;
}

std::string Regex::sub(std::string_view subject, const Repl& repl) const {
  std::string out;
  size_t last = 0;
  size_t pos = 0;
  bool last_empty = false;
  bool any = false;
  while (pos <= subject.size()) {
    Match m;
    if (!run(code_, subject, pos, last_empty ? PCRE2_NOTEMPTY_ATSTART : 0, &m)) break;
    if (!any) out.reserve(subject.size() + 16);
    any = true;
    out.append(subject.substr(last, m.start() - last));
    out.append(repl(m));
    last = m.end();
    last_empty = m.end() == m.start();
    pos = m.end();
  }
  if (!any) return std::string(subject);
  out.append(subject.substr(last));
  return out;
}

std::string Regex::sub(std::string_view subject, std::string_view tmpl) const {
  return sub(subject, [&](const Match& m) {
    std::string r;
    for (size_t i = 0; i < tmpl.size(); ++i) {
      if (tmpl[i] == '\\' && i + 1 < tmpl.size() && tmpl[i + 1] >= '0' && tmpl[i + 1] <= '9') {
        r.append(m.group(tmpl[i + 1] - '0'));
        ++i;
      } else {
        r.push_back(tmpl[i]);
      }
    }
    return r;
  });
}

std::vector<std::string> Regex::split(std::string_view subject) const {
  std::vector<std::string> out;
  size_t last = 0;
  for (const Match& m : finditer(subject)) {
    out.emplace_back(subject.substr(last, m.start() - last));
    for (size_t g = 1; g <= m.groups(); ++g) out.emplace_back(m.group(static_cast<int>(g)));
    last = m.end();
  }
  out.emplace_back(subject.substr(last));
  return out;
}

}  // namespace kt2n::text
