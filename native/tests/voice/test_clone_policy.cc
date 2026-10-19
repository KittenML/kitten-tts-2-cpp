// L11 unit tests for the clone policy ports (src/voice/clone_policy.h). CPU only.
//   g++ -std=c++20 -O1 -pthread -Isrc/voice tests/voice/test_clone_policy.cc -o /tmp/test_clone_policy && /tmp/test_clone_policy
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <set>
#include <thread>
#include <vector>

#include "clone_policy.h"

using namespace kt2n::voice;
using namespace std::chrono_literals;

static int failures = 0;
#define CHECK(c)                                                    \
  do {                                                              \
    if (!(c)) {                                                     \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++failures;                                                   \
    }                                                               \
  } while (0)

static void admission_limits() {
  CloneAdmission a(4, 16);  // live: 4 slots, queue 16
  for (int i = 0; i < 4; ++i) CHECK(a.acquire(120.0) == AdmitResult::kOk);
  CHECK(a.active() == 4);
  // 16 waiters fill the queue; the 17th is refused at once
  std::vector<std::thread> ws;
  std::atomic<int> admitted{0};
  std::vector<int> order;
  std::mutex om;
  for (int i = 0; i < 16; ++i) {
    ws.emplace_back([&, i] {
      if (a.acquire(10.0) == AdmitResult::kOk) {
        {
          std::lock_guard<std::mutex> g(om);
          order.push_back(i);
        }
        ++admitted;
      }
    });
    while (a.waiting() < i + 1) std::this_thread::sleep_for(1ms);  // deterministic arrival order
  }
  CHECK(a.waiting() == 16);
  auto t0 = Clock::now();
  CHECK(a.acquire(120.0) == AdmitResult::kBusyQueueFull);
  CHECK(Clock::now() - t0 < 20ms);  // refused without waiting
  // bypass_depth (the lazy rebuild) may wait even when full: here it times out on its own bound
  CHECK(a.acquire(0.1, std::nullopt, {}, true) == AdmitResult::kBusyWaitBound);
  // releases admit waiters in FIFO order
  for (int i = 0; i < 16; ++i) {
    a.release();
    while (admitted.load() < i + 1) std::this_thread::sleep_for(1ms);
  }
  for (auto& t : ws) t.join();
  CHECK(admitted.load() == 16);
  bool fifo = true;
  for (int i = 0; i < 16; ++i) fifo = fifo && order[i] == i;
  CHECK(fifo);
  for (int i = 0; i < 4; ++i) a.release();
  CHECK(a.active() == 0);
  auto s = a.stats();
  CHECK(s.refused_queue_full == 1 && s.refused_wait_bound == 1 && s.admitted == 20 && s.admitted_after_wait == 16);
  CHECK(s.max_active == 4 && s.max_waiting == 17);
}

static void admission_bounds() {
  CloneAdmission a(1, 4);
  CHECK(a.acquire(120.0) == AdmitResult::kOk);
  auto t0 = Clock::now();
  CHECK(a.acquire(0.2) == AdmitResult::kBusyWaitBound);  // CC_CLONE_WAIT_SECONDS bound
  auto dt = Clock::now() - t0;
  CHECK(dt >= 200ms && dt < 400ms);
  CHECK(a.acquire(120.0, Clock::now() + 100ms) == AdmitResult::kBusyDeadline);  // request deadline
  std::atomic<bool> gone{false};
  std::thread th([&] {
    std::this_thread::sleep_for(80ms);
    gone = true;
  });
  CHECK(a.acquire(120.0, std::nullopt, [&] { return gone.load(); }) == AdmitResult::kCancelled);
  th.join();
  CHECK(a.waiting() == 0);  // every refused / cancelled waiter left the queue
  a.release();
  CHECK(a.acquire(0.0) == AdmitResult::kOk);  // free slot, nobody waiting: immediate
  a.release();
  bool threw = false;
  try {
    a.release();
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);
}

static void lru_basics() {
  ActiveSafeLru<std::string, int> c(3);
  int loads = 0;
  auto loader = [&](int v) { return [&, v] { ++loads; return v; }; };
  { auto a = c.acquire("a", loader(1)); CHECK(*a == 1); }
  { auto b = c.acquire("b", loader(2)); }
  { auto d = c.acquire("c", loader(3)); }
  { auto a = c.acquire("a", loader(9)); CHECK(*a == 1); }  // hit, moves "a" to MRU
  CHECK(loads == 3);
  { auto e = c.acquire("d", loader(4)); }  // evicts LRU = "b"
  CHECK(c.entries() == 3 && !c.contains("b") && c.contains("a"));
  // active entries are never evicted: hold all, add more -> exceeds bound, release restores it
  auto la = c.acquire("a", loader(0));
  auto lc = c.acquire("c", loader(0));
  auto ld = c.acquire("d", loader(0));
  auto le = c.acquire("e", loader(5));
  CHECK(c.entries() == 4);
  la.reset();
  CHECK(c.entries() == 3 && !c.contains("a"));
  lc.reset();
  ld.reset();
  le.reset();
  CHECK(c.entries() == 3 && c.active_total() == 0);
}

static void lru_single_flight() {
  ActiveSafeLru<int, int> c(32);  // live: 32
  std::atomic<int> loads{0};
  std::vector<std::thread> ts;
  std::atomic<int> ok{0};
  for (int i = 0; i < 8; ++i)
    ts.emplace_back([&] {
      auto l = c.acquire(7, [&] {
        ++loads;
        std::this_thread::sleep_for(50ms);
        return 42;
      });
      ok += *l == 42;
    });
  for (auto& t : ts) t.join();
  CHECK(loads.load() == 1 && ok.load() == 8);
  // a failing leader: its caller gets the exception; a follower retries and loads
  std::atomic<int> attempt{0};
  std::atomic<int> got{0}, threw{0};
  std::thread t1([&] {
    try {
      auto l = c.acquire(8, [&] {
        ++attempt;
        std::this_thread::sleep_for(50ms);
        throw std::runtime_error("decode failed");
        return 0;
      });
    } catch (const std::runtime_error&) {
      ++threw;
    }
  });
  std::this_thread::sleep_for(10ms);
  std::thread t2([&] {
    auto l = c.acquire(8, [&] {
      ++attempt;
      return 5;
    });
    got = *l;
  });
  t1.join();
  t2.join();
  CHECK(threw.load() == 1 && got.load() == 5 && attempt.load() == 2);
}

static void lru_invalidate() {
  ActiveSafeLru<int, int> c(4);
  auto l = c.acquire(1, [] { return 1; });
  c.invalidate(1);
  CHECK(c.contains(1));  // still in use: kept
  l.reset();
  CHECK(!c.contains(1));  // removed when the last borrower leaves
  bool threw = false;
  try {
    c.acquire(1, [] { return 2; });
  } catch (const CacheEntryInvalidated&) {
    threw = true;
  }
  CHECK(threw);
}

static void inline_index() {
  InlineVoiceIndex idx(3);
  std::set<uint64_t> busy;
  auto in_use = [&](uint64_t id) { return busy.count(id) != 0; };
  std::vector<uint64_t> ev;
  CHECK(idx.lookup("a") == 0);
  CHECK(idx.insert("a", 1, in_use, &ev) == 1);
  CHECK(idx.insert("b", 2, in_use, &ev) == 2);
  CHECK(idx.insert("c", 3, in_use, &ev) == 3);
  CHECK(ev.empty() && idx.size() == 3);
  CHECK(idx.lookup("a") == 1);  // a becomes most recent: order b, c, a
  CHECK(idx.insert("d", 4, in_use, &ev) == 4);
  CHECK(ev.size() == 1 && ev[0] == 2);  // LRU idle entry goes
  ev.clear();
  busy = {3, 1, 4};  // everything in use (and e is protected by its own insert): stays above capacity
  CHECK(idx.insert("e", 5, in_use, &ev) == 5);
  CHECK(ev.empty() && idx.size() == 4);
  busy = {4};  // two go idle: one trim pass drops the LRU idle one (c), back to capacity
  idx.trim(in_use, &ev);
  CHECK(ev.size() == 1 && ev[0] == 3 && idx.size() == 3);
  ev.clear();
  CHECK(idx.insert("a", 99, in_use, &ev) == 1);  // existing entry wins (caller drops its duplicate)
  CHECK(ev.empty());
  idx.erase_voice(1);
  CHECK(idx.lookup("a") == 0 && idx.size() == 2);
  InlineVoiceIndex zero(0);  // capacity 0: kept while its first request runs, dropped on the next trim
  CHECK(zero.insert("x", 7, {}, &ev) == 7 && ev.empty() && zero.size() == 1);
  zero.trim({}, &ev);
  CHECK(ev.size() == 1 && ev[0] == 7 && zero.size() == 0);
}

// production _pyannote_embedding's result check: 512 finite values with L2 norm in [0.999, 1.001]
static void speaker_check() {
  std::vector<float> e(512, 0.f);
  e[0] = 1.f;
  CHECK(speaker_embedding_ok(e));                               // unit vector
  std::vector<float> spread(512, float(1.0 / std::sqrt(512.0)));
  CHECK(speaker_embedding_ok(spread));
  CHECK(!speaker_embedding_ok(std::vector<float>(512, 0.f)));   // zero (1/sqrt(0) upstream gives NaN; also caught)
  auto bad = e;
  bad[7] = std::numeric_limits<float>::quiet_NaN();
  CHECK(!speaker_embedding_ok(bad));
  bad = e;
  bad[7] = std::numeric_limits<float>::infinity();
  CHECK(!speaker_embedding_ok(bad));
  bad = e;
  bad[0] = 0.5f;                                                // norm 0.5
  CHECK(!speaker_embedding_ok(bad));
  bad[0] = 0.9985f;                                             // just under 0.999
  CHECK(!speaker_embedding_ok(bad));
  bad[0] = 0.9995f;                                             // inside
  CHECK(speaker_embedding_ok(bad));
  bad[0] = 1.0005f;                                             // inside
  CHECK(speaker_embedding_ok(bad));
  bad[0] = 1.0015f;                                             // just over 1.001
  CHECK(!speaker_embedding_ok(bad));
  CHECK(!speaker_embedding_ok(std::vector<float>(511, float(1.0 / std::sqrt(511.0)))));  // wrong length
  CHECK(!speaker_embedding_ok(std::vector<float>(513, float(1.0 / std::sqrt(513.0)))));
  CHECK(!speaker_embedding_ok(nullptr, 512));
}

int main() {
  speaker_check();
  inline_index();
  admission_limits();
  admission_bounds();
  lru_basics();
  lru_single_flight();
  lru_invalidate();
  std::printf("clone policy tests: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
