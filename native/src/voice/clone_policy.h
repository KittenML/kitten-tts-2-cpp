// L11: host-side clone policy, ported from production (kittenml-api ebcdd3f):
//   CloneAdmission  <- kitten_tts_09/clone_admission.py CloneAdmission (live: 4 slots, queue 16, 120 s wait)
//   StageGate       <- clone_admission.py StageGate (FIFO-fair bounded section; nothing refused)
//   ActiveSafeLru   <- kitten_tts_09/custom_voices.py ActiveSafeLRU (inline LRU, live 32 entries: never evicts an
//                      entry in use; concurrent misses for one key run the loader once)
//   speaker_embedding_ok <- runtime.py _pyannote_embedding's result check (every clone path, consent create too)
// Standard library only; header-only. Unit tests: tests/voice/test_clone_policy.cc.
#pragma once

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace kt2n::voice {

// Timed waits. Production builds wait on the steady clock (pthread_cond_clockwait). GCC 11's TSan does not
// intercept pthread_cond_clockwait and then reports false races / "double lock" on these mutexes (seen 10 Oct on
// the box), so the TSan build (-DKT2N_TSAN_COMPAT=1, as L8's engine) waits on the system clock through
// pthread_cond_timedwait, which it does intercept. The waits are bounded polls, so the clock choice is not
// observable.
// Production's check on every speaker embedding (runtime.py ebcdd3f _pyannote_embedding, ~:11345): 512 values,
// all finite, L2 norm in [0.999, 1.001]; anything else is 503 custom_voice_service_error ("The speaker encoder
// returned an invalid result."). It runs on every path that embeds an upload: inline, saved and consent create.
// A zero or non-finite vector must never be stored as a consent speaker: np.dot with NaN is NaN, and NaN < 0.70 is
// false, so every later speaker verification against it would pass. The norm is summed in double (production sums
// float32); the two differ only within ~1e-7 of the bounds, and a normalised embedding sits at 1 +- 1e-7.
inline bool speaker_embedding_ok(const float* e, size_t n) {
  if (n != 512 || e == nullptr) return false;
  double nn = 0.0;
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(e[i])) return false;
    nn += double(e[i]) * e[i];
  }
  const double norm = std::sqrt(nn);
  return norm >= 0.999 && norm <= 1.001;
}
inline bool speaker_embedding_ok(const std::vector<float>& e) { return speaker_embedding_ok(e.data(), e.size()); }

template <class Rep, class Period>
inline void policy_wait_for(std::condition_variable& cv, std::unique_lock<std::mutex>& lk,
                            std::chrono::duration<Rep, Period> d) {
#if KT2N_TSAN_COMPAT
  cv.wait_until(lk, std::chrono::system_clock::now() + d);
#else
  cv.wait_for(lk, d);
#endif
}

using Clock = std::chrono::steady_clock;

enum class AdmitResult : int {
  kOk = 0,
  kBusyQueueFull = 1,  // 429 custom_voice_creation_busy
  kBusyDeadline = 2,   // 429 custom_voice_creation_busy
  kBusyWaitBound = 3,  // 429 custom_voice_creation_busy
  kCancelled = 4       // client gone / request cancelled while waiting (GenerationCancelled)
};

inline const char* admit_reason(AdmitResult r) {
  switch (r) {
    case AdmitResult::kOk: return "ok";
    case AdmitResult::kBusyQueueFull: return "queue_full";
    case AdmitResult::kBusyDeadline: return "deadline";
    case AdmitResult::kBusyWaitBound: return "wait_bound";
    case AdmitResult::kCancelled: return "client_disconnected";
  }
  return "?";
}

class CloneAdmission {
 public:
  static constexpr std::chrono::milliseconds kPoll{50};  // POLL_SECONDS = 0.05

  CloneAdmission(int slots, int depth) : slots_(slots), depth_(depth) {
    if (slots < 1) throw std::invalid_argument("clone slots must be positive");
    if (depth < 0) throw std::invalid_argument("clone queue depth must not be negative");
  }

  struct Stats {
    int64_t admitted = 0, admitted_after_wait = 0, refused_queue_full = 0, refused_wait_bound = 0,
            refused_deadline = 0, cancelled_waiting = 0;
    int max_active = 0, max_waiting = 0;
    double max_wait_ms = 0.0;
  };

  // Same order of checks as production: free slot and nobody waiting -> admit; queue full -> busy; else wait FIFO
  // (head of the queue takes the next free slot) until cancelled / deadline / max_wait.
  AdmitResult acquire(double max_wait_s, std::optional<Clock::time_point> deadline = std::nullopt,
                      const std::function<bool()>& cancelled = {}, bool bypass_depth = false,
                      double* waited_s = nullptr) {
    const auto started = Clock::now();
    std::unique_lock<std::mutex> lk(mu_);
    if (active_ < slots_ && waiters_.empty()) {
      take_locked(0.0);
      if (waited_s) *waited_s = 0.0;
      return AdmitResult::kOk;
    }
    if (!bypass_depth && static_cast<int>(waiters_.size()) >= depth_) {
      ++stats_.refused_queue_full;
      return AdmitResult::kBusyQueueFull;
    }
    const uint64_t ticket = next_ticket_++;
    waiters_.push_back(ticket);
    stats_.max_waiting = std::max(stats_.max_waiting, static_cast<int>(waiters_.size()));
    auto leave = [&](AdmitResult r) {
      for (auto it = waiters_.begin(); it != waiters_.end(); ++it)
        if (*it == ticket) {
          waiters_.erase(it);
          break;
        }
      cv_.notify_all();
      return r;
    };
    for (;;) {
      if (active_ < slots_ && waiters_.front() == ticket) {
        waiters_.pop_front();
        double w = std::chrono::duration<double>(Clock::now() - started).count();
        take_locked(w);
        ++stats_.admitted_after_wait;
        cv_.notify_all();
        if (waited_s) *waited_s = w;
        return AdmitResult::kOk;
      }
      if (cancelled) {
        lk.unlock();
        bool c = cancelled();
        lk.lock();
        if (c) {
          ++stats_.cancelled_waiting;
          return leave(AdmitResult::kCancelled);
        }
        if (active_ < slots_ && waiters_.front() == ticket) continue;
      }
      auto now = Clock::now();
      if (deadline && now >= *deadline) {
        ++stats_.refused_deadline;
        return leave(AdmitResult::kBusyDeadline);
      }
      if (std::chrono::duration<double>(now - started).count() >= max_wait_s) {
        ++stats_.refused_wait_bound;
        return leave(AdmitResult::kBusyWaitBound);
      }
      policy_wait_for(cv_, lk, kPoll);
    }
  }

  void release() {
    std::lock_guard<std::mutex> lk(mu_);
    if (active_ <= 0) throw std::logic_error("clone admission released more often than acquired");
    --active_;
    cv_.notify_all();
  }

  int active() const {
    std::lock_guard<std::mutex> lk(mu_);
    return active_;
  }
  int waiting() const {
    std::lock_guard<std::mutex> lk(mu_);
    return static_cast<int>(waiters_.size());
  }
  Stats stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stats_;
  }

 private:
  void take_locked(double waited) {
    ++active_;
    ++stats_.admitted;
    stats_.max_active = std::max(stats_.max_active, active_);
    stats_.max_wait_ms = std::max(stats_.max_wait_ms, waited * 1000.0);
  }
  const int slots_, depth_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  int active_ = 0;
  std::deque<uint64_t> waiters_;
  uint64_t next_ticket_ = 0;
  Stats stats_;
};

// FIFO-fair bounded section; callers queue, nothing is refused (cancel ends the wait).
class StageGate {
 public:
  explicit StageGate(int size) : size_(size) {
    if (size < 1) throw std::invalid_argument("gate size must be positive");
  }
  bool enter(const std::function<bool()>& cancelled = {}) {
    std::unique_lock<std::mutex> lk(mu_);
    if (active_ < size_ && waiters_.empty()) {
      ++active_;
      return true;
    }
    const uint64_t ticket = next_++;
    waiters_.push_back(ticket);
    while (!(active_ < size_ && waiters_.front() == ticket)) {
      if (cancelled && cancelled()) {
        for (auto it = waiters_.begin(); it != waiters_.end(); ++it)
          if (*it == ticket) {
            waiters_.erase(it);
            break;
          }
        cv_.notify_all();
        return false;
      }
      policy_wait_for(cv_, lk, std::chrono::milliseconds(50));
    }
    waiters_.pop_front();
    ++active_;
    cv_.notify_all();
    return true;
  }
  void leave() {
    std::lock_guard<std::mutex> lk(mu_);
    --active_;
    cv_.notify_all();
  }

 private:
  const int size_;
  std::mutex mu_;
  std::condition_variable cv_;
  int active_ = 0;
  std::deque<uint64_t> waiters_;
  uint64_t next_ = 0;
};

class CacheEntryInvalidated : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Bounded LRU that never evicts an entry while a borrower holds it. It may exceed max_entries only while every
// eviction candidate is active; releasing the last borrower restores the bound. Concurrent misses for one key
// run the loader exactly once (the leader); followers wait and then retry (and become the leader if it failed).
template <class K, class V>
class ActiveSafeLru {
 public:
  explicit ActiveSafeLru(int max_entries) : max_(max_entries) {
    if (max_entries < 1) throw std::invalid_argument("max_entries must be positive");
  }

  class Lease {
   public:
    Lease() = default;
    Lease(ActiveSafeLru* c, K k, std::shared_ptr<V> v, uint64_t gen) : c_(c), k_(std::move(k)), v_(std::move(v)), gen_(gen) {}
    Lease(Lease&& o) noexcept : c_(o.c_), k_(std::move(o.k_)), v_(std::move(o.v_)), gen_(o.gen_) { o.c_ = nullptr; }
    Lease& operator=(Lease&& o) noexcept {
      if (this != &o) {
        reset();
        c_ = o.c_;
        k_ = std::move(o.k_);
        v_ = std::move(o.v_);
        gen_ = o.gen_;
        o.c_ = nullptr;
      }
      return *this;
    }
    ~Lease() { reset(); }
    const V& operator*() const { return *v_; }
    const V* operator->() const { return v_.get(); }
    explicit operator bool() const { return c_ != nullptr; }
    void reset() {
      if (c_) c_->release(k_, gen_);
      c_ = nullptr;
    }

   private:
    ActiveSafeLru* c_ = nullptr;
    K k_{};
    std::shared_ptr<V> v_;
    uint64_t gen_ = 0;
  };

  // loader() returns V or throws; the exception reaches the leader's caller.
  Lease acquire(const K& key, const std::function<V()>& loader) {
    for (;;) {
      std::unique_lock<std::mutex> lk(mu_);
      if (invalidated_.count(key)) throw CacheEntryInvalidated("cache entry has been invalidated");
      auto it = index_.find(key);
      if (it != index_.end()) {
        Entry& e = *it->second;
        ++e.active;
        order_.splice(order_.end(), order_, it->second);  // move to MRU
        return Lease(this, key, e.value, e.gen);
      }
      if (loading_.count(key)) {
        cv_.wait(lk, [&] { return loading_.count(key) == 0; });
        continue;  // retry: hit, or lead a new load if the leader failed
      }
      loading_.insert(key);
      lk.unlock();
      std::shared_ptr<V> value;
      try {
        value = std::make_shared<V>(loader());
      } catch (...) {
        lk.lock();
        loading_.erase(key);
        cv_.notify_all();
        throw;
      }
      lk.lock();
      loading_.erase(key);
      cv_.notify_all();
      if (invalidated_.count(key)) throw CacheEntryInvalidated("cache entry was invalidated while loading");
      order_.push_back(Entry{key, value, 1, ++gen_counter_});
      auto pos = std::prev(order_.end());
      index_[key] = pos;
      evict_inactive_locked();
      return Lease(this, key, value, pos->gen);
    }
  }

  void invalidate(const K& key) {
    std::lock_guard<std::mutex> lk(mu_);
    invalidated_.insert(key);
    auto it = index_.find(key);
    if (it != index_.end() && it->second->active == 0) {
      order_.erase(it->second);
      index_.erase(it);
    }
  }

  int entries() const {
    std::lock_guard<std::mutex> lk(mu_);
    return static_cast<int>(order_.size());
  }
  int active_total() const {
    std::lock_guard<std::mutex> lk(mu_);
    int a = 0;
    for (const auto& e : order_) a += e.active;
    return a;
  }
  bool contains(const K& key) const {
    std::lock_guard<std::mutex> lk(mu_);
    return index_.count(key) != 0;
  }

 private:
  struct Entry {
    K key;
    std::shared_ptr<V> value;
    int active;
    uint64_t gen;
  };
  void release(const K& key, uint64_t gen) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = index_.find(key);
    if (it == index_.end() || it->second->gen != gen) return;  // entry replaced (current is not ours)
    Entry& e = *it->second;
    if (--e.active < 0) throw std::logic_error("custom voice cache active count underflow");
    if (invalidated_.count(key) && e.active == 0) {
      order_.erase(it->second);
      index_.erase(it);
    }
    evict_inactive_locked();
  }
  void evict_inactive_locked() {
    if (static_cast<int>(order_.size()) <= max_) return;
    for (auto it = order_.begin(); it != order_.end() && static_cast<int>(order_.size()) > max_;) {
      if (it->active == 0) {
        index_.erase(it->key);
        it = order_.erase(it);
      } else {
        ++it;
      }
    }
  }
  const int max_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::list<Entry> order_;  // LRU first, MRU last
  std::map<K, typename std::list<Entry>::iterator> index_;
  std::set<K> loading_;
  std::set<K> invalidated_;
  uint64_t gen_counter_ = 0;
};

// The engine-level inline clone table: production's ActiveSafeLRU (CUSTOM_VOICE_CACHE_ENTRIES = 32, entries in use
// are never evicted) applied to engine voice ids, keyed by the inline identity (engine.h identity_key). The engine
// owns the voices (LM prefix K/V + render state); this only decides which idle ones to drop.
//   lookup(key)             0 on a miss; a hit becomes the most recently used
//   insert(key, id, in_use) registers a fresh build; returns the id the key maps to afterwards (an existing entry
//                           wins: the caller then drops its duplicate) and appends the ids to evict to *evict:
//                           least recently used first, only idle ones, until size <= capacity. The new entry is
//                           never evicted by its own insert (its request is about to use it: production's lease)
//   trim(in_use)            the same eviction pass; call when a voice goes idle (production evicts on release)
//   erase_voice(id)         forget an id the engine evicted for another reason
// The caller runs kt2n_voice_evict on the returned ids outside its own locks. Thread-safe.
class InlineVoiceIndex {
 public:
  explicit InlineVoiceIndex(int capacity = 32) : cap_(capacity < 0 ? 0 : capacity) {}
  uint64_t lookup(const std::string& key) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = index_.find(key);
    if (it == index_.end()) return 0;
    order_.splice(order_.end(), order_, it->second);
    return it->second->second;
  }
  uint64_t insert(const std::string& key, uint64_t voice, const std::function<bool(uint64_t)>& in_use,
                  std::vector<uint64_t>* evict) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = index_.find(key);
    if (it != index_.end()) {
      order_.splice(order_.end(), order_, it->second);
      return it->second->second;
    }
    order_.emplace_back(key, voice);
    index_[key] = std::prev(order_.end());
    trim_locked(in_use, evict, voice);
    return voice;
  }
  void trim(const std::function<bool(uint64_t)>& in_use, std::vector<uint64_t>* evict) {
    std::lock_guard<std::mutex> lk(mu_);
    trim_locked(in_use, evict, 0);
  }
  void erase_voice(uint64_t voice) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto it = order_.begin(); it != order_.end(); ++it)
      if (it->second == voice) {
        index_.erase(it->first);
        order_.erase(it);
        return;
      }
  }
  int size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return static_cast<int>(order_.size());
  }

 private:
  void trim_locked(const std::function<bool(uint64_t)>& in_use, std::vector<uint64_t>* evict, uint64_t keep) {
    for (auto it = order_.begin(); it != order_.end() && static_cast<int>(order_.size()) > cap_;) {
      if ((keep && it->second == keep) || (in_use && in_use(it->second))) {
        ++it;
        continue;
      }
      if (evict) evict->push_back(it->second);
      index_.erase(it->first);
      it = order_.erase(it);
    }
  }
  const int cap_;
  mutable std::mutex mu_;
  std::list<std::pair<std::string, uint64_t>> order_;  // LRU first
  std::map<std::string, std::list<std::pair<std::string, uint64_t>>::iterator> index_;
};

}  // namespace kt2n::voice
