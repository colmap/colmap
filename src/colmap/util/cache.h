// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/util/logging.h"

#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace colmap {

// Least Recently Used cache implementation. Whenever the cache size is
// exceeded, the least recently used (by Get and GetMutable) is deleted.
template <typename key_t, typename value_t>
class LRUCache {
 public:
  using LoadFn = std::function<std::shared_ptr<value_t>(const key_t&)>;

  LRUCache(size_t max_num_elems, LoadFn load_fn);
  virtual ~LRUCache() = default;

  // The number of elements in the cache.
  virtual size_t NumElems() const;
  virtual size_t MaxNumElems() const;

  // Check whether the element with the given key exists.
  virtual bool Exists(const key_t& key) const;

  // Get the value of an element either from the cache or compute the new value.
  virtual std::shared_ptr<value_t> Get(const key_t& key);

  // Manually set the value of an element.
  virtual void Set(const key_t& key, std::shared_ptr<value_t> value);

  // Manually evict an element from the cache.
  // Returns true if the element was evicted.
  virtual bool Evict(const key_t& key) {
    const auto it = elems_map_.find(key);
    if (it != elems_map_.end()) {
      elems_list_.erase(it->second);
      elems_map_.erase(it);
      return true;
    }
    return false;
  }

  // Pop least recently used element from cache.
  virtual void Pop();

  // Clear all elements from cache.
  virtual void Clear();

 protected:
  typedef typename std::pair<key_t, std::shared_ptr<value_t>> key_value_pair_t;
  typedef typename std::list<key_value_pair_t>::iterator list_iterator_t;

  // Maximum number of least-recently-used elements the cache remembers.
  const size_t max_num_elems_;

  // List to keep track of the least-recently-used elements.
  std::list<key_value_pair_t> elems_list_;

  // Mapping from key to location in the list.
  std::unordered_map<key_t, list_iterator_t> elems_map_;

  // Function to compute new values if not in the cache.
  const LoadFn load_fn_;
};

// Least Recently Used cache implementation that is constrained by a maximum
// memory limitation of its elements. Whenever the memory limit is exceeded, the
// least recently used (by Get and GetMutable) is deleted. Each element must
// implement a `size_t NumBytes()` method that returns its size in memory.
template <typename key_t, typename value_t>
class MemoryConstrainedLRUCache : public LRUCache<key_t, value_t> {
 public:
  using typename LRUCache<key_t, value_t>::LoadFn;

  MemoryConstrainedLRUCache(size_t max_num_bytes, LoadFn load_fn);

  size_t NumBytes() const;
  size_t MaxNumBytes() const;
  void UpdateNumBytes(const key_t& key);

  bool Evict(const key_t& key) override {
    elems_num_bytes_.erase(key);
    return LRUCache<key_t, value_t>::Evict(key);
  }

  void Set(const key_t& key, std::shared_ptr<value_t> value) override;
  void Pop() override;
  void Clear() override;

 private:
  using typename LRUCache<key_t, value_t>::key_value_pair_t;
  using typename LRUCache<key_t, value_t>::list_iterator_t;
  using LRUCache<key_t, value_t>::max_num_elems_;
  using LRUCache<key_t, value_t>::elems_list_;
  using LRUCache<key_t, value_t>::elems_map_;
  using LRUCache<key_t, value_t>::load_fn_;

  const size_t max_num_bytes_;
  size_t num_bytes_;
  std::unordered_map<key_t, size_t> elems_num_bytes_;
};

template <typename key_t, typename value_t>
class ThreadSafeLRUCache : public LRUCache<key_t, value_t> {
 public:
  using typename LRUCache<key_t, value_t>::LoadFn;

  ThreadSafeLRUCache(size_t max_num_elems, LoadFn load_fn)
      : cache_(max_num_elems, std::move(load_fn)) {}

  size_t NumElems() const override;
  size_t MaxNumElems() const override;

  bool Exists(const key_t& key) const override;

  std::shared_ptr<value_t> Get(const key_t& key) override;

  void Set(const key_t& key, std::shared_ptr<value_t> value) override;

  bool Evict(const key_t& key) override;

  void Pop() override;

  void Clear() override;

 protected:
  struct Entry {
    Entry() : future(promise.get_future()) {}
    std::promise<std::shared_ptr<value_t>> promise;
    std::shared_future<std::shared_ptr<value_t>> future;
    bool is_loading = false;
  };

  std::mutex cache_mutex_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename key_t, typename value_t>
LRUCache<key_t, value_t>::LRUCache(const size_t max_num_elems, LoadFn load_fn)
    : max_num_elems_(max_num_elems), load_fn_(std::move(load_fn)) {
  THROW_CHECK_NOTNULL(load_fn_);
  THROW_CHECK_GT(max_num_elems, 0);
}

template <typename key_t, typename value_t>
size_t LRUCache<key_t, value_t>::NumElems() const {
  return elems_map_.size();
}

template <typename key_t, typename value_t>
size_t LRUCache<key_t, value_t>::MaxNumElems() const {
  return max_num_elems_;
}

template <typename key_t, typename value_t>
bool LRUCache<key_t, value_t>::Exists(const key_t& key) const {
  return elems_map_.find(key) != elems_map_.end();
}

template <typename key_t, typename value_t>
std::shared_ptr<value_t> LRUCache<key_t, value_t>::Get(const key_t& key) {
  const auto it = elems_map_.find(key);
  if (it == elems_map_.end()) {
    Set(key, load_fn_(key));
    return elems_map_[key]->second;
  } else {
    elems_list_.splice(elems_list_.begin(), elems_list_, it->second);
    return it->second->second;
  }
}

template <typename key_t, typename value_t>
void LRUCache<key_t, value_t>::Set(const key_t& key,
                                   std::shared_ptr<value_t> value) {
  THROW_CHECK_NOTNULL(value);

  auto it = elems_map_.find(key);
  elems_list_.emplace_front(key, std::move(value));
  if (it != elems_map_.end()) {
    elems_list_.erase(it->second);
    elems_map_.erase(it);
  }
  elems_map_.emplace_hint(it, key, elems_list_.begin());
  if (elems_map_.size() > max_num_elems_) {
    Pop();
  }
}

template <typename key_t, typename value_t>
bool LRUCache<key_t, value_t>::Evict(const key_t& key) {
  const auto it = elems_map_.find(key);
  if (it != elems_map_.end()) {
    elems_list_.erase(it->second);
    elems_map_.erase(it);
    return true;
  }
  return false;
}

template <typename key_t, typename value_t>
void LRUCache<key_t, value_t>::Pop() {
  if (!elems_list_.empty()) {
    auto last = elems_list_.end();
    --last;
    elems_map_.erase(last->first);
    elems_list_.pop_back();
  }
}

template <typename key_t, typename value_t>
void LRUCache<key_t, value_t>::Clear() {
  elems_list_.clear();
  elems_map_.clear();
}

template <typename key_t, typename value_t>
MemoryConstrainedLRUCache<key_t, value_t>::MemoryConstrainedLRUCache(
    const size_t max_num_bytes, LoadFn load_fn)
    : LRUCache<key_t, value_t>(std::numeric_limits<size_t>::max(),
                               std::move(load_fn)),
      max_num_bytes_(max_num_bytes),
      num_bytes_(0) {
  THROW_CHECK_GT(max_num_bytes, 0);
}

template <typename key_t, typename value_t>
size_t MemoryConstrainedLRUCache<key_t, value_t>::NumBytes() const {
  return num_bytes_;
}

template <typename key_t, typename value_t>
size_t MemoryConstrainedLRUCache<key_t, value_t>::MaxNumBytes() const {
  return max_num_bytes_;
}

template <typename key_t, typename value_t>
void MemoryConstrainedLRUCache<key_t, value_t>::Set(
    const key_t& key, std::shared_ptr<value_t> value) {
  THROW_CHECK_NOTNULL(value);

  const size_t num_bytes = value->NumBytes();
  auto it = elems_map_.find(key);
  elems_list_.emplace_front(key, std::move(value));
  if (it != elems_map_.end()) {
    elems_list_.erase(it->second);
    elems_map_.erase(it);
  }
  elems_map_[key] = elems_list_.begin();

  num_bytes_ += num_bytes;
  elems_num_bytes_.emplace(key, num_bytes);

  while (num_bytes_ > max_num_bytes_ && elems_map_.size() > 1) {
    Pop();
  }
}

template <typename key_t, typename value_t>
void MemoryConstrainedLRUCache<key_t, value_t>::Pop() {
  if (!elems_list_.empty()) {
    auto last = elems_list_.end();
    --last;
    num_bytes_ -= elems_num_bytes_.at(last->first);
    THROW_CHECK_GE(num_bytes_, 0);
    elems_num_bytes_.erase(last->first);
    elems_map_.erase(last->first);
    elems_list_.pop_back();
  }
}

template <typename key_t, typename value_t>
void MemoryConstrainedLRUCache<key_t, value_t>::UpdateNumBytes(
    const key_t& key) {
  auto& num_bytes = elems_num_bytes_.at(key);
  num_bytes_ -= num_bytes;
  THROW_CHECK_GE(num_bytes_, 0);
  num_bytes = LRUCache<key_t, value_t>::Get(key)->NumBytes();
  num_bytes_ += num_bytes;

  while (num_bytes_ > max_num_bytes_ && elems_map_.size() > 1) {
    Pop();
  }
}

template <typename key_t, typename value_t>
void MemoryConstrainedLRUCache<key_t, value_t>::Clear() {
  LRUCache<key_t, value_t>::Clear();
  num_bytes_ = 0;
  elems_num_bytes_.clear();
}

template <typename key_t, typename value_t>
std::shared_ptr<value_t> ThreadSafeLRUCache<key_t, value_t>::Get(
    const key_t& key) {
  bool should_load = false;
  std::shared_ptr<Entry> entry;
  std::shared_future<std::shared_ptr<value_t>> shared_future;

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    entry = cache_.Get(key);
    if (entry == nullptr) {
      return nullptr;
    }
    shared_future = entry->future;
    if (!entry->is_loading) {
      should_load = true;
      entry->is_loading = true;
    }
  }

  if (should_load) {
    try {
      entry->promise.set_value(THROW_CHECK_NOTNULL(load_fn_(key)));
    } catch (...) {
      // Evict the cache entry after load failed and set the exception.
      std::lock_guard<std::mutex> lock(cache_mutex_);
      cache_.Evict(key);
      entry->promise.set_exception(std::current_exception());
    }
  }

  return shared_future.get();
}

template <typename key_t, typename value_t>
bool ThreadSafeLRUCache<key_t, value_t>::Evict(const key_t& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return cache_.Evict(key);
}

template <typename key_t, typename value_t>
void ThreadSafeLRUCache<key_t, value_t>::Clear() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return cache_.Clear();
}

template <typename key_t, typename value_t>
int ThreadSafeLRUCache<key_t, value_t>::NumElems() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return cache_.NumElems();
}

template <typename key_t, typename value_t>
int ThreadSafeLRUCache<key_t, value_t>::MaxNumElems() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return cache_.MaxNumElems();
}

}  // namespace colmap
