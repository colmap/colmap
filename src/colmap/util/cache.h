// Copyright (c), ETH Zurich and UNC Chapel Hill.
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
#include <shared_mutex>
#include <unordered_map>

namespace colmap {

// Least Recently Used cache implementation. Whenever the cache size is
// exceeded, the least recently used (by Get) is deleted.
template <typename key_t, typename value_t>
class LRUCache {
 public:
  using LoadFn = std::function<std::shared_ptr<value_t>(const key_t&)>;

  LRUCache(size_t max_num_elems, LoadFn load_fn);

  // The number of elements in the cache.
  size_t NumElems() const;
  size_t MaxNumElems() const;

  // Check whether the element with the given key exists.
  bool Exists(const key_t& key) const;

  // Get the value of an element either from the cache or compute the new value.
  std::shared_ptr<value_t> Get(const key_t& key);

  // Manually evict an element from the cache.
  // Returns true if the element was evicted.
  bool Evict(const key_t& key);

  // Pop least recently used element from cache.
  void Pop();

  // Clear all elements from cache.
  void Clear();

 private:
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

// Thread-safe Least Recently Used cache implementation.
template <typename key_t, typename value_t>
class ThreadSafeLRUCache {
 public:
  using LoadFn = std::function<std::shared_ptr<value_t>(const key_t&)>;

  ThreadSafeLRUCache(size_t max_num_elems, LoadFn load_fn);

  // The number of elements in the cache.
  size_t NumElems() const;
  size_t MaxNumElems() const;

  // Check whether the element with the given key exists.
  bool Exists(const key_t& key) const;

  // Get the value of an element either from the cache or compute the new value.
  std::shared_ptr<value_t> Get(const key_t& key);

  // Manually evict an element from the cache.
  // Returns true if the element was evicted.
  bool Evict(const key_t& key);

  // Pop least recently used element from cache.
  void Pop();

  // Clear all elements from cache.
  void Clear();

 protected:
  struct Entry {
    Entry() : future(promise.get_future()) {}
    std::promise<std::shared_ptr<value_t>> promise;
    std::shared_future<std::shared_ptr<value_t>> future;
    bool is_loading = false;
  };

  // Function to compute new values if not in the cache.
  const LoadFn load_fn_;

  mutable std::shared_mutex cache_mutex_;
  LRUCache<key_t, Entry> cache_;
};

// Least Recently Used cache implementation that is constrained by a maximum
// memory limitation of its elements. Whenever the memory limit is exceeded, the
// least recently used (by Get) is deleted. Each element must implement a
// `size_t NumBytes()` method that returns its size in memory.
template <typename key_t, typename value_t>
class MemoryConstrainedLRUCache {
 public:
  using LoadFn = std::function<std::shared_ptr<value_t>(const key_t&)>;

  MemoryConstrainedLRUCache(size_t max_num_bytes, LoadFn load_fn);

  // The size in bytes of the elements in the cache.
  size_t NumBytes() const;
  size_t MaxNumBytes() const;
  void UpdateNumBytes(const key_t& key);

  // The number of elements in the cache.
  size_t NumElems() const;
  size_t MaxNumElems() const;

  // Check whether the element with the given key exists.
  bool Exists(const key_t& key) const;

  // Get the value of an element either from the cache or compute the new value.
  std::shared_ptr<value_t> Get(const key_t& key);

  // Manually evict an element from the cache.
  // Returns true if the element was evicted.
  bool Evict(const key_t& key);

  // Pop least recently used element from cache.
  void Pop();

  // Clear all elements from cache.
  void Clear();

 private:
  typedef typename std::pair<key_t, std::shared_ptr<value_t>> key_value_pair_t;
  typedef typename std::list<key_value_pair_t>::iterator list_iterator_t;

  const size_t max_num_bytes_;
  size_t num_bytes_;

  // List to keep track of the least-recently-used elements.
  std::list<key_value_pair_t> elems_list_;

  // Mapping from key to (location in list, num_bytes).
  std::unordered_map<key_t, std::pair<list_iterator_t, size_t>> elems_map_;

  // Function to compute new values if not in the cache.
  const LoadFn load_fn_;
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
    auto it = elems_map_.find(key);
    elems_list_.emplace_front(key, load_fn_(key));
    if (it != elems_map_.end()) {
      elems_list_.erase(it->second);
      elems_map_.erase(it);
    }
    it = elems_map_.emplace_hint(it, key, elems_list_.begin());
    if (elems_map_.size() > max_num_elems_) {
      Pop();
    }
    return it->second->second;
  } else {
    elems_list_.splice(elems_list_.begin(), elems_list_, it->second);
    return it->second->second;
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
ThreadSafeLRUCache<key_t, value_t>::ThreadSafeLRUCache(
    const size_t max_num_elems, LoadFn load_fn)
    : load_fn_(load_fn), cache_(max_num_elems, [](const key_t&) {
        return std::make_shared<Entry>();
      }) {
  THROW_CHECK_NOTNULL(load_fn_);
}

template <typename key_t, typename value_t>
size_t ThreadSafeLRUCache<key_t, value_t>::NumElems() const {
  std::shared_lock lock(cache_mutex_);
  return cache_.NumElems();
}

template <typename key_t, typename value_t>
size_t ThreadSafeLRUCache<key_t, value_t>::MaxNumElems() const {
  std::shared_lock lock(cache_mutex_);
  return cache_.MaxNumElems();
}

template <typename key_t, typename value_t>
bool ThreadSafeLRUCache<key_t, value_t>::Exists(const key_t& key) const {
  std::shared_lock lock(cache_mutex_);
  return cache_.Exists(key);
}

template <typename key_t, typename value_t>
std::shared_ptr<value_t> ThreadSafeLRUCache<key_t, value_t>::Get(
    const key_t& key) {
  bool should_load = false;
  std::shared_ptr<Entry> entry;
  std::shared_future<std::shared_ptr<value_t>> shared_future;

  {
    std::unique_lock lock(cache_mutex_);
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
      std::unique_lock lock(cache_mutex_);
      cache_.Evict(key);
      entry->promise.set_exception(std::current_exception());
    }
  }

  return shared_future.get();
}

template <typename key_t, typename value_t>
bool ThreadSafeLRUCache<key_t, value_t>::Evict(const key_t& key) {
  std::unique_lock lock(cache_mutex_);
  return cache_.Evict(key);
}

template <typename key_t, typename value_t>
void ThreadSafeLRUCache<key_t, value_t>::Pop() {
  std::unique_lock lock(cache_mutex_);
  return cache_.Pop();
}

template <typename key_t, typename value_t>
void ThreadSafeLRUCache<key_t, value_t>::Clear() {
  std::unique_lock lock(cache_mutex_);
  return cache_.Clear();
}

template <typename key_t, typename value_t>
MemoryConstrainedLRUCache<key_t, value_t>::MemoryConstrainedLRUCache(
    const size_t max_num_bytes, LoadFn load_fn)
    : max_num_bytes_(max_num_bytes),
      num_bytes_(0),
      load_fn_(std::move(load_fn)) {
  THROW_CHECK_NOTNULL(load_fn_);
  THROW_CHECK_GT(max_num_bytes, 0);
}

template <typename key_t, typename value_t>
size_t MemoryConstrainedLRUCache<key_t, value_t>::NumElems() const {
  return elems_map_.size();
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
bool MemoryConstrainedLRUCache<key_t, value_t>::Exists(const key_t& key) const {
  return elems_map_.find(key) != elems_map_.end();
}

template <typename key_t, typename value_t>
std::shared_ptr<value_t> MemoryConstrainedLRUCache<key_t, value_t>::Get(
    const key_t& key) {
  const auto it = elems_map_.find(key);
  if (it == elems_map_.end()) {
    std::shared_ptr<value_t> value = load_fn_(key);
    const size_t num_bytes = value->NumBytes();
    auto it = elems_map_.find(key);
    elems_list_.emplace_front(key, std::move(value));
    if (it != elems_map_.end()) {
      elems_list_.erase(it->second.first);
      elems_map_.erase(it);
    }
    it = elems_map_.emplace_hint(
        it, key, std::make_pair(elems_list_.begin(), num_bytes));

    num_bytes_ += num_bytes;
    while (num_bytes_ > max_num_bytes_ && elems_map_.size() > 1) {
      Pop();
    }

    return it->second.first->second;
  } else {
    elems_list_.splice(elems_list_.begin(), elems_list_, it->second.first);
    return it->second.first->second;
  }
}

template <typename key_t, typename value_t>
bool MemoryConstrainedLRUCache<key_t, value_t>::Evict(const key_t& key) {
  const auto it = elems_map_.find(key);
  if (it != elems_map_.end()) {
    num_bytes_ -= it->second.second;
    elems_list_.erase(it->second.first);
    elems_map_.erase(it);
    return true;
  }
  return false;
}

template <typename key_t, typename value_t>
void MemoryConstrainedLRUCache<key_t, value_t>::Pop() {
  if (!elems_list_.empty()) {
    auto last = elems_list_.end();
    --last;
    const auto it = elems_map_.find(last->first);
    num_bytes_ -= it->second.second;
    THROW_CHECK_GE(num_bytes_, 0);
    elems_map_.erase(it);
    elems_list_.pop_back();
  }
}

template <typename key_t, typename value_t>
void MemoryConstrainedLRUCache<key_t, value_t>::UpdateNumBytes(
    const key_t& key) {
  size_t& num_bytes = elems_map_.at(key).second;
  num_bytes_ -= num_bytes;
  THROW_CHECK_GE(num_bytes_, 0);
  num_bytes = Get(key)->NumBytes();
  num_bytes_ += num_bytes;

  while (num_bytes_ > max_num_bytes_ && elems_map_.size() > 1) {
    Pop();
  }
}

template <typename key_t, typename value_t>
void MemoryConstrainedLRUCache<key_t, value_t>::Clear() {
  elems_list_.clear();
  elems_map_.clear();
  num_bytes_ = 0;
}

}  // namespace colmap
