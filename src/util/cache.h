// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_UTIL_CACHE_H_
#define COLMAP_SRC_UTIL_CACHE_H_

#include <iostream>
#include <list>
#include <unordered_map>

#include "util/logging.h"

namespace colmap {

// Least Recently Used cache implementation. Whenever the cache size is
// exceeded, the least recently used (by Get and GetMutable) is deleted.
template <typename key_t, typename value_t>
class LRUCache {
 public:
  LRUCache(const size_t max_num_elems,
           const std::function<value_t(const key_t&)>& getter_func);
  virtual ~LRUCache() = default;

  // The number of elements in the cache.
  size_t NumElems() const;
  size_t MaxNumElems() const;

  // Check whether the element with the given key exists.
  bool Exists(const key_t& key) const;

  // Get the value of an element either from the cache or compute the new value.
  const value_t& Get(const key_t& key);
  value_t& GetMutable(const key_t& key);

  // Manually set the value of an element. Note that the ownership of the value
  // is moved to the cache, which invalidates the object on the caller side.
  virtual void Set(const key_t& key, value_t&& value);

  // Pop least recently used element from cache.
  virtual void Pop();

  // Clear all elements from cache.
  virtual void Clear();

 protected:
  typedef typename std::pair<key_t, value_t> key_value_pair_t;
  typedef typename std::list<key_value_pair_t>::iterator list_iterator_t;

  // Maximum number of least-recently-used elements the cache remembers.
  const size_t max_num_elems_;

  // List to keep track of the least-recently-used elements.
  std::list<key_value_pair_t> elems_list_;

  // Mapping from key to location in the list.
  std::unordered_map<key_t, list_iterator_t> elems_map_;

  // Function to compute new values if not in the cache.
  const std::function<value_t(const key_t&)> getter_func_;
};

// Least Recently Used cache implementation that is constrained by a maximum
// memory limitation of its elements. Whenever the memory limit is exceeded, the
// least recently used (by Get and GetMutable) is deleted. Each element must
// implement a `size_t NumBytes()` method that returns its size in memory.
template <typename key_t, typename value_t>
class MemoryConstrainedLRUCache : public LRUCache<key_t, value_t> {
 public:
  MemoryConstrainedLRUCache(
      const size_t max_num_bytes,
      const std::function<value_t(const key_t&)>& getter_func);

  size_t NumBytes() const;
  size_t MaxNumBytes() const;
  void UpdateNumBytes(const key_t& key);

  void Set(const key_t& key, value_t&& value) override;
  void Pop() override;
  void Clear() override;

 private:
  using typename LRUCache<key_t, value_t>::key_value_pair_t;
  using typename LRUCache<key_t, value_t>::list_iterator_t;
  using LRUCache<key_t, value_t>::max_num_elems_;
  using LRUCache<key_t, value_t>::elems_list_;
  using LRUCache<key_t, value_t>::elems_map_;
  using LRUCache<key_t, value_t>::getter_func_;

  const size_t max_num_bytes_;
  size_t num_bytes_;
  std::unordered_map<key_t, size_t> elems_num_bytes_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename key_t, typename value_t>
LRUCache<key_t, value_t>::LRUCache(
    const size_t max_num_elems,
    const std::function<value_t(const key_t&)>& getter_func)
    : max_num_elems_(max_num_elems), getter_func_(getter_func) {
  CHECK(getter_func);
  CHECK_GT(max_num_elems, 0);
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
const value_t& LRUCache<key_t, value_t>::Get(const key_t& key) {
  return GetMutable(key);
}

template <typename key_t, typename value_t>
value_t& LRUCache<key_t, value_t>::GetMutable(const key_t& key) {
  const auto it = elems_map_.find(key);
  if (it == elems_map_.end()) {
    Set(key, std::move(getter_func_(key)));
    return elems_map_[key]->second;
  } else {
    elems_list_.splice(elems_list_.begin(), elems_list_, it->second);
    return it->second->second;
  }
}

template <typename key_t, typename value_t>
void LRUCache<key_t, value_t>::Set(const key_t& key, value_t&& value) {
  auto it = elems_map_.find(key);
  elems_list_.push_front(key_value_pair_t(key, std::move(value)));
  if (it != elems_map_.end()) {
    elems_list_.erase(it->second);
    elems_map_.erase(it);
  }
  elems_map_[key] = elems_list_.begin();
  if (elems_map_.size() > max_num_elems_) {
    Pop();
  }
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
    const size_t max_num_bytes,
    const std::function<value_t(const key_t&)>& getter_func)
    : LRUCache<key_t, value_t>(std::numeric_limits<size_t>::max(), getter_func),
      max_num_bytes_(max_num_bytes),
      num_bytes_(0) {
  CHECK_GT(max_num_bytes, 0);
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
void MemoryConstrainedLRUCache<key_t, value_t>::Set(const key_t& key,
                                                    value_t&& value) {
  auto it = elems_map_.find(key);
  elems_list_.push_front(key_value_pair_t(key, std::move(value)));
  if (it != elems_map_.end()) {
    elems_list_.erase(it->second);
    elems_map_.erase(it);
  }
  elems_map_[key] = elems_list_.begin();

  const size_t num_bytes = value.NumBytes();
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
    CHECK_GE(num_bytes_, 0);
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
  CHECK_GE(num_bytes_, 0);
  num_bytes = LRUCache<key_t, value_t>::Get(key).NumBytes();
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

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_CACHE_H_
