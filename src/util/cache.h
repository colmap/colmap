// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_UTIL_CACHE_H_
#define COLMAP_SRC_UTIL_CACHE_H_

#include <list>
#include <unordered_map>

#include "util/logging.h"

namespace colmap {

template <typename key_t, typename value_t>
class LRUCache {
 public:
  LRUCache(const size_t max_num_elems,
           const std::function<value_t(const key_t&)>& getter_func);

  // The number of elements in the cache.
  size_t NumElems() const;

  // Check whether the element with the given key exists.
  bool Exists(const key_t& key) const;

  // Manually set the value of an element.
  void Set(const key_t& key, const value_t& value);

  // Get the value of an element either from the cache or compute the new value.
  const value_t& Get(const key_t& key);

 private:
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

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename key_t, typename value_t>
LRUCache<key_t, value_t>::LRUCache(
    const size_t max_num_elems,
    const std::function<value_t(const key_t&)>& getter_func)
    : max_num_elems_(max_num_elems), getter_func_(getter_func) {
  CHECK(getter_func);
}

template <typename key_t, typename value_t>
size_t LRUCache<key_t, value_t>::NumElems() const {
  return elems_map_.size();
}

template <typename key_t, typename value_t>
bool LRUCache<key_t, value_t>::Exists(const key_t& key) const {
  return elems_map_.find(key) != elems_map_.end();
}

template <typename key_t, typename value_t>
void LRUCache<key_t, value_t>::Set(const key_t& key, const value_t& value) {
  auto it = elems_map_.find(key);
  elems_list_.push_front(key_value_pair_t(key, value));
  if (it != elems_map_.end()) {
    elems_list_.erase(it->second);
    elems_map_.erase(it);
  }
  elems_map_[key] = elems_list_.begin();

  if (elems_map_.size() > max_num_elems_) {
    auto last = elems_list_.end();
    --last;
    elems_map_.erase(last->first);
    elems_list_.pop_back();
  }
}

template <typename key_t, typename value_t>
const value_t& LRUCache<key_t, value_t>::Get(const key_t& key) {
  const auto it = elems_map_.find(key);
  if (it == elems_map_.end()) {
    Set(key, getter_func_(key));
    return elems_map_[key]->second;
  } else {
    elems_list_.splice(elems_list_.begin(), elems_list_, it->second);
    return it->second->second;
  }
}

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_CACHE_H_
