// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/hash_containers.h"

#include <cstddef>
#include <functional>
#include <optional>

namespace colmap {

// Helper class to perform union-find operations. The optional Hash parameter
// allows keying on types without a std::hash specialization (e.g., std::pair,
// via colmap::PairHash).
template <typename T, typename Hash = std::hash<T>>
class UnionFind {
 public:
  void Reserve(size_t capacity) { parent_.reserve(capacity); }

  // Find the root of the element x. If x is not in the structure, it is
  // inserted as its own parent.
  T Find(const T& x) {
    auto parent_it = parent_.find(x);
    if (parent_it == parent_.end()) {
      parent_.emplace_hint(parent_it, x, x);
      return x;
    }
    // Path compression.
    if (parent_it->second != x) {
      parent_it->second = Find(parent_it->second);
    }
    return parent_it->second;
  }

  // Find the root of x only if x already exists in the structure, otherwise
  // returns std::nullopt. Does not insert new elements.
  std::optional<T> FindIfExists(const T& x) const {
    auto it = parent_.find(x);
    if (it == parent_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  // Unite the sets containing x and y.
  void Union(const T& x, const T& y) {
    const T root_x = Find(x);
    const T root_y = Find(y);
    if (root_x != root_y) {
      parent_[root_x] = root_y;
    }
  }

  // Path-compress all elements so each points directly to its root.
  // Call this once after all Union operations and before iterating Parents().
  void Compress() {
    for (auto& [elem, parent] : parent_) {
      parent = Find(elem);
    }
  }

  // Access all elements and their parents in the union-find structure.
  const NodeHashMap<T, T, Hash>& Parents() const { return parent_; }

 private:
  // Map to store the parent of each element.
  NodeHashMap<T, T, Hash> parent_;
};

}  // namespace colmap
