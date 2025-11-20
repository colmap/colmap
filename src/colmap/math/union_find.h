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

#include <unordered_map>

namespace colmap {

// Helper class to perform union-find operations.
template <typename T>
class UnionFind {
 public:
  explicit UnionFind(size_t expected_size = 0) {
    parent_.reserve(expected_size);
  }

  // Find the root of the element x.
  T Find(const T& x) {
    // If x is not in parent map, initialize it with x as its parent.
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

  // Unite the sets containing x and y.
  void Union(const T& x, const T& y) {
    const T root_x = Find(x);
    const T root_y = Find(y);
    if (root_x != root_y) {
      parent_[root_x] = root_y;
    }
  }

 private:
  // Map to store the parent of each element.
  std::unordered_map<T, T> parent_;
};

}  // namespace
