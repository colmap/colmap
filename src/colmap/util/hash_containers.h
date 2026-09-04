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

// Hash container aliases for COLMAP.
//
// This header centralizes the hash map/set implementation used across the
// performance-critical scene and SfM data structures, so that the underlying
// container can be swapped without touching call sites.
//
// Two families of aliases are provided:
//
//   FlatHashMap / FlatHashSet
//       Open-addressing (flat) containers. Fastest and lowest-memory, but they
//       INVALIDATE references, pointers, and iterators to elements on any
//       insertion that triggers a rehash, and on erase. Use only where no
//       long-lived reference to a stored element is held across a mutation of
//       the same container, and avoid iterator-based erase loops (erase by key
//       instead).
//
//   NodeHashMap / NodeHashSet
//       Node-based containers whose element references/pointers remain valid
//       across insert/erase of *other* elements (std::unordered_* semantics).
//       Use as the drop-in replacement for element stores that may hand out
//       long-lived references.
//
// The default hash is std::hash<K>, so the custom std::hash specializations and
// colmap::PairHash (see util/types.h) are reused unchanged. boost::unordered
// internally re-mixes a non-avalanching hash such as the identity
// std::hash<uint32_t>.
//
// These aliases are data members of classes in public headers, so their layout
// is part of COLMAP's ABI, and nothing about it reaches the linker. There is
// deliberately no build option to swap them: two COLMAP builds that disagreed
// would corrupt memory when loaded into one process instead of failing to link.
// boost::unordered_node_map requires Boost >= 1.84; see FETCH_BOOST_UNORDERED
// in cmake/FindDependencies.cmake for how that is guaranteed.

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <boost/unordered/unordered_node_map.hpp>
#include <boost/unordered/unordered_node_set.hpp>
#include <functional>

namespace colmap {

template <class Key,
          class Value,
          class Hash = std::hash<Key>,
          class Eq = std::equal_to<Key>>
using FlatHashMap = boost::unordered_flat_map<Key, Value, Hash, Eq>;

template <class Key, class Hash = std::hash<Key>, class Eq = std::equal_to<Key>>
using FlatHashSet = boost::unordered_flat_set<Key, Hash, Eq>;

template <class Key,
          class Value,
          class Hash = std::hash<Key>,
          class Eq = std::equal_to<Key>>
using NodeHashMap = boost::unordered_node_map<Key, Value, Hash, Eq>;

template <class Key, class Hash = std::hash<Key>, class Eq = std::equal_to<Key>>
using NodeHashSet = boost::unordered_node_set<Key, Hash, Eq>;

}  // namespace colmap
