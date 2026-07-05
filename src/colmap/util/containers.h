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

// Backend-selectable hash container aliases for COLMAP.
//
// This header centralizes the hash map/set implementation used across the
// performance-critical scene and SfM data structures so that the underlying
// container can be swapped at compile time for benchmarking, without touching
// call sites. The backend is selected by exactly one of the following macros,
// set by CMake via the COLMAP_HASH_MAP_BACKEND cache variable:
//
//   COLMAP_HASH_STD    -> std::unordered_map / std::unordered_set (default)
//   COLMAP_HASH_BOOST  -> boost::unordered_flat_* / boost::unordered_node_*
//   COLMAP_HASH_ABSL   -> absl::flat_hash_* / absl::node_hash_*
//   COLMAP_HASH_ANKERL -> ankerl::unordered_dense::map / ::set (flat only)
//
// Two families of aliases are provided:
//
//   FlatHashMap / FlatHashSet
//       Open-addressing (flat) containers. Fastest and lowest-memory, but they
//       INVALIDATE references, pointers, and iterators to elements on any
//       insertion that triggers a rehash (and, for erase, on some backends).
//       Use only where no long-lived reference to a stored element is held
//       across a mutation of the same container.
//
//   NodeHashMap / NodeHashSet
//       Node-based containers whose element references/pointers remain valid
//       across insert/erase of *other* elements (std::unordered_* semantics).
//       Use as the drop-in replacement for element stores that may hand out
//       long-lived references.
//
// The default hash is std::hash<K> for every backend, so the existing custom
// std::hash specializations and colmap::PairHash (see util/types.h) are reused
// unchanged and the hashing semantics are held constant across backends -- the
// experiment varies only the container implementation. Note: boost::unordered
// and ankerl::unordered_dense internally re-mix a non-avalanching hash such as
// the identity std::hash<uint32_t>, whereas absl does not; absl numbers for
// integer-keyed maps should therefore be read as a conservative lower bound.

#include <functional>

#if defined(COLMAP_HASH_ABSL)
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/node_hash_map.h>
#include <absl/container/node_hash_set.h>
#elif defined(COLMAP_HASH_BOOST)
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <boost/unordered/unordered_node_map.hpp>
#include <boost/unordered/unordered_node_set.hpp>
#elif defined(COLMAP_HASH_ANKERL)
#include <ankerl/unordered_dense.h>
#include <unordered_map>
#include <unordered_set>
#else
// Default to the standard library when no backend macro is defined.
#ifndef COLMAP_HASH_STD
#define COLMAP_HASH_STD
#endif
#include <unordered_map>
#include <unordered_set>
#endif

namespace colmap {

#if defined(COLMAP_HASH_ABSL)

template <class Key,
          class Value,
          class Hash = std::hash<Key>,
          class Eq = std::equal_to<Key>>
using FlatHashMap = absl::flat_hash_map<Key, Value, Hash, Eq>;

template <class Key, class Hash = std::hash<Key>, class Eq = std::equal_to<Key>>
using FlatHashSet = absl::flat_hash_set<Key, Hash, Eq>;

template <class Key,
          class Value,
          class Hash = std::hash<Key>,
          class Eq = std::equal_to<Key>>
using NodeHashMap = absl::node_hash_map<Key, Value, Hash, Eq>;

template <class Key, class Hash = std::hash<Key>, class Eq = std::equal_to<Key>>
using NodeHashSet = absl::node_hash_set<Key, Hash, Eq>;

#elif defined(COLMAP_HASH_BOOST)

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

#elif defined(COLMAP_HASH_ANKERL)

template <class Key,
          class Value,
          class Hash = std::hash<Key>,
          class Eq = std::equal_to<Key>>
using FlatHashMap = ankerl::unordered_dense::map<Key, Value, Hash, Eq>;

template <class Key, class Hash = std::hash<Key>, class Eq = std::equal_to<Key>>
using FlatHashSet = ankerl::unordered_dense::set<Key, Hash, Eq>;

// ankerl::unordered_dense stores values in a contiguous vector and therefore
// has no reference-stable node variant. Fall back to std::unordered_* for the
// Node* aliases so the safe reference-stability policy is preserved.
template <class Key,
          class Value,
          class Hash = std::hash<Key>,
          class Eq = std::equal_to<Key>>
using NodeHashMap = std::unordered_map<Key, Value, Hash, Eq>;

template <class Key, class Hash = std::hash<Key>, class Eq = std::equal_to<Key>>
using NodeHashSet = std::unordered_set<Key, Hash, Eq>;

#else  // COLMAP_HASH_STD

template <class Key,
          class Value,
          class Hash = std::hash<Key>,
          class Eq = std::equal_to<Key>>
using FlatHashMap = std::unordered_map<Key, Value, Hash, Eq>;

template <class Key, class Hash = std::hash<Key>, class Eq = std::equal_to<Key>>
using FlatHashSet = std::unordered_set<Key, Hash, Eq>;

template <class Key,
          class Value,
          class Hash = std::hash<Key>,
          class Eq = std::equal_to<Key>>
using NodeHashMap = std::unordered_map<Key, Value, Hash, Eq>;

template <class Key, class Hash = std::hash<Key>, class Eq = std::equal_to<Key>>
using NodeHashSet = std::unordered_set<Key, Hash, Eq>;

#endif

}  // namespace colmap
