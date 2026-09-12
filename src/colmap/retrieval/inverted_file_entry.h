// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/retrieval/geometry.h"
#include "colmap/util/endian.h"

#include <bitset>

namespace colmap {
namespace retrieval {

// An inverted file entry. The template defines the dimensionality of the binary
// string used to approximate the descriptor in the Hamming space.
// This class is based on an original implementation by Torsten Sattler.
template <int N>
struct InvertedFileEntry {
  void Read(std::istream* in);
  void Write(std::ostream* out) const;

  // The identifier of the image this entry is associated with.
  int image_id = -1;

  // The index of the feature within the image's keypoints list.
  int feature_idx = -1;

  // The geometry of the feature, used for spatial verification.
  FeatureGeometry geometry;

  // The binary signature in the Hamming embedding.
  std::bitset<N> descriptor;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <int N>
void InvertedFileEntry<N>::Read(std::istream* in) {
  static_assert(N <= 64, "Dimensionality too large");
  static_assert(sizeof(unsigned long long) >= 8,
                "Expected unsigned long to be at least 8 byte");
  static_assert(sizeof(FeatureGeometry) == 16, "Geometry type size mismatch");

  image_id = ReadBinaryLittleEndian<int>(in);
  feature_idx = ReadBinaryLittleEndian<int>(in);

  in->read(reinterpret_cast<char*>(&geometry), sizeof(FeatureGeometry));

  const uint64_t descriptor_data = ReadBinaryLittleEndian<uint64_t>(in);
  descriptor = std::bitset<N>(descriptor_data);
}

template <int N>
void InvertedFileEntry<N>::Write(std::ostream* out) const {
  static_assert(N <= 64, "Dimensionality too large");
  static_assert(sizeof(unsigned long long) >= 8,
                "Expected unsigned long to be at least 8 byte");
  static_assert(sizeof(FeatureGeometry) == 16, "Geometry type size mismatch");

  WriteBinaryLittleEndian<int>(out, image_id);
  WriteBinaryLittleEndian<int>(out, feature_idx);

  out->write(reinterpret_cast<const char*>(&geometry), sizeof(FeatureGeometry));

  const uint64_t descriptor_data =
      static_cast<uint64_t>(descriptor.to_ullong());
  WriteBinaryLittleEndian<uint64_t>(out, descriptor_data);
}

}  // namespace retrieval
}  // namespace colmap
