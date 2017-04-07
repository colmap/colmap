// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#ifndef COLMAP_SRC_RETRIEVAL_INVERTED_FILE_ENTRY_H_
#define COLMAP_SRC_RETRIEVAL_INVERTED_FILE_ENTRY_H_

#include <bitset>
#include <fstream>

namespace colmap {
namespace retrieval {

// Models an inverted file entry. The template defines the dimensionality of
// the binary string used to approximate the descriptor.
// This class is based on an original implementation by Torsten Sattler.
template <int N>
struct InvertedFileEntry {
  void Read(std::istream* ifs);
  void Write(std::ostream* ofs) const;

  // The identifier of the image this entry is associated with.
  int image_id = -1;

  // The binary signature in the Hamming embedding.
  std::bitset<N> descriptor;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <int N>
void InvertedFileEntry<N>::Read(std::istream* ifs) {
  static_assert(N <= 64, "Dimensionality too large");
  static_assert(sizeof(unsigned long long) >= 8,
                "Expected unsigned long to be at least 8 byte");

  int32_t image_id_data = 0;
  ifs->read(reinterpret_cast<char*>(&image_id_data), sizeof(int32_t));
  image_id = static_cast<int>(image_id_data);

  uint64_t descriptor_data = 0;
  ifs->read(reinterpret_cast<char*>(&descriptor_data), sizeof(uint64_t));
  descriptor = std::bitset<N>(descriptor_data);
}

template <int N>
void InvertedFileEntry<N>::Write(std::ostream* ofs) const {
  static_assert(N <= 64, "Dimensionality too large");
  static_assert(sizeof(unsigned long long) >= 8,
                "Expected unsigned long to be at least 8 byte");

  ofs->write(reinterpret_cast<const char*>(&image_id), sizeof(int32_t));

  const uint64_t descriptor_data =
      static_cast<uint64_t>(descriptor.to_ullong());
  ofs->write(reinterpret_cast<const char*>(&descriptor_data), sizeof(uint64_t));
}

}  // namespace retrieval
}  // namespace colmap

#endif  // COLMAP_SRC_RETRIEVAL_INVERTED_FILE_ENTRY_H_
