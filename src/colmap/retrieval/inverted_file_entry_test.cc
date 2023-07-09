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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/retrieval/inverted_file_entry.h"

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {

TEST(InvertedFileEntry, Empty) {
  InvertedFileEntry<10> entry;
  EXPECT_EQ(entry.image_id, -1);
  EXPECT_EQ(entry.feature_idx, -1);
  EXPECT_EQ(entry.geometry.x, 0);
  EXPECT_EQ(entry.geometry.y, 0);
  EXPECT_EQ(entry.geometry.scale, 0);
  EXPECT_EQ(entry.geometry.orientation, 0);
  EXPECT_EQ(entry.descriptor.size(), 10);
}

TEST(InvertedFileEntry, ReadWrite) {
  InvertedFileEntry<10> entry;
  entry.image_id = 99;
  entry.feature_idx = 100;
  entry.geometry.x = 0.123;
  entry.geometry.y = 0.456;
  entry.geometry.scale = 0.789;
  entry.geometry.orientation = -0.1;
  for (size_t i = 0; i < entry.descriptor.size(); ++i) {
    entry.descriptor[i] = (i % 2) == 0;
  }
  std::stringstream file;
  entry.Write(&file);

  InvertedFileEntry<10> read_entry;
  read_entry.Read(&file);
  EXPECT_EQ(entry.image_id, read_entry.image_id);
  EXPECT_EQ(entry.feature_idx, read_entry.feature_idx);
  EXPECT_EQ(entry.geometry.x, read_entry.geometry.x);
  EXPECT_EQ(entry.geometry.y, read_entry.geometry.y);
  EXPECT_EQ(entry.geometry.scale, read_entry.geometry.scale);
  EXPECT_EQ(entry.geometry.orientation, read_entry.geometry.orientation);
  for (size_t i = 0; i < entry.descriptor.size(); ++i) {
    EXPECT_EQ(entry.descriptor[i], read_entry.descriptor[i]);
  }
}

}  // namespace retrieval
}  // namespace colmap
