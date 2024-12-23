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

#include "colmap/scene/track.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(TrackElement, Empty) {
  TrackElement track_el;
  EXPECT_EQ(track_el.image_id, kInvalidImageId);
  EXPECT_EQ(track_el.point2D_idx, kInvalidPoint2DIdx);
}

TEST(TrackElement, Equals) {
  TrackElement track_el;
  TrackElement other = track_el;
  EXPECT_EQ(track_el, other);
  track_el.image_id = 1;
  EXPECT_NE(track_el, other);
  other.image_id = 1;
  EXPECT_EQ(track_el, other);
}

TEST(TrackElement, Print) {
  TrackElement track_el(1, 2);
  std::ostringstream stream;
  stream << track_el;
  EXPECT_EQ(stream.str(), "TrackElement(image_id=1, point2D_idx=2)");
}

TEST(Track, Default) {
  Track track;
  EXPECT_EQ(track.Length(), 0);
  EXPECT_EQ(track.Elements().size(), track.Length());
}

TEST(Track, Equals) {
  Track track;
  Track other = track;
  EXPECT_EQ(track, other);
  track.AddElement(0, 1);
  EXPECT_NE(track, other);
  other.AddElement(0, 1);
  EXPECT_EQ(track, other);
}

TEST(Track, Print) {
  Track track;
  track.AddElement(1, 2);
  track.AddElement(2, 3);
  std::ostringstream stream;
  stream << track;
  EXPECT_EQ(stream.str(),
            "Track(elements=[TrackElement(image_id=1, point2D_idx=2), "
            "TrackElement(image_id=2, point2D_idx=3)])");
}

TEST(Track, SetElements) {
  Track track;
  std::vector<TrackElement> elements;
  elements.emplace_back(0, 1);
  elements.emplace_back(0, 2);
  track.SetElements(elements);
  EXPECT_EQ(track.Length(), 2);
  EXPECT_EQ(track.Elements().size(), track.Length());
  EXPECT_EQ(track.Element(0).image_id, 0);
  EXPECT_EQ(track.Element(0).point2D_idx, 1);
  EXPECT_EQ(track.Element(1).image_id, 0);
  EXPECT_EQ(track.Element(1).point2D_idx, 2);
  for (size_t i = 0; i < track.Length(); ++i) {
    EXPECT_EQ(track.Element(i).image_id, track.Elements()[i].image_id);
    EXPECT_EQ(track.Element(i).point2D_idx, track.Elements()[i].point2D_idx);
  }
}

TEST(Track, AddElement) {
  Track track;
  track.AddElement(0, 1);
  track.AddElement(TrackElement(0, 2));
  std::vector<TrackElement> elements;
  elements.emplace_back(0, 1);
  elements.emplace_back(0, 2);
  track.AddElements(elements);
  EXPECT_EQ(track.Length(), 4);
  EXPECT_EQ(track.Elements().size(), track.Length());
  EXPECT_EQ(track.Element(0).image_id, 0);
  EXPECT_EQ(track.Element(0).point2D_idx, 1);
  EXPECT_EQ(track.Element(1).image_id, 0);
  EXPECT_EQ(track.Element(1).point2D_idx, 2);
  EXPECT_EQ(track.Element(2).image_id, 0);
  EXPECT_EQ(track.Element(2).point2D_idx, 1);
  EXPECT_EQ(track.Element(3).image_id, 0);
  EXPECT_EQ(track.Element(3).point2D_idx, 2);
  for (size_t i = 0; i < track.Length(); ++i) {
    EXPECT_EQ(track.Element(i).image_id, track.Elements()[i].image_id);
    EXPECT_EQ(track.Element(i).point2D_idx, track.Elements()[i].point2D_idx);
  }
}

TEST(Track, DeleteElement) {
  Track track;
  track.AddElement(0, 1);
  track.AddElement(0, 2);
  track.AddElement(0, 3);
  track.AddElement(0, 3);
  EXPECT_EQ(track.Length(), 4);
  EXPECT_EQ(track.Elements().size(), track.Length());
  track.DeleteElement(0);
  EXPECT_EQ(track.Length(), 3);
  EXPECT_EQ(track.Elements().size(), track.Length());
  EXPECT_EQ(track.Element(0).image_id, 0);
  EXPECT_EQ(track.Element(0).point2D_idx, 2);
  EXPECT_EQ(track.Element(1).image_id, 0);
  EXPECT_EQ(track.Element(1).point2D_idx, 3);
  EXPECT_EQ(track.Element(2).image_id, 0);
  EXPECT_EQ(track.Element(2).point2D_idx, 3);
  track.DeleteElement(0, 3);
  EXPECT_EQ(track.Length(), 1);
  EXPECT_EQ(track.Elements().size(), track.Length());
  EXPECT_EQ(track.Element(0).image_id, 0);
  EXPECT_EQ(track.Element(0).point2D_idx, 2);
}

TEST(Track, Reserve) {
  Track track;
  track.Reserve(2);
  EXPECT_EQ(track.Elements().capacity(), 2);
}

TEST(Track, Compress) {
  Track track;
  track.AddElement(0, 1);
  track.AddElement(0, 2);
  track.AddElement(0, 3);
  track.AddElement(0, 3);
  EXPECT_EQ(track.Elements().capacity(), 4);
  track.DeleteElement(0);
  track.DeleteElement(0);
  EXPECT_EQ(track.Elements().capacity(), 4);
  track.Compress();
  EXPECT_EQ(track.Elements().capacity(), 2);
}

}  // namespace
}  // namespace colmap
