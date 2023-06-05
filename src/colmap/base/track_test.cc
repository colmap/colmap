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

#define TEST_NAME "base/track"
#include "colmap/base/track.h"

#include "colmap/util/testing.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestTrackElement) {
  TrackElement track_el;
  BOOST_CHECK_EQUAL(track_el.image_id, kInvalidImageId);
  BOOST_CHECK_EQUAL(track_el.point2D_idx, kInvalidPoint2DIdx);
}

BOOST_AUTO_TEST_CASE(TestDefault) {
  Track track;
  BOOST_CHECK_EQUAL(track.Length(), 0);
  BOOST_CHECK_EQUAL(track.Elements().size(), track.Length());
}

BOOST_AUTO_TEST_CASE(TestSetElements) {
  Track track;
  std::vector<TrackElement> elements;
  elements.emplace_back(0, 1);
  elements.emplace_back(0, 2);
  track.SetElements(elements);
  BOOST_CHECK_EQUAL(track.Length(), 2);
  BOOST_CHECK_EQUAL(track.Elements().size(), track.Length());
  BOOST_CHECK_EQUAL(track.Element(0).image_id, 0);
  BOOST_CHECK_EQUAL(track.Element(0).point2D_idx, 1);
  BOOST_CHECK_EQUAL(track.Element(1).image_id, 0);
  BOOST_CHECK_EQUAL(track.Element(1).point2D_idx, 2);
  for (size_t i = 0; i < track.Length(); ++i) {
    BOOST_CHECK_EQUAL(track.Element(i).image_id, track.Elements()[i].image_id);
    BOOST_CHECK_EQUAL(track.Element(i).point2D_idx,
                      track.Elements()[i].point2D_idx);
  }
}

BOOST_AUTO_TEST_CASE(TestAddElement) {
  Track track;
  track.AddElement(0, 1);
  track.AddElement(TrackElement(0, 2));
  std::vector<TrackElement> elements;
  elements.emplace_back(0, 1);
  elements.emplace_back(0, 2);
  track.AddElements(elements);
  BOOST_CHECK_EQUAL(track.Length(), 4);
  BOOST_CHECK_EQUAL(track.Elements().size(), track.Length());
  BOOST_CHECK_EQUAL(track.Element(0).image_id, 0);
  BOOST_CHECK_EQUAL(track.Element(0).point2D_idx, 1);
  BOOST_CHECK_EQUAL(track.Element(1).image_id, 0);
  BOOST_CHECK_EQUAL(track.Element(1).point2D_idx, 2);
  BOOST_CHECK_EQUAL(track.Element(2).image_id, 0);
  BOOST_CHECK_EQUAL(track.Element(2).point2D_idx, 1);
  BOOST_CHECK_EQUAL(track.Element(3).image_id, 0);
  BOOST_CHECK_EQUAL(track.Element(3).point2D_idx, 2);
  for (size_t i = 0; i < track.Length(); ++i) {
    BOOST_CHECK_EQUAL(track.Element(i).image_id, track.Elements()[i].image_id);
    BOOST_CHECK_EQUAL(track.Element(i).point2D_idx,
                      track.Elements()[i].point2D_idx);
  }
}

BOOST_AUTO_TEST_CASE(TestDeleteElement) {
  Track track;
  track.AddElement(0, 1);
  track.AddElement(0, 2);
  track.AddElement(0, 3);
  track.AddElement(0, 3);
  BOOST_CHECK_EQUAL(track.Length(), 4);
  BOOST_CHECK_EQUAL(track.Elements().size(), track.Length());
  track.DeleteElement(0);
  BOOST_CHECK_EQUAL(track.Length(), 3);
  BOOST_CHECK_EQUAL(track.Elements().size(), track.Length());
  BOOST_CHECK_EQUAL(track.Element(0).image_id, 0);
  BOOST_CHECK_EQUAL(track.Element(0).point2D_idx, 2);
  BOOST_CHECK_EQUAL(track.Element(1).image_id, 0);
  BOOST_CHECK_EQUAL(track.Element(1).point2D_idx, 3);
  BOOST_CHECK_EQUAL(track.Element(2).image_id, 0);
  BOOST_CHECK_EQUAL(track.Element(2).point2D_idx, 3);
  track.DeleteElement(0, 3);
  BOOST_CHECK_EQUAL(track.Length(), 1);
  BOOST_CHECK_EQUAL(track.Elements().size(), track.Length());
  BOOST_CHECK_EQUAL(track.Element(0).image_id, 0);
  BOOST_CHECK_EQUAL(track.Element(0).point2D_idx, 2);
}

BOOST_AUTO_TEST_CASE(TestReserve) {
  Track track;
  track.Reserve(2);
  BOOST_CHECK_EQUAL(track.Elements().capacity(), 2);
}

BOOST_AUTO_TEST_CASE(TestCompress) {
  Track track;
  track.AddElement(0, 1);
  track.AddElement(0, 2);
  track.AddElement(0, 3);
  track.AddElement(0, 3);
  BOOST_CHECK_EQUAL(track.Elements().capacity(), 4);
  track.DeleteElement(0);
  track.DeleteElement(0);
  BOOST_CHECK_EQUAL(track.Elements().capacity(), 4);
  track.Compress();
  BOOST_CHECK_EQUAL(track.Elements().capacity(), 2);
}
