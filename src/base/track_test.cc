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

#define TEST_NAME "base/track"
#include "util/testing.h"

#include "base/track.h"

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
