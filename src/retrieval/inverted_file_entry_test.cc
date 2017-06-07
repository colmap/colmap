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

#define TEST_NAME "retrieval/inverted_file_entry"
#include "util/testing.h"

#include "retrieval/inverted_file_entry.h"

using namespace colmap::retrieval;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  InvertedFileEntry<10> entry;
  BOOST_CHECK_EQUAL(entry.image_id, -1);
  BOOST_CHECK_EQUAL(entry.geometry.x, 0);
  BOOST_CHECK_EQUAL(entry.geometry.y, 0);
  BOOST_CHECK_EQUAL(entry.geometry.scale, 0);
  BOOST_CHECK_EQUAL(entry.geometry.orientation, 0);
  BOOST_CHECK_EQUAL(entry.descriptor.size(), 10);
  BOOST_CHECK_EQUAL(entry.descriptor.size(), 10);
}

BOOST_AUTO_TEST_CASE(TestReadWrite) {
  InvertedFileEntry<10> entry;
  entry.image_id = 99;
  entry.geometry.x = 0.123;
  entry.geometry.x = 0.456;
  entry.geometry.scale = 0.789;
  entry.geometry.orientation = -0.1;
  for (size_t i = 0; i < entry.descriptor.size(); ++i) {
    entry.descriptor[i] = (i % 2) == 0;
  }
  std::stringstream file;
  entry.Write(&file);

  InvertedFileEntry<10> read_entry;
  read_entry.Read(&file);
  BOOST_CHECK_EQUAL(entry.image_id, read_entry.image_id);
  BOOST_CHECK_EQUAL(entry.geometry.x, read_entry.geometry.x);
  BOOST_CHECK_EQUAL(entry.geometry.y, read_entry.geometry.y);
  BOOST_CHECK_EQUAL(entry.geometry.scale, read_entry.geometry.scale);
  BOOST_CHECK_EQUAL(entry.geometry.orientation,
                    read_entry.geometry.orientation);
  for (size_t i = 0; i < entry.descriptor.size(); ++i) {
    BOOST_CHECK_EQUAL(entry.descriptor[i], read_entry.descriptor[i]);
  }
}
