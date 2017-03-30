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

#define TEST_NAME "base/reconstruction_manager"
#include "util/testing.h"

#include "base/reconstruction_manager.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  ReconstructionManager reconstruction_manager;
  BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 0);
}

BOOST_AUTO_TEST_CASE(TestAddGet) {
  ReconstructionManager reconstruction_manager;
  BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 0);
  for (size_t i = 0; i < 10; ++i) {
    const size_t idx = reconstruction_manager.Add();
    BOOST_CHECK_EQUAL(reconstruction_manager.Size(), i + 1);
    BOOST_CHECK_EQUAL(idx, i);
    BOOST_CHECK_EQUAL(reconstruction_manager.Get(idx).NumCameras(), 0);
    BOOST_CHECK_EQUAL(reconstruction_manager.Get(idx).NumImages(), 0);
    BOOST_CHECK_EQUAL(reconstruction_manager.Get(idx).NumPoints3D(), 0);
  }
}

BOOST_AUTO_TEST_CASE(TestDelete) {
  ReconstructionManager reconstruction_manager;
  BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 0);
  for (size_t i = 0; i < 10; ++i) {
    reconstruction_manager.Add();
  }

  BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 10);
  for (size_t i = 0; i < 10; ++i) {
    reconstruction_manager.Delete(0);
    BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 9 - i);
  }
}

BOOST_AUTO_TEST_CASE(TestClear) {
  ReconstructionManager reconstruction_manager;
  BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 0);
  for (size_t i = 0; i < 10; ++i) {
    reconstruction_manager.Add();
  }

  BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 10);
  reconstruction_manager.Clear();
  BOOST_CHECK_EQUAL(reconstruction_manager.Size(), 0);
}
