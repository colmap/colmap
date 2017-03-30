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

#define TEST_NAME "estimators/coordinate_frame"
#include "util/testing.h"

#include "estimators/coordinate_frame.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  Reconstruction reconstruction;
  std::string image_path;
  BOOST_CHECK_EQUAL(EstimateCoordinateFrame(CoordinateFrameEstimationOptions(),
                                            reconstruction, image_path),
                    Eigen::Matrix3d::Zero());
}
