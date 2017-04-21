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

#include "estimators/coordinate_frame.h"
#include "util/misc.h"
#include "util/option_manager.h"

using namespace colmap;

int main(int argc, char* argv[]) {
  InitializeGlog(argv);

  std::string input_path;
  std::string output_path;

  CoordinateFrameEstimationOptions frame_estimation_options;

  OptionManager options;
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("max_image_size",
                           &frame_estimation_options.max_image_size);
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  const Eigen::Matrix3d frame = EstimateCoordinateFrame(
      frame_estimation_options, reconstruction, *options.image_path);

  PrintHeading1("Aligning Reconstruction");

  Eigen::Matrix3d tform;
  if (frame.col(0).nonZeros() == 0) {
    std::cout << "Only aligning vertical axis" << std::endl;
    tform = RotationFromUnitVectors(frame.col(1), Eigen::Vector3d(0, -1, 0));
  } else if (frame.col(1).nonZeros() == 0) {
    tform = RotationFromUnitVectors(frame.col(0), Eigen::Vector3d(1, 0, 0));
    std::cout << "Only aligning horizontal axis" << std::endl;
  } else {
    tform = frame.transpose();
    std::cout << "Aligning horizontal and vertical axes" << std::endl;
  }

  std::cout << "using the rotation matrix:" << std::endl;
  std::cout << tform << std::endl;

  reconstruction.Transform(1, RotationMatrixToQuaternion(tform),
                           Eigen::Vector3d(0, 0, 0));

  std::cout << "Writing aligned reconstruction..." << std::endl;
  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}
