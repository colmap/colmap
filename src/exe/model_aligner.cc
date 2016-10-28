// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "base/reconstruction.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"

#include <sstream>

using namespace colmap;

void ReadReferenceCameras(
    const std::string& path, std::vector<std::string>* reference_image_names,
    std::vector<Eigen::Vector3d>* reference_camera_positions) {
  std::vector<std::string> lines = ReadTextFileLines(path);

  for (const auto line : lines) {
    std::stringstream line_parser(line);
    std::string image_name = "";
    Eigen::Vector3d camera_position;
    line_parser >> image_name >> camera_position[0] >> camera_position[1]
                >> camera_position[2];
    reference_image_names->push_back(image_name);
    reference_camera_positions->push_back(camera_position);
  }
}

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string input_model_path;
  std::string reference_cameras_path;
  std::string output_model_path;
  int min_common_images = 3;

  OptionManager options;
  options.AddRequiredOption("input_model_path", &input_model_path);
  options.AddRequiredOption("reference_cameras_path", &reference_cameras_path);
  options.AddRequiredOption("output_model_path", &output_model_path);
  options.AddDefaultOption("min_common_images", min_common_images,
                           &min_common_images);

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  std::vector<std::string> reference_image_names;
  std::vector<Eigen::Vector3d> reference_camera_positions;
  ReadReferenceCameras(reference_cameras_path, &reference_image_names,
                       &reference_camera_positions);

  Reconstruction reconstruction;
  reconstruction.Read(input_model_path);
  PrintHeading2("Reconstruction ");
  std::cout << StringPrintf("Images: %d", reconstruction.NumRegImages())
            << std::endl;
  std::cout << StringPrintf("Points: %d", reconstruction.NumPoints3D())
            << std::endl;

  PrintHeading2("Aligning reconstruction");
  if (reconstruction.AlignToCameraPositions(reference_image_names,
                                            reference_camera_positions,
                                            min_common_images)) {
    std::cout << "=> Alignment succeeded" << std::endl;
    reconstruction.Write(output_model_path);
  } else {
    std::cout << "=> Alignment failed" << std::endl;
  }

  return EXIT_SUCCESS;
}
