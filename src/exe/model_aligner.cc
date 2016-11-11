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

void ReadReferenceImages(const std::string& path,
                         std::vector<std::string>* ref_image_names,
                         std::vector<Eigen::Vector3d>* ref_locations) {
  std::vector<std::string> lines = ReadTextFileLines(path);
  for (const auto line : lines) {
    std::stringstream line_parser(line);
    std::string image_name = "";
    Eigen::Vector3d camera_position;
    line_parser >> image_name >> camera_position[0] >> camera_position[1] >>
        camera_position[2];
    ref_image_names->push_back(image_name);
    ref_locations->push_back(camera_position);
  }
}

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string input_path;
  std::string ref_images_path;
  std::string output_path;
  int min_common_images = 3;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("ref_images_path", &ref_images_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("min_common_images", min_common_images,
                           &min_common_images);

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  std::vector<std::string> ref_image_names;
  std::vector<Eigen::Vector3d> ref_locations;
  ReadReferenceImages(ref_images_path, &ref_image_names, &ref_locations);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  PrintHeading2("Aligning reconstruction");

  std::cout << StringPrintf(" => Using %d reference images",
                            ref_image_names.size())
            << std::endl;

  if (reconstruction.Align(ref_image_names, ref_locations, min_common_images)) {
    std::cout << " => Alignment succeeded" << std::endl;
    reconstruction.Write(output_path);

    double positional_error = 0;
    size_t num_aligned = 0;
    for (size_t i = 0; i < ref_image_names.size(); ++i) {
      const Image* image = reconstruction.FindImageWithName(ref_image_names[i]);
      if (image != nullptr) {
        positional_error +=
            (image->ProjectionCenter() - ref_locations[i]).norm();
        num_aligned += 1;
      }
    }

    std::cout << StringPrintf(" => Alignment error: %f",
                              positional_error / num_aligned)
              << std::endl;
  } else {
    std::cout << " => Alignment failed" << std::endl;
  }

  return EXIT_SUCCESS;
}
