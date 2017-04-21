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

#include "base/reconstruction.h"
#include "util/misc.h"
#include "util/option_manager.h"

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
  bool robust_alignment = true;
  RANSACOptions ransac_options;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("ref_images_path", &ref_images_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("min_common_images", &min_common_images);
  options.AddDefaultOption("robust_alignment", &robust_alignment);
  options.AddDefaultOption("robust_alignment_max_error",
                           &ransac_options.max_error);
  options.Parse(argc, argv);

  if (robust_alignment && ransac_options.max_error <= 0) {
    std::cout << "ERROR: You must provide a maximum alignment error > 0"
              << std::endl;
    return EXIT_FAILURE;
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

  bool alignment_success;
  if (robust_alignment) {
    alignment_success = reconstruction.AlignRobust(
        ref_image_names, ref_locations, min_common_images, ransac_options);
  } else {
    alignment_success =
        reconstruction.Align(ref_image_names, ref_locations, min_common_images);
  }

  if (alignment_success) {
    std::cout << " => Alignment succeeded" << std::endl;
    reconstruction.Write(output_path);

    std::vector<double> errors;
    errors.reserve(ref_image_names.size());

    for (size_t i = 0; i < ref_image_names.size(); ++i) {
      const Image* image = reconstruction.FindImageWithName(ref_image_names[i]);
      if (image != nullptr) {
        errors.push_back((image->ProjectionCenter() - ref_locations[i]).norm());
      }
    }

    std::cout << StringPrintf(" => Alignment error: %f (mean), %f (median)",
                              Mean(errors), Median(errors))
              << std::endl;
  } else {
    std::cout << " => Alignment failed" << std::endl;
  }

  return EXIT_SUCCESS;
}
