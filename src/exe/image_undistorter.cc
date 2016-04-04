// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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
#include "base/undistortion.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"

using namespace colmap;

namespace config = boost::program_options;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string input_path;
  std::string output_path;
  std::string output_type;

  UndistortCameraOptions undistort_camera_options;

  OptionManager options;
  options.AddImageOptions();
  options.desc->add_options()(
      "input_path", config::value<std::string>(&input_path)->required());
  options.desc->add_options()(
      "output_path", config::value<std::string>(&output_path)->required());
  options.desc->add_options()(
      "output_type", config::value<std::string>(&output_type)->required(),
      "{'Default', 'PMVS', 'CMP-MVS'}");
  options.desc->add_options()(
      "blank_pixels",
      config::value<double>(&undistort_camera_options.blank_pixels)
          ->default_value(undistort_camera_options.blank_pixels));
  options.desc->add_options()(
      "min_scale", config::value<double>(&undistort_camera_options.min_scale)
                       ->default_value(undistort_camera_options.min_scale));
  options.desc->add_options()(
      "max_scale", config::value<double>(&undistort_camera_options.max_scale)
                       ->default_value(undistort_camera_options.max_scale));
  options.desc->add_options()(
      "max_image_size",
      config::value<int>(&undistort_camera_options.max_image_size)
          ->default_value(undistort_camera_options.max_image_size));

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  ImageUndistorter* undistorter = nullptr;

  if (output_type == "Default") {
    undistorter = new ImageUndistorter(undistort_camera_options, reconstruction,
                                       *options.image_path, output_path);
  } else if (output_type == "PMVS") {
    undistorter = new PMVSUndistorter(undistort_camera_options, reconstruction,
                                      *options.image_path, output_path);
  } else if (output_type == "CMP-MVS") {
    undistorter =
        new CMPMVSUndistorter(undistort_camera_options, reconstruction,
                              *options.image_path, output_path);
  } else {
    std::cerr << "ERROR: Invalid `output_type`" << std::endl;
    return EXIT_FAILURE;
  }

  undistorter->start();
  undistorter->wait();

  return EXIT_SUCCESS;
}
