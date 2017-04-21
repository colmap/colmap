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
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string input_path1;
  std::string input_path2;
  std::string output_path;
  int min_common_images = 3;

  OptionManager options;
  options.AddRequiredOption("input_path1", &input_path1);
  options.AddRequiredOption("input_path2", &input_path2);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("min_common_images", &min_common_images);
  options.Parse(argc, argv);

  Reconstruction reconstruction1;
  reconstruction1.Read(input_path1);
  PrintHeading2("Reconstruction 1");
  std::cout << StringPrintf("Images: %d", reconstruction1.NumRegImages())
            << std::endl;
  std::cout << StringPrintf("Points: %d", reconstruction1.NumPoints3D())
            << std::endl;

  Reconstruction reconstruction2;
  reconstruction2.Read(input_path2);
  PrintHeading2("Reconstruction 2");
  std::cout << StringPrintf("Images: %d", reconstruction2.NumRegImages())
            << std::endl;
  std::cout << StringPrintf("Points: %d", reconstruction2.NumPoints3D())
            << std::endl;

  PrintHeading2("Merging reconstructions");
  if (reconstruction1.Merge(reconstruction2, min_common_images)) {
    std::cout << "=> Merge succeeded" << std::endl;
    PrintHeading2("Merged reconstruction");
    std::cout << StringPrintf("Images: %d", reconstruction1.NumRegImages())
              << std::endl;
    std::cout << StringPrintf("Points: %d", reconstruction1.NumPoints3D())
              << std::endl;
  } else {
    std::cout << "=> Merge failed" << std::endl;
  }

  reconstruction1.Write(output_path);

  return EXIT_SUCCESS;
}
