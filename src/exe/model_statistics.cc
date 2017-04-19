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
#include "util/option_manager.h"

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string path;

  OptionManager options;
  options.AddRequiredOption("path", &path);
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(path);

  std::cout << StringPrintf("Cameras: %d", reconstruction.NumCameras())
            << std::endl;
  std::cout << StringPrintf("Images: %d", reconstruction.NumImages())
            << std::endl;
  std::cout << StringPrintf("Registered images: %d",
                            reconstruction.NumRegImages())
            << std::endl;
  std::cout << StringPrintf("Points: %d", reconstruction.NumPoints3D())
            << std::endl;
  std::cout << StringPrintf("Observations: %d",
                            reconstruction.ComputeNumObservations())
            << std::endl;
  std::cout << StringPrintf("Mean track length: %f",
                            reconstruction.ComputeMeanTrackLength())
            << std::endl;
  std::cout << StringPrintf("Mean observations per image: %f",
                            reconstruction.ComputeMeanObservationsPerRegImage())
            << std::endl;
  std::cout << StringPrintf("Mean reprojection error: %fpx",
                            reconstruction.ComputeMeanReprojectionError())
            << std::endl;

  return EXIT_SUCCESS;
}
