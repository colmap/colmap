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

#ifndef COLMAP_SRC_MVS_MESHING_H_
#define COLMAP_SRC_MVS_MESHING_H_

#include <string>

namespace colmap {
namespace mvs {

struct PoissonReconstructionOptions {
  // This floating point value specifies the importance that interpolation of
  // the point samples is given in the formulation of the screened Poisson
  // equation. The results of the original (unscreened) Poisson Reconstruction
  // can be obtained by setting this value to 0.
  double point_weight = 1.0;

  // This integer is the maximum depth of the tree that will be used for surface
  // reconstruction. Running at depth d corresponds to solving on a voxel grid
  // whose resolution is no larger than 2^d x 2^d x 2^d. Note that since the
  // reconstructor adapts the octree to the sampling density, the specified
  // reconstruction depth is only an upper bound.
  int depth = 13;

  // If specified, the reconstruction code assumes that the input is equipped
  // with colors and will extrapolate the color values to the vertices of the
  // reconstructed mesh. The floating point value specifies the relative
  // importance of finer color estimates over lower ones.
  double color = 32.0;

  // This floating point values specifies the value for mesh trimming. The
  // subset of the mesh with signal value less than the trim value is discarded.
  double trim = 10.0;

  // The number of threads used for the Poisson reconstruction.
  int num_threads = -1;

  bool Check() const;
};

// Perform Poisson surface reconstruction and return true if successful.
bool PoissonReconstruction(const PoissonReconstructionOptions& options,
                           const std::string& input_path,
                           const std::string& output_path);

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_MESHING_H_
