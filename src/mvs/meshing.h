// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

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
