// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#pragma once

#include "colmap/util/ply.h"

namespace colmap {
namespace mvs {

struct MeshSimplificationOptions {
  // Fraction of faces to retain, in (0, 1].
  double target_face_ratio = 0.1;

  // Maximum quadric error per collapse; 0 = disabled.
  double max_error = 0.0;

  // Penalty weight for boundary edges; 0 = disabled.
  double boundary_weight = 1000.0;

  // Blend colors on collapse vs. pick lower-error vertex.
  bool interpolate_colors = true;

  // The number of threads to use for initialization. Default is all threads.
  int num_threads = -1;

  bool Check() const;
};

// Simplify a triangle mesh using Quadric Error Metric (QEM) decimation
// (Garland & Heckbert, SIGGRAPH 1997).
PlyMesh SimplifyMesh(const PlyMesh& mesh,
                     const MeshSimplificationOptions& options);

}  // namespace mvs
}  // namespace colmap
