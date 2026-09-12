// SPDX-License-Identifier: BSD-3-Clause

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
