#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"

namespace glomap {

struct TrackFilter {
  static int FilterObservationsWithLargeReprojectionError(
      colmap::Reconstruction& reconstruction,
      double max_reprojection_error = 1e-2,
      bool in_normalized_image = true);

  static int FilterObservationsWithLargeAngularError(
      colmap::Reconstruction& reconstruction, double max_angle_error = 1.);

  static int FilterTracksWithSmallTriangulationAngle(
      colmap::Reconstruction& reconstruction, double min_angle = 1.);
};

}  // namespace glomap
