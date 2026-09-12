// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/logging.h"

namespace colmap {

struct RenderOptions {
  enum ProjectionType {
    PERSPECTIVE,
    ORTHOGRAPHIC,
  };

  // Minimum track length for a point to be rendered.
  int min_track_len = 3;

  // Maximum error for a point to be rendered.
  double max_error = 2;

  // The rate of registered images at which to refresh.
  int refresh_rate = 1;

  // Whether to automatically adjust the refresh rate. The bigger the
  // reconstruction gets, the less frequently the scene is rendered.
  bool adapt_refresh_rate = true;

  // Whether to show camera orientation (triangle in top-left corner).
  bool show_camera_orientation = false;

  // Whether to visualize image connections.
  bool image_connections = false;

  // Whether to render the mesh as wireframe.
  bool mesh_wireframe = false;

  // Whether to render mesh colors/textures (false = uniform gray).
  bool mesh_color = true;

  // The projection type of the renderer.
  int projection_type = ProjectionType::PERSPECTIVE;

  inline bool Check() const {
    CHECK_OPTION_GE(min_track_len, 0);
    CHECK_OPTION_GE(max_error, 0);
    CHECK_OPTION_GT(refresh_rate, 0);
    CHECK_OPTION(projection_type == ProjectionType::PERSPECTIVE ||
                 projection_type == ProjectionType::ORTHOGRAPHIC);
    return true;
  }
};

}  // namespace colmap
