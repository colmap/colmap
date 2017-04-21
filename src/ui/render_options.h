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

#ifndef COLMAP_SRC_UI_RENDER_OPTIONS_H_
#define COLMAP_SRC_UI_RENDER_OPTIONS_H_

#include <iostream>

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

  // Whether to visualize image connections.
  bool image_connections = false;

  // The projection type of the renderer.
  int projection_type = ProjectionType::PERSPECTIVE;

  bool Check() const;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_RENDER_OPTIONS_H_
