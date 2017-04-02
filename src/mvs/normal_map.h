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

#ifndef COLMAP_SRC_MVS_NORMAL_MAP_H_
#define COLMAP_SRC_MVS_NORMAL_MAP_H_

#include <string>
#include <vector>

#include "mvs/mat.h"
#include "util/bitmap.h"

namespace colmap {
namespace mvs {

// Normal map class that stores per-pixel normals as a MxNx3 image.
class NormalMap : public Mat<float> {
 public:
  NormalMap();
  NormalMap(const size_t width, const size_t height);
  explicit NormalMap(const Mat<float>& mat);

  void Rescale(const float factor);
  void Downsize(const size_t max_width, const size_t max_height);

  Bitmap ToBitmap() const;
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_NORMAL_MAP_H_
