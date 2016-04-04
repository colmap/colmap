// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#ifndef COLMAP_SRC_BASE_WARP_H_
#define COLMAP_SRC_BASE_WARP_H_

#include "base/camera.h"
#include "util/bitmap.h"

namespace colmap {

// Warp source image to target image by projecting the pixels of the target
// image up to infinity and projecting it down into the source image
// (i.e. an inverse mapping). The function allocates the target image.
void WarpImageBetweenCameras(const Camera& source_camera,
                             const Camera& target_camera,
                             const Bitmap& source_image, Bitmap* target_image);

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_WARP_H_
