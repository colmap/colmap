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

#include "base/track.h"

namespace colmap {

Track::Track() {}

TrackElement::TrackElement()
    : image_id(kInvalidImageId), point2D_idx(kInvalidPoint2DIdx) {}

TrackElement::TrackElement(const image_t image_id, const point2D_t point2D_idx)
    : image_id(image_id), point2D_idx(point2D_idx) {}

void Track::DeleteElement(const image_t image_id, const point2D_t point2D_idx) {
  elements_.erase(
      std::remove_if(elements_.begin(), elements_.end(),
                     [image_id, point2D_idx](const TrackElement& element) {
                       return element.image_id == image_id &&
                              element.point2D_idx == point2D_idx;
                     }),
      elements_.end());
}

}  // namespace colmap
