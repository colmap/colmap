// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/track.h"

namespace colmap {

Track::Track() {}

TrackElement::TrackElement()
    : image_id(kInvalidImageId), point2D_idx(kInvalidPoint2DIdx) {}

TrackElement::TrackElement(const image_t image_id, const point2D_t point2D_idx)
    : image_id(image_id), point2D_idx(point2D_idx) {}

void Track::DeleteElement(const image_t image_id, const point2D_t point2D_idx) {
  elements_.erase(
      std::remove_if(elements_.begin(),
                     elements_.end(),
                     [image_id, point2D_idx](const TrackElement& element) {
                       return element.image_id == image_id &&
                              element.point2D_idx == point2D_idx;
                     }),
      elements_.end());
}

std::ostream& operator<<(std::ostream& stream, const TrackElement& track_el) {
  stream << "TrackElement(image_id=" << track_el.image_id
         << ", point2D_idx=" << track_el.point2D_idx << ")";
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Track& track) {
  stream << "Track(elements=[";
  for (auto it = track.Elements().begin(); it != track.Elements().end();) {
    stream << *it;
    if (++it != track.Elements().end()) {
      stream << ", ";
    }
  }
  stream << "])";
  return stream;
}

}  // namespace colmap
