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
  for (const auto& track_el : track.Elements()) {
    stream << track_el << ", ";
  }
  if (track.Length() > 0) {
    stream.seekp(-2, std::ios_base::end);
  }
  stream << "])";
  return stream;
}

}  // namespace colmap
