// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_BASE_TRACK_H_
#define COLMAP_SRC_BASE_TRACK_H_

#include <vector>

#include "util/logging.h"
#include "util/types.h"

namespace colmap {

// Track class stores all observations of a 3D point.
struct TrackElement {
  TrackElement();
  TrackElement(const image_t image_id, const point2D_t point2D_idx);
  // The image in which the track element is observed.
  image_t image_id;
  // The point in the image that the track element is observed.
  point2D_t point2D_idx;
};

class Track {
 public:
  Track();

  // The number of track elements.
  inline size_t Length() const;

  // Access all elements.
  inline const std::vector<TrackElement>& Elements() const;
  inline std::vector<TrackElement>& Elements();
  inline void SetElements(const std::vector<TrackElement>& elements);

  // Access specific elements.
  inline const TrackElement& Element(const size_t idx) const;
  inline TrackElement& Element(const size_t idx);
  inline void SetElement(const size_t idx, const TrackElement& element);

  // Append new elements.
  inline void AddElement(const TrackElement& element);
  inline void AddElement(const image_t image_id, const point2D_t point2D_idx);
  inline void AddElements(const std::vector<TrackElement>& elements);

  // Delete existing element.
  inline void DeleteElement(const size_t idx);
  void DeleteElement(const image_t image_id, const point2D_t point2D_idx);

  // Requests that the track capacity be at least enough to contain the
  // specified number of elements.
  inline void Reserve(const size_t num_elements);

  // Shrink the capacity of track vector to fit its size to save memory.
  inline void Compress();

 private:
  std::vector<TrackElement> elements_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t Track::Length() const { return elements_.size(); }

const std::vector<TrackElement>& Track::Elements() const { return elements_; }

std::vector<TrackElement>& Track::Elements() { return elements_; }

void Track::SetElements(const std::vector<TrackElement>& elements) {
  elements_ = elements;
}

// Access specific elements.
const TrackElement& Track::Element(const size_t idx) const {
  return elements_.at(idx);
}

TrackElement& Track::Element(const size_t idx) { return elements_.at(idx); }

void Track::SetElement(const size_t idx, const TrackElement& element) {
  elements_.at(idx) = element;
}

void Track::AddElement(const TrackElement& element) {
  elements_.push_back(element);
}

void Track::AddElement(const image_t image_id, const point2D_t point2D_idx) {
  elements_.emplace_back(image_id, point2D_idx);
}

void Track::AddElements(const std::vector<TrackElement>& elements) {
  elements_.insert(elements_.end(), elements.begin(), elements.end());
}

void Track::DeleteElement(const size_t idx) {
  CHECK_LT(idx, elements_.size());
  elements_.erase(elements_.begin() + idx);
}

void Track::Reserve(const size_t num_elements) {
  elements_.reserve(num_elements);
}

void Track::Compress() { elements_.shrink_to_fit(); }

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_TRACK_H_
