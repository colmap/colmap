// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <vector>

namespace colmap {

// Track class stores all observations of a 3D point.
struct TrackElement {
  TrackElement();
  TrackElement(image_t image_id, point2D_t point2D_idx);
  // The image in which the track element is observed.
  image_t image_id;
  // The point in the image that the track element is observed.
  point2D_t point2D_idx;

  inline bool operator==(const TrackElement& other) const;
  inline bool operator!=(const TrackElement& other) const;
};

class Track {
 public:
  Track();

  // The number of track elements.
  inline size_t Length() const;

  // Access all elements.
  inline const std::vector<TrackElement>& Elements() const;
  inline std::vector<TrackElement>& Elements();
  inline void SetElements(std::vector<TrackElement> elements);

  // Access specific elements.
  inline const TrackElement& Element(size_t idx) const;
  inline TrackElement& Element(size_t idx);
  inline void SetElement(size_t idx, const TrackElement& element);

  // Append new elements.
  inline void AddElement(const TrackElement& element);
  inline void AddElement(image_t image_id, point2D_t point2D_idx);
  inline void AddElements(const std::vector<TrackElement>& elements);

  // Delete existing element.
  inline void DeleteElement(size_t idx);
  void DeleteElement(image_t image_id, point2D_t point2D_idx);

  // Requests that the track capacity be at least enough to contain the
  // specified number of elements.
  inline void Reserve(size_t num_elements);

  // Shrink the capacity of track vector to fit its size to save memory.
  inline void Compress();

  inline bool operator==(const Track& other) const;
  inline bool operator!=(const Track& other) const;

 private:
  std::vector<TrackElement> elements_;
};

std::ostream& operator<<(std::ostream& stream, const TrackElement& track_el);
std::ostream& operator<<(std::ostream& stream, const Track& track);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

bool TrackElement::operator==(const TrackElement& other) const {
  return image_id == other.image_id && point2D_idx == other.point2D_idx;
}

bool TrackElement::operator!=(const TrackElement& other) const {
  return !(*this == other);
}

size_t Track::Length() const { return elements_.size(); }

const std::vector<TrackElement>& Track::Elements() const { return elements_; }

std::vector<TrackElement>& Track::Elements() { return elements_; }

void Track::SetElements(std::vector<TrackElement> elements) {
  elements_ = std::move(elements);
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
  THROW_CHECK_LT(idx, elements_.size());
  elements_.erase(elements_.begin() + idx);
}

void Track::Reserve(const size_t num_elements) {
  elements_.reserve(num_elements);
}

void Track::Compress() { elements_.shrink_to_fit(); }

bool Track::operator==(const Track& other) const {
  return elements_ == other.elements_;
}

bool Track::operator!=(const Track& other) const { return !(*this == other); }

}  // namespace colmap
