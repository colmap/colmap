#pragma once

#include "colmap/scene/frame.h"

namespace glomap {

struct Frame : public colmap::Frame {
  Frame() : colmap::Frame() {}
  explicit Frame(const colmap::Frame& frame) : colmap::Frame(frame) {}

  // whether the frame is within the largest connected component
  bool is_registered = false;
  int cluster_id = -1;
};

}  // namespace glomap
