#pragma once

#include "colmap/scene/frame.h"

namespace glomap {

struct Frame : public colmap::Frame {
  Frame() : colmap::Frame() {}
  explicit Frame(const colmap::Frame& frame) : colmap::Frame(frame) {}
};

}  // namespace glomap
