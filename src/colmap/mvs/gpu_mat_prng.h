// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/gpu_mat.h"

namespace colmap {
namespace mvs {

class GpuMatPRNG : public GpuMat<curandState> {
 public:
  GpuMatPRNG(const int width, const int height);

 private:
  void InitRandomState();
};

}  // namespace mvs
}  // namespace colmap
