// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/mvs/image.h"

#include "colmap/scene/projection.h"
#include "colmap/util/logging.h"

#include <Eigen/Core>

namespace colmap {
namespace mvs {

Image::Image() {}

Image::Image(const std::string& path,
             const size_t width,
             const size_t height,
             const float* K,
             const float* R,
             const float* T)
    : path_(path), width_(width), height_(height) {
  memcpy(K_, K, 9 * sizeof(float));
  memcpy(R_, R, 9 * sizeof(float));
  memcpy(T_, T, 3 * sizeof(float));
  ComposeProjectionMatrix(K_, R_, T_, P_);
  ComposeInverseProjectionMatrix(K_, R_, T_, inv_P_);
}

void Image::SetBitmap(const Bitmap& bitmap) {
  bitmap_ = bitmap;
  CHECK_EQ(width_, bitmap_.Width());
  CHECK_EQ(height_, bitmap_.Height());
}

void Image::Rescale(const float factor) { Rescale(factor, factor); }

void Image::Rescale(const float factor_x, const float factor_y) {
  const size_t new_width = std::round(width_ * factor_x);
  const size_t new_height = std::round(height_ * factor_y);

  if (bitmap_.Data() != nullptr) {
    bitmap_.Rescale(new_width, new_height);
  }

  const float scale_x = new_width / static_cast<float>(width_);
  const float scale_y = new_height / static_cast<float>(height_);
  K_[0] *= scale_x;
  K_[2] *= scale_x;
  K_[4] *= scale_y;
  K_[5] *= scale_y;
  ComposeProjectionMatrix(K_, R_, T_, P_);
  ComposeInverseProjectionMatrix(K_, R_, T_, inv_P_);

  width_ = new_width;
  height_ = new_height;
}

void Image::Downsize(const size_t max_width, const size_t max_height) {
  if (width_ <= max_width && height_ <= max_height) {
    return;
  }
  const float factor_x = static_cast<float>(max_width) / width_;
  const float factor_y = static_cast<float>(max_height) / height_;
  Rescale(std::min(factor_x, factor_y));
}

void ComputeRelativePose(const float R1[9],
                         const float T1[3],
                         const float R2[9],
                         const float T2[3],
                         float R[9],
                         float T[3]) {
  const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R1_m(R1);
  const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R2_m(R2);
  const Eigen::Map<const Eigen::Matrix<float, 3, 1>> T1_m(T1);
  const Eigen::Map<const Eigen::Matrix<float, 3, 1>> T2_m(T2);
  Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R_m(R);
  Eigen::Map<Eigen::Vector3f> T_m(T);

  R_m = R2_m * R1_m.transpose();
  T_m = T2_m - R_m * T1_m;
}

void ComposeProjectionMatrix(const float K[9],
                             const float R[9],
                             const float T[3],
                             float P[12]) {
  Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> P_m(P);
  P_m.leftCols<3>() =
      Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(R);
  P_m.rightCols<1>() = Eigen::Map<const Eigen::Vector3f>(T);
  P_m = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K) * P_m;
}

void ComposeInverseProjectionMatrix(const float K[9],
                                    const float R[9],
                                    const float T[3],
                                    float inv_P[12]) {
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> P;
  ComposeProjectionMatrix(K, R, T, P.data());
  P.row(3) = Eigen::Vector4f(0, 0, 0, 1);
  const Eigen::Matrix4f inv_P_temp = P.inverse();
  Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> inv_P_m(inv_P);
  inv_P_m = inv_P_temp.topRows<3>();
}

void ComputeProjectionCenter(const float R[9], const float T[3], float C[3]) {
  const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R_m(R);
  const Eigen::Map<const Eigen::Matrix<float, 3, 1>> T_m(T);
  Eigen::Map<Eigen::Vector3f> C_m(C);
  C_m = -R_m.transpose() * T_m;
}

void RotatePose(const float RR[9], float R[9], float T[3]) {
  Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R_m(R);
  Eigen::Map<Eigen::Matrix<float, 3, 1>> T_m(T);
  const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> RR_m(RR);
  R_m = RR_m * R_m;
  T_m = RR_m * T_m;
}

}  // namespace mvs
}  // namespace colmap
