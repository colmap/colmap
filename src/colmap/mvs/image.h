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

#pragma once

#include "colmap/sensor/bitmap.h"

#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace colmap {
namespace mvs {

class Image {
 public:
  Image();
  Image(const std::string& path,
        size_t width,
        size_t height,
        const float* K,
        const float* R,
        const float* T);

  inline size_t GetWidth() const;
  inline size_t GetHeight() const;

  void SetBitmap(const Bitmap& bitmap);
  inline const Bitmap& GetBitmap() const;

  inline const std::string& GetPath() const;
  inline const float* GetR() const;
  inline const float* GetT() const;
  inline const float* GetK() const;
  inline const float* GetP() const;
  inline const float* GetInvP() const;
  inline const float* GetViewingDirection() const;

  void Rescale(float factor);
  void Rescale(float factor_x, float factor_y);
  void Downsize(size_t max_width, size_t max_height);

 private:
  std::string path_;
  size_t width_;
  size_t height_;
  float K_[9];
  float R_[9];
  float T_[3];
  float P_[12];
  float inv_P_[12];
  Bitmap bitmap_;
};

void ComputeRelativePose(const float R1[9],
                         const float T1[3],
                         const float R2[9],
                         const float T2[3],
                         float R[9],
                         float T[3]);

void ComposeProjectionMatrix(const float K[9],
                             const float R[9],
                             const float T[3],
                             float P[12]);

void ComposeInverseProjectionMatrix(const float K[9],
                                    const float R[9],
                                    const float T[3],
                                    float inv_P[12]);

void ComputeProjectionCenter(const float R[9], const float T[3], float C[3]);

void RotatePose(const float RR[9], float R[9], float T[3]);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t Image::GetWidth() const { return width_; }

size_t Image::GetHeight() const { return height_; }

const Bitmap& Image::GetBitmap() const { return bitmap_; }

const std::string& Image::GetPath() const { return path_; }

const float* Image::GetR() const { return R_; }

const float* Image::GetT() const { return T_; }

const float* Image::GetK() const { return K_; }

const float* Image::GetP() const { return P_; }

const float* Image::GetInvP() const { return inv_P_; }

const float* Image::GetViewingDirection() const { return &R_[6]; }

}  // namespace mvs
}  // namespace colmap
