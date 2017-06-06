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

#ifndef COLMAP_SRC_MVS_IMAGE_H_
#define COLMAP_SRC_MVS_IMAGE_H_

#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "util/bitmap.h"

namespace colmap {
namespace mvs {

class Image {
 public:
  Image();
  Image(const std::string& path, const size_t width, const size_t height,
        const float* K, const float* R, const float* T);

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

  void Rescale(const float factor);
  void Rescale(const float factor_x, const float factor_y);
  void Downsize(const size_t max_width, const size_t max_height);

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

void ComputeRelativePose(const float R1[9], const float T1[3],
                         const float R2[9], const float T2[3], float R[9],
                         float T[3]);

void ComposeProjectionMatrix(const float K[9], const float R[9],
                             const float T[3], float P[12]);

void ComposeInverseProjectionMatrix(const float K[9], const float R[9],
                                    const float T[3], float inv_P[12]);

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

#endif  // COLMAP_SRC_MVS_IMAGE_H_
