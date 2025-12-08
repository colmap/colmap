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

#pragma once

#include "colmap/util/string.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace colmap {

// Templated bitmap color class.
template <typename T>
struct BitmapColor {
  BitmapColor();
  explicit BitmapColor(T gray);
  BitmapColor(T r, T g, T b);

  template <typename D>
  BitmapColor<D> Cast() const;

  bool operator==(const BitmapColor<T>& rhs) const;
  bool operator!=(const BitmapColor<T>& rhs) const;

  template <typename D>
  friend std::ostream& operator<<(std::ostream& output,
                                  const BitmapColor<D>& color);

  T r;
  T g;
  T b;
};

// Wrapper class around bitmaps.
class Bitmap {
 public:
  Bitmap();

  // Construct bitmap with given dimensions.
  Bitmap(int width, int height, bool as_rgb, bool linear_colorspace = false);

  Bitmap(const Bitmap& other);
  Bitmap(Bitmap&& other) noexcept;

  Bitmap& operator=(const Bitmap& other);
  Bitmap& operator=(Bitmap&& other) noexcept;

  // Dimensions of bitmap.
  inline int Width() const;
  inline int Height() const;
  inline int Channels() const;

  // Number of bits per pixel. This is 8 for grey and 24 for RGB images.
  inline int BitsPerPixel() const;

  // Number of bytes required to store image.
  inline size_t NumBytes() const;

  // Scan line size in bytes, also known as stride.
  inline int Pitch() const;

  // Check whether the image is empty (i.e., width/height=0).
  inline bool IsEmpty() const;

  // Check whether image is grey- or colorscale.
  inline bool IsRGB() const;
  inline bool IsGrey() const;

  // Access raw image data array.
  inline std::vector<uint8_t>& RowMajorData();
  inline const std::vector<uint8_t>& RowMajorData() const;

  // Manipulate individual pixels. For grayscale images, only the red element
  // of the RGB color is used.
  inline bool GetPixel(int x, int y, BitmapColor<uint8_t>* color) const;
  inline bool SetPixel(int x, int y, const BitmapColor<uint8_t>& color);

  // Fill entire bitmap with uniform color. For grayscale images, the first
  // element of the vector is used.
  void Fill(const BitmapColor<uint8_t>& color);

  // Interpolate color at given floating point position.
  bool InterpolateNearestNeighbor(double x,
                                  double y,
                                  BitmapColor<uint8_t>* color) const;
  bool InterpolateBilinear(double x, double y, BitmapColor<float>* color) const;

  // Extract EXIF information from bitmap. Returns false if no EXIF information
  // is embedded in the bitmap.
  bool ExifCameraModel(std::string* camera_model) const;
  bool ExifFocalLength(double* focal_length) const;
  bool ExifLatitude(double* latitude) const;
  bool ExifLongitude(double* longitude) const;
  bool ExifAltitude(double* altitude) const;
  bool ExifGravity(double gravity[3]) const;

  // Read bitmap at given path and convert to grey- or colorscale. Defaults to
  // keeping the original colorspace (potentially non-linear) for image
  // processing.
  bool Read(const std::string& path,
            bool as_rgb = true,
            bool linearize_colorspace = false);

  // Write bitmap to file at given path. Defaults to converting to sRGB
  // colorspace for file storage.
  bool Write(const std::string& path, bool delinearize_colorspace = true) const;

  // Rescale image to the new dimensions.
  enum class RescaleFilter {
    kBilinear,
    kBox,
  };
  void Rescale(int new_width,
               int new_height,
               RescaleFilter filter = RescaleFilter::kBilinear);

  // Clone the image to a new bitmap object.
  Bitmap Clone() const;
  Bitmap CloneAsGrey() const;
  Bitmap CloneAsRGB() const;

  // Access metadata information (EXIF).
  void SetMetaData(const std::string_view& name,
                   const std::string_view& type,
                   const void* value);
  void SetMetaData(const std::string_view& name, const std::string_view& value);
  bool GetMetaData(const std::string_view& name,
                   const std::string_view& type,
                   void* value) const;
  bool GetMetaData(const std::string_view& name, std::string_view* value) const;

  // Clone metadata from this bitmap object to another target bitmap object.
  void CloneMetadata(Bitmap* target) const;

  struct MetaData {
    virtual ~MetaData() = default;
  };

 private:
  int width_;
  int height_;
  int channels_;
  bool linear_colorspace_;
  std::vector<uint8_t> data_;
  std::unique_ptr<MetaData> meta_data_;
};

std::ostream& operator<<(std::ostream& stream, const Bitmap& bitmap);

// Jet colormap inspired by Matlab. Grayvalues are expected in the range [0, 1]
// and are converted to RGB values in the same range.
class JetColormap {
 public:
  static float Red(float gray);
  static float Green(float gray);
  static float Blue(float gray);

 private:
  static float Interpolate(float val, float y0, float x0, float y1, float x1);
  static float Base(float val);
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

namespace internal {

template <typename T1, typename T2>
T2 BitmapColorCast(const T1 value) {
  return std::min(static_cast<T1>(std::numeric_limits<T2>::max()),
                  std::max(static_cast<T1>(std::numeric_limits<T2>::min()),
                           std::round(value)));
}

}  // namespace internal

template <typename T>
BitmapColor<T>::BitmapColor() : r(0), g(0), b(0) {}

template <typename T>
BitmapColor<T>::BitmapColor(const T gray) : r(gray), g(gray), b(gray) {}

template <typename T>
BitmapColor<T>::BitmapColor(const T r, const T g, const T b)
    : r(r), g(g), b(b) {}

template <typename T>
template <typename D>
BitmapColor<D> BitmapColor<T>::Cast() const {
  BitmapColor<D> color;
  color.r = internal::BitmapColorCast<T, D>(r);
  color.g = internal::BitmapColorCast<T, D>(g);
  color.b = internal::BitmapColorCast<T, D>(b);
  return color;
}

template <typename T>
bool BitmapColor<T>::operator==(const BitmapColor<T>& rhs) const {
  return r == rhs.r && g == rhs.g && b == rhs.b;
}

template <typename T>
bool BitmapColor<T>::operator!=(const BitmapColor<T>& rhs) const {
  return r != rhs.r || g != rhs.g || b != rhs.b;
}

template <typename T>
std::ostream& operator<<(std::ostream& output, const BitmapColor<T>& color) {
  output << StringPrintf("RGB(%f, %f, %f)",
                         static_cast<double>(color.r),
                         static_cast<double>(color.g),
                         static_cast<double>(color.b));
  return output;
}

int Bitmap::Width() const { return width_; }

int Bitmap::Height() const { return height_; }

int Bitmap::Channels() const { return channels_; }

size_t Bitmap::NumBytes() const { return data_.size(); }

int Bitmap::BitsPerPixel() const { return channels_ * 8; }

int Bitmap::Pitch() const { return width_ * channels_; }

bool Bitmap::IsEmpty() const { return NumBytes() == 0; }

bool Bitmap::IsRGB() const { return channels_ == 3; }

bool Bitmap::IsGrey() const { return channels_ == 1; }

std::vector<uint8_t>& Bitmap::RowMajorData() { return data_; }

const std::vector<uint8_t>& Bitmap::RowMajorData() const { return data_; }

bool Bitmap::GetPixel(const int x,
                      const int y,
                      BitmapColor<uint8_t>* color) const {
  if (x < 0 || x >= width_ || y < 0 || y >= height_) {
    return false;
  }

  if (IsGrey()) {
    color->r = data_[y * width_ + x];
    color->g = color->r;
    color->b = color->r;
    return true;
  } else if (IsRGB()) {
    const uint8_t* pixel = &data_[(y * width_ + x) * channels_];
    color->r = pixel[0];
    color->g = pixel[1];
    color->b = pixel[2];
    return true;
  }

  return false;
}

bool Bitmap::SetPixel(const int x,
                      const int y,
                      const BitmapColor<uint8_t>& color) {
  if (x < 0 || x >= width_ || y < 0 || y >= height_) {
    return false;
  }

  if (IsGrey()) {
    data_[y * width_ + x] = color.r;
    return true;
  } else if (IsRGB()) {
    uint8_t* pixel = &data_[(y * width_ + x) * channels_];
    pixel[0] = color.r;
    pixel[1] = color.g;
    pixel[2] = color.b;
    return true;
  }

  return false;
}

}  // namespace colmap
