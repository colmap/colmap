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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
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
  inline std::optional<BitmapColor<uint8_t>> GetPixel(int x, int y) const;
  inline bool SetPixel(int x, int y, const BitmapColor<uint8_t>& color);

  // Fill entire bitmap with uniform color. For grayscale images, the first
  // element of the vector is used.
  void Fill(const BitmapColor<uint8_t>& color);

  // Interpolate color at given floating point position.
  inline std::optional<BitmapColor<uint8_t>> InterpolateNearestNeighbor(
      double x, double y) const;
  inline std::optional<BitmapColor<float>> InterpolateBilinear(double x,
                                                               double y) const;

  // Extract EXIF information from bitmap. Returns std::nullopt if no EXIF
  // information is embedded in the bitmap.
  std::optional<int> ExifOrientation() const;
  std::optional<std::string> ExifCameraModel() const;
  std::optional<double> ExifFocalLength() const;
  std::optional<double> ExifLatitude() const;
  std::optional<double> ExifLongitude() const;
  std::optional<double> ExifAltitude() const;

  // Read bitmap at given path and convert to grey- or colorscale. Defaults to
  // keeping the original colorspace (potentially non-linear) for image
  // processing.
  bool Read(const std::filesystem::path& path,
            bool as_rgb = true,
            bool linearize_colorspace = false);

  // Write bitmap to file at given path. Defaults to converting to sRGB
  // colorspace for file storage.
  bool Write(const std::filesystem::path& path,
             bool delinearize_colorspace = true) const;

  // Rescale image to the new dimensions.
  enum class RescaleFilter {
    kBilinear,
    kBox,
  };
  void Rescale(int new_width,
               int new_height,
               RescaleFilter filter = RescaleFilter::kBilinear);

  // Downscale the image in place so that neither dimension exceeds
  // `max_image_size`, preserving the aspect ratio. Images that already fit
  // within the bound are left unchanged. Returns the scale factor that was
  // applied (1 if no rescaling was necessary).
  double Thumbnail(int max_image_size,
                   RescaleFilter filter = RescaleFilter::kBilinear);

  // Rotate image by k * 90 degrees counter-clockwise.
  void Rot90(int k);

  // Clone the image to a new bitmap object.
  Bitmap Clone() const;
  Bitmap CloneAsGrey() const;
  Bitmap CloneAsRGB() const;

  // Set compression quality when writing to JPEG in the range [1, 100].
  // Lower values reduce quality and file size. By default, bitmaps are
  // written in superb (100) quality, if not otherwise specified.
  void SetJpegQuality(int quality);

  // Access metadata information (e.g., EXIF).
  void SetMetaData(const std::string_view& name,
                   const std::string_view& type,
                   const void* value);
  void SetMetaData(const std::string_view& name, const std::string_view& value);
  bool GetMetaData(const std::string_view& name,
                   const std::string_view& type,
                   void* value) const;
  std::optional<std::string> GetMetaData(const std::string_view& name) const;

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
std::ostream& operator<<(std::ostream& stream, const BitmapColor<T>& color) {
  if (std::is_same<T, char>::value || std::is_same<T, unsigned char>::value) {
    stream << "RGB(" << static_cast<int>(color.r) << ", "
           << static_cast<int>(color.g) << ", " << static_cast<int>(color.b)
           << ")";
  } else {
    stream << "RGB(" << color.r << ", " << color.g << ", " << color.b << ")";
  }
  return stream;
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

std::optional<BitmapColor<uint8_t>> Bitmap::GetPixel(const int x,
                                                     const int y) const {
  if (x < 0 || x >= width_ || y < 0 || y >= height_) {
    return std::nullopt;
  }

  if (IsGrey()) {
    const uint8_t v = data_[y * width_ + x];
    return BitmapColor<uint8_t>(v, v, v);
  } else if (IsRGB()) {
    const uint8_t* pixel = &data_[(y * width_ + x) * channels_];
    return BitmapColor<uint8_t>(pixel[0], pixel[1], pixel[2]);
  }

  return std::nullopt;
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

std::optional<BitmapColor<uint8_t>> Bitmap::InterpolateNearestNeighbor(
    const double x, const double y) const {
  const int xx = static_cast<int>(std::round(x));
  const int yy = static_cast<int>(std::round(y));
  return GetPixel(xx, yy);
}

std::optional<BitmapColor<float>> Bitmap::InterpolateBilinear(
    const double x, const double y) const {
  const int x0 = static_cast<int>(std::floor(x));
  const int x1 = x0 + 1;
  const int y0 = static_cast<int>(std::floor(y));
  const int y1 = y0 + 1;

  if (x0 < 0 || x1 >= width_ || y0 < 0 || y1 >= height_) {
    return std::nullopt;
  }

  const double dx = x - x0;
  const double dy = y - y0;
  const double dx_1 = 1 - dx;
  const double dy_1 = 1 - dy;

  const int pitch = width_ * channels_;
  const uint8_t* line0 = &data_[y0 * pitch];
  const uint8_t* line1 = &data_[y1 * pitch];

  if (IsGrey()) {
    // Top row, column-wise linear interpolation.
    const double v0 = dx_1 * line0[x0] + dx * line0[x1];

    // Bottom row, column-wise linear interpolation.
    const double v1 = dx_1 * line1[x0] + dx * line1[x1];

    // Row-wise linear interpolation.
    const float r = dy_1 * v0 + dy * v1;
    return BitmapColor<float>(r, r, r);
  } else if (IsRGB()) {
    const uint8_t* p00 = &line0[3 * x0];
    const uint8_t* p01 = &line0[3 * x1];
    const uint8_t* p10 = &line1[3 * x0];
    const uint8_t* p11 = &line1[3 * x1];

    // Top row, column-wise linear interpolation.
    const double v0_r = dx_1 * p00[0] + dx * p01[0];
    const double v0_g = dx_1 * p00[1] + dx * p01[1];
    const double v0_b = dx_1 * p00[2] + dx * p01[2];

    // Bottom row, column-wise linear interpolation.
    const double v1_r = dx_1 * p10[0] + dx * p11[0];
    const double v1_g = dx_1 * p10[1] + dx * p11[1];
    const double v1_b = dx_1 * p10[2] + dx * p11[2];

    // Row-wise linear interpolation.
    return BitmapColor<float>(dy_1 * v0_r + dy * v1_r,
                              dy_1 * v0_g + dy * v1_g,
                              dy_1 * v0_b + dy * v1_b);
  }

  return std::nullopt;
}

}  // namespace colmap
