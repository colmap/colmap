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

#pragma once

#include <algorithm>
#include <cmath>
#include <ios>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif
#include "colmap/util/string.h"

#include <FreeImage.h>

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

// Wrapper class around FreeImage bitmaps.
class Bitmap {
 public:
  Bitmap();

  // Copy constructor.
  Bitmap(const Bitmap& other);
  // Move constructor.
  Bitmap(Bitmap&& other) noexcept;

  // Create bitmap object from existing FreeImage bitmap object. Note that
  // this class takes ownership of the object.
  explicit Bitmap(FIBITMAP* data);

  // Copy assignment.
  Bitmap& operator=(const Bitmap& other);
  // Move assignment.
  Bitmap& operator=(Bitmap&& other) noexcept;

  // Allocate bitmap by overwriting the existing data.
  bool Allocate(int width, int height, bool as_rgb);

  // Deallocate the bitmap by releasing the existing data.
  void Deallocate();

  // Get pointer to underlying FreeImage object.
  inline const FIBITMAP* Data() const;
  inline FIBITMAP* Data();

  // Dimensions of bitmap.
  inline int Width() const;
  inline int Height() const;
  inline int Channels() const;

  // Number of bits per pixel. This is 8 for grey and 24 for RGB image.
  inline unsigned int BitsPerPixel() const;

  // Scan width of bitmap which differs from the actual image width to achieve
  // 32 bit aligned memory. Also known as pitch or stride.
  inline unsigned int ScanWidth() const;

  // Check whether image is grey- or colorscale.
  inline bool IsRGB() const;
  inline bool IsGrey() const;

  // Number of bytes required to store image.
  size_t NumBytes() const;

  // Copy raw image data to array.
  std::vector<uint8_t> ConvertToRawBits() const;
  std::vector<uint8_t> ConvertToRowMajorArray() const;
  std::vector<uint8_t> ConvertToColMajorArray() const;

  // Manipulate individual pixels. For grayscale images, only the red element
  // of the RGB color is used.
  bool GetPixel(int x, int y, BitmapColor<uint8_t>* color) const;
  bool SetPixel(int x, int y, const BitmapColor<uint8_t>& color);

  // Get pointer to y-th scanline, where the 0-th scanline is at the top.
  const uint8_t* GetScanline(int y) const;

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

  // Read bitmap at given path and convert to grey- or colorscale.
  bool Read(const std::string& path, bool as_rgb = true);

  // Write image to file. Flags can be used to set e.g. the JPEG quality.
  // Consult the FreeImage documentation for all available flags.
  bool Write(const std::string& path,
             FREE_IMAGE_FORMAT format = FIF_UNKNOWN,
             int flags = 0) const;

  // Smooth the image using a Gaussian kernel.
  void Smooth(float sigma_x, float sigma_y);

  // Rescale image to the new dimensions.
  void Rescale(int new_width,
               int new_height,
               FREE_IMAGE_FILTER filter = FILTER_BILINEAR);

  // Clone the image to a new bitmap object.
  Bitmap Clone() const;
  Bitmap CloneAsGrey() const;
  Bitmap CloneAsRGB() const;

  // Clone metadata from this bitmap object to another target bitmap object.
  void CloneMetadata(Bitmap* target) const;

  // Read specific EXIF tag.
  bool ReadExifTag(FREE_IMAGE_MDMODEL model,
                   const std::string& tag_name,
                   std::string* result) const;

 private:
  typedef std::unique_ptr<FIBITMAP, decltype(&FreeImage_Unload)> FIBitmapPtr;

  void SetPtr(FIBITMAP* data);

  static bool IsPtrGrey(FIBITMAP* data);
  static bool IsPtrRGB(FIBITMAP* data);
  static bool IsPtrSupported(FIBITMAP* data);

  FIBitmapPtr data_;
  int width_;
  int height_;
  int channels_;
};

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

FIBITMAP* Bitmap::Data() { return data_.get(); }
const FIBITMAP* Bitmap::Data() const { return data_.get(); }

int Bitmap::Width() const { return width_; }
int Bitmap::Height() const { return height_; }
int Bitmap::Channels() const { return channels_; }

unsigned int Bitmap::BitsPerPixel() const {
  return FreeImage_GetBPP(data_.get());
}

unsigned int Bitmap::ScanWidth() const {
  return FreeImage_GetPitch(data_.get());
}

bool Bitmap::IsRGB() const { return channels_ == 3; }

bool Bitmap::IsGrey() const { return channels_ == 1; }

}  // namespace colmap
