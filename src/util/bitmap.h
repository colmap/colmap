// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#ifndef COLMAP_SRC_UTIL_BITMAP_H_
#define COLMAP_SRC_UTIL_BITMAP_H_

#include <ios>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif
#include <FreeImage.h>

#include "util/types.h"

namespace colmap {

// Wrapper class around FreeImage bitmaps.
class Bitmap {
 public:
  Bitmap();

  // Create bitmap object from existing FreeImage bitmap object. Note that
  // this class takes ownership of the object.
  explicit Bitmap(FIBITMAP* data);

  // Allocate bitmap by overwriting the existing data.
  bool Allocate(const int width, const int height, const bool as_rgb);

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

  // Copy raw image data to array.
  std::vector<uint8_t> ConvertToRawBits() const;
  std::vector<uint8_t> ConvertToRowMajorArray() const;
  std::vector<uint8_t> ConvertToColMajorArray() const;

  // Manipulate individual pixels. For grayscale images, the first element of
  // the vector is used.
  bool GetPixel(const int x, const int y, Eigen::Vector3ub* color) const;
  bool SetPixel(const int x, const int y, const Eigen::Vector3ub& color);

  // Get pointer to y-th scanline, where the 0-th scanline is at the top.
  const uint8_t* GetScanline(const int y) const;

  // Fill entire bitmap with uniform color. For grayscale images, the first
  // element of the vector is used.
  void Fill(const Eigen::Vector3ub& color);

  // Interpolate color at given floating point position.
  bool InterpolateNearestNeighbor(const double x, const double y,
                                  Eigen::Vector3ub* color) const;
  bool InterpolateBilinear(const double x, const double y,
                           Eigen::Vector3d* color) const;

  // Extract EXIF information from bitmap. Returns false if no EXIF information
  // is embedded in the bitmap.
  bool ExifFocalLength(double* focal_length);
  bool ExifLatitude(double* latitude);
  bool ExifLongitude(double* longitude);
  bool ExifAltitude(double* altitude);

  // Read bitmap at given path and convert to grey- or colorscale.
  bool Read(const std::string& path, const bool as_rgb = true);

  // Write image to file. Flags can be used to set e.g. the JPEG quality.
  // Consult the FreeImage documentation for all available flags.
  bool Write(const std::string& path,
             const FREE_IMAGE_FORMAT format = FIF_UNKNOWN,
             const int flags = 0) const;

  // Smooth the image using a Gaussian kernel.
  void Smooth(const float sigma_x, const float sigma_y);

  // Rescale image to the new dimensions.
  void Rescale(const int new_width, const int new_height,
               const FREE_IMAGE_FILTER filter = FILTER_BILINEAR);

  // Clone the image to a new bitmap object.
  Bitmap Clone() const;
  Bitmap CloneAsGrey() const;
  Bitmap CloneAsRGB() const;

  // Clone metadata from this bitmap object to another target bitmap object.
  void CloneMetadata(Bitmap* target) const;

  // Read specific EXIF tag.
  bool ReadExifTag(const FREE_IMAGE_MDMODEL model, const std::string& tag_name,
                   std::string* result) const;

 private:
  typedef std::unique_ptr<FIBITMAP, decltype(&FreeImage_Unload)> FIBitmapPtr;

  void SetPtr(FIBITMAP* data);

  FIBitmapPtr data_;
  int width_;
  int height_;
  int channels_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

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

#endif  // COLMAP_SRC_UTIL_BITMAP_H_
