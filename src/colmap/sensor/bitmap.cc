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

#include "colmap/sensor/bitmap.h"

#include "colmap/math/math.h"
#include "colmap/sensor/database.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include "thirdparty/VLFeat/imopv.h"

#include <regex>
#include <unordered_map>

namespace colmap {
namespace {

#ifdef FREEIMAGE_LIB  // Only needed for static FreeImage.

struct FreeImageInitializer {
  FreeImageInitializer() { FreeImage_Initialise(); }
  ~FreeImageInitializer() { FreeImage_DeInitialise(); }
};

const static auto initializer = FreeImageInitializer();

#endif  // FREEIMAGE_LIB

}  // namespace

Bitmap::Bitmap()
    : data_(nullptr, &FreeImage_Unload), width_(0), height_(0), channels_(0) {}

Bitmap::Bitmap(const Bitmap& other) : Bitmap() {
  if (other.data_) {
    SetPtr(FreeImage_Clone(other.data_.get()));
  }
}

Bitmap::Bitmap(Bitmap&& other) noexcept : Bitmap() {
  data_ = std::move(other.data_);
  width_ = other.width_;
  height_ = other.height_;
  channels_ = other.channels_;
}

Bitmap::Bitmap(FIBITMAP* data) : Bitmap() { SetPtr(data); }

Bitmap& Bitmap::operator=(const Bitmap& other) {
  if (other.data_) {
    SetPtr(FreeImage_Clone(other.data_.get()));
  }
  return *this;
}

Bitmap& Bitmap::operator=(Bitmap&& other) noexcept {
  if (this != &other) {
    data_ = std::move(other.data_);
    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;
  }
  return *this;
}

bool Bitmap::Allocate(const int width, const int height, const bool as_rgb) {
  FIBITMAP* data = nullptr;
  width_ = width;
  height_ = height;
  if (as_rgb) {
    const int kNumBitsPerPixel = 24;
    data = FreeImage_Allocate(width, height, kNumBitsPerPixel);
    channels_ = 3;
  } else {
    const int kNumBitsPerPixel = 8;
    data = FreeImage_Allocate(width, height, kNumBitsPerPixel);
    channels_ = 1;
  }
  data_ = FIBitmapPtr(data, &FreeImage_Unload);
  return data != nullptr;
}

void Bitmap::Deallocate() {
  data_.reset();
  width_ = 0;
  height_ = 0;
  channels_ = 0;
}

size_t Bitmap::NumBytes() const {
  if (data_) {
    return ScanWidth() * height_;
  } else {
    return 0;
  }
}

std::vector<uint8_t> Bitmap::ConvertToRawBits() const {
  const unsigned int scan_width = ScanWidth();
  const unsigned int bpp = BitsPerPixel();
  const bool kTopDown = true;
  std::vector<uint8_t> raw_bits(scan_width * height_, 0);
  FreeImage_ConvertToRawBits(raw_bits.data(),
                             data_.get(),
                             scan_width,
                             bpp,
                             FI_RGBA_RED_MASK,
                             FI_RGBA_GREEN_MASK,
                             FI_RGBA_BLUE_MASK,
                             kTopDown);
  return raw_bits;
}

std::vector<uint8_t> Bitmap::ConvertToRowMajorArray() const {
  std::vector<uint8_t> array(width_ * height_ * channels_);
  size_t i = 0;
  for (int y = 0; y < height_; ++y) {
    const uint8_t* line = FreeImage_GetScanLine(data_.get(), height_ - 1 - y);
    for (int x = 0; x < width_; ++x) {
      for (int d = 0; d < channels_; ++d) {
        array[i] = line[x * channels_ + d];
        i += 1;
      }
    }
  }
  return array;
}

std::vector<uint8_t> Bitmap::ConvertToColMajorArray() const {
  std::vector<uint8_t> array(width_ * height_ * channels_);
  size_t i = 0;
  for (int d = 0; d < channels_; ++d) {
    for (int x = 0; x < width_; ++x) {
      for (int y = 0; y < height_; ++y) {
        const uint8_t* line =
            FreeImage_GetScanLine(data_.get(), height_ - 1 - y);
        array[i] = line[x * channels_ + d];
        i += 1;
      }
    }
  }
  return array;
}

bool Bitmap::GetPixel(const int x,
                      const int y,
                      BitmapColor<uint8_t>* color) const {
  if (x < 0 || x >= width_ || y < 0 || y >= height_) {
    return false;
  }

  const uint8_t* line = FreeImage_GetScanLine(data_.get(), height_ - 1 - y);

  if (IsGrey()) {
    color->r = line[x];
    return true;
  } else if (IsRGB()) {
    color->r = line[3 * x + FI_RGBA_RED];
    color->g = line[3 * x + FI_RGBA_GREEN];
    color->b = line[3 * x + FI_RGBA_BLUE];
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

  uint8_t* line = FreeImage_GetScanLine(data_.get(), height_ - 1 - y);

  if (IsGrey()) {
    line[x] = color.r;
    return true;
  } else if (IsRGB()) {
    line[3 * x + FI_RGBA_RED] = color.r;
    line[3 * x + FI_RGBA_GREEN] = color.g;
    line[3 * x + FI_RGBA_BLUE] = color.b;
    return true;
  }

  return false;
}

const uint8_t* Bitmap::GetScanline(const int y) const {
  CHECK_GE(y, 0);
  CHECK_LT(y, height_);
  return FreeImage_GetScanLine(data_.get(), height_ - 1 - y);
}

void Bitmap::Fill(const BitmapColor<uint8_t>& color) {
  for (int y = 0; y < height_; ++y) {
    uint8_t* line = FreeImage_GetScanLine(data_.get(), height_ - 1 - y);
    for (int x = 0; x < width_; ++x) {
      if (IsGrey()) {
        line[x] = color.r;
      } else if (IsRGB()) {
        line[3 * x + FI_RGBA_RED] = color.r;
        line[3 * x + FI_RGBA_GREEN] = color.g;
        line[3 * x + FI_RGBA_BLUE] = color.b;
      }
    }
  }
}

bool Bitmap::InterpolateNearestNeighbor(const double x,
                                        const double y,
                                        BitmapColor<uint8_t>* color) const {
  const int xx = static_cast<int>(std::round(x));
  const int yy = static_cast<int>(std::round(y));
  return GetPixel(xx, yy, color);
}

bool Bitmap::InterpolateBilinear(const double x,
                                 const double y,
                                 BitmapColor<float>* color) const {
  // FreeImage's coordinate system origin is in the lower left of the image.
  const double inv_y = height_ - 1 - y;

  const int x0 = static_cast<int>(std::floor(x));
  const int x1 = x0 + 1;
  const int y0 = static_cast<int>(std::floor(inv_y));
  const int y1 = y0 + 1;

  if (x0 < 0 || x1 >= width_ || y0 < 0 || y1 >= height_) {
    return false;
  }

  const double dx = x - x0;
  const double dy = inv_y - y0;
  const double dx_1 = 1 - dx;
  const double dy_1 = 1 - dy;

  const uint8_t* line0 = FreeImage_GetScanLine(data_.get(), y0);
  const uint8_t* line1 = FreeImage_GetScanLine(data_.get(), y1);

  if (IsGrey()) {
    // Top row, column-wise linear interpolation.
    const double v0 = dx_1 * line0[x0] + dx * line0[x1];

    // Bottom row, column-wise linear interpolation.
    const double v1 = dx_1 * line1[x0] + dx * line1[x1];

    // Row-wise linear interpolation.
    color->r = dy_1 * v0 + dy * v1;
    return true;
  } else if (IsRGB()) {
    const uint8_t* p00 = &line0[3 * x0];
    const uint8_t* p01 = &line0[3 * x1];
    const uint8_t* p10 = &line1[3 * x0];
    const uint8_t* p11 = &line1[3 * x1];

    // Top row, column-wise linear interpolation.
    const double v0_r = dx_1 * p00[FI_RGBA_RED] + dx * p01[FI_RGBA_RED];
    const double v0_g = dx_1 * p00[FI_RGBA_GREEN] + dx * p01[FI_RGBA_GREEN];
    const double v0_b = dx_1 * p00[FI_RGBA_BLUE] + dx * p01[FI_RGBA_BLUE];

    // Bottom row, column-wise linear interpolation.
    const double v1_r = dx_1 * p10[FI_RGBA_RED] + dx * p11[FI_RGBA_RED];
    const double v1_g = dx_1 * p10[FI_RGBA_GREEN] + dx * p11[FI_RGBA_GREEN];
    const double v1_b = dx_1 * p10[FI_RGBA_BLUE] + dx * p11[FI_RGBA_BLUE];

    // Row-wise linear interpolation.
    color->r = dy_1 * v0_r + dy * v1_r;
    color->g = dy_1 * v0_g + dy * v1_g;
    color->b = dy_1 * v0_b + dy * v1_b;
    return true;
  }

  return false;
}

bool Bitmap::ExifCameraModel(std::string* camera_model) const {
  // Read camera make and model
  std::string make_str;
  std::string model_str;
  std::string focal_length;
  *camera_model = "";
  if (ReadExifTag(FIMD_EXIF_MAIN, "Make", &make_str)) {
    *camera_model += (make_str + "-");
  } else {
    *camera_model = "";
    return false;
  }
  if (ReadExifTag(FIMD_EXIF_MAIN, "Model", &model_str)) {
    *camera_model += (model_str + "-");
  } else {
    *camera_model = "";
    return false;
  }
  if (ReadExifTag(FIMD_EXIF_EXIF, "FocalLengthIn35mmFilm", &focal_length) ||
      ReadExifTag(FIMD_EXIF_EXIF, "FocalLength", &focal_length)) {
    *camera_model += (focal_length + "-");
  } else {
    *camera_model = "";
    return false;
  }
  *camera_model += (std::to_string(width_) + "x" + std::to_string(height_));
  return true;
}

bool Bitmap::ExifFocalLength(double* focal_length) const {
  const double max_size = std::max(width_, height_);

  //////////////////////////////////////////////////////////////////////////////
  // Focal length in 35mm equivalent
  //////////////////////////////////////////////////////////////////////////////

  std::string focal_length_35mm_str;
  if (ReadExifTag(
          FIMD_EXIF_EXIF, "FocalLengthIn35mmFilm", &focal_length_35mm_str)) {
    const std::regex regex(".*?([0-9.]+).*?mm.*?");
    std::cmatch result;
    if (std::regex_search(focal_length_35mm_str.c_str(), result, regex)) {
      const double focal_length_35 = std::stold(result[1]);
      if (focal_length_35 > 0) {
        *focal_length = focal_length_35 / 35.0 * max_size;
        return true;
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Focal length in mm
  //////////////////////////////////////////////////////////////////////////////

  std::string focal_length_str;
  if (ReadExifTag(FIMD_EXIF_EXIF, "FocalLength", &focal_length_str)) {
    std::regex regex(".*?([0-9.]+).*?mm");
    std::cmatch result;
    if (std::regex_search(focal_length_str.c_str(), result, regex)) {
      const double focal_length_mm = std::stold(result[1]);

      // Lookup sensor width in database.
      std::string make_str;
      std::string model_str;
      if (ReadExifTag(FIMD_EXIF_MAIN, "Make", &make_str) &&
          ReadExifTag(FIMD_EXIF_MAIN, "Model", &model_str)) {
        CameraDatabase database;
        double sensor_width;
        if (database.QuerySensorWidth(make_str, model_str, &sensor_width)) {
          *focal_length = focal_length_mm / sensor_width * max_size;
          return true;
        }
      }

      // Extract sensor width from EXIF.
      std::string pixel_x_dim_str;
      std::string x_res_str;
      std::string res_unit_str;
      if (ReadExifTag(FIMD_EXIF_EXIF, "PixelXDimension", &pixel_x_dim_str) &&
          ReadExifTag(FIMD_EXIF_EXIF, "FocalPlaneXResolution", &x_res_str) &&
          ReadExifTag(
              FIMD_EXIF_EXIF, "FocalPlaneResolutionUnit", &res_unit_str)) {
        regex = std::regex(".*?([0-9.]+).*?");
        if (std::regex_search(pixel_x_dim_str.c_str(), result, regex)) {
          const double pixel_x_dim = std::stold(result[1]);
          regex = std::regex(".*?([0-9.]+).*?/.*?([0-9.]+).*?");
          if (std::regex_search(x_res_str.c_str(), result, regex)) {
            const double x_res = std::stold(result[2]) / std::stold(result[1]);
            // Use PixelXDimension instead of actual width of image, since
            // the image might have been resized, but the EXIF data preserved.
            const double ccd_width = x_res * pixel_x_dim;
            if (ccd_width > 0 && focal_length_mm > 0) {
              if (res_unit_str == "cm") {
                *focal_length = focal_length_mm / (ccd_width * 10.0) * max_size;
                return true;
              } else if (res_unit_str == "inches") {
                *focal_length = focal_length_mm / (ccd_width * 25.4) * max_size;
                return true;
              }
            }
          }
        }
      }
    }
  }

  return false;
}

bool Bitmap::ExifLatitude(double* latitude) const {
  std::string str;
  double sign = 1.0;
  if (ReadExifTag(FIMD_EXIF_GPS, "GPSLatitudeRef", &str)) {
    StringTrim(&str);
    StringToLower(&str);
    if (!str.empty() && str[0] == 's') {
      sign = -1.0;
    }
  }
  if (ReadExifTag(FIMD_EXIF_GPS, "GPSLatitude", &str)) {
    const std::regex regex(".*?([0-9.]+):([0-9.]+):([0-9.]+).*?");
    std::cmatch result;
    if (std::regex_search(str.c_str(), result, regex)) {
      const double hours = std::stold(result[1]);
      const double minutes = std::stold(result[2]);
      const double seconds = std::stold(result[3]);
      double value = hours + minutes / 60.0 + seconds / 3600.0;
      if (value > 0 && sign < 0) {
        value *= sign;
      }
      *latitude = value;
      return true;
    }
  }
  return false;
}

bool Bitmap::ExifLongitude(double* longitude) const {
  std::string str;
  double sign = 1.0;
  if (ReadExifTag(FIMD_EXIF_GPS, "GPSLongitudeRef", &str)) {
    StringTrim(&str);
    StringToLower(&str);
    if (!str.empty() && str[0] == 'w') {
      sign = -1.0;
    }
  }
  if (ReadExifTag(FIMD_EXIF_GPS, "GPSLongitude", &str)) {
    const std::regex regex(".*?([0-9.]+):([0-9.]+):([0-9.]+).*?");
    std::cmatch result;
    if (std::regex_search(str.c_str(), result, regex)) {
      const double hours = std::stold(result[1]);
      const double minutes = std::stold(result[2]);
      const double seconds = std::stold(result[3]);
      double value = hours + minutes / 60.0 + seconds / 3600.0;
      if (value > 0 && sign < 0) {
        value *= sign;
      }
      *longitude = value;
      return true;
    }
  }
  return false;
}

bool Bitmap::ExifAltitude(double* altitude) const {
  std::string str;
  if (ReadExifTag(FIMD_EXIF_GPS, "GPSAltitude", &str)) {
    const std::regex regex(".*?([0-9.]+).*?/.*?([0-9.]+).*?");
    std::cmatch result;
    if (std::regex_search(str.c_str(), result, regex)) {
      *altitude = std::stold(result[1]) / std::stold(result[2]);
      return true;
    }
  }
  return false;
}

bool Bitmap::Read(const std::string& path, const bool as_rgb) {
  if (!ExistsFile(path)) {
    return false;
  }

  const FREE_IMAGE_FORMAT format = FreeImage_GetFileType(path.c_str(), 0);

  if (format == FIF_UNKNOWN) {
    return false;
  }

  FIBITMAP* fi_bitmap = FreeImage_Load(format, path.c_str());
  if (fi_bitmap == nullptr) {
    return false;
  }

  data_ = FIBitmapPtr(fi_bitmap, &FreeImage_Unload);

  if (!IsPtrRGB(data_.get()) && as_rgb) {
    FIBITMAP* converted_bitmap = FreeImage_ConvertTo24Bits(fi_bitmap);
    data_ = FIBitmapPtr(converted_bitmap, &FreeImage_Unload);
  } else if (!IsPtrGrey(data_.get()) && !as_rgb) {
    FIBITMAP* converted_bitmap = FreeImage_ConvertToGreyscale(fi_bitmap);
    data_ = FIBitmapPtr(converted_bitmap, &FreeImage_Unload);
  }

  if (!IsPtrSupported(data_.get())) {
    data_.reset();
    return false;
  }

  width_ = FreeImage_GetWidth(data_.get());
  height_ = FreeImage_GetHeight(data_.get());
  channels_ = as_rgb ? 3 : 1;

  return true;
}

bool Bitmap::Write(const std::string& path,
                   const FREE_IMAGE_FORMAT format,
                   const int flags) const {
  FREE_IMAGE_FORMAT save_format;
  if (format == FIF_UNKNOWN) {
    save_format = FreeImage_GetFIFFromFilename(path.c_str());
    if (save_format == FIF_UNKNOWN) {
      // If format could not be deduced, save as PNG by default.
      save_format = FIF_PNG;
    }
  } else {
    save_format = format;
  }

  int save_flags = flags;
  if (save_format == FIF_JPEG && flags == 0) {
    // Use superb JPEG quality by default to avoid artifacts.
    save_flags = JPEG_QUALITYSUPERB;
  }

  bool success = false;
  if (save_flags == 0) {
    success = FreeImage_Save(save_format, data_.get(), path.c_str());
  } else {
    success =
        FreeImage_Save(save_format, data_.get(), path.c_str(), save_flags);
  }

  return success;
}

void Bitmap::Smooth(const float sigma_x, const float sigma_y) {
  std::vector<float> array(width_ * height_);
  std::vector<float> array_smoothed(width_ * height_);
  for (int d = 0; d < channels_; ++d) {
    size_t i = 0;
    for (int y = 0; y < height_; ++y) {
      const uint8_t* line = FreeImage_GetScanLine(data_.get(), height_ - 1 - y);
      for (int x = 0; x < width_; ++x) {
        array[i] = line[x * channels_ + d];
        i += 1;
      }
    }

    vl_imsmooth_f(array_smoothed.data(),
                  width_,
                  array.data(),
                  width_,
                  height_,
                  width_,
                  sigma_x,
                  sigma_y);

    i = 0;
    for (int y = 0; y < height_; ++y) {
      uint8_t* line = FreeImage_GetScanLine(data_.get(), height_ - 1 - y);
      for (int x = 0; x < width_; ++x) {
        line[x * channels_ + d] =
            TruncateCast<float, uint8_t>(array_smoothed[i]);
        i += 1;
      }
    }
  }
}

void Bitmap::Rescale(const int new_width,
                     const int new_height,
                     const FREE_IMAGE_FILTER filter) {
  SetPtr(FreeImage_Rescale(data_.get(), new_width, new_height, filter));
}

Bitmap Bitmap::Clone() const { return Bitmap(FreeImage_Clone(data_.get())); }

Bitmap Bitmap::CloneAsGrey() const {
  if (IsGrey()) {
    return Clone();
  } else {
    return Bitmap(FreeImage_ConvertToGreyscale(data_.get()));
  }
}

Bitmap Bitmap::CloneAsRGB() const {
  if (IsRGB()) {
    return Clone();
  } else {
    return Bitmap(FreeImage_ConvertTo24Bits(data_.get()));
  }
}

void Bitmap::CloneMetadata(Bitmap* target) const {
  CHECK_NOTNULL(target);
  CHECK_NOTNULL(target->Data());
  FreeImage_CloneMetadata(data_.get(), target->Data());
}

bool Bitmap::ReadExifTag(const FREE_IMAGE_MDMODEL model,
                         const std::string& tag_name,
                         std::string* result) const {
  FITAG* tag = nullptr;
  FreeImage_GetMetadata(model, data_.get(), tag_name.c_str(), &tag);
  if (tag == nullptr) {
    *result = "";
    return false;
  } else {
    if (tag_name == "FocalPlaneXResolution") {
      // This tag seems to be in the wrong category.
      *result = std::string(FreeImage_TagToString(FIMD_EXIF_INTEROP, tag));
    } else {
      *result = FreeImage_TagToString(model, tag);
    }
    return true;
  }
}

void Bitmap::SetPtr(FIBITMAP* data) {
  if (!IsPtrSupported(data)) {
    FIBITMAP* temp_data = data;
    data = FreeImage_ConvertTo24Bits(temp_data);
    FreeImage_Unload(temp_data);
  }

  data_ = FIBitmapPtr(data, &FreeImage_Unload);
  width_ = FreeImage_GetWidth(data);
  height_ = FreeImage_GetHeight(data);
  channels_ = IsPtrRGB(data) ? 3 : 1;
}

bool Bitmap::IsPtrGrey(FIBITMAP* data) {
  return FreeImage_GetColorType(data) == FIC_MINISBLACK &&
         FreeImage_GetBPP(data) == 8;
}

bool Bitmap::IsPtrRGB(FIBITMAP* data) {
  return FreeImage_GetColorType(data) == FIC_RGB &&
         FreeImage_GetBPP(data) == 24;
}

bool Bitmap::IsPtrSupported(FIBITMAP* data) {
  return IsPtrGrey(data) || IsPtrRGB(data);
}

float JetColormap::Red(const float gray) { return Base(gray - 0.25f); }

float JetColormap::Green(const float gray) { return Base(gray); }

float JetColormap::Blue(const float gray) { return Base(gray + 0.25f); }

float JetColormap::Base(const float val) {
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (val <= 0.125f) {
    return 0.0f;
  } else if (val <= 0.375f) {
    return Interpolate(2.0f * val - 1.0f, 0.0f, -0.75f, 1.0f, -0.25f);
  } else if (val <= 0.625f) {
    return 1.0f;
  } else if (val <= 0.87f) {
    return Interpolate(2.0f * val - 1.0f, 1.0f, 0.25f, 0.0f, 0.75f);
  } else {
    return 0.0f;
  }
}

float JetColormap::Interpolate(const float val,
                               const float y0,
                               const float x0,
                               const float y1,
                               const float x1) {
  return (val - x0) * (y1 - y0) / (x1 - x0) + y0;
}

}  // namespace colmap
