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

#include "util/bitmap.h"

#include <unordered_map>

#include <boost/filesystem/operations.hpp>
#include <boost/regex.hpp>

#include "base/camera_database.h"
#include "ext/VLFeat/imopv.h"
#include "util/logging.h"
#include "util/math.h"
#include "util/misc.h"

namespace colmap {

Bitmap::Bitmap()
    : data_(nullptr, &FreeImage_Unload), width_(0), height_(0), channels_(0) {}

Bitmap::Bitmap(const Bitmap& other) : Bitmap() {
  if (other.data_) {
    SetPtr(FreeImage_Clone(other.data_.get()));
  }
}

Bitmap::Bitmap(Bitmap&& other) : Bitmap() {
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

Bitmap& Bitmap::operator=(Bitmap&& other) {
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
  FreeImage_ConvertToRawBits(raw_bits.data(), data_.get(), scan_width, bpp,
                             FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK,
                             FI_RGBA_BLUE_MASK, kTopDown);
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

bool Bitmap::GetPixel(const int x, const int y,
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

bool Bitmap::SetPixel(const int x, const int y,
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

bool Bitmap::InterpolateNearestNeighbor(const double x, const double y,
                                        BitmapColor<uint8_t>* color) const {
  const int xx = static_cast<int>(std::round(x));
  const int yy = static_cast<int>(std::round(y));
  return GetPixel(xx, yy, color);
}

bool Bitmap::InterpolateBilinear(const double x, const double y,
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

bool Bitmap::ExifFocalLength(double* focal_length) {
  const double max_size = std::max(width_, height_);

  //////////////////////////////////////////////////////////////////////////////
  // Focal length in 35mm equivalent
  //////////////////////////////////////////////////////////////////////////////

  std::string focal_length_35mm_str;
  if (ReadExifTag(FIMD_EXIF_EXIF, "FocalLengthIn35mmFilm",
                  &focal_length_35mm_str)) {
    const boost::regex regex(".*?([0-9.]+).*?mm.*?");
    boost::cmatch result;
    if (boost::regex_search(focal_length_35mm_str.c_str(), result, regex)) {
      const double focal_length_35 = boost::lexical_cast<double>(result[1]);
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
    boost::regex regex(".*?([0-9.]+).*?mm");
    boost::cmatch result;
    if (boost::regex_search(focal_length_str.c_str(), result, regex)) {
      const double focal_length_mm = boost::lexical_cast<double>(result[1]);

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
          ReadExifTag(FIMD_EXIF_EXIF, "FocalPlaneResolutionUnit",
                      &res_unit_str)) {
        regex = boost::regex(".*?([0-9.]+).*?");
        if (boost::regex_search(pixel_x_dim_str.c_str(), result, regex)) {
          const double pixel_x_dim = boost::lexical_cast<double>(result[1]);
          regex = boost::regex(".*?([0-9.]+).*?/.*?([0-9.]+).*?");
          if (boost::regex_search(x_res_str.c_str(), result, regex)) {
            const double x_res = boost::lexical_cast<double>(result[2]) /
                                 boost::lexical_cast<double>(result[1]);
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

bool Bitmap::ExifLatitude(double* latitude) {
  std::string str;
  if (ReadExifTag(FIMD_EXIF_GPS, "GPSLatitude", &str)) {
    const boost::regex regex(".*?([0-9.]+):([0-9.]+):([0-9.]+).*?");
    boost::cmatch result;
    if (boost::regex_search(str.c_str(), result, regex)) {
      const double hours = boost::lexical_cast<double>(result[1]);
      const double minutes = boost::lexical_cast<double>(result[2]);
      const double seconds = boost::lexical_cast<double>(result[3]);
      *latitude = hours + minutes / 60.0 + seconds / 3600.0;
      return true;
    }
  }
  return false;
}

bool Bitmap::ExifLongitude(double* longitude) {
  std::string str;
  if (ReadExifTag(FIMD_EXIF_GPS, "GPSLongitude", &str)) {
    const boost::regex regex(".*?([0-9.]+):([0-9.]+):([0-9.]+).*?");
    boost::cmatch result;
    if (boost::regex_search(str.c_str(), result, regex)) {
      const double hours = boost::lexical_cast<double>(result[1]);
      const double minutes = boost::lexical_cast<double>(result[2]);
      const double seconds = boost::lexical_cast<double>(result[3]);
      *longitude = hours + minutes / 60.0 + seconds / 3600.0;
      return true;
    }
  }
  return false;
}

bool Bitmap::ExifAltitude(double* altitude) {
  std::string str;
  if (ReadExifTag(FIMD_EXIF_GPS, "GPSAltitude", &str)) {
    const boost::regex regex(".*?([0-9.]+).*?/.*?([0-9.]+).*?");
    boost::cmatch result;
    if (boost::regex_search(str.c_str(), result, regex)) {
      *altitude = boost::lexical_cast<double>(result[1]) /
                  boost::lexical_cast<double>(result[2]);
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

bool Bitmap::Write(const std::string& path, const FREE_IMAGE_FORMAT format,
                   const int flags) const {
  FREE_IMAGE_FORMAT save_format;
  if (format == FIF_UNKNOWN) {
    save_format = FreeImage_GetFIFFromFilename(path.c_str());
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

    vl_imsmooth_f(array_smoothed.data(), width_, array.data(), width_, height_,
                  width_, sigma_x, sigma_y);

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

void Bitmap::Rescale(const int new_width, const int new_height,
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
  CHECK(IsPtrSupported(data));

  data_ = FIBitmapPtr(data, &FreeImage_Unload);
  width_ = FreeImage_GetWidth(data);
  height_ = FreeImage_GetHeight(data);

  if (!IsPtrGrey(data) && !IsPtrRGB(data)) {
    FIBITMAP* data_converted = FreeImage_ConvertTo24Bits(data);
    data_ = FIBitmapPtr(data_converted, &FreeImage_Unload);
    channels_ = 3;
  } else {
    channels_ = IsPtrRGB(data) ? 3 : 1;
  }
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

float JetColormap::Interpolate(const float val, const float y0, const float x0,
                               const float y1, const float x1) {
  return (val - x0) * (y1 - y0) / (x1 - x0) + y0;
}

}  // namespace colmap
