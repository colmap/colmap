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

#include "colmap/sensor/bitmap.h"

#include "colmap/math/math.h"
#include "colmap/sensor/database.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/oiio_utils.h"

#include <OpenImageIO/color.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>

namespace colmap {
namespace {

struct OIIOMetaData : public Bitmap::MetaData {
  OIIOMetaData() = default;

  OIIO::ImageSpec image_spec;

  static OIIOMetaData* Upcast(Bitmap::MetaData* meta_data) {
    return THROW_CHECK_NOTNULL(dynamic_cast<OIIOMetaData*>(meta_data));
  }

  static std::unique_ptr<MetaData> Clone(
      const std::unique_ptr<Bitmap::MetaData>& meta_data) {
    auto cloned = std::make_unique<OIIOMetaData>();
    *cloned = *Upcast(meta_data.get());
    return cloned;
  }
};

// For backwards compatibility with older OIIO versions without implicit
// conversion from std::string_view.
OIIO::string_view OIIOFromStdStringView(std::string_view value) {
  return {value.data(), value.size()};
}

std::vector<uint8_t> ConvertColorSpace(const uint8_t* src_data,
                                       int width,
                                       int height,
                                       int channels,
                                       const std::string_view& from,
                                       const std::string_view& to) {
  EnsureOpenImageIOInitialized();
  const OIIO::ImageSpec image_spec(
      width, height, channels, OIIO::TypeDesc::UINT8);
  const int pitch = width * channels;
  const OIIO::ImageBuf src(image_spec, const_cast<uint8_t*>(src_data));
  std::vector<uint8_t> tgt_data(height * pitch);
  OIIO::ImageBuf tgt(image_spec, tgt_data.data());
  THROW_CHECK(OIIO::ImageBufAlgo::colorconvert(
      tgt, src, OIIOFromStdStringView(from), OIIOFromStdStringView(to)));
  return tgt_data;
}

void SetImageSpecColorSpace(OIIO::ImageSpec& image_spec,
                            const OIIO::string_view& colorspace) {
  EnsureOpenImageIOInitialized();
#if OIIO_VERSION >= OIIO_MAKE_VERSION(3, 0, 0)
  image_spec.set_colorspace(colorspace);
#else
  // Extract logic from 3.0.0 version for backwards compatibility.
  const OIIO::string_view oldspace =
      image_spec.get_string_attribute("oiio:ColorSpace");
  if (oldspace.size() && colorspace.size() && oldspace == colorspace) {
    return;
  }

  if (colorspace.empty()) {
    image_spec.erase_attribute("oiio:ColorSpace");
  } else {
    image_spec.attribute("oiio:ColorSpace", colorspace);
  }

  if (colorspace != "sRGB") {
    image_spec.erase_attribute("Exif:ColorSpace");
  }

  image_spec.erase_attribute("tiff:ColorSpace");
  image_spec.erase_attribute("tiff:PhotometricInterpretation");
  image_spec.erase_attribute("oiio:Gamma");
#endif
}

bool IsEquivalentColorSpace(const std::string_view& colorspace1,
                            const std::string_view& colorspace2) {
  EnsureOpenImageIOInitialized();
#if OIIO_VERSION >= OIIO_MAKE_VERSION(3, 0, 0)
  return OIIO::equivalent_colorspace(colorspace1, colorspace2);
#else
  // Poor (wo)man's version of available functionality in recent OIIO versions.
  auto is_linear_srgb = [](const std::string_view& colorspace) {
    return colorspace == "linear" || colorspace == "lin_srgb" ||
           colorspace == "lin_rec709P";
  };
  if (is_linear_srgb(colorspace1) && is_linear_srgb(colorspace2)) {
    return true;
  } else {
    return colorspace1 == colorspace2;
  }
#endif
}

}  // namespace

Bitmap::Bitmap()
    : width_(0), height_(0), channels_(0), linear_colorspace_(true) {
  EnsureOpenImageIOInitialized();
}

Bitmap::Bitmap(const int width,
               const int height,
               const bool as_rgb,
               const bool linear_colorspace) {
  EnsureOpenImageIOInitialized();
  width_ = width;
  height_ = height;
  channels_ = as_rgb ? 3 : 1;
  linear_colorspace_ = linear_colorspace;
  data_.resize(width_ * height_ * channels_);
  auto meta_data = std::make_unique<OIIOMetaData>();
  meta_data->image_spec =
      OIIO::ImageSpec(width_, height_, channels_, OIIO::TypeDesc::UINT8);
  SetImageSpecColorSpace(meta_data->image_spec,
                         linear_colorspace ? "linear" : "sRGB");
  meta_data_ = std::move(meta_data);
}

Bitmap::Bitmap(const Bitmap& other) {
  width_ = other.width_;
  height_ = other.height_;
  channels_ = other.channels_;
  linear_colorspace_ = other.linear_colorspace_;
  data_ = other.data_;
  meta_data_ = OIIOMetaData::Clone(other.meta_data_);
}

Bitmap::Bitmap(Bitmap&& other) noexcept {
  width_ = other.width_;
  height_ = other.height_;
  channels_ = other.channels_;
  linear_colorspace_ = other.linear_colorspace_;
  data_ = std::move(other.data_);
  meta_data_ = std::move(other.meta_data_);
  other.width_ = 0;
  other.height_ = 0;
  other.channels_ = 0;
}

Bitmap& Bitmap::operator=(const Bitmap& other) {
  width_ = other.width_;
  height_ = other.height_;
  channels_ = other.channels_;
  linear_colorspace_ = other.linear_colorspace_;
  data_ = other.data_;
  meta_data_ = OIIOMetaData::Clone(other.meta_data_);
  return *this;
}

Bitmap& Bitmap::operator=(Bitmap&& other) noexcept {
  if (this != &other) {
    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;
    linear_colorspace_ = other.linear_colorspace_;
    data_ = std::move(other.data_);
    meta_data_ = std::move(other.meta_data_);
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
  }
  return *this;
}

void Bitmap::Fill(const BitmapColor<uint8_t>& color) {
  if (IsGrey()) {
    std::fill(data_.begin(), data_.end(), color.r);
  } else {
    THROW_CHECK_EQ(data_.size() % 3, 0);
    size_t i = 0;
    while (i < data_.size()) {
      data_[i++] = color.r;
      data_[i++] = color.g;
      data_[i++] = color.b;
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
  const int x0 = static_cast<int>(std::floor(x));
  const int x1 = x0 + 1;
  const int y0 = static_cast<int>(std::floor(y));
  const int y1 = y0 + 1;

  if (x0 < 0 || x1 >= width_ || y0 < 0 || y1 >= height_) {
    return false;
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
    color->r = dy_1 * v0 + dy * v1;
    return true;
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
    color->r = dy_1 * v0_r + dy * v1_r;
    color->g = dy_1 * v0_g + dy * v1_g;
    color->b = dy_1 * v0_b + dy * v1_b;
    return true;
  }

  return false;
}

bool Bitmap::ExifCameraModel(std::string* camera_model) const {
  // Read camera make and model
  std::string_view make_str;
  std::string_view model_str;
  float focal_length = 0;
  *camera_model = "";
  if (GetMetaData("Make", &make_str)) {
    *camera_model += std::string(make_str) + "-";
  } else {
    *camera_model = "";
    return false;
  }
  if (GetMetaData("Model", &model_str)) {
    *camera_model += std::string(model_str) + "-";
  } else {
    *camera_model = "";
    return false;
  }
  if (GetMetaData("Exif:FocalLengthIn35mmFilm", "float", &focal_length) ||
      GetMetaData("Exif:FocalLength", "float", &focal_length)) {
    *camera_model += std::to_string(focal_length) + "-";
  } else {
    *camera_model = "";
    return false;
  }
  *camera_model += std::to_string(width_) + "x" + std::to_string(height_);
  return true;
}

bool Bitmap::ExifFocalLength(double* focal_length) const {
  const double max_size = std::max(width_, height_);

  float focal_length_35mm = 0;
  if (GetMetaData("Exif:FocalLengthIn35mmFilm", "float", &focal_length_35mm)) {
    if (focal_length_35mm > 0) {
      // Based on https://en.wikipedia.org/wiki/35_mm_equivalent_focal_length
      // According to CIPA guidelines, 35 mm equivalent focal length is to be
      // calculated like this:
      // "focal length in 35 mm camera" =
      //   (Diagonal distance of image area in the 35 mm camera (43.27 mm) /
      //    Diagonal distance of image area on the image sensor of the DSC)
      //    * focal length of the lens of the DSC.
      const double diagonal = std::sqrt(width_ * width_ + height_ * height_);
      *focal_length = focal_length_35mm / 43.27 * diagonal;
      return true;
    }
  }

  float focal_length_mm = 0.f;
  if (GetMetaData("Exif:FocalLength", "float", &focal_length_mm)) {
    float focal_x_res = 0.f;
    int focal_x_res_unit = 0;
    if (GetMetaData("Exif:FocalPlaneXResolution", "float", &focal_x_res) &&
        GetMetaData(
            "Exif:FocalPlaneResolutionUnit", "int", &focal_x_res_unit)) {
      if (focal_length_mm > 0 && focal_x_res_unit > 1 &&
          focal_x_res_unit <= 5) {
        double pixels_per_mm = 0;
        switch (focal_x_res_unit) {
          case 2:  // inches
            pixels_per_mm = focal_x_res * 25.4;
            break;
          case 3:  // cm
            pixels_per_mm = focal_x_res * 10.0;
            break;
          case 4:  // mm
            pixels_per_mm = focal_x_res * 1.0;
            break;
          case 5:  // um
            pixels_per_mm = focal_x_res * 0.1;
            break;
          default:
            LOG(FATAL) << "Unexpected FocalPlaneXResolution value";
        }
        *focal_length = focal_length_mm / pixels_per_mm;
        return true;
      }
    }

    // Lookup sensor width in database.
    std::string_view make_str;
    std::string_view model_str;
    if (GetMetaData("Make", &make_str) && GetMetaData("Model", &model_str)) {
      CameraDatabase database;
      double sensor_width_mm;
      if (database.QuerySensorWidth(std::string(make_str),
                                    std::string(model_str),
                                    &sensor_width_mm)) {
        *focal_length = focal_length_mm / sensor_width_mm * max_size;
        return true;
      }
    }
  }

  return false;
}

bool Bitmap::ExifLatitude(double* latitude) const {
  std::string_view latitude_ref;
  double sign = 1.0;
  if (GetMetaData("GPS:LatitudeRef", &latitude_ref)) {
    if (latitude_ref == "N" || latitude_ref == "n") {
      sign = 1.0;
    } else if (latitude_ref == "S" || latitude_ref == "s") {
      sign = -1.0;
    }
  }
  float deg_min_sec[3] = {0.0};
  if (GetMetaData("GPS:Latitude", "point", &deg_min_sec)) {
    *latitude =
        deg_min_sec[0] + deg_min_sec[1] / 60.0 + deg_min_sec[2] / 3600.0;
    if (*latitude > 0 && sign < 0) {
      *latitude *= sign;
    }
    return true;
  }
  return false;
}

bool Bitmap::ExifLongitude(double* longitude) const {
  std::string_view longitude_ref;
  double sign = 1.0;
  if (GetMetaData("GPS:LongitudeRef", &longitude_ref)) {
    if (longitude_ref == "W" || longitude_ref == "w") {
      sign = 1.0;
    } else if (longitude_ref == "E" || longitude_ref == "e") {
      sign = -1.0;
    }
  }
  float deg_min_sec[3] = {0.0};
  if (GetMetaData("GPS:Longitude", "point", &deg_min_sec)) {
    *longitude =
        deg_min_sec[0] + deg_min_sec[1] / 60.0 + deg_min_sec[2] / 3600.0;
    if (*longitude > 0 && sign < 0) {
      *longitude *= sign;
    }
    return true;
  }
  return false;
}

bool Bitmap::ExifAltitude(double* altitude) const {
  std::string_view altitude_ref;
  double sign = 1.0;
  if (GetMetaData("GPS:AltitudeRef", &altitude_ref)) {
    if (altitude_ref == "0") {
      sign = 1.0;
    } else if (altitude_ref == "1") {
      sign = -1.0;
    }
  }
  float altitude_float = 0.f;
  if (GetMetaData("GPS:Altitude", "float", &altitude_float)) {
    *altitude = altitude_float;
    if (*altitude > 0 && sign < 0) {
      *altitude *= sign;
    }
    return true;
  }
  return false;
}

bool Bitmap::Read(const std::string& path,
                  const bool as_rgb,
                  const bool linearize_colorspace) {
  if (!ExistsFile(path)) {
    VLOG(3) << "Failed to read bitmap, because file does not exist";
    return false;
  }

  OIIO::ImageSpec config;
  config["oiio:reorient"] = 0;

  const auto input = OIIO::ImageInput::open(path, &config);
  if (!input) {
    VLOG(3) << "Failed to read bitmap specs";
    return false;
  }

  const OIIO::ImageSpec& image_spec = input->spec();
  width_ = image_spec.width;
  height_ = image_spec.height;
  channels_ = image_spec.nchannels;
  if (channels_ != 1 && channels_ != 3) {
    VLOG(3) << "Bitmap is not grayscale or RGB";
    return false;
  }

  data_.resize(width_ * height_ * channels_);
  input->read_image(0, 0, 0, channels_, OIIO::TypeDesc::UINT8, data_.data());
  input->close();

  auto meta_data = std::make_unique<OIIOMetaData>();
  meta_data->image_spec = image_spec;
  meta_data_ = std::move(meta_data);

  if (linearize_colorspace) {
    const std::string colorspace = image_spec["oiio:ColorSpace"];
    if (IsEquivalentColorSpace(colorspace, "linear")) {
      data_ = ConvertColorSpace(
          data_.data(), width_, height_, channels_, colorspace, "linear");
    }
  }

  if (as_rgb && channels_ != 3) {
    *this = CloneAsRGB();
  } else if (!as_rgb && channels_ != 1) {
    *this = CloneAsGrey();
  }

  return true;
}

bool Bitmap::Write(const std::string& path,
                   const bool delinearize_colorspace) const {
  const auto output = OIIO::ImageOutput::create(path);
  if (!output) {
    std::cerr << "Could not create an ImageOutput for " << path
              << ", error = " << OIIO::geterror() << "\n";
    return false;
  }

  auto* meta_data = OIIOMetaData::Upcast(meta_data_.get());

  const uint8_t* output_data_ptr = data_.data();
  std::vector<uint8_t> maybe_linearized_output_data;
  if (delinearize_colorspace && linear_colorspace_) {
    std::string_view colorspace;
    if (!GetMetaData("oiio:ColorSpace", &colorspace)) {
      // Assume sRGB color space if not specified.
      colorspace = "sRGB";
      SetImageSpecColorSpace(meta_data->image_spec,
                             OIIOFromStdStringView(colorspace));
    }

    maybe_linearized_output_data = ConvertColorSpace(
        data_.data(), width_, height_, channels_, "linear", colorspace);
    output_data_ptr = maybe_linearized_output_data.data();
  }

  if (HasFileExtension(path, ".jpg") || HasFileExtension(path, ".jpeg")) {
    std::string_view compression;
    if (!GetMetaData("Compression", &compression)) {
      // Save JPEG in superb quality by default to reduce compression artifacts.
      meta_data->image_spec["Compression"] = "jpeg:100";
    }
  }

  if (!output->open(path, meta_data->image_spec)) {
    VLOG(3) << "Could not open " << path << ", error = " << output->geterror()
            << "\n";
    return false;
  }

  if (!output->write_image(OIIO::TypeDesc::UINT8, output_data_ptr)) {
    VLOG(3) << "Could not write pixels to " << path
            << ", error = " << output->geterror() << "\n";
    return false;
  }

  if (!output->close()) {
    VLOG(3) << "Error closing " << path << ", error = " << output->geterror()
            << "\n";
    return false;
  }

  return true;
}

void Bitmap::Rescale(const int new_width,
                     const int new_height,
                     RescaleFilter filter) {
  const OIIO::ImageBuf buf(
      OIIO::ImageSpec(width_, height_, channels_, OIIO::TypeDesc::UINT8),
      data_.data());
  std::vector<uint8_t> new_data(new_width * new_height * channels_);
  OIIO::ImageBuf new_buf(
      OIIO::ImageSpec(new_width, new_height, channels_, OIIO::TypeDesc::UINT8),
      new_data.data());
  THROW_CHECK(OIIO::ImageBufAlgo::resize(new_buf, buf));

  width_ = new_width;
  height_ = new_height;
  data_ = std::move(new_data);
  auto* meta_data = OIIOMetaData::Upcast(meta_data_.get());
  meta_data->image_spec.width = new_width;
  meta_data->image_spec.height = new_height;
}

Bitmap Bitmap::Clone() const { return *this; }

Bitmap Bitmap::CloneAsGrey() const {
  if (IsGrey()) {
    return Clone();
  } else {
    Bitmap cloned;
    cloned.width_ = width_;
    cloned.height_ = height_;
    cloned.channels_ = 1;
    cloned.linear_colorspace_ = linear_colorspace_;
    cloned.data_.resize(width_ * height_);
    for (size_t i = 0; i < cloned.data_.size(); ++i) {
      cloned.data_[i] =
          std::round(.2126f * data_[3 * i + 0] + .7152f * data_[3 * i + 1] +
                     .0722f * data_[3 * i + 2]);
    }
    cloned.meta_data_ = OIIOMetaData::Clone(meta_data_);
    return cloned;
  }
}

Bitmap Bitmap::CloneAsRGB() const {
  if (IsRGB()) {
    return Clone();
  } else {
    THROW_CHECK_EQ(channels_, 1);
    Bitmap cloned;
    cloned.width_ = width_;
    cloned.height_ = height_;
    cloned.channels_ = 3;
    cloned.linear_colorspace_ = linear_colorspace_;
    cloned.data_.resize(width_ * height_ * 3);
    for (size_t i = 0; i < data_.size(); ++i) {
      cloned.data_[3 * i + 0] = data_[i];
      cloned.data_[3 * i + 1] = data_[i];
      cloned.data_[3 * i + 2] = data_[i];
    }
    cloned.meta_data_ = OIIOMetaData::Clone(meta_data_);
    return cloned;
  }
}

void Bitmap::SetMetaData(const std::string_view& name,
                         const std::string_view& type,
                         const void* value) {
  THROW_CHECK_NE(type, "string");
  auto* meta_data = OIIOMetaData::Upcast(meta_data_.get());
  OIIO::TypeDesc type_desc;
  type_desc.fromstring(OIIOFromStdStringView(type));
  THROW_CHECK_NE(type_desc, OIIO::TypeDesc::UNKNOWN);
  meta_data->image_spec.attribute(
      OIIOFromStdStringView(name), type_desc, value);
}

void Bitmap::SetMetaData(const std::string_view& name,
                         const std::string_view& value) {
  auto* meta_data = OIIOMetaData::Upcast(meta_data_.get());
  meta_data->image_spec.attribute(OIIOFromStdStringView(name),
                                  OIIOFromStdStringView(value));
}

bool Bitmap::GetMetaData(const std::string_view& name,
                         const std::string_view& type,
                         void* value) const {
  THROW_CHECK_NE(type, "string");
  auto* meta_data = OIIOMetaData::Upcast(meta_data_.get());
  OIIO::TypeDesc type_desc;
  type_desc.fromstring(OIIOFromStdStringView(type));
  THROW_CHECK_NE(type_desc, OIIO::TypeDesc::UNKNOWN);
  return meta_data->image_spec.getattribute(
      OIIOFromStdStringView(name), type_desc, value);
}

bool Bitmap::GetMetaData(const std::string_view& name,
                         std::string_view* value) const {
  auto* meta_data = OIIOMetaData::Upcast(meta_data_.get());
  OIIO::ustring ustring_value;
  if (meta_data->image_spec.getattribute(
          OIIOFromStdStringView(name), OIIO::TypeString, &ustring_value)) {
    *value = std::string_view(ustring_value.data(), ustring_value.size());
    return true;
  }
  return false;
}

void Bitmap::CloneMetadata(Bitmap* target) const {
  THROW_CHECK_NOTNULL(target);
  target->meta_data_ = OIIOMetaData::Clone(meta_data_);
  auto* target_meta_data = OIIOMetaData::Upcast(target->meta_data_.get());
  target_meta_data->image_spec.width = target->Width();
  target_meta_data->image_spec.height = target->Height();
}

std::ostream& operator<<(std::ostream& stream, const Bitmap& bitmap) {
  stream << "Bitmap(width=" << bitmap.Width() << ", height=" << bitmap.Height()
         << ", channels=" << bitmap.Channels() << ")";
  return stream;
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
