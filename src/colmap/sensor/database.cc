// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sensor/database.h"

#include "colmap/util/string.h"

namespace colmap {

const camera_specs_t CameraDatabase::specs_ = InitializeCameraSpecs();

bool CameraDatabase::QuerySensorWidth(const std::string& make,
                                      const std::string& model,
                                      double* sensor_width_mm) {
  // Clean the strings from all separators.
  std::string cleaned_make = make;
  std::string cleaned_model = model;
  cleaned_make = StringReplace(cleaned_make, " ", "");
  cleaned_model = StringReplace(cleaned_model, " ", "");
  cleaned_make = StringReplace(cleaned_make, "-", "");
  cleaned_model = StringReplace(cleaned_model, "-", "");
  StringToLower(&cleaned_make);
  StringToLower(&cleaned_model);

  // Make sure that make name is not duplicated.
  cleaned_model = StringReplace(cleaned_model, cleaned_make, "");

  // Check if cleaned_make exists in database: Test whether EXIF string is
  // substring of database entry and vice versa.
  size_t spec_matches = 0;
  for (const auto& [make_name, models] : specs_) {
    if (StringContains(cleaned_make, make_name) ||
        StringContains(make_name, cleaned_make)) {
      for (const auto& [model_name, sensor_width] : models) {
        if (StringContains(cleaned_model, model_name) ||
            StringContains(model_name, cleaned_model)) {
          *sensor_width_mm = sensor_width;
          if (cleaned_model == model_name) {
            // Model exactly matches, return immediately.
            return true;
          }
          spec_matches += 1;
          if (spec_matches > 1) {
            break;
          }
        }
      }
    }
  }

  // Only return unique results, if model does not exactly match.
  return spec_matches == 1;
}

}  // namespace colmap
