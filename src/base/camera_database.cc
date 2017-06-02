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

#include "base/camera_database.h"

#include "util/string.h"

namespace colmap {

const camera_specs_t CameraDatabase::specs_ = InitializeCameraSpecs();

CameraDatabase::CameraDatabase() {}

bool CameraDatabase::QuerySensorWidth(const std::string& make,
                                      const std::string& model,
                                      double* sensor_width) {
  // Clean the strings from all separators.
  std::string cleaned_make = make;
  std::string cleaned_model = model;
  StringReplace(cleaned_make, " ", "");
  StringReplace(cleaned_model, " ", "");
  StringReplace(cleaned_make, "-", "");
  StringReplace(cleaned_model, "-", "");
  StringToLower(&cleaned_make);
  StringToLower(&cleaned_model);

  // Make sure that make name is not duplicated.
  cleaned_model = StringReplace(cleaned_model, cleaned_make, "");

  // Check if cleaned_make exists in database: Test whether EXIF string is
  // substring of database entry and vice versa.
  size_t spec_matches = 0;
  for (const auto& make_elem : specs_) {
    if (cleaned_make.find(make_elem.first) != std::string::npos ||
        make_elem.first.find(cleaned_make) != std::string::npos) {
      for (const auto& model_elem : make_elem.second) {
        if (cleaned_model.find(model_elem.first) != std::string::npos ||
            model_elem.first.find(cleaned_model) != std::string::npos) {
          *sensor_width = model_elem.second;
          if (cleaned_model == model_elem.first) {
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
