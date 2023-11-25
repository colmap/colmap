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

#include "colmap/sensor/database.h"

#include "colmap/util/string.h"

namespace colmap {

const camera_specs_t CameraDatabase::specs_ = InitializeCameraSpecs();

bool CameraDatabase::QuerySensorWidth(const std::string& make,
                                      const std::string& model,
                                      double* sensor_width) {
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
  for (const auto& make_elem : specs_) {
    if (StringContains(cleaned_make, make_elem.first) ||
        StringContains(make_elem.first, cleaned_make)) {
      for (const auto& model_elem : make_elem.second) {
        if (StringContains(cleaned_model, model_elem.first) ||
            StringContains(model_elem.first, cleaned_model)) {
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
