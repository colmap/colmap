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

#ifndef COLMAP_SRC_UTIL_CAMERA_SPECS_H_
#define COLMAP_SRC_UTIL_CAMERA_SPECS_H_

#include <string>
#include <unordered_map>
#include <vector>

namespace colmap {

// { make1 : ({ model1 : sensor-width in mm }, ...), ... }
typedef std::vector<std::pair<std::string, float>> camera_make_specs_t;
typedef std::unordered_map<std::string, camera_make_specs_t> camera_specs_t;

camera_specs_t InitializeCameraSpecs();

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_CAMERA_SPECS_H_
