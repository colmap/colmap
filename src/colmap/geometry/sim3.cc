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

#include "colmap/geometry/sim3.h"

#include "colmap/util/logging.h"

#include <fstream>

namespace colmap {

void Sim3d::ToFile(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc);
  CHECK(file.good()) << path;
  // Ensure that we don't loose any precision by storing in text.
  file.precision(17);
  file << scale << " " << rotation.w() << " " << rotation.x() << " "
       << rotation.y() << " " << rotation.z() << " " << translation.x() << " "
       << translation.y() << " " << translation.z() << std::endl;
}

Sim3d Sim3d::FromFile(const std::string& path) {
  std::ifstream file(path);
  CHECK(file.good()) << path;
  Sim3d t;
  file >> t.scale;
  file >> t.rotation.w();
  file >> t.rotation.x();
  file >> t.rotation.y();
  file >> t.rotation.z();
  file >> t.translation(0);
  file >> t.translation(1);
  file >> t.translation(2);
  return t;
}

}  // namespace colmap
