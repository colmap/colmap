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

#include "util/version.h"

namespace colmap {

std::string GetVersionInfo() {
  return StringPrintf("COLMAP %s", COLMAP_VERSION.c_str());
}

std::string GetBuildInfo() {
#ifdef CUDA_ENABLED
  const std::string cuda_info = "with CUDA";
#else
  const std::string cuda_info = "without CUDA";
#endif
  return StringPrintf("Commit %s on %s %s", COLMAP_COMMIT_ID.c_str(),
                      COLMAP_COMMIT_DATE.c_str(), cuda_info.c_str());
}

}  // namespace colmap
