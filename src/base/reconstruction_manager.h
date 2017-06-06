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

#ifndef COLMAP_SRC_BASE_RECONSTRUCTION_MANAGER_H_
#define COLMAP_SRC_BASE_RECONSTRUCTION_MANAGER_H_

#include "base/reconstruction.h"

namespace colmap {

class OptionManager;

class ReconstructionManager {
 public:
  ReconstructionManager();

  // Move constructor and assignment.
  ReconstructionManager(ReconstructionManager&& other);
  ReconstructionManager& operator=(ReconstructionManager&& other);

  // The number of reconstructions managed.
  size_t Size() const;

  // Get a reference to a specific reconstruction.
  const Reconstruction& Get(const size_t idx) const;
  Reconstruction& Get(const size_t idx);

  // Add a new empty reconstruction and return its index.
  size_t Add();

  // Delete a specific reconstruction.
  void Delete(const size_t idx);

  // Delete all reconstructions.
  void Clear();

  // Read and add a new reconstruction and return its index.
  size_t Read(const std::string& path);

  // Write all managed reconstructions into sub-folders "0", "1", "2", ...
  // If the option manager object is not null, the options are written
  // to each respective reconstruction folder as well.
  void Write(const std::string& path, const OptionManager* options) const;

 private:
  NON_COPYABLE(ReconstructionManager)

  std::vector<std::unique_ptr<Reconstruction>> reconstructions_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_RECONSTRUCTION_MANAGER_H_
