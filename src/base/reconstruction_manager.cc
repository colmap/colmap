// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "base/reconstruction_manager.h"

#include "util/misc.h"

namespace colmap {

size_t ReconstructionManager::Size() const { return reconstructions_.size(); }

const Reconstruction& ReconstructionManager::Get(const size_t idx) const {
  return *reconstructions_.at(idx);
}

Reconstruction& ReconstructionManager::Get(const size_t idx) {
  return *reconstructions_.at(idx);
}

size_t ReconstructionManager::Add() {
  const size_t idx = Size();
  reconstructions_.emplace_back(new Reconstruction());
  return idx;
}

void ReconstructionManager::Delete(const size_t idx) {
  CHECK_LT(idx, reconstructions_.size());
  reconstructions_.erase(reconstructions_.begin() + idx);
}

void ReconstructionManager::Clear() {
  reconstructions_.clear();
}

size_t ReconstructionManager::Read(const std::string& path) {
  const size_t idx = Add();
  reconstructions_[idx]->Read(path);
  return idx;
}

void ReconstructionManager::Write(const std::string& path,
                                  const OptionManager* options) const {
  const std::string base_path = EnsureTrailingSlash(path);
  for (size_t i = 0; i < reconstructions_.size(); ++i) {
    const std::string reconstruction_path = base_path + std::to_string(i);
    CreateDirIfNotExists(reconstruction_path);
    reconstructions_[i]->Write(reconstruction_path);
    if (options != nullptr) {
      options->Write(reconstruction_path + "/project.ini");
    }
  }
}

}  // namespace colmap
