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

#include "colmap/base/reconstruction_manager.h"

#include "colmap/util/misc.h"
#include "colmap/util/option_manager.h"

namespace colmap {

ReconstructionManager::ReconstructionManager(
    ReconstructionManager&& other) noexcept
    : ReconstructionManager() {
  reconstructions_ = std::move(other.reconstructions_);
}

ReconstructionManager& ReconstructionManager::operator=(
    ReconstructionManager&& other) noexcept {
  if (this != &other) {
    reconstructions_ = std::move(other.reconstructions_);
  }
  return *this;
}

size_t ReconstructionManager::Size() const { return reconstructions_.size(); }

std::shared_ptr<const Reconstruction> ReconstructionManager::Get(
    const size_t idx) const {
  return reconstructions_.at(idx);
}

std::shared_ptr<Reconstruction> ReconstructionManager::Get(const size_t idx) {
  return reconstructions_.at(idx);
}

size_t ReconstructionManager::Add() {
  const size_t idx = Size();
  reconstructions_.push_back(std::make_shared<Reconstruction>());
  return idx;
}

void ReconstructionManager::Delete(const size_t idx) {
  CHECK_LT(idx, reconstructions_.size());
  reconstructions_.erase(reconstructions_.begin() + idx);
}

void ReconstructionManager::Clear() { reconstructions_.clear(); }

size_t ReconstructionManager::Read(const std::string& path) {
  const size_t idx = Add();
  reconstructions_[idx]->Read(path);
  return idx;
}

void ReconstructionManager::Write(const std::string& path,
                                  const OptionManager* options) const {
  std::vector<std::pair<size_t, size_t>> recon_sizes(reconstructions_.size());
  for (size_t i = 0; i < reconstructions_.size(); ++i) {
    recon_sizes[i] = std::make_pair(i, reconstructions_[i]->NumPoints3D());
  }
  std::sort(recon_sizes.begin(),
            recon_sizes.end(),
            [](const std::pair<size_t, size_t>& first,
               const std::pair<size_t, size_t>& second) {
              return first.second > second.second;
            });

  for (size_t i = 0; i < reconstructions_.size(); ++i) {
    const std::string reconstruction_path = JoinPaths(path, std::to_string(i));
    CreateDirIfNotExists(reconstruction_path);
    reconstructions_[recon_sizes[i].first]->Write(reconstruction_path);
    if (options != nullptr) {
      options->Write(JoinPaths(reconstruction_path, "project.ini"));
    }
  }
}

}  // namespace colmap
