// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/reconstruction_manager.h"

#include "colmap/util/file.h"

namespace colmap {

size_t ReconstructionManager::Size() const { return reconstructions_.size(); }

std::shared_ptr<const Reconstruction> ReconstructionManager::Get(
    const size_t idx) const {
  return reconstructions_.at(idx);
}

std::shared_ptr<Reconstruction>& ReconstructionManager::Get(const size_t idx) {
  return reconstructions_.at(idx);
}

size_t ReconstructionManager::Add() {
  const size_t idx = Size();
  reconstructions_.push_back(std::make_shared<Reconstruction>());
  return idx;
}

void ReconstructionManager::Delete(const size_t idx) {
  THROW_CHECK_LT(idx, reconstructions_.size());
  reconstructions_.erase(reconstructions_.begin() + idx);
}

void ReconstructionManager::Clear() { reconstructions_.clear(); }

size_t ReconstructionManager::Read(const std::filesystem::path& path) {
  const size_t idx = Add();
  reconstructions_[idx]->Read(path);
  return idx;
}

void ReconstructionManager::Write(const std::filesystem::path& path) const {
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
    const auto reconstruction_path = path / std::to_string(i);
    CreateDirIfNotExists(reconstruction_path);
    reconstructions_[recon_sizes[i].first]->Write(reconstruction_path);
  }
}

}  // namespace colmap
