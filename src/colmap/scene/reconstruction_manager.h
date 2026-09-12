// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/reconstruction.h"

#include <filesystem>
#include <memory>

namespace colmap {

class ReconstructionManager {
 public:
  // The number of reconstructions managed.
  size_t Size() const;

  // Get a reference to a specific reconstruction.
  std::shared_ptr<const Reconstruction> Get(size_t idx) const;
  std::shared_ptr<Reconstruction>& Get(size_t idx);

  // Add a new empty reconstruction and return its index.
  size_t Add();

  // Delete a specific reconstruction.
  void Delete(size_t idx);

  // Delete all reconstructions.
  void Clear();

  // Read and add a new reconstruction and return its index.
  size_t Read(const std::filesystem::path& path);

  // Write all managed reconstructions into sub-folders "0", "1", "2", ...
  void Write(const std::filesystem::path& path) const;

 private:
  std::vector<std::shared_ptr<Reconstruction>> reconstructions_;
};

}  // namespace colmap
