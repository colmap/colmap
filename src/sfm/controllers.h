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

#ifndef COLMAP_SRC_SFM_CONTROLLERS_H_
#define COLMAP_SRC_SFM_CONTROLLERS_H_

#include "sfm/incremental_mapper.h"
#include "util/option_manager.h"
#include "util/threading.h"
#include "util/timer.h"

namespace colmap {

// Class that controls the incremental mapping procedure by iteratively
// initializing reconstructions from the same scene graph.
// The following callbacks are available:
//  - "InitialImagePairRegistered"
//  - "NextImageRegistered"
//  - "LastImageRegistered"
//  - "Finished"
class IncrementalMapperController : public Thread {
 public:
  IncrementalMapperController(const OptionManager& options);
  IncrementalMapperController(const OptionManager& options,
                              Reconstruction* initial_reconstruction);

  // Access reconstructed models.
  inline size_t NumModels() const;
  inline const std::vector<std::unique_ptr<Reconstruction>>& Models() const;
  inline const Reconstruction& Model(const size_t idx) const;
  inline Reconstruction& Model(const size_t idx);

  // Add new model and return its index.
  size_t AddModel();

 private:
  void Run() override;

  const OptionManager options_;

  // Collection of reconstructed models.
  std::vector<std::unique_ptr<Reconstruction>> models_;
};

// Class that controls the global bundle adjustment procedure.
// The class implements the "Finished" callback.
class BundleAdjustmentController : public Thread {
 public:
  BundleAdjustmentController(const OptionManager& options);

  // The model to be adjusted, must be set prior to starting the thread.
  Reconstruction* reconstruction;

 private:
  void Run() override;

  const OptionManager options_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

inline size_t IncrementalMapperController::NumModels() const {
  return models_.size();
}

const std::vector<std::unique_ptr<Reconstruction>>&
IncrementalMapperController::Models() const {
  return models_;
}

inline const Reconstruction& IncrementalMapperController::Model(
    const size_t idx) const {
  return *models_.at(idx);
}

inline Reconstruction& IncrementalMapperController::Model(const size_t idx) {
  return *models_.at(idx);
}

}  // namespace colmap

#endif  // COLMAP_SRC_SFM_CONTROLLERS_H_
