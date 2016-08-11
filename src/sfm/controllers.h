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

#include "base/reconstruction_manager.h"
#include "sfm/incremental_mapper.h"
#include "util/option_manager.h"
#include "util/threading.h"
#include "util/timer.h"

namespace colmap {

// Class that controls the incremental mapping procedure by iteratively
// initializing reconstructions from the same scene graph.
class IncrementalMapperController : public Thread {
 public:
  enum {
    INITIAL_IMAGE_PAIR_REG_CALLBACK,
    NEXT_IMAGE_REG_CALLBACK,
    LAST_IMAGE_REG_CALLBACK,
  };

  IncrementalMapperController(const OptionManager& options,
                              ReconstructionManager* reconstruction_manager);

 private:
  void Run();

  const OptionManager options_;
  ReconstructionManager* reconstruction_manager_;
};

// Class that controls the global bundle adjustment procedure.
class BundleAdjustmentController : public Thread {
 public:
  BundleAdjustmentController(const OptionManager& options);

  // The model to be adjusted, must be set prior to starting the thread.
  Reconstruction* reconstruction;

 private:
  void Run();

  const OptionManager options_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_SFM_CONTROLLERS_H_
