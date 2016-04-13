// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#ifndef COLMAP_SRC_SFM_MAP_CONTROLLER_H_
#define COLMAP_SRC_SFM_MAP_CONTROLLER_H_

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <QAction>
#include <QMutex>
#include <QThread>
#include <QWaitCondition>

#include "sfm/incremental_mapper.h"
#include "util/option_manager.h"
#include "util/timer.h"

namespace colmap {

class IncrementalMapperController : public QThread {
 public:
  IncrementalMapperController(const OptionManager& options);
  IncrementalMapperController(const OptionManager& options,
                              class Reconstruction* initial_model);

  void run();

  void Stop();
  void Pause();
  void Resume();

  bool IsRunning();
  bool IsStarted();
  bool IsPaused();
  bool IsFinished();

  // Access reconstructed models.
  inline size_t NumModels() const;
  inline const std::vector<std::unique_ptr<Reconstruction>>& Models() const;
  inline const Reconstruction& Model(const size_t idx) const;
  inline Reconstruction& Model(const size_t idx);

  // Add new model and return its index.
  size_t AddModel();

  // Events that are triggered in the respective stages of the reconstruction.
  // The events are only triggered if the objects are not null. A render
  // event is triggered whenever a new image is registered. A render_now
  // event is triggered when the last image in the model was registered.
  // A finish event is triggered when the reconstruction finishes.
  QAction* action_render;
  QAction* action_render_now;
  QAction* action_finish;

 private:
  void Render();
  void RenderNow();
  void Finish();

  bool terminate_;
  bool pause_;
  bool running_;
  bool started_;
  bool finished_;

  QMutex control_mutex_;
  QWaitCondition pause_condition_;

  const OptionManager options_;

  // Collection of reconstructed models.
  std::vector<std::unique_ptr<class Reconstruction>> models_;
};

// Warning: This class is not feature-complete.
class InteractiveMapperController : public QThread {
 public:
  InteractiveMapperController(const OptionManager& options);

  void run();

 private:
  void PrintCommandPrompt();
  bool HandleHelp(const std::vector<std::string>& commands);
  bool HandleStats(const std::vector<std::string>& commands);
  bool HandleExport(const std::vector<std::string>& commands);
  bool HandleRegister(const std::vector<std::string>& commands);
  bool HandleBundleAdjustment(const std::vector<std::string>& commands);

  const OptionManager options_;
};

class BundleAdjustmentController : public QThread {
 public:
  BundleAdjustmentController(const OptionManager& options);

  void run();

  // Check whether bundle adjustment is running.
  bool IsRunning();

  // The model to be adjusted, must be set prior to starting the thread.
  Reconstruction* reconstruction;

  // Event that is triggered when the bundle adjustment finishes. The event
  // is only triggered if the object is not null.
  QAction* action_finish;

 private:
  const OptionManager options_;

  bool running_;

  QMutex mutex_;
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

#endif  // COLMAP_SRC_SFM_MAP_CONTROLLER_H_
