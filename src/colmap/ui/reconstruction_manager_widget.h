// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/reconstruction_manager.h"

#include <QtWidgets>

namespace colmap {

class ReconstructionManagerWidget : public QComboBox {
 public:
  const static size_t kNewestReconstructionIdx;

  ReconstructionManagerWidget(
      QWidget* parent,
      std::shared_ptr<const ReconstructionManager> reconstruction_manager);

  void Update();

  size_t SelectedReconstructionIdx() const;
  void SelectReconstruction(size_t idx);

 private:
  const std::shared_ptr<const ReconstructionManager> reconstruction_manager_;
};

}  // namespace colmap
