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

#include "colmap/ui/reconstruction_manager_widget.h"

namespace colmap {

const size_t ReconstructionManagerWidget::kNewestReconstructionIdx =
    std::numeric_limits<size_t>::max();

ReconstructionManagerWidget::ReconstructionManagerWidget(
    QWidget* parent,
    std::shared_ptr<const ReconstructionManager> reconstruction_manager)
    : QComboBox(parent),
      reconstruction_manager_(std::move(reconstruction_manager)) {
  QFont font;
  font.setPointSize(10);
  setFont(font);
}

void ReconstructionManagerWidget::Update() {
  if (view()->isVisible()) {
    return;
  }

  blockSignals(true);

  const int prev_idx = currentIndex() == -1 ? 0 : currentIndex();

  clear();

  addItem("Newest model");

  int max_width = 0;
  for (size_t i = 0; i < reconstruction_manager_->Size(); ++i) {
    const QString item = QString().asprintf(
        "Model %d (%d images, %d points)",
        static_cast<int>(i + 1),
        static_cast<int>(reconstruction_manager_->Get(i)->NumRegImages()),
        static_cast<int>(reconstruction_manager_->Get(i)->NumPoints3D()));
    QFontMetrics font_metrics(view()->font());
#if QT_VERSION >= QT_VERSION_CHECK(5, 11, 0)
    const auto width = font_metrics.horizontalAdvance(item);
#else
    const auto width = font_metrics.width(item);
#endif
    max_width = std::max(max_width, width);
    addItem(item);
  }

  view()->setMinimumWidth(max_width);

  if (reconstruction_manager_->Size() == 0) {
    setCurrentIndex(0);
  } else {
    setCurrentIndex(prev_idx);
  }

  blockSignals(false);
}

size_t ReconstructionManagerWidget::SelectedReconstructionIdx() const {
  if (reconstruction_manager_->Size() == 0) {
    return kNewestReconstructionIdx;
  } else {
    if (currentIndex() == 0) {
      return kNewestReconstructionIdx;
    } else {
      return currentIndex() - 1;
    }
  }
}

void ReconstructionManagerWidget::SelectReconstruction(const size_t idx) {
  if (reconstruction_manager_->Size() == 0) {
    blockSignals(true);
    setCurrentIndex(0);
    blockSignals(false);
  } else {
    blockSignals(true);
    setCurrentIndex(idx + 1);
    blockSignals(false);
  }
}

}  // namespace colmap
