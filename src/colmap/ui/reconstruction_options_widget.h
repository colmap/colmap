// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"
#include "colmap/ui/options_widget.h"

#include <QtCore>
#include <QtWidgets>
#include <functional>

namespace colmap {

// Selects which mapper the GUI runs when starting a reconstruction.
enum class MapperType { INCREMENTAL, HIERARCHICAL, GLOBAL };

class ReconstructionOptionsWidget : public QWidget {
 public:
  ReconstructionOptionsWidget(QWidget* parent,
                              OptionManager* options,
                              MapperType* mapper_type,
                              std::function<void()> on_mapper_type_changed);
};

}  // namespace colmap
