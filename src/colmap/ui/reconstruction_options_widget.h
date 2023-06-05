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

#ifndef COLMAP_SRC_UI_RECONSTRUCTION_OPTIONS_WIDGET_H_
#define COLMAP_SRC_UI_RECONSTRUCTION_OPTIONS_WIDGET_H_

#include "colmap/ui/options_widget.h"
#include "colmap/util/option_manager.h"

#include <QtCore>
#include <QtWidgets>

namespace colmap {

class MapperGeneralOptionsWidget : public OptionsWidget {
 public:
  MapperGeneralOptionsWidget(QWidget* parent, OptionManager* options);
};

class MapperTriangulationOptionsWidget : public OptionsWidget {
 public:
  MapperTriangulationOptionsWidget(QWidget* parent, OptionManager* options);
};

class MapperRegistrationOptionsWidget : public OptionsWidget {
 public:
  MapperRegistrationOptionsWidget(QWidget* parent, OptionManager* options);
};

class MapperInitializationOptionsWidget : public OptionsWidget {
 public:
  MapperInitializationOptionsWidget(QWidget* parent, OptionManager* options);
};

class MapperBundleAdjustmentOptionsWidget : public OptionsWidget {
 public:
  MapperBundleAdjustmentOptionsWidget(QWidget* parent, OptionManager* options);
};

class MapperFilteringOptionsWidget : public OptionsWidget {
 public:
  MapperFilteringOptionsWidget(QWidget* parent, OptionManager* options);
};

class ReconstructionOptionsWidget : public QWidget {
 public:
  ReconstructionOptionsWidget(QWidget* parent, OptionManager* options);
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_RECONSTRUCTION_OPTIONS_WIDGET_H_
