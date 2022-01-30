// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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

#ifndef COLMAP_SRC_UI_PROJECT_WIDGET_H_
#define COLMAP_SRC_UI_PROJECT_WIDGET_H_

#include <QtCore>
#include <QtWidgets>

#include "util/misc.h"
#include "util/option_manager.h"

namespace colmap {

class ProjectWidget : public QWidget {
 public:
  ProjectWidget(QWidget* parent, OptionManager* options);

  bool IsValid() const;
  void Reset();

  std::string GetDatabasePath() const;
  std::string GetImagePath() const;
  void SetDatabasePath(const std::string& path);
  void SetImagePath(const std::string& path);

 private:
  void Save();
  void SelectNewDatabasePath();
  void SelectExistingDatabasePath();
  void SelectImagePath();
  QString DefaultDirectory();

  OptionManager* options_;

  // Whether file dialog was opened previously.
  bool prev_selected_;

  // Text boxes that hold the currently selected paths.
  QLineEdit* database_path_text_;
  QLineEdit* image_path_text_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_PROJECT_WIDGET_H_
