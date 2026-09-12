// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#if defined(COLMAP_GUI_ENABLED)
#include <QApplication>
#else
// Dummy QApplication class when GUI is disabled.
class QApplication {
 public:
  QApplication(int argc, char** argv) {}
};
#endif

namespace colmap {

int RunGraphicalUserInterface(int argc, char** argv);
int RunProjectGenerator(int argc, char** argv);

}  // namespace colmap
