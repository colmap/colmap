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

#include <QtGui>

#include "sfm/controllers.h"
#include "ui/main_window.h"
#include "util/logging.h"

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  Q_INIT_RESOURCE(resources);

  QApplication app(argc, argv);

  OptionManager options;

  if (argc > 1) {
    options.AddAllOptions();
    if (!options.Parse(argc, argv)) {
      QMessageBox::critical(0, "Configuration error",
                            "There is an error in your configuration.");
      return EXIT_FAILURE;
    }

    if (options.ParseHelp(argc, argv)) {
      return EXIT_SUCCESS;
    }
  }

  MainWindow main_window(options);
  main_window.show();

  return app.exec();
}
