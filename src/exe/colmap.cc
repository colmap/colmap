// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "ui/main_window.h"

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

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

  Q_INIT_RESOURCE(resources);

  QApplication app(argc, argv);

  MainWindow main_window(options);
  main_window.show();

  return app.exec();
}
