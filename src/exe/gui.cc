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

#include "exe/gui.h"

#include "util/opengl_utils.h"
#include "util/option_manager.h"

namespace colmap {

int RunGraphicalUserInterface(int argc, char** argv) {
#ifndef GUI_ENABLED
  std::cerr << "ERROR: Cannot start colmap GUI; colmap was built without GUI "
               "support or QT dependency is missing."
            << std::endl;
  return EXIT_FAILURE;
#else
  using namespace colmap;

  OptionManager options;

  std::string import_path;

  if (argc > 1) {
    options.AddDefaultOption("import_path", &import_path);
    options.AddAllOptions();
    options.Parse(argc, argv);
  }

#if (QT_VERSION >= QT_VERSION_CHECK(5, 6, 0))
  QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
  QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif

  QApplication app(argc, argv);

  MainWindow main_window(options);
  main_window.show();

  if (!import_path.empty()) {
    main_window.ImportReconstruction(import_path);
  }

  return app.exec();
#endif
}

int RunProjectGenerator(int argc, char** argv) {
  std::string output_path;
  std::string quality = "high";

  OptionManager options;
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("quality", &quality, "{low, medium, high, extreme}");
  options.Parse(argc, argv);

  OptionManager output_options;
  output_options.AddAllOptions();

  StringToLower(&quality);
  if (quality == "low") {
    output_options.ModifyForLowQuality();
  } else if (quality == "medium") {
    output_options.ModifyForMediumQuality();
  } else if (quality == "high") {
    output_options.ModifyForHighQuality();
  } else if (quality == "extreme") {
    output_options.ModifyForExtremeQuality();
  } else {
    LOG(FATAL) << "Invalid quality provided";
  }

  output_options.Write(output_path);

  return EXIT_SUCCESS;
}

}  // namespace colmap
