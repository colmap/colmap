// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/gui.h"

#if defined(COLMAP_GUI_ENABLED)
#include "colmap/ui/main_window.h"
#endif
#include "colmap/controllers/option_manager.h"
#include "colmap/util/logging.h"

namespace colmap {

int RunGraphicalUserInterface(int argc, char** argv) {
#if !defined(COLMAP_GUI_ENABLED)
  LOG(ERROR)
      << "Cannot start graphical user interface. COLMAP was built without GUI "
         "support or Qt dependency was not found.";
  return EXIT_FAILURE;
#else
  OptionManager options;

  std::filesystem::path import_path;

  if (argc > 1) {
    options.AddDefaultOption("import_path", &import_path);
    options.AddAllOptions();
    if (!options.Parse(argc, argv)) {
      return EXIT_FAILURE;
    }
  }

  QApplication app(argc, argv);

#if (QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)) && \
    (QT_VERSION < QT_VERSION_CHECK(6, 0, 0))
  app.setAttribute(Qt::AA_EnableHighDpiScaling);
  app.setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif
  app.setAttribute(Qt::AA_DontShowIconsInMenus, false);

  MainWindow main_window(std::move(options));
  main_window.show();

  if (!import_path.empty()) {
    main_window.ImportReconstruction(import_path);
  }

  return app.exec();
#endif
}

int RunProjectGenerator(int argc, char** argv) {
  std::filesystem::path output_path;
  std::string quality = "high";

  OptionManager options;
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("quality", &quality, "{low, medium, high, extreme}");
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

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
    LOG(FATAL_THROW) << "Invalid quality provided";
  }

  output_options.Write(output_path);

  return EXIT_SUCCESS;
}

}  // namespace colmap
