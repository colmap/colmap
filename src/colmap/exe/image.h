// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace colmap {

int RunImageDeleter(int argc, char** argv);
int RunImageFilterer(int argc, char** argv);
int RunImageRectifier(int argc, char** argv);
int RunImageRegistrator(int argc, char** argv);
int RunImageUndistorter(int argc, char** argv);
int RunImageUndistorterStandalone(int argc, char** argv);

}  // namespace colmap
