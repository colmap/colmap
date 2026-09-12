// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace colmap {

int RunDatabaseCleaner(int argc, char** argv);
int RunDatabaseCreator(int argc, char** argv);
int RunDatabaseMerger(int argc, char** argv);
int RunRigConfigurator(int argc, char** argv);

}  // namespace colmap
