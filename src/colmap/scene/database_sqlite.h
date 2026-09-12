// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/database.h"

#include <filesystem>
#include <memory>

namespace colmap {

// Can be used to construct temporary in-memory database.
constexpr inline char kInMemorySqliteDatabasePath[] = ":memory:";

std::shared_ptr<Database> OpenSqliteDatabase(const std::filesystem::path& path);

}  // namespace colmap
