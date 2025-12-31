#pragma once

#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"

#include "glomap/scene/types.h"

namespace glomap {

// Initialize an empty reconstruction from the database.
// This adds cameras, rigs, frames, and images (without 3D points).
void InitializeEmptyReconstructionFromDatabase(
    const colmap::Database& database, colmap::Reconstruction& reconstruction);

}  // namespace glomap
