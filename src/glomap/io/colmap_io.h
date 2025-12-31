#pragma once

#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"

namespace glomap {

// Initialize an empty reconstruction from the database.
// This adds cameras, rigs, frames, and images (without 3D points).
void InitializeEmptyReconstructionFromDatabase(
    const colmap::Database& database, colmap::Reconstruction& reconstruction);

// Write a reconstruction to disk at path/0/.
void WriteReconstruction(const std::string& reconstruction_path,
                         const colmap::Reconstruction& reconstruction,
                         const std::string& output_format = "bin",
                         const std::string& image_path = "");

}  // namespace glomap
