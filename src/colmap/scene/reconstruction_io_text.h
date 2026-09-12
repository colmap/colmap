// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/reconstruction.h"

#include <filesystem>
#include <iostream>

namespace colmap {

// Note that cameras must be read before images.

void ReadRigsText(Reconstruction& reconstruction, std::istream& stream);
void ReadRigsText(Reconstruction& reconstruction,
                  const std::filesystem::path& path);

void ReadCamerasText(Reconstruction& reconstruction, std::istream& stream);
void ReadCamerasText(Reconstruction& reconstruction,
                     const std::filesystem::path& path);

void ReadFramesText(Reconstruction& reconstruction, std::istream& stream);
void ReadFramesText(Reconstruction& reconstruction,
                    const std::filesystem::path& path);

void ReadImagesText(Reconstruction& reconstruction, std::istream& stream);
void ReadImagesText(Reconstruction& reconstruction,
                    const std::filesystem::path& path);

void ReadPoints3DText(Reconstruction& reconstruction, std::istream& stream);
void ReadPoints3DText(Reconstruction& reconstruction,
                      const std::filesystem::path& path);

void WriteRigsText(const Reconstruction& reconstruction, std::ostream& stream);
void WriteRigsText(const Reconstruction& reconstruction,
                   const std::filesystem::path& path);

void WriteCamerasText(const Reconstruction& reconstruction,
                      std::ostream& stream);
void WriteCamerasText(const Reconstruction& reconstruction,
                      const std::filesystem::path& path);

void WriteFramesText(const Reconstruction& reconstruction,
                     std::ostream& stream);
void WriteFramesText(const Reconstruction& reconstruction,
                     const std::filesystem::path& path);

void WriteImagesText(const Reconstruction& reconstruction,
                     std::ostream& stream);
void WriteImagesText(const Reconstruction& reconstruction,
                     const std::filesystem::path& path);

void WritePoints3DText(const Reconstruction& reconstruction,
                       std::ostream& stream);
void WritePoints3DText(const Reconstruction& reconstruction,
                       const std::filesystem::path& path);

}  // namespace colmap
