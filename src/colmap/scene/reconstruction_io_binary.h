// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#pragma once

#include "colmap/scene/reconstruction.h"

#include <iostream>

namespace colmap {

// Note that cameras must be read before images.

void ReadRigsBinary(Reconstruction& reconstruction, std::istream& stream);
void ReadRigsBinary(Reconstruction& reconstruction, const std::string& path);

void ReadCamerasBinary(Reconstruction& reconstruction, std::istream& stream);
void ReadCamerasBinary(Reconstruction& reconstruction, const std::string& path);

void ReadFramesBinary(Reconstruction& reconstruction, std::istream& stream);
void ReadFramesBinary(Reconstruction& reconstruction, const std::string& path);

void ReadImagesBinary(Reconstruction& reconstruction, std::istream& stream);
void ReadImagesBinary(Reconstruction& reconstruction, const std::string& path);

void ReadPoints3DBinary(Reconstruction& reconstruction, std::istream& stream);
void ReadPoints3DBinary(Reconstruction& reconstruction,
                        const std::string& path);

void WriteRigsBinary(const Reconstruction& reconstruction,
                     std::ostream& stream);
void WriteRigsBinary(const Reconstruction& reconstruction,
                     const std::string& path);

void WriteFramesBinary(const Reconstruction& reconstruction,
                       std::ostream& stream);
void WriteFramesBinary(const Reconstruction& reconstruction,
                       const std::string& path);

void WriteCamerasBinary(const Reconstruction& reconstruction,
                        std::ostream& stream);
void WriteCamerasBinary(const Reconstruction& reconstruction,
                        const std::string& path);

void WriteImagesBinary(const Reconstruction& reconstruction,
                       std::ostream& stream);
void WriteImagesBinary(const Reconstruction& reconstruction,
                       const std::string& path);

void WritePoints3DBinary(const Reconstruction& reconstruction,
                         std::ostream& stream);
void WritePoints3DBinary(const Reconstruction& reconstruction,
                         const std::string& path);

}  // namespace colmap
