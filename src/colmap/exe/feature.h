// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/controllers/image_reader.h"

namespace colmap {

// This enum can be used as optional input for feature_extractor and
// feature_importer to ensure that the camera flags of ImageReader are set in an
// exclusive and unambigous way. The table below explains the corespondence of
// each setting with the flags
//
// -----------------------------------------------------------------------------------
// |            |                         ImageReaderOptions | | CameraMode |
// single_camera | single_camera_per_folder | single_camera_per_image |
// |------------|---------------|--------------------------|-------------------------|
// | AUTO       | false         | false                    | false | | SINGLE |
// true          | false                    | false                   | |
// PER_FOLDER | false         | true                     | false | | PER_IMAGE
// | false         | false                    | true                    |
// -----------------------------------------------------------------------------------
//
// Note: When using AUTO mode a camera model will be uniquely identified by the
// following 5 parameters from EXIF tags:
// 1. Camera Make
// 2. Camera Model
// 3. Focal Length
// 4. Image Width
// 5. Image Height
//
// If any of the tags is missing then a camera model is considered invalid and a
// new camera is created similar to the PER_IMAGE mode.
//
// If these considered fields are not sufficient to uniquely identify a camera
// then using the AUTO mode will lead to incorrect setup for the cameras, e.g.
// the same camera is used with same focal length but different principal point
// between captures. In these cases it is recommended to either use the
// PER_FOLDER or PER_IMAGE settings.
enum class CameraMode { AUTO = 0, SINGLE = 1, PER_FOLDER = 2, PER_IMAGE = 3 };

void UpdateImageReaderOptionsFromCameraMode(ImageReaderOptions& options,
                                            CameraMode mode);

bool VerifySiftGPUParams(bool use_gpu);

bool VerifyCameraParams(const std::string& camera_model,
                        const std::string& params);

int RunFeatureExtractor(int argc, char** argv);
int RunFeatureImporter(int argc, char** argv);
int RunExhaustiveMatcher(int argc, char** argv);
int RunMatchesImporter(int argc, char** argv);
int RunSequentialMatcher(int argc, char** argv);
int RunSpatialMatcher(int argc, char** argv);
int RunTransitiveMatcher(int argc, char** argv);
int RunVocabTreeMatcher(int argc, char** argv);

}  // namespace colmap
