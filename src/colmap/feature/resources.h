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

#include <string>

namespace colmap {

#ifdef COLMAP_DOWNLOAD_ENABLED
const static std::string kDefaultALIKEDN16RotFeatureExtractorUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "aliked-n16rot.onnx;"
    "aliked-n16rot.onnx;"
    "dbd26aadb66b1a3d38fbbba2efd52d61040abc3747ed4f2f185cfd30a0ad1d7e";
const static std::string kDefaultALIKEDN32FeatureExtractorUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "aliked-n32.onnx;"
    "aliked-n32.onnx;"
    "93d8e85f08e48254da554208f6d50ce2e113c1058df12ad580f86bda5525600f";
const static std::string kDefaultXFeatExtractorUri =
    "https://github.com/colmap/colmap/releases/download/3.12.5/"
    "xfeat_extractor.onnx;"
    "xfeat_extractor.onnx;"
    "4a3e421a9ad202cbe99c147fa18157ddba095db7fc97cd3d53d01443705d93c5";
const static std::string kDefaultONNXBruteForceMatcherUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "bruteforce-matcher.onnx;"
    "bruteforce-matcher.onnx;"
    "bc8b01e4bb2099adb634083dfa5e8663b733a22d1b778852cd74f74236126873";
const static std::string kDefaultXFeatLighterGlueMatcherUri =
    "https://github.com/colmap/colmap/releases/download/3.12.5/"
    "xfeat_lighterglue_matcher.onnx;"
    "xfeat_lighterglue_matcher.onnx;"
    "43fa66b70930c8e681e79af765cae4119da6605db02f0cd56c9d2e7e41e0c5cc";
#else
const static std::string kDefaultXFeatExtractorUri = "";
const static std::string kDefaultXFeatBruteForceMatcherUri = "";
const static std::string kDefaultXFeatLighterGlueMatcherUri = "";
#endif

// ALIKED uses local model paths only (no download URL).
const static std::string kDefaultAlikedExtractorUri = "";
const static std::string kDefaultAlikedBruteForceMatcherUri = "";

}  // namespace colmap
