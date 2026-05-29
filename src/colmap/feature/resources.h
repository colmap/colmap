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
inline const std::string kDefaultAlikedN16RotFeatureExtractorUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "aliked-n16rot.onnx;"
    "aliked-n16rot.onnx;"
    "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547";
inline const std::string kDefaultAlikedN32FeatureExtractorUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "aliked-n32.onnx;"
    "aliked-n32.onnx;"
    "a077728a02d2de1a775c66df6de8cfeb7c6b51ca57572c64c680131c988c8b3c";
inline const std::string kDefaultAlikedLightGlueFeatureMatcherUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "aliked-lightglue.onnx;"
    "aliked-lightglue.onnx;"
    "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d";
inline const std::string kDefaultBruteForceONNXMatcherUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "bruteforce-matcher.onnx;"
    "bruteforce-matcher.onnx;"
    "3c1282f96d83f5ffc861a873298d08bbe5219f59af59223f5ceab5c41a182a47";
inline const std::string kDefaultSiftLightGlueFeatureMatcherUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "sift-lightglue.onnx;"
    "sift-lightglue.onnx;"
    "e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e";
#else
inline const std::string kDefaultAlikedN16RotFeatureExtractorUri = "";
inline const std::string kDefaultAlikedN32FeatureExtractorUri = "";
inline const std::string kDefaultAlikedLightGlueFeatureMatcherUri = "";
inline const std::string kDefaultBruteForceONNXMatcherUri = "";
inline const std::string kDefaultSiftLightGlueFeatureMatcherUri = "";
#endif

}  // namespace colmap
