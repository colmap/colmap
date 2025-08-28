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
const static std::string kDefaultXFeatExtractorUri =
    "https://github.com/colmap/colmap/releases/download/3.12.5/"
    "xfeat_extractor.onnx;"
    "xfeat_extractor.onnx;"
    "027898e280e30021af4fcf7a47cd010f5cf975f765dae6bff132f766b84fb6c1";
const static std::string kDefaultXFeatBruteForceMatcherUri =
    "https://github.com/colmap/colmap/releases/download/3.12.5/"
    "xfeat_bruteforce_matcher.onnx;"
    "xfeat_bruteforce_matcher.onnx;"
    "85d69d867fc9685e5ab0e19b7f2c30c8a23aaa0e756c21c7312578d0ded19f57";
#else
const static std::string kDefaultXFeatUri = "";
const static std::string kDefaultLightGlueXFeatUri = "";
const static std::string kDefaultLightGlueSiftUri = "";
#endif

}  // namespace colmap
