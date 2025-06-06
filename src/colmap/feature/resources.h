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
const static std::string kDefaultAlikedUri =
    "https://github.com/colmap/colmap/releases/download/3.11.1/aliked_n32.pt;"
    "aliked_n32.pt;"
    "bc64763cfa2dc7b3356bb03053d2b2b277498c64e89786001dd391daa577a7d6";
const static std::string kDefaultLightGlueAlikedUri =
    "https://github.com/colmap/colmap/releases/download/3.11.1/"
    "lightglue_aliked.pt;"
    "lightglue_aliked.pt;"
    "01ce35141db9d91e0e4fe39ede3435b1f8dd61929f9d32ae609e95172e2fa402";
const static std::string kDefaultLightGlueSiftUri =
    "https://github.com/colmap/colmap/releases/download/3.11.1/"
    "lightglue_sift.pt;"
    "lightglue_sift.pt;"
    "5a3c6ec3f941cf7600fb465b65f0f0a341383e069fec77d510381945b3de41a7";
#else
const static std::string kDefaultAlikedUri = "";
const static std::string kDefaultLightGlueAlikedUri = "";
const static std::string kDefaultLightGlueSiftUri = "";
#endif

}  // namespace colmap
