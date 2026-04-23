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

#include "colmap/feature/types.h"

#include <filesystem>

namespace colmap {

#ifdef COLMAP_DOWNLOAD_ENABLED
inline const std::filesystem::path kDefaultSiftVocabTreeUri =
    "https://github.com/colmap/colmap/releases/download/3.11.1/"
    "vocab_tree_faiss_flickr100K_words256K.bin;"
    "vocab_tree_faiss_flickr100K_words256K.bin;"
    "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc";
inline const std::filesystem::path kDefaultAlikedN16RotVocabTreeUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin;"
    "vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin;"
    "8b2f9bdc44ca7204d8543bb3adab4c03ba9336c84ef41220b5007991036f075e";
inline const std::filesystem::path kDefaultAlikedN32VocabTreeUri =
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin;"
    "vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin;"
    "65619481045b8f933268f10c31ad180eb1ee7881182873efe0f5753972ef6a20";
#else
inline const std::filesystem::path kDefaultSiftVocabTreeUri = "";
inline const std::filesystem::path kDefaultAlikedN16RotVocabTreeUri = "";
inline const std::filesystem::path kDefaultAlikedN32VocabTreeUri = "";
#endif

const std::filesystem::path& GetVocabTreeUriForFeatureType(
    FeatureExtractorType feature_type);

}  // namespace colmap
