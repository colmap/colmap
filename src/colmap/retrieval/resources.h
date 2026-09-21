// SPDX-License-Identifier: BSD-3-Clause

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
inline const std::filesystem::path kDefaultLomaBVocabTreeUri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "vocab_tree_loma_b_flickr100k.bin;"
    "vocab_tree_loma_b_flickr100k.bin;"
    "06e41d7f4a2cf710f5699a410b3865a0894be15d856b999e91fe4e1f5d5a2989";
inline const std::filesystem::path kDefaultLomaB128VocabTreeUri =
    "https://github.com/davnords/storage/releases/download/loma/"
    "vocab_tree_loma_b128_flickr100k.bin;"
    "vocab_tree_loma_b128_flickr100k.bin;"
    "4259491c3d1379ab981656592d504c788849d16f1ad5c26e31e8198f0b4a2855";
#else
inline const std::filesystem::path kDefaultSiftVocabTreeUri = "";
inline const std::filesystem::path kDefaultAlikedN16RotVocabTreeUri = "";
inline const std::filesystem::path kDefaultAlikedN32VocabTreeUri = "";
inline const std::filesystem::path kDefaultLomaBVocabTreeUri = "";
inline const std::filesystem::path kDefaultLomaB128VocabTreeUri = "";
#endif

const std::filesystem::path& GetVocabTreeUriForFeatureType(
    FeatureExtractorType feature_type);

}  // namespace colmap
