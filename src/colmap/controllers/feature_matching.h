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

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/pairing.h"
#include "colmap/feature/sift.h"
#include "colmap/util/threading.h"

#include <memory>
#include <string>

namespace colmap {

// Exhaustively match images by processing each block in the exhaustive match
// matrix in one batch:
//
// +----+----+-----------------> images[i]
// |#000|0000|
// |1#00|1000| <- Above the main diagonal, the block diagonal is not matched
// |11#0|1100|                                                             ^
// |111#|1110|                                                             |
// +----+----+                                                             |
// |1000|#000|\                                                            |
// |1100|1#00| \ One block                                                 |
// |1110|11#0| / of image pairs                                            |
// |1111|111#|/                                                            |
// +----+----+                                                             |
// |  ^                                                                    |
// |  |                                                                    |
// | Below the main diagonal, the block diagonal is matched <--------------+
// |
// v
// images[i]
//
// Pairs will only be matched if 1, to avoid duplicate pairs. Pairs with #
// are on the main diagonal and denote pairs of the same image.
std::unique_ptr<Thread> CreateExhaustiveFeatureMatcher(
    const ExhaustiveMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path);

// Match each image against its nearest neighbors using a vocabulary tree.
std::unique_ptr<Thread> CreateVocabTreeFeatureMatcher(
    const VocabTreeMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path);

// Sequentially match images within neighborhood:
//
// +-------------------------------+-----------------------> images[i]
//                      ^          |           ^
//                      |   Current image[i]   |
//                      |          |           |
//                      +----------+-----------+
//                                 |
//                        Match image_i against
//
//                    image_[i - o, i + o]        with o = [1 .. overlap]
//                    image_[i - 2^o, i + 2^o]    (for quadratic overlap)
//
// Sequential order is determined based on the image names in ascending order.
//
// Invoke loop detection if `(i mod loop_detection_period) == 0`, retrieve
// most similar `loop_detection_num_images` images from vocabulary tree,
// and perform matching and verification.
std::unique_ptr<Thread> CreateSequentialFeatureMatcher(
    const SequentialMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path);

// Match images against spatial nearest neighbors using prior location
// information, e.g. provided manually or extracted from EXIF.
std::unique_ptr<Thread> CreateSpatialFeatureMatcher(
    const SpatialMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path);

// Match transitive image pairs in a database with existing feature matches.
// This matcher transitively closes loops/triplets. For example, if image pairs
// A-B and B-C match but A-C has not been matched, then this matcher attempts to
// match A-C. This procedure is performed for multiple iterations.
std::unique_ptr<Thread> CreateTransitiveFeatureMatcher(
    const TransitiveMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path);

// Match images manually specified in a list of image pairs.
//
// Read matches file with the following format:
//
//    image_name1 image_name2
//    image_name1 image_name3
//    image_name2 image_name3
//    ...
//
std::unique_ptr<Thread> CreateImagePairsFeatureMatcher(
    const ImagePairsMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path);

// Import feature matches from a text file.
//
// Read matches file with the following format:
//
//      image_name1 image_name2
//      0 1
//      1 2
//      2 3
//      <empty line>
//      image_name1 image_name3
//      0 1
//      1 2
//      2 3
//      ...
//
std::unique_ptr<Thread> CreateFeaturePairsFeatureMatcher(
    const FeaturePairsMatchingOptions& options,
    const SiftMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path);

}  // namespace colmap
