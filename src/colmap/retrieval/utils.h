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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include <array>
#include <cmath>

namespace colmap {
namespace retrieval {

struct ImageScore {
  int image_id = -1;
  float score = 0.0f;
};

// Implements the weighting function used to derive a voting weight from the
// Hamming distance of two binary signatures. See Eqn. 4 in
// Arandjelovic, Zisserman. DisLocation: Scalable descriptor distinctiveness for
// location recognition. ACCV 2014.
// The template is the length of the Hamming embedding vectors.
// This class is based on an original implementation by Torsten Sattler.
template <int N, int kSigma = 16>
class HammingDistWeightFunctor {
 public:
  static const size_t kMaxHammingDistance = static_cast<size_t>(1.5f * kSigma);

  HammingDistWeightFunctor() {
    // Fills the look-up table.
    const float sigma_squared = kSigma * kSigma;
    for (int n = 0; n <= N; ++n) {
      const float hamming_dist = static_cast<float>(n);
      if (hamming_dist <= kMaxHammingDistance) {
        look_up_table_.at(n) =
            std::exp(-hamming_dist * hamming_dist / sigma_squared);
      } else {
        look_up_table_.at(n) = 0.0f;
      }
    }
  }

  // Returns the weight for Hamming distance h and standard deviation sigma.
  // Does not perform a range check when performing the look-up.
  inline float operator()(const size_t hamming_dist) const {
    return look_up_table_.at(hamming_dist);
  }

 private:
  // In order to avoid wasting computations, we once compute a look-up table
  // storing all function values for all possible values of the standard
  // deviation \sigma. This is implemented as a (N + 1) vector.
  std::array<float, N + 1> look_up_table_;
};

}  // namespace retrieval
}  // namespace colmap
