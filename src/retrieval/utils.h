// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_RETRIEVAL_UTILS_H_
#define COLMAP_SRC_RETRIEVAL_UTILS_H_

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
  HammingDistWeightFunctor() {
    // Fills the look-up table.
    const float sigma_squared = kSigma * kSigma;
    const float max_hamming_dist = 1.5f * kSigma;
    for (int n = 0; n <= N; ++n) {
      const float hamming_dist = static_cast<float>(n);
      if (hamming_dist <= max_hamming_dist) {
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

#endif  // COLMAP_SRC_RETRIEVAL_UTILS_H_
