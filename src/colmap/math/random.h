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

#include "colmap/util/logging.h"

#include <chrono>
#include <memory>
#include <random>
#include <thread>

namespace colmap {

extern thread_local std::unique_ptr<std::mt19937> PRNG;

extern int kDefaultPRNGSeed;

// Initialize the PRNG with the given seed.
//
// @param seed   The seed for the PRNG. If the seed is -1, the current time
//               is used as the seed.
void SetPRNGSeed(unsigned seed = kDefaultPRNGSeed);

// Generate uniformly distributed random integer number.
//
// This implementation is unbiased and thread-safe in contrast to `rand()`.
template <typename T>
T RandomInteger(T min, T max);

// Generate uniformly distributed random real number.
//
// This implementation is unbiased and thread-safe in contrast to `rand()`.
template <typename T>
T RandomReal(T min, T max);

// Generate Gaussian distributed random real number.
//
// This implementation is unbiased and thread-safe in contrast to `rand()`.
template <typename T>
T RandomGaussian(T mean, T stddev);

// Fisher-Yates shuffling.
//
// Note that the vector may not contain more values than UINT32_MAX. This
// restriction comes from the fact that the 32-bit version of the
// Mersenne Twister PRNG is significantly faster.
//
// @param elems            Vector of elements to shuffle.
// @param num_to_shuffle   Optional parameter, specifying the number of first
//                         N elements in the vector to shuffle.
template <typename T>
void Shuffle(uint32_t num_to_shuffle, std::vector<T>* elems);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
T RandomInteger(const T min, const T max) {
  if (PRNG == nullptr) {
    SetPRNGSeed();
  }

  std::uniform_int_distribution<T> distribution(min, max);

  return distribution(*PRNG);
}

template <typename T>
T RandomReal(const T min, const T max) {
  if (PRNG == nullptr) {
    SetPRNGSeed();
  }

  std::uniform_real_distribution<T> distribution(min, max);

  return distribution(*PRNG);
}

template <typename T>
T RandomGaussian(const T mean, const T stddev) {
  if (PRNG == nullptr) {
    SetPRNGSeed();
  }

  std::normal_distribution<T> distribution(mean, stddev);
  return distribution(*PRNG);
}

template <typename T>
void Shuffle(const uint32_t num_to_shuffle, std::vector<T>* elems) {
  CHECK_LE(num_to_shuffle, elems->size());
  const uint32_t last_idx = static_cast<uint32_t>(elems->size() - 1);
  for (uint32_t i = 0; i < num_to_shuffle; ++i) {
    const auto j = RandomInteger<uint32_t>(i, last_idx);
    std::swap((*elems)[i], (*elems)[j]);
  }
}

}  // namespace colmap
