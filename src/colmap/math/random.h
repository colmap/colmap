// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/logging.h"

#include <memory>
#include <random>

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
T RandomUniformInteger(T min, T max);

// Generate uniformly distributed random real number.
//
// This implementation is unbiased and thread-safe in contrast to `rand()`.
template <typename T>
T RandomUniformReal(T min, T max);

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
T RandomUniformInteger(const T min, const T max) {
  if (COLMAP_PREDICT_FALSE(PRNG == nullptr)) {
    SetPRNGSeed();
  }

  std::uniform_int_distribution<T> distribution(min, max);
  return distribution(*PRNG);
}

template <typename T>
T RandomUniformReal(const T min, const T max) {
  if (COLMAP_PREDICT_FALSE(PRNG == nullptr)) {
    SetPRNGSeed();
  }

  std::uniform_real_distribution<T> distribution(min, max);
  return distribution(*PRNG);
}

template <typename T>
T RandomGaussian(const T mean, const T stddev) {
  if (COLMAP_PREDICT_FALSE(PRNG == nullptr)) {
    SetPRNGSeed();
  }

  std::normal_distribution<T> distribution(mean, stddev);
  return distribution(*PRNG);
}

template <typename T>
void Shuffle(const uint32_t num_to_shuffle, std::vector<T>* elems) {
  THROW_CHECK_LE(num_to_shuffle, elems->size());
  const uint32_t last_idx = static_cast<uint32_t>(elems->size() - 1);
  for (uint32_t i = 0; i < num_to_shuffle; ++i) {
    const auto j = RandomUniformInteger<uint32_t>(i, last_idx);
    std::swap((*elems)[i], (*elems)[j]);
  }
}

}  // namespace colmap
