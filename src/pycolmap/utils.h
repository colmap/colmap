#pragma once

#include "colmap/feature/types.h"
#include "colmap/util/logging.h"

#include <iostream>
#include <regex>
#include <string>

#include <Eigen/Core>

using namespace colmap;

enum class Device { AUTO = -1, CPU = 0, CUDA = 1 };

inline bool IsGPU(Device device) {
  if (device == Device::AUTO) {
#ifdef COLMAP_CUDA_ENABLED
    return true;
#else
    return false;
#endif
  } else {
    return static_cast<bool>(device);
  }
}

inline void VerifyGPUParams(const bool use_gpu) {
#ifndef COLMAP_CUDA_ENABLED
  if (use_gpu) {
    LOG(FATAL_THROW)
        << "Cannot use Sift GPU without CUDA support; set device='auto' "
           "or device='cpu'.";
  }
#endif
}

typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> PyInlierMask;

inline PyInlierMask ToPythonMask(const std::vector<char>& mask_char) {
  return Eigen::Map<const Eigen::Matrix<char, Eigen::Dynamic, 1>>(
             mask_char.data(), mask_char.size())
      .cast<bool>();
}

typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, 2, Eigen::RowMajor>
    PyFeatureMatches;

inline PyFeatureMatches FeatureMatchesToMatrix(const FeatureMatches& matches) {
  PyFeatureMatches matrix(matches.size(), 2);
  for (size_t i = 0; i < matches.size(); i++) {
    matrix(i, 0) = matches[i].point2D_idx1;
    matrix(i, 1) = matches[i].point2D_idx2;
  }
  return matrix;
}

inline FeatureMatches FeatureMatchesFromMatrix(const PyFeatureMatches& matrix) {
  FeatureMatches matches(matrix.rows());
  for (size_t i = 0; i < matches.size(); i++) {
    matches[i].point2D_idx1 = matrix(i, 0);
    matches[i].point2D_idx2 = matrix(i, 1);
  }
  return matches;
}
