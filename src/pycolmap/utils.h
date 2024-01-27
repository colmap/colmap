#pragma once

#include "pycolmap/log_exceptions.h"

#include <iostream>
#include <regex>
#include <string>

enum class Device { AUTO = -1, CPU = 0, CUDA = 1 };

bool IsGPU(Device device) {
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

void VerifyGPUParams(const bool use_gpu) {
#ifndef COLMAP_CUDA_ENABLED
  if (use_gpu) {
    THROW_EXCEPTION(std::invalid_argument,
                    "Cannot use Sift GPU without CUDA support; "
                    "set device='auto' or device='cpu'.")
  }
#endif
}

typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> PyInlierMask;

PyInlierMask ToPythonMask(const std::vector<char>& mask_char) {
  return Eigen::Map<const Eigen::Matrix<char, Eigen::Dynamic, 1>>(
             mask_char.data(), mask_char.size())
      .cast<bool>();
}
