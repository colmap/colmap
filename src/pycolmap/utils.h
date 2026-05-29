#pragma once

#include <Eigen/Core>

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

typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> PyInlierMask;

inline PyInlierMask ToPythonMask(const std::vector<char>& mask_char) {
  return Eigen::Map<const Eigen::Matrix<char, Eigen::Dynamic, 1>>(
             mask_char.data(), mask_char.size())
      .cast<bool>();
}
