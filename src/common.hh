#include <Eigen/Core>
#include <iostream>

namespace colmap {

template<typename T>
static inline bool is_ray(T &p) {
  return p(2) != 1;
}

} //namespace colmap
