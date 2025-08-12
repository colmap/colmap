#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindLogging(py::module& m);
void BindTimer(py::module& m);
void BindUtilTypes(py::module& m);
#if defined(COLMAP_CUDA_ENABLED)
void BindCudaUtils(py::module& m);
#endif  // COLMAP_CUDA_ENABLED

void BindUtil(py::module& m) {
  BindUtilTypes(m);
  BindLogging(m);
  BindTimer(m);
#if defined(COLMAP_CUDA_ENABLED)
  BindCudaUtils(m);
#endif  // COLMAP_CUDA_ENABLED
}
