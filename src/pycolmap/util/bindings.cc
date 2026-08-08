#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindLogging(py::module& m);
void BindOpenImageIO(py::module& m);
void BindTimer(py::module& m);
void BindTimestamp(py::module& m);
void BindUtilTypes(py::module& m);
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
void BindCudaUtils(py::module& m);
#endif  // COLMAP_CUDA_ENABLED || COLMAP_HIP_ENABLED

void BindUtil(py::module& m) {
  BindUtilTypes(m);
  BindTimestamp(m);
  BindLogging(m);
  BindOpenImageIO(m);
  BindTimer(m);
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
  BindCudaUtils(m);
#endif  // COLMAP_CUDA_ENABLED || COLMAP_HIP_ENABLED
}
