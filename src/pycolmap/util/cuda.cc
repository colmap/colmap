#include "colmap/util/cuda.h"

#include <pybind11/pybind11.h>

using namespace colmap;
namespace py = pybind11;

#if defined(COLMAP_CUDA_ENABLED)

void BindCudaUtils(py::module& m) {
  m.def("get_num_cuda_devices",
        &GetNumCudaDevices,
        "Get the number of available CUDA devices.");
}

#endif  // COLMAP_CUDA_ENABLED
