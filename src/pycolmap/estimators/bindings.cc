#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindAbsolutePoseEstimator(py::module& m);
void BindAffineTransformEstimator(py::module& m);
void BindAlignmentEstimator(py::module& m);
void BindBundleAdjuster(py::module& m);
void BindCeres(py::module& m);
void BindCostFunctions(py::module& m);
void BindCovarianceEstimator(py::module& m);
void BindEssentialMatrixEstimator(py::module& m);
void BindFundamentalMatrixEstimator(py::module& m);
void BindGeneralizedAbsolutePoseEstimator(py::module& m);
void BindHomographyMatrixEstimator(py::module& m);
void BindManifold(py::module& m);
void BindSimilarityTransformEstimator(py::module& m);
void BindTriangulationEstimator(py::module& m);
void BindTwoViewGeometryEstimator(py::module& m);

void BindEstimators(py::module& m) {
  BindCeres(m);
  BindAbsolutePoseEstimator(m);
  BindAffineTransformEstimator(m);
  BindAlignmentEstimator(m);
  BindBundleAdjuster(m);
  BindCostFunctions(m);
  BindCovarianceEstimator(m);
  BindEssentialMatrixEstimator(m);
  BindFundamentalMatrixEstimator(m);
  BindGeneralizedAbsolutePoseEstimator(m);
  BindHomographyMatrixEstimator(m);
  BindManifold(m);
  BindSimilarityTransformEstimator(m);
  BindTriangulationEstimator(m);
  BindTwoViewGeometryEstimator(m);
}
