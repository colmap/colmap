#include "colmap/mvs/meshing.h"

#include "colmap/util/misc.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindMeshing(py::module& m) {
  using PoissonMOpts = mvs::PoissonMeshingOptions;
  auto PyPoissonMeshingOptions =
      py::class_<PoissonMOpts>(m, "PoissonMeshingOptions")
          .def(py::init<>())
          .def_readwrite("point_weight",
                         &PoissonMOpts::point_weight,
                         "This floating point value specifies the importance "
                         "that interpolation of"
                         "the point samples is given in the formulation of the "
                         "screened Poisson"
                         "equation. The results of the original (unscreened) "
                         "Poisson Reconstruction"
                         "can be obtained by setting this value to 0.")
          .def_readwrite("depth",
                         &PoissonMOpts::depth,
                         "This integer is the maximum depth of the tree that "
                         "will be used for surface"
                         "reconstruction. Running at depth d corresponds to "
                         "solving on a voxel grid"
                         "whose resolution is no larger than 2^d x 2^d x 2^d. "
                         "Note that since the"
                         "reconstructor adapts the octree to the sampling "
                         "density, the specified"
                         "reconstruction depth is only an upper bound.")
          .def_readwrite("color",
                         &PoissonMOpts::color,
                         "If specified, the reconstruction code assumes that "
                         "the input is equipped"
                         "with colors and will extrapolate the color values to "
                         "the vertices of the"
                         "reconstructed mesh. The floating point value "
                         "specifies the relative"
                         "importance of finer color estimates over lower ones.")
          .def_readwrite("trim",
                         &PoissonMOpts::trim,
                         "This floating point values specifies the value for "
                         "mesh trimming. The"
                         "subset of the mesh with signal value less than the "
                         "trim value is discarded.")
          .def_readwrite(
              "num_threads",
              &PoissonMOpts::num_threads,
              "The number of threads used for the Poisson reconstruction.");
  MakeDataclass(PyPoissonMeshingOptions);
  auto poisson_options = PyPoissonMeshingOptions().cast<PoissonMOpts>();

  using DMOpts = mvs::DelaunayMeshingOptions;
  auto PyDelaunayMeshingOptions =
      py::class_<DMOpts>(m, "DelaunayMeshingOptions")
          .def(py::init<>())
          .def_readwrite("max_proj_dist",
                         &DMOpts::max_proj_dist,
                         "Unify input points into one cell in the Delaunay "
                         "triangulation that fall"
                         "within a reprojected radius of the given pixels.")
          .def_readwrite("max_depth_dist",
                         &DMOpts::max_depth_dist,
                         "Maximum relative depth difference between input "
                         "point and a vertex of an"
                         "existing cell in the Delaunay triangulation, "
                         "otherwise a new vertex is"
                         "created in the triangulation.")
          .def_readwrite("visibility_sigma",
                         &DMOpts::visibility_sigma,
                         "The standard deviation of wrt. the number of images "
                         "seen by each point."
                         "Increasing this value decreases the influence of "
                         "points seen in few images.")
          .def_readwrite("distance_sigma_factor",
                         &DMOpts::distance_sigma_factor,
                         "The factor that is applied to the computed distance "
                         "sigma, which is"
                         "automatically computed as the 25th percentile of "
                         "edge lengths. A higher"
                         "value will increase the smoothness of the surface.")
          .def_readwrite(
              "quality_regularization",
              &DMOpts::quality_regularization,
              "A higher quality regularization leads to a smoother surface.")
          .def_readwrite("max_side_length_factor",
                         &DMOpts::max_side_length_factor,
                         "Filtering thresholds for outlier surface mesh faces. "
                         "If the longest side of"
                         "a mesh face (longest out of 3) exceeds the side "
                         "lengths of all faces at a"
                         "certain percentile by the given factor, then it is "
                         "considered an outlier"
                         "mesh face and discarded.")
          .def_readwrite("max_side_length_percentile",
                         &DMOpts::max_side_length_percentile,
                         "Filtering thresholds for outlier surface mesh faces. "
                         "If the longest side of"
                         "a mesh face (longest out of 3) exceeds the side "
                         "lengths of all faces at a"
                         "certain percentile by the given factor, then it is "
                         "considered an outlier"
                         "mesh face and discarded.")
          .def_readwrite("num_threads",
                         &DMOpts::num_threads,
                         "The number of threads to use for reconstruction. "
                         "Default is all threads.");
  MakeDataclass(PyDelaunayMeshingOptions);
  auto delaunay_options = PyDelaunayMeshingOptions().cast<DMOpts>();

  m.def(
      "poisson_meshing",
      [](const std::string& input_path,
         const std::string& output_path,
         const PoissonMOpts& options) -> void {
        THROW_CHECK_HAS_FILE_EXTENSION(input_path, ".ply");
        THROW_CHECK_FILE_EXISTS(input_path);
        THROW_CHECK_HAS_FILE_EXTENSION(output_path, ".ply");
        THROW_CHECK_PATH_OPEN(output_path);
        mvs::PoissonMeshing(options, input_path, output_path);
      },
      "input_path"_a,
      "output_path"_a,
      "options"_a = poisson_options,
      "Perform Poisson surface reconstruction and return true if successful.");

#ifdef COLMAP_CGAL_ENABLED
  m.def("sparse_delaunay_meshing",
        &mvs::SparseDelaunayMeshing,
        "input_path"_a,
        "output_path"_a,
        "options"_a = delaunay_options,
        "Delaunay meshing of sparse COLMAP reconstructions.");

  m.def("dense_delaunay_meshing",
        &mvs::DenseDelaunayMeshing,
        "input_path"_a,
        "output_path"_a,
        "options"_a = delaunay_options,
        "Delaunay meshing of dense COLMAP reconstructions.");
#endif
};
