#include "colmap/mvs/advancing_front_meshing.h"
#include "colmap/mvs/delaunay_meshing.h"
#include "colmap/mvs/mesh_simplification.h"
#include "colmap/mvs/poisson_meshing.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/ply.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <filesystem>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindMeshing(py::module& m) {
  using PoissonMOpts = mvs::PoissonMeshingOptions;
  auto PyPoissonMeshingOptions =
      py::classh<PoissonMOpts>(m, "PoissonMeshingOptions")
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
              "The number of threads used for the Poisson reconstruction.")
          .def("check", &PoissonMOpts::Check);
  MakeDataclass(PyPoissonMeshingOptions);

  using DMOpts = mvs::DelaunayMeshingOptions;
  auto PyDelaunayMeshingOptions =
      py::classh<DMOpts>(m, "DelaunayMeshingOptions")
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
                         "Default is all threads.")
          .def("check", &DMOpts::Check);
  MakeDataclass(PyDelaunayMeshingOptions);

  m.def(
      "poisson_meshing",
      [](const std::filesystem::path& input_path,
         const std::filesystem::path& output_path,
         const PoissonMOpts& options) -> void {
        mvs::PoissonMeshing(options, input_path, output_path);
      },
      "input_path"_a,
      "output_path"_a,
      py::arg_v(
          "options", mvs::PoissonMeshingOptions(), "PoissonMeshingOptions()"),
      "Perform Poisson surface reconstruction and return true if successful.");

#ifdef COLMAP_CGAL_ENABLED
  m.def(
      "sparse_delaunay_meshing",
      [](const std::filesystem::path& input_path,
         const std::filesystem::path& output_path,
         const DMOpts& options) -> void {
        mvs::SparseDelaunayMeshing(options, input_path, output_path);
      },
      "input_path"_a,
      "output_path"_a,
      py::arg_v(
          "options", mvs::DelaunayMeshingOptions(), "DelaunayMeshingOptions()"),
      "Delaunay meshing of sparse COLMAP reconstructions.");

  m.def(
      "dense_delaunay_meshing",
      [](const std::filesystem::path& input_path,
         const std::filesystem::path& output_path,
         const DMOpts& options) -> void {
        mvs::DenseDelaunayMeshing(options, input_path, output_path);
      },
      "input_path"_a,
      "output_path"_a,
      py::arg_v(
          "options", mvs::DelaunayMeshingOptions(), "DelaunayMeshingOptions()"),
      "Delaunay meshing of dense COLMAP reconstructions.");

  using AFMOpts = mvs::AdvancingFrontMeshingOptions;
  auto PyAdvancingFrontMeshingOptions =
      py::classh<AFMOpts>(m, "AdvancingFrontMeshingOptions")
          .def(py::init<>())
          .def_readwrite("max_edge_length",
                         &AFMOpts::max_edge_length,
                         "Maximum edge length constraint for triangles "
                         "(in world units). Set to 0 to disable.")
          .def_readwrite("visibility_filtering",
                         &AFMOpts::visibility_filtering,
                         "Whether to use visibility-based filtering.")
          .def_readwrite("visibility_filtering_max_intersections",
                         &AFMOpts::visibility_filtering_max_intersections,
                         "Maximum number of visibility ray intersections "
                         "before a facet is rejected.")
          .def_readwrite("visibility_post_filtering",
                         &AFMOpts::visibility_post_filtering,
                         "If true, post-filter via AABB tree. "
                         "If false, pre-filter via Priority functor.")
          .def_readwrite("visibility_ray_trim_offset",
                         &AFMOpts::visibility_ray_trim_offset,
                         "Offset distance (in world units) by which visibility "
                         "rays are trimmed at the target end.")
          .def_readwrite("block_size",
                         &AFMOpts::block_size,
                         "Block size for parallel processing "
                         "(in world units, 0 to disable).")
          .def_readwrite("block_overlap",
                         &AFMOpts::block_overlap,
                         "Overlap margin as a fraction of block_size.")
          .def_readwrite("num_threads",
                         &AFMOpts::num_threads,
                         "The number of threads to use. -1 = all threads.")
          .def("check", &AFMOpts::Check);
  MakeDataclass(PyAdvancingFrontMeshingOptions);

  m.def(
      "advancing_front_meshing",
      [](const std::filesystem::path& input_path,
         const std::filesystem::path& output_path,
         const AFMOpts& options) -> void {
        mvs::AdvancingFrontMeshing(options, input_path, output_path);
      },
      "input_path"_a,
      "output_path"_a,
      py::arg_v("options",
                mvs::AdvancingFrontMeshingOptions(),
                "AdvancingFrontMeshingOptions()"),
      "Advancing front surface reconstruction of dense COLMAP "
      "reconstructions.");
#endif

  using MSOpts = mvs::MeshSimplificationOptions;
  auto PyMeshSimplificationOptions =
      py::classh<MSOpts>(m, "MeshSimplificationOptions")
          .def(py::init<>())
          .def_readwrite("target_face_ratio",
                         &MSOpts::target_face_ratio,
                         "Fraction of faces to retain, in (0, 1].")
          .def_readwrite("max_error",
                         &MSOpts::max_error,
                         "Maximum quadric error per collapse; 0 = disabled.")
          .def_readwrite("boundary_weight",
                         &MSOpts::boundary_weight,
                         "Penalty weight for boundary edges; 0 = disabled.")
          .def_readwrite(
              "interpolate_colors",
              &MSOpts::interpolate_colors,
              "Blend colors on collapse vs. pick lower-error vertex.")
          .def_readwrite("num_threads",
                         &MSOpts::num_threads,
                         "The number of threads to use for initialization. "
                         "-1 = all threads.")
          .def("check", &MSOpts::Check);
  MakeDataclass(PyMeshSimplificationOptions);

  m.def(
      "simplify_mesh",
      [](const std::filesystem::path& input_path,
         const std::filesystem::path& output_path,
         const MSOpts& options) -> void {
        THROW_CHECK_HAS_FILE_EXTENSION(input_path, ".ply");
        THROW_CHECK_FILE_EXISTS(input_path);
        THROW_CHECK_HAS_FILE_EXTENSION(output_path, ".ply");
        const PlyMesh mesh = ReadPlyMesh(input_path).mesh;
        const PlyMesh result = mvs::SimplifyMesh(mesh, options);
        WriteBinaryPlyMesh(output_path, PlyTexturedMesh{result});
      },
      "input_path"_a,
      "output_path"_a,
      py::arg_v("options",
                mvs::MeshSimplificationOptions(),
                "MeshSimplificationOptions()"),
      "Read a PLY mesh, simplify it using QEM decimation, and write "
      "the result.",
      py::call_guard<py::gil_scoped_release>());
}
