#include "colmap/controllers/hierarchical_pipeline.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/scene/scene_clustering.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindHierarchicalMapper(py::module& m) {
  {
    using Opts = SceneClustering::Options;
    auto PyOpts = py::classh<Opts>(m, "SceneClusteringOptions");
    PyOpts.def(py::init<>())
        .def_readwrite("is_hierarchical",
                       &Opts::is_hierarchical,
                       "Flag for hierarchical vs. flat clustering.")
        .def_readwrite("branching",
                       &Opts::branching,
                       "The branching factor of the hierarchical clustering.")
        .def_readwrite(
            "image_overlap",
            &Opts::image_overlap,
            "The number of overlapping images between child clusters.")
        .def_readwrite(
            "num_image_matches",
            &Opts::num_image_matches,
            "The max related images matches to look for in a flat cluster.")
        .def_readwrite(
            "leaf_max_num_images",
            &Opts::leaf_max_num_images,
            "The maximum number of images in a leaf node cluster, otherwise "
            "the cluster is further partitioned using the given branching "
            "factor. Note that a cluster leaf node will have at most "
            "`leaf_max_num_images + image_overlap` images to satisfy the "
            "overlap constraint.")
        .def("check", &Opts::Check);
    MakeDataclass(PyOpts);
  }

  {
    using Opts = HierarchicalPipelineOptions;
    auto PyOpts = py::classh<Opts>(m, "HierarchicalPipelineOptions");
    PyOpts.def(py::init<>())
        .def_readwrite(
            "image_path",
            &Opts::image_path,
            "The image path at which to find the images to extract point "
            "colors. If not specified, all point colors will be black.")
        .def_readwrite("init_num_trials",
                       &Opts::init_num_trials,
                       "The maximum number of trials to initialize a cluster.")
        .def_readwrite(
            "num_threads",
            &Opts::num_threads,
            "The total number of threads for the hierarchical pipeline. This "
            "budget is divided across workers to avoid thread "
            "oversubscription. Note that incremental_options.num_threads is "
            "ignored in favor of this option.")
        .def_readwrite(
            "num_workers",
            &Opts::num_workers,
            "The number of workers used to reconstruct clusters in parallel.")
        .def_readwrite("clustering_options",
                       &Opts::clustering_options,
                       "Options for clustering the scene graph.")
        .def_readwrite("incremental_options",
                       &Opts::incremental_options,
                       "Options used to reconstruct each cluster individually.")
        .def("check", &Opts::Check);
    MakeDataclass(PyOpts);
  }

  py::classh<HierarchicalPipeline>(
      m,
      "HierarchicalPipeline",
      "Class that controls the hierarchical mapping procedure by first "
      "partitioning the scene into multiple overlapping clusters, then "
      "reconstructing them separately using incremental mapping, and finally "
      "merging them all into a globally consistent reconstruction. This is "
      "especially useful for larger-scale scenes, since incremental mapping "
      "becomes slow with an increasing number of images.")
      .def(py::init<const HierarchicalPipelineOptions&,
                    std::shared_ptr<Database>,
                    std::shared_ptr<ReconstructionManager>>(),
           "options"_a,
           "database"_a,
           "reconstruction_manager"_a,
           py::call_guard<py::gil_scoped_release>())
      .def("run",
           &HierarchicalPipeline::Run,
           py::call_guard<py::gil_scoped_release>(),
           "Run the full hierarchical mapping pipeline.");
}
