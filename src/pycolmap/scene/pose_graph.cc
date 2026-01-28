#include "colmap/scene/pose_graph.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/types.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindPoseGraph(py::module& m) {
  auto PyPoseGraphEdge =
      py::classh<PoseGraph::Edge>(m, "PoseGraphEdge")
          .def(py::init<>())
          .def(py::init<const Rigid3d&>(), "cam2_from_cam1"_a)
          .def_readwrite("cam2_from_cam1",
                         &PoseGraph::Edge::cam2_from_cam1,
                         "Relative pose from image 1 to image 2.")
          .def_readwrite("num_matches",
                         &PoseGraph::Edge::num_matches,
                         "Number of two-view matches used to compute the "
                         "relative pose.")
          .def_readwrite("valid",
                         &PoseGraph::Edge::valid,
                         "Whether this edge is valid for reconstruction.")
          .def("invert",
               &PoseGraph::Edge::Invert,
               "Invert the geometry to match swapped image order.");
  MakeDataclass(PyPoseGraphEdge);

  py::bind_map<PoseGraphEdgeMap>(m, "PoseGraphEdgeMap");

  py::classh<PoseGraph>(m, "PoseGraph")
      .def(py::init<>())
      .def_property_readonly("edges",
                             py::overload_cast<>(&PoseGraph::Edges),
                             py::return_value_policy::reference_internal,
                             "Access to all edges in the pose graph.")
      .def_property_readonly("num_edges",
                             &PoseGraph::NumEdges,
                             "Number of edges in the pose graph.")
      .def_property_readonly(
          "empty", &PoseGraph::Empty, "Whether the pose graph has no edges.")
      .def("clear", &PoseGraph::Clear, "Remove all edges.")
      .def("load",
           &PoseGraph::Load,
           "corr_graph"_a,
           "Load edges from a correspondence graph.")
      .def("add_edge",
           &PoseGraph::AddEdge,
           "image_id1"_a,
           "image_id2"_a,
           "edge"_a,
           py::return_value_policy::reference_internal,
           "Add a new edge between two images. Throws if edge already exists.")
      .def("has_edge",
           &PoseGraph::HasEdge,
           "image_id1"_a,
           "image_id2"_a,
           "Check if an edge exists between two images.")
      .def("get_edge",
           &PoseGraph::GetEdge,
           "image_id1"_a,
           "image_id2"_a,
           "Get a copy of the edge between two images. Automatically handles "
           "geometric inversion if image order was swapped.")
      .def("delete_edge",
           &PoseGraph::DeleteEdge,
           "image_id1"_a,
           "image_id2"_a,
           "Delete the edge between two images. Returns True if deleted.")
      .def("update_edge",
           &PoseGraph::UpdateEdge,
           "image_id1"_a,
           "image_id2"_a,
           "edge"_a,
           "Update an existing edge. Throws if edge does not exist.")
      .def("is_valid",
           &PoseGraph::IsValid,
           "pair_id"_a,
           "Check if an edge is marked as valid.")
      .def("set_valid_edge",
           &PoseGraph::SetValidEdge,
           "pair_id"_a,
           "Mark an edge as valid.")
      .def("set_invalid_edge",
           &PoseGraph::SetInvalidEdge,
           "pair_id"_a,
           "Mark an edge as invalid.")
      .def("compute_largest_connected_frame_component",
           &PoseGraph::ComputeLargestConnectedFrameComponent,
           "reconstruction"_a,
           "filter_unregistered"_a = true,
           "Compute the largest connected component of frames. If "
           "filter_unregistered is True, only considers frames with poses.")
      .def("invalidate_pairs_outside_active_image_ids",
           &PoseGraph::InvalidatePairsOutsideActiveImageIds,
           "active_image_ids"_a,
           "Mark image pairs as invalid if either image is not in the active "
           "set.")
      .def(
          "mark_connected_components",
          [](const PoseGraph& self,
             const Reconstruction& reconstruction,
             int min_num_images) {
            std::unordered_map<frame_t, int> cluster_ids;
            int num_components = self.MarkConnectedComponents(
                reconstruction, cluster_ids, min_num_images);
            return py::dict("num_components"_a = num_components,
                            "cluster_ids"_a = cluster_ids);
          },
          "reconstruction"_a,
          "min_num_images"_a = -1,
          "Mark connected clusters of images. Returns dict with num_components "
          "and cluster_ids mapping frame IDs to cluster IDs.");
}
