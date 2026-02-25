#include "colmap/scene/correspondence_graph.h"

#include "colmap/feature/types.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindCorrespondenceGraph(py::module& m) {
  py::classh<CorrespondenceGraph::Correspondence> PyCorrespondence(
      m, "Correspondence");
  PyCorrespondence.def(py::init<>())
      .def(py::init<image_t, point2D_t>(), "image_id"_a, "point2D_idx"_a)
      .def_readwrite("image_id", &CorrespondenceGraph::Correspondence::image_id)
      .def_readwrite("point2D_idx",
                     &CorrespondenceGraph::Correspondence::point2D_idx);
  MakeDataclass(PyCorrespondence);

  py::class_<CorrespondenceGraph::CorrespondenceRange>(m, "CorrespondenceRange")
      .def_property_readonly(
          "empty",
          [](const CorrespondenceGraph::CorrespondenceRange& self) {
            return self.beg == self.end;
          },
          "Whether the range is empty.")
      .def(
          "to_list",
          [](const CorrespondenceGraph::CorrespondenceRange& self) {
            return std::vector<CorrespondenceGraph::Correspondence>(self.beg,
                                                                    self.end);
          },
          "Convert range to list of correspondences.");

  auto PyCorrespondenceGraph =
      py::classh<CorrespondenceGraph>(m, "CorrespondenceGraph");
  PyCorrespondenceGraph.def(py::init<>())
      .def("finalize", &CorrespondenceGraph::Finalize)
      .def("num_images", &CorrespondenceGraph::NumImages)
      .def("num_image_pairs", &CorrespondenceGraph::NumImagePairs)
      .def("num_observations_for_image",
           &CorrespondenceGraph::NumObservationsForImage,
           "image_id"_a)
      .def("num_correspondences_for_image",
           &CorrespondenceGraph::NumCorrespondencesForImage,
           "image_id"_a)
      .def("num_matches_between_images",
           py::overload_cast<image_t, image_t>(
               &CorrespondenceGraph::NumMatchesBetweenImages, py::const_),
           "image_id1"_a,
           "image_id2"_a)
      .def("num_matches_between_all_images",
           py::overload_cast<>(&CorrespondenceGraph::NumMatchesBetweenAllImages,
                               py::const_))
      .def("exists_image", &CorrespondenceGraph::ExistsImage, "image_id"_a)
      .def("image_pairs", &CorrespondenceGraph::ImagePairs)
      .def("add_image",
           &CorrespondenceGraph::AddImage,
           "image_id"_a,
           "num_points2D"_a)
      .def("add_two_view_geometry",
           &CorrespondenceGraph::AddTwoViewGeometry,
           "image_id1"_a,
           "image_id2"_a,
           "two_view_geometry"_a)
      .def(
          "extract_correspondences",
          [](const CorrespondenceGraph& self,
             const image_t image_id,
             const point2D_t point2D_idx) {
            std::vector<CorrespondenceGraph::Correspondence> correspondences;
            self.ExtractCorrespondences(
                image_id, point2D_idx, &correspondences);
            return correspondences;
          },
          "image_id"_a,
          "point2D_idx"_a)
      .def(
          "extract_transitive_correspondences",
          [](const CorrespondenceGraph& self,
             const image_t image_id,
             const point2D_t point2D_idx,
             const size_t transitivity) {
            std::vector<CorrespondenceGraph::Correspondence> correspondences;
            self.ExtractTransitiveCorrespondences(
                image_id, point2D_idx, transitivity, &correspondences);
            return correspondences;
          },
          "image_id"_a,
          "point2D_idx"_a,
          "transitivity"_a)
      .def(
          "extract_matches_between_images",
          [](const CorrespondenceGraph& self,
             const image_t image_id1,
             const image_t image_id2) -> FeatureMatchesMatrix {
            FeatureMatches matches;
            self.ExtractMatchesBetweenImages(image_id1, image_id2, matches);
            return MatchesToMatrix(matches);
          },
          "image_id1"_a,
          "image_id2"_a)
      .def("extract_two_view_geometry",
           &CorrespondenceGraph::ExtractTwoViewGeometry,
           "image_id1"_a,
           "image_id2"_a,
           "extract_inlier_matches"_a)
      .def("has_correspondences",
           &CorrespondenceGraph::HasCorrespondences,
           "image_id"_a,
           "point2D_idx"_a)
      .def("find_correspondences",
           &CorrespondenceGraph::FindCorrespondences,
           "image_id"_a,
           "point2D_idx"_a,
           "Find range of correspondences of an image observation.")
      .def("is_two_view_observation",
           &CorrespondenceGraph::IsTwoViewObservation,
           "image_id"_a,
           "point2D_idx"_a)
      .def("__copy__",
           [](const CorrespondenceGraph& self) {
             return CorrespondenceGraph(self);
           })
      .def("__deepcopy__",
           [](const CorrespondenceGraph& self, const py::dict&) {
             return CorrespondenceGraph(self);
           })
      .def("__repr__", &CreateRepresentation<CorrespondenceGraph>);
  DefDeprecation(PyCorrespondenceGraph,
                 "num_correspondences_between_images",
                 "num_matches_between_images");
  DefDeprecation(PyCorrespondenceGraph,
                 "num_correspondences_between_all_images",
                 "num_matches_between_all_images");
  DefDeprecation(PyCorrespondenceGraph,
                 "find_correspondences_between_images",
                 "extract_matches_between_images");
}
