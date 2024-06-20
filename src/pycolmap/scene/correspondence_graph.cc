#include "colmap/scene/correspondence_graph.h"

#include "colmap/feature/types.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include "pycolmap/utils.h"

#include <memory>
#include <sstream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindCorrespondenceGraph(py::module& m) {
  py::class_<CorrespondenceGraph::Correspondence,
             std::shared_ptr<CorrespondenceGraph::Correspondence>>(
      m, "Correspondence")
      .def(py::init<>())
      .def(py::init<image_t, point2D_t>(), "image_id"_a, "point2D_idx"_a)
      .def_readwrite("image_id", &CorrespondenceGraph::Correspondence::image_id)
      .def_readwrite("point2D_idx",
                     &CorrespondenceGraph::Correspondence::point2D_idx)
      .def("__copy__",
           [](const CorrespondenceGraph::Correspondence& self) {
             return CorrespondenceGraph::Correspondence(self);
           })
      .def(
          "__deepcopy__",
          [](const CorrespondenceGraph::Correspondence& self, const py::dict&) {
            return CorrespondenceGraph::Correspondence(self);
          })
      .def("__repr__", [](const CorrespondenceGraph::Correspondence& self) {
        return "Correspondence(image_id=" + std::to_string(self.image_id) +
               ", point2D_idx=" + std::to_string(self.point2D_idx) + ")";
      });

  py::class_<CorrespondenceGraph, std::shared_ptr<CorrespondenceGraph>>(
      m, "CorrespondenceGraph")
      .def(py::init<>())
      .def("num_images", &CorrespondenceGraph::NumImages)
      .def("num_image_pairs", &CorrespondenceGraph::NumImagePairs)
      .def("exists_image", &CorrespondenceGraph::ExistsImage, "image_id"_a)
      .def("num_observations_for_image",
           &CorrespondenceGraph::NumObservationsForImage,
           "image_id"_a)
      .def("num_correspondences_for_image",
           &CorrespondenceGraph::NumCorrespondencesForImage,
           "image_id"_a)
      .def("num_correspondences_between_images",
           py::overload_cast<image_t, image_t>(
               &CorrespondenceGraph::NumCorrespondencesBetweenImages,
               py::const_),
           "image_id1"_a,
           "image_id2"_a)
      .def("num_correspondences_between_all_images",
           py::overload_cast<>(
               &CorrespondenceGraph::NumCorrespondencesBetweenImages,
               py::const_))
      .def("finalize", &CorrespondenceGraph::Finalize)
      .def("add_image",
           &CorrespondenceGraph::AddImage,
           "image_id"_a,
           "num_points2D"_a)
      .def(
          "add_correspondences",
          [](CorrespondenceGraph& self,
             const image_t image_id1,
             const image_t image_id2,
             const PyFeatureMatches& corrs) {
            FeatureMatches matches = FeatureMatchesFromMatrix(corrs);
            self.AddCorrespondences(image_id1, image_id2, matches);
          },
          "image_id1"_a,
          "image_id2"_a,
          "correspondences"_a)
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
          "find_correspondences_between_images",
          [](const CorrespondenceGraph& self,
             const image_t image_id1,
             const image_t image_id2) -> PyFeatureMatches {
            const FeatureMatches matches =
                self.FindCorrespondencesBetweenImages(image_id1, image_id2);
            return FeatureMatchesToMatrix(matches);
          },
          "image_id1"_a,
          "image_id2"_a)
      .def("has_correspondences",
           &CorrespondenceGraph::HasCorrespondences,
           "image_id"_a,
           "point2D_idx"_a)
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
      .def("__repr__", [](const CorrespondenceGraph& self) {
        std::stringstream ss;
        ss << "CorrespondenceGraph(num_images=" << self.NumImages()
           << ", num_image_pairs=" << self.NumImagePairs() << ")";
        return ss.str();
      });
}
