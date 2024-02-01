#pragma once

#include "colmap/feature/types.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

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
      .def(py::init<image_t, point2D_t>())
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
      .def("exists_image", &CorrespondenceGraph::ExistsImage)
      .def("num_observations_for_image",
           &CorrespondenceGraph::NumObservationsForImage)
      .def("num_correspondences_for_image",
           &CorrespondenceGraph::NumCorrespondencesForImage)
      .def("num_correspondences_between_images",
           [](const CorrespondenceGraph& self,
              const image_t image_id1,
              const image_t image_id2) {
             return self.NumCorrespondencesBetweenImages(image_id1, image_id2);
           })
      .def("finalize", &CorrespondenceGraph::Finalize)
      .def("add_image", &CorrespondenceGraph::AddImage)
      .def(
          "add_correspondences",
          [](CorrespondenceGraph& self,
             const image_t image_id1,
             const image_t image_id2,
             const Eigen::Ref<Eigen::Matrix<point2D_t, -1, 2, Eigen::RowMajor>>&
                 corrs) {
            FeatureMatches matches;
            matches.reserve(corrs.rows());
            for (Eigen::Index idx = 0; idx < corrs.rows(); ++idx) {
              matches.push_back(FeatureMatch(corrs(idx, 0), corrs(idx, 1)));
            }
            self.AddCorrespondences(image_id1, image_id2, matches);
          })
      .def("extract_correspondences",
           &CorrespondenceGraph::ExtractCorrespondences)
      .def("extract_transitive_correspondences",
           &CorrespondenceGraph::ExtractTransitiveCorrespondences)
      .def("find_correspondences_between_images",
           [](const CorrespondenceGraph& self,
              const image_t image_id1,
              const image_t image_id2) {
             const FeatureMatches matches =
                 self.FindCorrespondencesBetweenImages(image_id1, image_id2);
             Eigen::Matrix<point2D_t, Eigen::Dynamic, 2, Eigen::RowMajor> corrs(
                 matches.size(), 2);
             for (size_t idx = 0; idx < matches.size(); ++idx) {
               corrs(idx, 0) = matches[idx].point2D_idx1;
               corrs(idx, 1) = matches[idx].point2D_idx2;
             }
             return corrs;
           })
      .def("has_correspondences", &CorrespondenceGraph::HasCorrespondences)
      .def("is_two_view_observation",
           &CorrespondenceGraph::IsTwoViewObservation)
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
