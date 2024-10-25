#include "colmap/sfm/observation_manager.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <memory>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
namespace py = pybind11;

void BindObservationManager(py::module& m) {
  using ImagePairStat = ObservationManager::ImagePairStat;
  py::class_ext_<ImagePairStat, std::shared_ptr<ImagePairStat>>(m,
                                                                "ImagePairStat")
      .def(py::init<>())
      .def_readwrite("num_tri_corrs", &ImagePairStat::num_tri_corrs)
      .def_readwrite("num_total_corrs", &ImagePairStat::num_total_corrs);

  py::class_<ObservationManager, std::shared_ptr<ObservationManager>>(
      m, "ObservationManager")
      .def(py::init<Reconstruction&,
                    std::shared_ptr<const CorrespondenceGraph>>(),
           "reconstruction"_a,
           "correspondence_graph"_a = py::none(),
           py::keep_alive<1, 2>())
      .def_property_readonly("image_pairs", &ObservationManager::ImagePairs)
      .def("add_point3D",
           &ObservationManager::AddPoint3D,
           "xyz"_a,
           "track"_a,
           "color"_a = Eigen::Vector3ub::Zero(),
           "Add new 3D object, and return its unique ID.")
      .def("add_observation",
           &ObservationManager::AddObservation,
           "point3D_id"_a,
           "track_element"_a,
           "Add observation to existing 3D point.")
      .def("delete_point3D",
           &ObservationManager::DeletePoint3D,
           "point3D_id"_a,
           "Delete a 3D point, and all its references in the observed images.")
      .def("delete_observation",
           &ObservationManager::DeleteObservation,
           "image_id"_a,
           "point2D_idx"_a,
           "Delete one observation from an image and the corresponding 3D "
           "point. Note that this deletes the entire 3D point, if the track has"
           " two elements prior to calling this method.")
      .def(
          "merge_points3D",
          &ObservationManager::MergePoints3D,
          "point3D_id1"_a,
          "point3D_id2"_a,
          "Merge two 3D points and return new identifier of new 3D point."
          "The location of the merged 3D point is a weighted average of the "
          "two original 3D point's locations according to their track lengths.")
      .def("filter_points3D",
           &ObservationManager::FilterPoints3D,
           "max_reproj_error"_a,
           "min_tri_angle"_a,
           "point3D_ids"_a,
           "Filter 3D points with large reprojection error, negative depth, or"
           "insufficient triangulation angle. Return the number of filtered "
           "observations.")
      .def("filter_points3D_in_images",
           &ObservationManager::FilterPoints3DInImages,
           "max_reproj_error"_a,
           "min_tri_angle"_a,
           "image_ids"_a,
           "Filter 3D points with large reprojection error, negative depth, or"
           "insufficient triangulation angle. Return the number of filtered "
           "observations.")
      .def("filter_all_points3D",
           &ObservationManager::FilterAllPoints3D,
           "max_reproj_error"_a,
           "min_tri_angle"_a,
           "Filter 3D points with large reprojection error, negative depth, or"
           "insufficient triangulation angle. Return the number of filtered "
           "observations.")
      .def("filter_observations_with_negative_depth",
           &ObservationManager::FilterObservationsWithNegativeDepth,
           "Filter observations that have negative depth. Return the number of "
           "filtered observations.")
      .def("filter_images",
           &ObservationManager::FilterImages,
           "min_focal_length_ratio"_a,
           "max_focal_length_ratio"_a,
           "max_extra_param"_a,
           "Filter images without observations or bogus camera parameters."
           "Return the identifiers of the filtered images.")
      .def("deregister_image",
           &ObservationManager::DeRegisterImage,
           "image_id"_a,
           "De-register an existing image, and all its references.")
      .def("num_observations",
           &ObservationManager::NumObservations,
           "image_id"_a,
           "Number of observations, i.e. the number of image points that"
           "have at least one correspondence to another image.")
      .def("num_correspondences",
           &ObservationManager::NumCorrespondences,
           "image_id"_a,
           "Number of correspondences for all image points.")
      .def("num_visible_points3D",
           &ObservationManager::NumVisiblePoints3D,
           "image_id"_a,
           "Get the number of observations that see a triangulated point, i.e. "
           "the number of image points that have at least one correspondence to"
           "a triangulated point in another image.")
      .def("point3D_visibility_score",
           &ObservationManager::Point3DVisibilityScore,
           "image_id"_a,
           "Get the score of triangulated observations. In contrast to"
           "`NumVisiblePoints3D`, this score also captures the distribution"
           "of triangulated observations in the image. This is useful to "
           "select the next best image in incremental reconstruction, because a"
           "more uniform distribution of observations results in more robust "
           "registration.")
      .def("increment_correspondence_has_point3D",
           &ObservationManager::IncrementCorrespondenceHasPoint3D,
           "image_id"_a,
           "point2D_idx"_a,
           "Indicate that another image has a point that is triangulated and "
           "has a correspondence to this image point.")
      .def("decrement_correspondence_has_point3D",
           &ObservationManager::DecrementCorrespondenceHasPoint3D,
           "image_id"_a,
           "point2D_idx"_a,
           "Indicate that another image has a point that is not triangulated "
           "any more and has a correspondence to this image point. This assumes"
           "that `IncrementCorrespondenceHasPoint3D` was called for the same"
           "image point and correspondence before.")
      .def("__repr__", &CreateRepresentation<ObservationManager>);
}
