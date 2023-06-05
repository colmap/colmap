// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_BASE_RECONSTRUCTION_H_
#define COLMAP_SRC_BASE_RECONSTRUCTION_H_

#include "colmap/base/camera.h"
#include "colmap/base/database.h"
#include "colmap/base/image.h"
#include "colmap/base/point2d.h"
#include "colmap/base/point3d.h"
#include "colmap/base/similarity_transform.h"
#include "colmap/base/track.h"
#include "colmap/estimators/similarity_transform.h"
#include "colmap/optim/loransac.h"
#include "colmap/util/types.h"

#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

namespace colmap {

struct PlyPoint;
struct RANSACOptions;
class DatabaseCache;
class CorrespondenceGraph;

// Reconstruction class holds all information about a single reconstructed
// model. It is used by the mapping and bundle adjustment classes and can be
// written to and read from disk.
class Reconstruction {
 public:
  struct ImagePairStat {
    // The number of triangulated correspondences between two images.
    size_t num_tri_corrs = 0;
    // The number of total correspondences/matches between two images.
    size_t num_total_corrs = 0;
  };

  Reconstruction();

  // Get number of objects.
  inline size_t NumCameras() const;
  inline size_t NumImages() const;
  inline size_t NumRegImages() const;
  inline size_t NumPoints3D() const;
  inline size_t NumImagePairs() const;

  // Get const objects.
  inline const class Camera& Camera(const camera_t camera_id) const;
  inline const class Image& Image(const image_t image_id) const;
  inline const class Point3D& Point3D(const point3D_t point3D_id) const;
  inline const ImagePairStat& ImagePair(const image_pair_t pair_id) const;
  inline ImagePairStat& ImagePair(const image_t image_id1,
                                  const image_t image_id2);

  // Get mutable objects.
  inline class Camera& Camera(const camera_t camera_id);
  inline class Image& Image(const image_t image_id);
  inline class Point3D& Point3D(const point3D_t point3D_id);
  inline ImagePairStat& ImagePair(const image_pair_t pair_id);
  inline const ImagePairStat& ImagePair(const image_t image_id1,
                                        const image_t image_id2) const;

  // Get reference to all objects.
  inline const std::unordered_map<camera_t, class Camera>& Cameras() const;
  inline const std::unordered_map<image_t, class Image>& Images() const;
  inline const std::vector<image_t>& RegImageIds() const;
  inline const std::unordered_map<point3D_t, class Point3D>& Points3D() const;
  inline const std::unordered_map<image_pair_t, ImagePairStat>& ImagePairs()
      const;

  // Identifiers of all 3D points.
  std::unordered_set<point3D_t> Point3DIds() const;

  // Check whether specific object exists.
  inline bool ExistsCamera(const camera_t camera_id) const;
  inline bool ExistsImage(const image_t image_id) const;
  inline bool ExistsPoint3D(const point3D_t point3D_id) const;
  inline bool ExistsImagePair(const image_pair_t pair_id) const;

  // Load data from given `DatabaseCache`.
  void Load(const DatabaseCache& database_cache);

  // Setup all relevant data structures before reconstruction. Note the
  // correspondence graph object must live until `TearDown` is called.
  void SetUp(const CorrespondenceGraph* correspondence_graph);

  // Finalize the Reconstruction after the reconstruction has finished.
  //
  // Once a scene has been finalized, it cannot be used for reconstruction.
  //
  // This removes all not yet registered images and unused cameras, in order to
  // save memory.
  void TearDown();

  // Add new camera. There is only one camera per image, while multiple images
  // might be taken by the same camera.
  void AddCamera(class Camera camera);

  // Add new image.
  void AddImage(class Image image);

  // Add new 3D object, and return its unique ID.
  point3D_t AddPoint3D(
      const Eigen::Vector3d& xyz,
      Track track,
      const Eigen::Vector3ub& color = Eigen::Vector3ub::Zero());

  // Add observation to existing 3D point.
  void AddObservation(const point3D_t point3D_id, const TrackElement& track_el);

  // Merge two 3D points and return new identifier of new 3D point.
  // The location of the merged 3D point is a weighted average of the two
  // original 3D point's locations according to their track lengths.
  point3D_t MergePoints3D(const point3D_t point3D_id1,
                          const point3D_t point3D_id2);

  // Delete a 3D point, and all its references in the observed images.
  void DeletePoint3D(const point3D_t point3D_id);

  // Delete one observation from an image and the corresponding 3D point.
  // Note that this deletes the entire 3D point, if the track has two elements
  // prior to calling this method.
  void DeleteObservation(const image_t image_id, const point2D_t point2D_idx);

  // Delete all 2D points of all images and all 3D points.
  void DeleteAllPoints2DAndPoints3D();

  // Register an existing image.
  void RegisterImage(const image_t image_id);

  // De-register an existing image, and all its references.
  void DeRegisterImage(const image_t image_id);

  // Check if image is registered.
  inline bool IsImageRegistered(const image_t image_id) const;

  // Normalize scene by scaling and translation to avoid degenerate
  // visualization after bundle adjustment and to improve numerical
  // stability of algorithms.
  //
  // Translates scene such that the mean of the camera centers or point
  // locations are at the origin of the coordinate system.
  //
  // Scales scene such that the minimum and maximum camera centers are at the
  // given `extent`, whereas `p0` and `p1` determine the minimum and
  // maximum percentiles of the camera centers considered.
  void Normalize(const double extent = 10.0,
                 const double p0 = 0.1,
                 const double p1 = 0.9,
                 const bool use_images = true);

  // Compute the centroid of the 3D points
  Eigen::Vector3d ComputeCentroid(const double p0 = 0.1,
                                  const double p1 = 0.9) const;

  // Compute the bounding box corners of the 3D points
  std::pair<Eigen::Vector3d, Eigen::Vector3d> ComputeBoundingBox(
      const double p0 = 0.0, const double p1 = 1.0) const;

  // Apply the 3D similarity transformation to all images and points.
  void Transform(const SimilarityTransform3& tform);

  // Creates a cropped reconstruction using the input bounds as corner points
  // of the bounding box containing the included 3D points of the new
  // reconstruction. Only the cameras and images of the included points are
  // registered.
  Reconstruction Crop(
      const std::pair<Eigen::Vector3d, Eigen::Vector3d>& bbox) const;

  // Merge the given reconstruction into this reconstruction by registering the
  // images registered in the given but not in this reconstruction and by
  // merging the two clouds and their tracks. The coordinate frames of the two
  // reconstructions are aligned using the projection centers of common
  // registered images. Return true if the two reconstructions could be merged.
  bool Merge(const Reconstruction& reconstruction,
             const double max_reproj_error);

  // Align the given reconstruction with a set of pre-defined camera positions.
  // Assuming that locations[i] gives the 3D coordinates of the center
  // of projection of the image with name image_names[i].
  template <bool kEstimateScale = true>
  bool Align(const std::vector<std::string>& image_names,
             const std::vector<Eigen::Vector3d>& locations,
             const int min_common_images,
             SimilarityTransform3* tform = nullptr);

  // Robust alignment using RANSAC.
  template <bool kEstimateScale = true>
  bool AlignRobust(const std::vector<std::string>& image_names,
                   const std::vector<Eigen::Vector3d>& locations,
                   const int min_common_images,
                   const RANSACOptions& ransac_options,
                   SimilarityTransform3* tform = nullptr);

  // Find specific image by name. Note that this uses linear search.
  const class Image* FindImageWithName(const std::string& name) const;

  // Find images that are both present in this and the given reconstruction.
  std::vector<image_t> FindCommonRegImageIds(
      const Reconstruction& reconstruction) const;

  // Update the image identifiers to match the ones in the database by matching
  // the names of the images.
  void TranscribeImageIdsToDatabase(const Database& database);

  // Filter 3D points with large reprojection error, negative depth, or
  // insufficient triangulation angle.
  //
  // @param max_reproj_error    The maximum reprojection error.
  // @param min_tri_angle       The minimum triangulation angle.
  // @param point3D_ids         The points to be filtered.
  //
  // @return                    The number of filtered observations.
  size_t FilterPoints3D(const double max_reproj_error,
                        const double min_tri_angle,
                        const std::unordered_set<point3D_t>& point3D_ids);
  size_t FilterPoints3DInImages(const double max_reproj_error,
                                const double min_tri_angle,
                                const std::unordered_set<image_t>& image_ids);
  size_t FilterAllPoints3D(const double max_reproj_error,
                           const double min_tri_angle);

  // Filter observations that have negative depth.
  //
  // @return    The number of filtered observations.
  size_t FilterObservationsWithNegativeDepth();

  // Filter images without observations or bogus camera parameters.
  //
  // @return    The identifiers of the filtered images.
  std::vector<image_t> FilterImages(const double min_focal_length_ratio,
                                    const double max_focal_length_ratio,
                                    const double max_extra_param);

  // Compute statistics for scene.
  size_t ComputeNumObservations() const;
  double ComputeMeanTrackLength() const;
  double ComputeMeanObservationsPerRegImage() const;
  double ComputeMeanReprojectionError() const;

  // Read data from text or binary file. Prefer binary data if it exists.
  void Read(const std::string& path);
  void Write(const std::string& path) const;

  // Read data from binary/text file.
  void ReadText(const std::string& path);
  void ReadBinary(const std::string& path);

  // Write data from binary/text file.
  void WriteText(const std::string& path) const;
  void WriteBinary(const std::string& path) const;

  // Convert 3D points in reconstruction to PLY point cloud.
  std::vector<PlyPoint> ConvertToPLY() const;

  // Import from other data formats. Note that these import functions are
  // only intended for visualization of data and usable for reconstruction.
  void ImportPLY(const std::string& path);
  void ImportPLY(const std::vector<PlyPoint>& ply_points);

  // Export to other data formats.

  // Exports in NVM format http://ccwu.me/vsfm/doc.html#nvm. Only supports
  // SIMPLE_RADIAL camera model when exporting distortion parameters. When
  // skip_distortion == true it supports all camera models with the caveat that
  // it's using the mean focal length which will be inaccurate for camera models
  // with two focal lengths and distortion.
  bool ExportNVM(const std::string& path, bool skip_distortion = false) const;

  // Exports in CAM format which is a simple text file that contains pose
  // information and camera intrinsics for each image and exports one file per
  // image; it does not include information on the 3D points. The format is as
  // follows (2 lines of text with space separated numbers):
  // <Tvec; 3 values> <Rotation matrix in row-major format; 9 values>
  // <focal_length> <k1> <k2> 1.0 <principal point X> <principal point Y>
  // Note that focal length is relative to the image max(width, height),
  // and principal points x and y are relative to width and height respectively.
  //
  // Only supports SIMPLE_RADIAL and RADIAL camera models when exporting
  // distortion parameters. When skip_distortion == true it supports all camera
  // models with the caveat that it's using the mean focal length which will be
  // inaccurate for camera models with two focal lengths and distortion.
  bool ExportCam(const std::string& path, bool skip_distortion = false) const;

  // Exports in Recon3D format which consists of three text files with the
  // following format and content:
  // 1) imagemap_0.txt: a list of image numeric IDs with one entry per line.
  // 2) urd-images.txt: A list of images with one entry per line as:
  //    <image file name> <width> <height>
  // 3) synth_0.out: Contains information for image poses, camera intrinsics,
  //    and 3D points as:
  //    <N; num images> <M; num points>
  //    <N lines of image entries>
  //    <M lines of point entries>
  //
  //    Each image entry consists of 5 lines as:
  //    <focal length> <k1> <k2>
  //    <Rotation matrix; 3x3 array>
  //    <Tvec; 3 values>
  //    Note that the focal length is scaled by 1 / max(width, height)
  //
  //    Each point entry consists of 3 lines as:
  //    <point x, y, z coordinates>
  //    <point RGB color>
  //    <K; num track elements> <Track Element 1> ... <Track Element K>
  //
  //    Each track elemenet is a sequence of 5 values as:
  //    <image ID> <2D point ID> -1.0 <X> <Y>
  //    Note that the 2D point coordinates are centered around the principal
  //    point and scaled by 1 / max(width, height).
  //
  // When skip_distortion == true it supports all camera models with the
  // caveat that it's using the mean focal length which will be inaccurate
  // for camera models with two focal lengths and distortion.
  bool ExportRecon3D(const std::string& path,
                     bool skip_distortion = false) const;

  // Exports in Bundler format https://www.cs.cornell.edu/~snavely/bundler/.
  // Supports SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL and RADIAL camera models
  // when exporting distortion parameters. When skip_distortion == true it
  // supports all camera models with the caveat that it's using the mean focal
  // length which will be inaccurate for camera models with two focal lengths
  // and distortion.
  bool ExportBundler(const std::string& path,
                     const std::string& list_path,
                     bool skip_distortion = false) const;

  // Exports 3D points only in PLY format.
  void ExportPLY(const std::string& path) const;

  // Exports in VRML format https://en.wikipedia.org/wiki/VRML.
  void ExportVRML(const std::string& images_path,
                  const std::string& points3D_path,
                  const double image_scale,
                  const Eigen::Vector3d& image_rgb) const;

  // Extract colors for 3D points of given image. Colors will be extracted
  // only for 3D points which are completely black.
  //
  // @param image_id      Identifier of the image for which to extract colors.
  // @param path          Absolute or relative path to root folder of image.
  //                      The image path is determined by concatenating the
  //                      root path and the name of the image.
  //
  // @return              True if image could be read at given path.
  bool ExtractColorsForImage(const image_t image_id, const std::string& path);

  // Extract colors for all 3D points by computing the mean color of all images.
  //
  // @param path          Absolute or relative path to root folder of image.
  //                      The image path is determined by concatenating the
  //                      root path and the name of the image.
  void ExtractColorsForAllImages(const std::string& path);

  // Create all image sub-directories in the given path.
  void CreateImageDirs(const std::string& path) const;

 private:
  size_t FilterPoints3DWithSmallTriangulationAngle(
      const double min_tri_angle,
      const std::unordered_set<point3D_t>& point3D_ids);
  size_t FilterPoints3DWithLargeReprojectionError(
      const double max_reproj_error,
      const std::unordered_set<point3D_t>& point3D_ids);

  std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>
  ComputeBoundsAndCentroid(const double p0,
                           const double p1,
                           const bool use_images) const;

  void ReadCamerasText(const std::string& path);
  void ReadImagesText(const std::string& path);
  void ReadPoints3DText(const std::string& path);
  void ReadCamerasBinary(const std::string& path);
  void ReadImagesBinary(const std::string& path);
  void ReadPoints3DBinary(const std::string& path);

  void WriteCamerasText(const std::string& path) const;
  void WriteImagesText(const std::string& path) const;
  void WritePoints3DText(const std::string& path) const;
  void WriteCamerasBinary(const std::string& path) const;
  void WriteImagesBinary(const std::string& path) const;
  void WritePoints3DBinary(const std::string& path) const;

  void SetObservationAsTriangulated(const image_t image_id,
                                    const point2D_t point2D_idx,
                                    const bool is_continued_point3D);
  void ResetTriObservations(const image_t image_id,
                            const point2D_t point2D_idx,
                            const bool is_deleted_point3D);

  const CorrespondenceGraph* correspondence_graph_;

  std::unordered_map<camera_t, class Camera> cameras_;
  std::unordered_map<image_t, class Image> images_;
  std::unordered_map<point3D_t, class Point3D> points3D_;

  std::unordered_map<image_pair_t, ImagePairStat> image_pair_stats_;

  // { image_id, ... } where `images_.at(image_id).registered == true`.
  std::vector<image_t> reg_image_ids_;

  // Total number of added 3D points, used to generate unique identifiers.
  point3D_t num_added_points3D_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t Reconstruction::NumCameras() const { return cameras_.size(); }

size_t Reconstruction::NumImages() const { return images_.size(); }

size_t Reconstruction::NumRegImages() const { return reg_image_ids_.size(); }

size_t Reconstruction::NumPoints3D() const { return points3D_.size(); }

size_t Reconstruction::NumImagePairs() const {
  return image_pair_stats_.size();
}

const class Camera& Reconstruction::Camera(const camera_t camera_id) const {
  return cameras_.at(camera_id);
}

const class Image& Reconstruction::Image(const image_t image_id) const {
  return images_.at(image_id);
}

const class Point3D& Reconstruction::Point3D(const point3D_t point3D_id) const {
  return points3D_.at(point3D_id);
}

const Reconstruction::ImagePairStat& Reconstruction::ImagePair(
    const image_pair_t pair_id) const {
  return image_pair_stats_.at(pair_id);
}

const Reconstruction::ImagePairStat& Reconstruction::ImagePair(
    const image_t image_id1, const image_t image_id2) const {
  const auto pair_id = Database::ImagePairToPairId(image_id1, image_id2);
  return image_pair_stats_.at(pair_id);
}

class Camera& Reconstruction::Camera(const camera_t camera_id) {
  return cameras_.at(camera_id);
}

class Image& Reconstruction::Image(const image_t image_id) {
  return images_.at(image_id);
}

class Point3D& Reconstruction::Point3D(const point3D_t point3D_id) {
  return points3D_.at(point3D_id);
}

Reconstruction::ImagePairStat& Reconstruction::ImagePair(
    const image_pair_t pair_id) {
  return image_pair_stats_.at(pair_id);
}

Reconstruction::ImagePairStat& Reconstruction::ImagePair(
    const image_t image_id1, const image_t image_id2) {
  const auto pair_id = Database::ImagePairToPairId(image_id1, image_id2);
  return image_pair_stats_.at(pair_id);
}

const std::unordered_map<camera_t, Camera>& Reconstruction::Cameras() const {
  return cameras_;
}

const std::unordered_map<image_t, class Image>& Reconstruction::Images() const {
  return images_;
}

const std::vector<image_t>& Reconstruction::RegImageIds() const {
  return reg_image_ids_;
}

const std::unordered_map<point3D_t, Point3D>& Reconstruction::Points3D() const {
  return points3D_;
}

const std::unordered_map<image_pair_t, Reconstruction::ImagePairStat>&
Reconstruction::ImagePairs() const {
  return image_pair_stats_;
}

bool Reconstruction::ExistsCamera(const camera_t camera_id) const {
  return cameras_.find(camera_id) != cameras_.end();
}

bool Reconstruction::ExistsImage(const image_t image_id) const {
  return images_.find(image_id) != images_.end();
}

bool Reconstruction::ExistsPoint3D(const point3D_t point3D_id) const {
  return points3D_.find(point3D_id) != points3D_.end();
}

bool Reconstruction::ExistsImagePair(const image_pair_t pair_id) const {
  return image_pair_stats_.find(pair_id) != image_pair_stats_.end();
}

bool Reconstruction::IsImageRegistered(const image_t image_id) const {
  return Image(image_id).IsRegistered();
}

template <bool kEstimateScale>
bool Reconstruction::Align(const std::vector<std::string>& image_names,
                           const std::vector<Eigen::Vector3d>& locations,
                           const int min_common_images,
                           SimilarityTransform3* tform) {
  CHECK_GE(min_common_images, 3);
  CHECK_EQ(image_names.size(), locations.size());

  // Find out which images are contained in the reconstruction and get the
  // positions of their camera centers.
  std::unordered_set<image_t> common_image_ids;
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < image_names.size(); ++i) {
    const class Image* image = FindImageWithName(image_names[i]);
    if (image == nullptr) {
      continue;
    }

    if (!IsImageRegistered(image->ImageId())) {
      continue;
    }

    // Ignore duplicate images.
    if (common_image_ids.count(image->ImageId()) > 0) {
      continue;
    }

    common_image_ids.insert(image->ImageId());
    src.push_back(image->ProjectionCenter());
    dst.push_back(locations[i]);
  }

  // Only compute the alignment if there are enough correspondences.
  if (common_image_ids.size() < static_cast<size_t>(min_common_images)) {
    return false;
  }

  SimilarityTransform3 transform;
  if (!transform.Estimate<kEstimateScale>(src, dst)) {
    return false;
  }

  Transform(transform);

  if (tform != nullptr) {
    *tform = transform;
  }

  return true;
}

template <bool kEstimateScale>
bool Reconstruction::AlignRobust(const std::vector<std::string>& image_names,
                                 const std::vector<Eigen::Vector3d>& locations,
                                 const int min_common_images,
                                 const RANSACOptions& ransac_options,
                                 SimilarityTransform3* tform) {
  CHECK_GE(min_common_images, 3);
  CHECK_EQ(image_names.size(), locations.size());

  // Find out which images are contained in the reconstruction and get the
  // positions of their camera centers.
  std::unordered_set<image_t> common_image_ids;
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < image_names.size(); ++i) {
    const class Image* image = FindImageWithName(image_names[i]);
    if (image == nullptr) {
      continue;
    }

    if (!IsImageRegistered(image->ImageId())) {
      continue;
    }

    // Ignore duplicate images.
    if (common_image_ids.count(image->ImageId()) > 0) {
      continue;
    }

    common_image_ids.insert(image->ImageId());
    src.push_back(image->ProjectionCenter());
    dst.push_back(locations[i]);
  }

  // Only compute the alignment if there are enough correspondences.
  if (common_image_ids.size() < static_cast<size_t>(min_common_images)) {
    return false;
  }

  LORANSAC<SimilarityTransformEstimator<3, kEstimateScale>,
           SimilarityTransformEstimator<3, kEstimateScale>>
      ransac(ransac_options);

  const auto report = ransac.Estimate(src, dst);

  if (report.support.num_inliers < static_cast<size_t>(min_common_images)) {
    return false;
  }

  SimilarityTransform3 transform = SimilarityTransform3(report.model);
  Transform(transform);

  if (tform != nullptr) {
    *tform = transform;
  }

  return true;
}

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_RECONSTRUCTION_H_
