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

#include "colmap/scene/synthetic.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/pose.h"
#include "colmap/math/random.h"
#include "colmap/scene/projection.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Geometry>

namespace colmap {
namespace {

void AddOutlierMatches(double inlier_ratio,
                       int num_points2D1,
                       int num_points2D2,
                       FeatureMatches* matches) {
  const int num_outliers = matches->size() * (1.0 - inlier_ratio);
  for (int i = 0; i < num_outliers; ++i) {
    matches->emplace_back(
        RandomUniformInteger<point2D_t>(0, num_points2D1 - 1),
        RandomUniformInteger<point2D_t>(0, num_points2D2 - 2));
  }
  std::shuffle(matches->begin(), matches->end(), *PRNG);
}

void SynthesizeExhaustiveMatches(double inlier_match_ratio,
                                 Reconstruction* reconstruction,
                                 Database* database) {
  const std::vector<image_t>& reg_image_ids = reconstruction->RegImageIds();
  for (size_t image_idx1 = 0; image_idx1 < reg_image_ids.size(); ++image_idx1) {
    const auto& image1 = reconstruction->Image(reg_image_ids[image_idx1]);
    const Eigen::Matrix3d K1 =
        reconstruction->Camera(image1.CameraId()).CalibrationMatrix();
    const auto num_points2D1 = image1.NumPoints2D();
    for (size_t image_idx2 = 0; image_idx2 < image_idx1; ++image_idx2) {
      const auto& image2 = reconstruction->Image(reg_image_ids[image_idx2]);
      const Eigen::Matrix3d K2 =
          reconstruction->Camera(image2.CameraId()).CalibrationMatrix();
      const auto num_points2D2 = image2.NumPoints2D();

      TwoViewGeometry two_view_geometry;
      two_view_geometry.config = TwoViewGeometry::CALIBRATED;
      two_view_geometry.cam2_from_cam1 =
          image2.CamFromWorld() * Inverse(image1.CamFromWorld());
      two_view_geometry.E =
          EssentialMatrixFromPose(two_view_geometry.cam2_from_cam1);
      two_view_geometry.F =
          FundamentalFromEssentialMatrix(K2, two_view_geometry.E, K1);
      for (point2D_t point2D_idx1 = 0; point2D_idx1 < num_points2D1;
           ++point2D_idx1) {
        const auto& point2D1 = image1.Point2D(point2D_idx1);
        if (!point2D1.HasPoint3D()) {
          continue;
        }
        for (point2D_t point2D_idx2 = 0; point2D_idx2 < num_points2D2;
             ++point2D_idx2) {
          const auto& point2D2 = image2.Point2D(point2D_idx2);
          if (point2D1.point3D_id == point2D2.point3D_id) {
            two_view_geometry.inlier_matches.emplace_back(point2D_idx1,
                                                          point2D_idx2);
            break;
          }
        }
      }

      FeatureMatches matches = two_view_geometry.inlier_matches;
      AddOutlierMatches(
          inlier_match_ratio, num_points2D1, num_points2D2, &matches);

      database->WriteMatches(image1.ImageId(), image2.ImageId(), matches);
      database->WriteTwoViewGeometry(
          image1.ImageId(), image2.ImageId(), two_view_geometry);
    }
  }
}

void SynthesizeChainedMatches(double inlier_match_ratio,
                              Reconstruction* reconstruction,
                              Database* database) {
  std::unordered_map<image_pair_t, TwoViewGeometry> two_view_geometries;
  for (const auto& point3D : reconstruction->Points3D()) {
    std::vector<TrackElement> track_elements = point3D.second.track.Elements();
    std::shuffle(track_elements.begin(), track_elements.end(), *PRNG);
    for (size_t i = 1; i < track_elements.size(); ++i) {
      const auto& prev_track_el = track_elements[i - 1];
      const auto& curr_track_el = track_elements[i];
      const image_pair_t pair_id = Database::ImagePairToPairId(
          prev_track_el.image_id, curr_track_el.image_id);
      if (Database::SwapImagePair(prev_track_el.image_id,
                                  curr_track_el.image_id)) {
        two_view_geometries[pair_id].inlier_matches.emplace_back(
            curr_track_el.point2D_idx, prev_track_el.point2D_idx);
      } else {
        two_view_geometries[pair_id].inlier_matches.emplace_back(
            prev_track_el.point2D_idx, curr_track_el.point2D_idx);
      }
    }
  }

  for (auto& two_view_geometry : two_view_geometries) {
    const auto image_pair =
        Database::PairIdToImagePair(two_view_geometry.first);
    const auto& image1 = reconstruction->Image(image_pair.first);
    const auto& camera1 = reconstruction->Camera(image1.CameraId());
    const auto& image2 = reconstruction->Image(image_pair.second);
    const auto& camera2 = reconstruction->Camera(image2.CameraId());
    two_view_geometry.second.config = TwoViewGeometry::CALIBRATED;
    two_view_geometry.second.cam2_from_cam1 =
        image2.CamFromWorld() * Inverse(image1.CamFromWorld());
    two_view_geometry.second.E =
        EssentialMatrixFromPose(two_view_geometry.second.cam2_from_cam1);
    two_view_geometry.second.F =
        FundamentalFromEssentialMatrix(camera2.CalibrationMatrix(),
                                       two_view_geometry.second.E,
                                       camera1.CalibrationMatrix());

    FeatureMatches matches = two_view_geometry.second.inlier_matches;
    AddOutlierMatches(inlier_match_ratio,
                      image1.NumPoints2D(),
                      image2.NumPoints2D(),
                      &matches);

    database->WriteMatches(image1.ImageId(), image2.ImageId(), matches);
    database->WriteTwoViewGeometry(
        image1.ImageId(), image2.ImageId(), two_view_geometry.second);
  }
}

}  // namespace

void SynthesizeDataset(const SyntheticDatasetOptions& options,
                       Reconstruction* reconstruction,
                       Database* database) {
  THROW_CHECK_GT(options.num_cameras, 0);
  THROW_CHECK_GT(options.num_images, 0);
  THROW_CHECK_LE(options.num_cameras, options.num_images);
  THROW_CHECK_GE(options.num_points3D, 0);
  THROW_CHECK_GE(options.num_points2D_without_point3D, 0);
  THROW_CHECK_GE(options.point2D_stddev, 0);

  // Synthesize cameras.
  std::vector<camera_t> camera_ids(options.num_cameras);
  for (int camera_idx = 0; camera_idx < options.num_cameras; ++camera_idx) {
    Camera camera;
    camera.width = options.camera_width;
    camera.height = options.camera_height;
    camera.model_id = options.camera_model_id;
    camera.params = options.camera_params;
    THROW_CHECK(camera.VerifyParams());
    const camera_t camera_id =
        (database == nullptr) ? camera_idx + 1 : database->WriteCamera(camera);
    camera_ids[camera_idx] = camera_id;
    camera.camera_id = camera_id;
    reconstruction->AddCamera(std::move(camera));
  }

  // Synthesize 3D points on unit sphere centered at origin.
  for (int point3D_idx = 0; point3D_idx < options.num_points3D; ++point3D_idx) {
    reconstruction->AddPoint3D(Eigen::Vector3d::Random().normalized(),
                               /*track=*/{});
  }

  // Synthesize images.

  const int existing_num_images =
      (database == nullptr) ? 0 : database->NumImages();
  int total_num_descriptors =
      (database == nullptr) ? 0 : database->NumDescriptors();
  for (int image_idx = 0; image_idx < options.num_images; ++image_idx) {
    Image image;
    image.SetName("image" + std::to_string(existing_num_images + image_idx));
    image.SetCameraId(camera_ids[image_idx % options.num_cameras]);
    // Synthesize image poses with projection centers on sphere with radious 5
    // centered at origin.
    const Eigen::Vector3d view_dir = -Eigen::Vector3d::Random().normalized();
    const Eigen::Vector3d proj_center = -5 * view_dir;
    image.CamFromWorld().rotation =
        Eigen::Quaterniond::FromTwoVectors(view_dir, Eigen::Vector3d(0, 0, 1));
    image.CamFromWorld().translation =
        image.CamFromWorld().rotation * -proj_center;

    const Camera& camera = reconstruction->Camera(image.CameraId());

    std::vector<Point2D> points2D;
    points2D.reserve(options.num_points3D +
                     options.num_points2D_without_point3D);

    // Create 3D point observations by project all 3D points to the image.
    for (auto& point3D : reconstruction->Points3D()) {
      Point2D point2D;
      point2D.xy = camera.ImgFromCam(
          (image.CamFromWorld() * point3D.second.xyz).hnormalized());
      if (options.point2D_stddev > 0) {
        const Eigen::Vector2d noise(
            RandomGaussian<double>(0, options.point2D_stddev),
            RandomGaussian<double>(0, options.point2D_stddev));
        point2D.xy += noise;
      }
      if (point2D.xy(0) >= 0 && point2D.xy(1) >= 0 &&
          point2D.xy(0) <= camera.width && point2D.xy(1) <= camera.height) {
        point2D.point3D_id = point3D.first;
        points2D.push_back(point2D);
      }
    }

    // Synthesize uniform random 2D points without 3D points.
    for (int i = 0; i < options.num_points2D_without_point3D; ++i) {
      Point2D point2D;
      point2D.xy = Eigen::Vector2d(RandomUniformReal<double>(0, camera.width),
                                   RandomUniformReal<double>(0, camera.height));
      points2D.push_back(point2D);
    }

    // Shuffle 2D points, so each image has another order of observed 3D points.
    std::shuffle(points2D.begin(), points2D.end(), *PRNG);

    const image_t image_id =
        (database == nullptr) ? image_idx + 1 : database->WriteImage(image);
    if (database != nullptr) {
      // Create keypoints to add to database.
      FeatureKeypoints keypoints;
      keypoints.reserve(points2D.size());
      FeatureDescriptors descriptors(points2D.size(), 128);
      std::uniform_int_distribution<int> feature_distribution(0, 255);
      for (point2D_t point2D_idx = 0; point2D_idx < points2D.size();
           ++point2D_idx) {
        const auto& point2D = points2D[point2D_idx];
        keypoints.emplace_back(point2D.xy(0), point2D.xy(1));
        // Generate a unique descriptor for each 3D point. If the 2D point does
        // not observe a 3D point, generate a random unique descriptor.
        std::mt19937 feature_generator(point2D.HasPoint3D()
                                           ? point2D.point3D_id
                                           : options.num_points3D +
                                                 (++total_num_descriptors));
        for (int d = 0; d < descriptors.cols(); ++d) {
          descriptors(point2D_idx, d) = feature_distribution(feature_generator);
        }
      }
      database->WriteKeypoints(image_id, keypoints);
      database->WriteDescriptors(image_id, descriptors);
    }

    for (point2D_t point2D_idx = 0; point2D_idx < points2D.size();
         ++point2D_idx) {
      const auto& point2D = points2D[point2D_idx];
      if (point2D.HasPoint3D()) {
        auto& point3D = reconstruction->Point3D(point2D.point3D_id);
        point3D.track.AddElement(image_id, point2D_idx);
      }
    }

    image.SetImageId(image_id);
    image.SetPoints2D(points2D);
    reconstruction->AddImage(std::move(image));
    reconstruction->RegisterImage(image_id);
  }

  if (database != nullptr) {
    switch (options.match_config) {
      case SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE:
        SynthesizeExhaustiveMatches(
            options.inlier_match_ratio, reconstruction, database);
        break;
      case SyntheticDatasetOptions::MatchConfig::CHAINED:
        SynthesizeChainedMatches(
            options.inlier_match_ratio, reconstruction, database);
        break;
      default:
        LOG(FATAL_THROW) << "Invalid MatchConfig specified";
    }
  }

  reconstruction->UpdatePoint3DErrors();
}

}  // namespace colmap
