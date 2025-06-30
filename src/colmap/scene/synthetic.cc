// Copyright (c), ETH Zurich and UNC Chapel Hill.
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
  for (const image_t image_id1 : reconstruction->RegImageIds()) {
    const auto& image1 = reconstruction->Image(image_id1);
    const Eigen::Matrix3d K1 = image1.CameraPtr()->CalibrationMatrix();
    const auto num_points2D1 = image1.NumPoints2D();
    for (const image_t image_id2 : reconstruction->RegImageIds()) {
      if (image_id1 == image_id2) {
        break;
      }
      const auto& image2 = reconstruction->Image(image_id2);
      const Eigen::Matrix3d K2 = image2.CameraPtr()->CalibrationMatrix();
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
    std::sort(track_elements.begin(),
              track_elements.end(),
              [](const TrackElement& left, const TrackElement& right) {
                return left.image_id < right.image_id;
              });
    for (size_t i = 1; i < track_elements.size(); ++i) {
      const auto& prev_track_el = track_elements[i - 1];
      const auto& curr_track_el = track_elements[i];
      if (curr_track_el.image_id != prev_track_el.image_id + 1) {
        continue;
      }
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
    const auto& camera1 = *image1.CameraPtr();
    const auto& image2 = reconstruction->Image(image_pair.second);
    const auto& camera2 = *image2.CameraPtr();
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
  THROW_CHECK_GT(options.num_rigs, 0);
  THROW_CHECK_GT(options.num_cameras_per_rig, 0);
  THROW_CHECK_GT(options.num_frames_per_rig, 0);
  THROW_CHECK_GE(options.num_points3D, 0);
  THROW_CHECK_GE(options.num_points2D_without_point3D, 0);
  THROW_CHECK_GE(options.sensor_from_rig_translation_stddev, 0.);
  THROW_CHECK_GE(options.sensor_from_rig_rotation_stddev, 0.);
  THROW_CHECK_GE(options.point2D_stddev, 0.);
  THROW_CHECK_GE(options.prior_position_stddev, 0.);

  if (PRNG == nullptr) {
    SetPRNGSeed();
  }

  // Synthesize 3D points on unit sphere centered at origin.
  for (int point3D_idx = 0; point3D_idx < options.num_points3D; ++point3D_idx) {
    reconstruction->AddPoint3D(Eigen::Vector3d::Random().normalized(),
                               /*track=*/{});
  }

  int total_num_images = (database == nullptr) ? 0 : database->NumImages();
  int total_num_descriptors =
      (database == nullptr) ? 0 : database->NumDescriptors();

  for (int rig_idx = 0; rig_idx < options.num_rigs; ++rig_idx) {
    Rig rig;

    std::vector<sensor_t> camera_sensor_ids;
    camera_sensor_ids.reserve(options.num_cameras_per_rig);
    for (int camera_idx = 0; camera_idx < options.num_cameras_per_rig;
         ++camera_idx) {
      Camera camera;
      camera.width = options.camera_width;
      camera.height = options.camera_height;
      camera.model_id = options.camera_model_id;
      camera.params = options.camera_params;
      THROW_CHECK(camera.VerifyParams());
      camera.has_prior_focal_length = options.camera_has_prior_focal_length;
      camera.camera_id =
          (database == nullptr)
              ? (rig_idx * options.num_cameras_per_rig + camera_idx + 1)
              : database->WriteCamera(camera);
      reconstruction->AddCamera(camera);

      if (rig.NumSensors() == 0) {
        rig.AddRefSensor(camera.SensorId());
      } else {
        Rigid3d sensor_from_rig;
        if (options.sensor_from_rig_rotation_stddev > 0) {
          // Generate a random rotation around the Z-axis.
          // This is to avoid 2D points fall behind the camera.
          const double angle =
              std::clamp(RandomGaussian<double>(
                             0, options.sensor_from_rig_rotation_stddev),
                         -180.0,
                         180.0);
          sensor_from_rig.rotation = Eigen::Quaterniond(
              Eigen::AngleAxisd(DegToRad(angle), Eigen::Vector3d(0, 0, 1)));
        }
        if (options.sensor_from_rig_translation_stddev > 0) {
          sensor_from_rig.translation = Eigen::Vector3d(
              RandomGaussian<double>(
                  0, options.sensor_from_rig_translation_stddev),
              RandomGaussian<double>(
                  0, options.sensor_from_rig_translation_stddev),
              RandomGaussian<double>(
                  0, options.sensor_from_rig_translation_stddev));
        }
        rig.AddSensor(camera.SensorId(), sensor_from_rig);
      }

      camera_sensor_ids.push_back(camera.SensorId());
    }

    const rig_t rig_id =
        (database == nullptr) ? rig_idx + 1 : database->WriteRig(rig);
    rig.SetRigId(rig_id);
    reconstruction->AddRig(rig);

    for (int frame_idx = 0; frame_idx < options.num_frames_per_rig;
         ++frame_idx) {
      Frame frame;
      frame.SetRigId(rig.RigId());

      // Synthesize frames as sphere centered at world origin.
      const Eigen::Vector3d view_dir = -Eigen::Vector3d::Random().normalized();
      const Eigen::Vector3d proj_center = -5 * view_dir;
      Rigid3d rig_from_world;
      rig_from_world.rotation = Eigen::Quaterniond::FromTwoVectors(
          view_dir, Eigen::Vector3d(0, 0, 1));
      rig_from_world.translation = rig_from_world.rotation * -proj_center;

      frame.SetRigFromWorld(rig_from_world);

      std::vector<Image> images;
      images.reserve(options.num_cameras_per_rig);
      for (const auto& sensor_id : camera_sensor_ids) {
        ++total_num_images;

        Image& image = images.emplace_back();
        image.SetName(
            StringPrintf("camera%06d_frame%06d", sensor_id.id, frame_idx));
        image.SetCameraId(sensor_id.id);
        const image_t image_id = (database == nullptr)
                                     ? total_num_images
                                     : database->WriteImage(image);
        image.SetImageId(image_id);

        frame.AddDataId(image.DataId());
      }

      const frame_t frame_id =
          (database == nullptr)
              ? (rig_idx * options.num_frames_per_rig + frame_idx + 1)
              : database->WriteFrame(frame);
      frame.SetFrameId(frame_id);
      reconstruction->AddFrame(std::move(frame));

      for (Image& image : images) {
        image.SetFrameId(frame_id);

        const Camera& camera = reconstruction->Camera(image.CameraId());
        const Rigid3d sensor_from_rig =
            rig.IsRefSensor(camera.SensorId())
                ? Rigid3d()
                : rig.SensorFromRig(camera.SensorId());
        const Rigid3d cam_from_world = sensor_from_rig * rig_from_world;

        std::vector<Point2D> points2D;
        points2D.reserve(options.num_points3D +
                         options.num_points2D_without_point3D);

        // Create 3D point observations by project all 3D points to the image.
        for (auto& [point3D_id, point3D] : reconstruction->Points3D()) {
          Point2D point2D;
          const std::optional<Eigen::Vector2d> proj_point2D =
              camera.ImgFromCam(cam_from_world * point3D.xyz);
          if (!proj_point2D.has_value()) {
            continue;  // Point is behind the camera.
          }
          point2D.xy = proj_point2D.value();
          if (options.point2D_stddev > 0) {
            const Eigen::Vector2d noise(
                RandomGaussian<double>(0, options.point2D_stddev),
                RandomGaussian<double>(0, options.point2D_stddev));
            point2D.xy += noise;
          }
          if (point2D.xy(0) >= 0 && point2D.xy(1) >= 0 &&
              point2D.xy(0) <= camera.width && point2D.xy(1) <= camera.height) {
            point2D.point3D_id = point3D_id;
            points2D.push_back(point2D);
          }
        }

        // Synthesize uniform random 2D points without 3D points.
        for (int i = 0; i < options.num_points2D_without_point3D; ++i) {
          Point2D point2D;
          point2D.xy =
              Eigen::Vector2d(RandomUniformReal<double>(0, camera.width),
                              RandomUniformReal<double>(0, camera.height));
          points2D.push_back(point2D);
        }

        // Shuffle 2D points, so each image has 3D points ordered differently.
        std::shuffle(points2D.begin(), points2D.end(), *PRNG);
        image.SetPoints2D(points2D);

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
            // Generate a unique descriptor for each 3D point. If the 2D point
            // does not observe a 3D point, generate a random unique descriptor.
            std::mt19937 feature_generator(point2D.HasPoint3D()
                                               ? point2D.point3D_id
                                               : options.num_points3D +
                                                     (++total_num_descriptors));
            for (int d = 0; d < descriptors.cols(); ++d) {
              descriptors(point2D_idx, d) =
                  feature_distribution(feature_generator);
            }
          }
          database->WriteKeypoints(image.ImageId(), keypoints);
          database->WriteDescriptors(image.ImageId(), descriptors);
        }

        for (point2D_t point2D_idx = 0; point2D_idx < points2D.size();
             ++point2D_idx) {
          const auto& point2D = points2D[point2D_idx];
          if (point2D.HasPoint3D()) {
            auto& point3D = reconstruction->Point3D(point2D.point3D_id);
            point3D.track.AddElement(image.ImageId(), point2D_idx);
          }
        }

        if (options.use_prior_position) {
          PosePrior noisy_prior(proj_center,
                                PosePrior::CoordinateSystem::CARTESIAN);

          if (options.prior_position_stddev > 0.) {
            noisy_prior.position += Eigen::Vector3d(
                RandomGaussian<double>(0, options.prior_position_stddev),
                RandomGaussian<double>(0, options.prior_position_stddev),
                RandomGaussian<double>(0, options.prior_position_stddev));
            noisy_prior.position_covariance = options.prior_position_stddev *
                                              options.prior_position_stddev *
                                              Eigen::Matrix3d::Identity();
          } else {
            noisy_prior.position_covariance = Eigen::Matrix3d::Identity();
          }

          if (options.use_geographic_coords_prior) {
            static const GPSTransform gps_trans;

            static const double lat0 = 47.37851943807808;
            static const double lon0 = 8.549099927632087;
            static const double alt0 = 451.5;

            noisy_prior.position = gps_trans.ENUToEllipsoid(
                {noisy_prior.position}, lat0, lon0, alt0)[0];
            noisy_prior.coordinate_system = PosePrior::CoordinateSystem::WGS84;
          }

          database->WritePosePrior(image.ImageId(), noisy_prior);
        }

        if (database != nullptr) {
          database->UpdateImage(image);
        }
        reconstruction->AddImage(image);
      }
    }
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
