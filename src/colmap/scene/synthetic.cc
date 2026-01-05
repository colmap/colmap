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
#include "colmap/geometry/gps.h"
#include "colmap/math/random.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/file.h"

#include <filesystem>

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
          EssentialMatrixFromPose(*two_view_geometry.cam2_from_cam1);
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

      if (!database->ExistsMatches(image_id1, image_id2)) {
        database->WriteMatches(image_id1, image_id2, matches);
      }
      if (!database->ExistsTwoViewGeometry(image_id1, image_id2)) {
        database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
      }
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
      const image_pair_t pair_id =
          ImagePairToPairId(prev_track_el.image_id, curr_track_el.image_id);
      if (ShouldSwapImagePair(prev_track_el.image_id, curr_track_el.image_id)) {
        two_view_geometries[pair_id].inlier_matches.emplace_back(
            curr_track_el.point2D_idx, prev_track_el.point2D_idx);
      } else {
        two_view_geometries[pair_id].inlier_matches.emplace_back(
            prev_track_el.point2D_idx, curr_track_el.point2D_idx);
      }
    }
  }

  for (auto& [pair_id, two_view_geometry] : two_view_geometries) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const auto& image1 = reconstruction->Image(image_id1);
    const auto& camera1 = *image1.CameraPtr();
    const auto& image2 = reconstruction->Image(image_id2);
    const auto& camera2 = *image2.CameraPtr();
    two_view_geometry.config = TwoViewGeometry::CALIBRATED;
    two_view_geometry.cam2_from_cam1 =
        image2.CamFromWorld() * Inverse(image1.CamFromWorld());
    two_view_geometry.E =
        EssentialMatrixFromPose(*two_view_geometry.cam2_from_cam1);
    two_view_geometry.F =
        FundamentalFromEssentialMatrix(camera2.CalibrationMatrix(),
                                       two_view_geometry.E,
                                       camera1.CalibrationMatrix());

    FeatureMatches matches = two_view_geometry.inlier_matches;
    AddOutlierMatches(inlier_match_ratio,
                      image1.NumPoints2D(),
                      image2.NumPoints2D(),
                      &matches);

    if (!database->ExistsMatches(image_id1, image_id2)) {
      database->WriteMatches(image_id1, image_id2, matches);
    }
    if (!database->ExistsTwoViewGeometry(image_id1, image_id2)) {
      database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
    }
  }
}

static const GPSTransform kGPSTransform;
static constexpr double kLat0 = 47.37851943807808;
static constexpr double kLon0 = 8.549099927632087;
static constexpr double kAlt0 = 451.5;

void PosePriorPositionCartesianToWGS84(PosePrior& pose_prior) {
  pose_prior.position = kGPSTransform.ENUToEllipsoid(
      {pose_prior.position}, kLat0, kLon0, kAlt0)[0];
  pose_prior.coordinate_system = PosePrior::CoordinateSystem::WGS84;
}

void PosePriorPositionWGS84ToCartesian(PosePrior& pose_prior) {
  pose_prior.position = kGPSTransform.EllipsoidToENU(
      {Eigen::Vector3d(kLat0, kLon0, kAlt0), pose_prior.position},
      kLat0,
      kLon0)[1];
  pose_prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
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

  if (PRNG == nullptr) {
    SetPRNGSeed();
  }

  // Synthesize 3D points on unit sphere centered at origin.
  std::unordered_set<point3D_t> new_points3D_ids;
  new_points3D_ids.reserve(options.num_points3D);
  for (int point3D_idx = 0; point3D_idx < options.num_points3D; ++point3D_idx) {
    new_points3D_ids.insert(
        reconstruction->AddPoint3D(Eigen::Vector3d::Random().normalized(),
                                   /*track=*/{}));
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
      std::vector<Rigid3d> cams_from_world;
      cams_from_world.reserve(options.num_cameras_per_rig);
      for (const auto& sensor_id : camera_sensor_ids) {
        ++total_num_images;

        Image& image = images.emplace_back();
        image.SetName(
            StringPrintf("camera%06d_frame%06d.png", sensor_id.id, frame_idx));
        image.SetCameraId(sensor_id.id);
        const image_t image_id = (database == nullptr)
                                     ? total_num_images
                                     : database->WriteImage(image);
        image.SetImageId(image_id);

        frame.AddDataId(image.DataId());

        // Need to compose cam_from_world manually, because the frame/rig
        // pointer references are not yet set up.
        const Camera& camera = reconstruction->Camera(image.CameraId());
        const Rigid3d sensor_from_rig =
            rig.IsRefSensor(camera.SensorId())
                ? Rigid3d()
                : rig.SensorFromRig(camera.SensorId());
        const Rigid3d cam_from_world = sensor_from_rig * rig_from_world;
        cams_from_world.push_back(cam_from_world);

        if (options.prior_position || options.prior_gravity) {
          PosePrior pose_prior;

          if (options.prior_position) {
            pose_prior.position = cam_from_world.TgtOriginInSrc();
            pose_prior.coordinate_system =
                PosePrior::CoordinateSystem::CARTESIAN;
            switch (options.prior_position_coordinate_system) {
              case PosePrior::CoordinateSystem::CARTESIAN:
                break;
              case PosePrior::CoordinateSystem::WGS84:
                PosePriorPositionCartesianToWGS84(pose_prior);
                break;
              default:
                LOG(FATAL) << "Invalid PosePrior::CoordinateSystem specified";
            }
          }

          if (options.prior_gravity) {
            pose_prior.gravity =
                (cam_from_world.rotation * options.prior_gravity_in_world)
                    .normalized();
          }

          pose_prior.corr_data_id = image.DataId();
          pose_prior.pose_prior_id = database->WritePosePrior(pose_prior);
        }
      }

      const frame_t frame_id =
          (database == nullptr)
              ? (rig_idx * options.num_frames_per_rig + frame_idx + 1)
              : database->WriteFrame(frame);
      frame.SetFrameId(frame_id);
      reconstruction->AddFrame(std::move(frame));

      for (int camera_idx = 0; camera_idx < options.num_cameras_per_rig;
           ++camera_idx) {
        Image& image = images[camera_idx];
        const Camera& camera = reconstruction->Camera(image.CameraId());
        const Rigid3d& cam_from_world = cams_from_world[camera_idx];

        image.SetFrameId(frame_id);

        std::vector<Point2D> points2D;
        points2D.reserve(options.num_points3D +
                         options.num_points2D_without_point3D);

        // Create 3D point observations by projecting 3D points to the image.
        for (auto& [point3D_id, point3D] : reconstruction->Points3D()) {
          if (new_points3D_ids.count(point3D_id) == 0) {
            // If a non-empty reconstruction is given, only add tracks for
            // newly added images and 3D points.
            continue;
          }

          Point2D point2D;
          const std::optional<Eigen::Vector2d> proj_point2D =
              camera.ImgFromCam(cam_from_world * point3D.xyz);
          if (!proj_point2D.has_value()) {
            continue;  // Point is behind the camera.
          }
          point2D.xy = proj_point2D.value();
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
            // does not observe a 3D point, generate a random unique
            // descriptor.
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

void SynthesizeNoise(const SyntheticNoiseOptions& options,
                     Reconstruction* reconstruction,
                     Database* database) {
  THROW_CHECK_GE(options.rig_from_world_translation_stddev, 0.);
  THROW_CHECK_GE(options.rig_from_world_rotation_stddev, 0.);
  THROW_CHECK_GE(options.point3D_stddev, 0.);
  THROW_CHECK_GE(options.point2D_stddev, 0.);
  THROW_CHECK_GE(options.prior_position_stddev, 0.);
  THROW_CHECK_GE(options.prior_gravity_stddev, 0.);

  for (const frame_t frame_id : reconstruction->RegFrameIds()) {
    Rigid3d& rig_from_world = reconstruction->Frame(frame_id).RigFromWorld();

    if (options.rig_from_world_rotation_stddev > 0.0) {
      const double angle = std::clamp(
          RandomGaussian<double>(0, options.rig_from_world_rotation_stddev),
          -180.0,
          180.0);
      rig_from_world.rotation *= Eigen::Quaterniond(
          Eigen::AngleAxisd(DegToRad(angle), Eigen::Vector3d::UnitZ()));
    }

    if (options.rig_from_world_translation_stddev > 0.0) {
      rig_from_world.translation += Eigen::Vector3d(
          RandomGaussian<double>(0, options.rig_from_world_translation_stddev),
          RandomGaussian<double>(0, options.rig_from_world_translation_stddev),
          RandomGaussian<double>(0, options.rig_from_world_translation_stddev));
    }
  }

  if (options.point2D_stddev > 0.0) {
    for (const auto& [image_id, _] : reconstruction->Images()) {
      Image& image = reconstruction->Image(image_id);
      for (auto& point2D : image.Points2D()) {
        point2D.xy +=
            Eigen::Vector2d(RandomGaussian<double>(0, options.point2D_stddev),
                            RandomGaussian<double>(0, options.point2D_stddev));
      }
      if (database != nullptr) {
        std::vector<FeatureKeypoint> keypoints =
            database->ReadKeypoints(image_id);
        for (point2D_t point2D_idx = 0; point2D_idx < keypoints.size();
             ++point2D_idx) {
          keypoints[point2D_idx].x = image.Point2D(point2D_idx).xy(0);
          keypoints[point2D_idx].y = image.Point2D(point2D_idx).xy(1);
        }
        database->UpdateKeypoints(image.ImageId(), keypoints);
      }
    }
  }

  if (options.point3D_stddev > 0.0) {
    for (auto& [point3D_id, _] : reconstruction->Points3D()) {
      reconstruction->Point3D(point3D_id).xyz +=
          Eigen::Vector3d(RandomGaussian<double>(0, options.point3D_stddev),
                          RandomGaussian<double>(0, options.point3D_stddev),
                          RandomGaussian<double>(0, options.point3D_stddev));
    }
  }

  if (database != nullptr && (options.prior_position_stddev > 0.0 ||
                              options.prior_gravity_stddev > 0.0)) {
    for (auto& pose_prior : database->ReadAllPosePriors()) {
      if (options.prior_position_stddev > 0.) {
        const bool prior_in_wgs84 =
            pose_prior.coordinate_system == PosePrior::CoordinateSystem::WGS84;
        if (prior_in_wgs84) {
          PosePriorPositionWGS84ToCartesian(pose_prior);
        }
        pose_prior.position += Eigen::Vector3d(
            RandomGaussian<double>(0, options.prior_position_stddev),
            RandomGaussian<double>(0, options.prior_position_stddev),
            RandomGaussian<double>(0, options.prior_position_stddev));
        if (!pose_prior.HasPositionCov()) {
          pose_prior.position_covariance = Eigen::Matrix3d::Zero();
        }
        pose_prior.position_covariance += options.prior_position_stddev *
                                          options.prior_position_stddev *
                                          Eigen::Matrix3d::Identity();
        if (prior_in_wgs84) {
          PosePriorPositionCartesianToWGS84(pose_prior);
        }
      }
      if (options.prior_gravity_stddev > 0.) {
        const double angle =
            RandomGaussian<double>(0, DegToRad(options.prior_gravity_stddev));
        const Eigen::Vector3d axis =
            pose_prior.gravity.cross(Eigen::Vector3d::Random()).normalized();
        pose_prior.gravity =
            (Eigen::AngleAxisd(angle, axis) * pose_prior.gravity).normalized();
      }
      database->UpdatePosePrior(pose_prior);
    }
  }

  reconstruction->UpdatePoint3DErrors();
}

void SynthesizeImages(const SyntheticImageOptions& options,
                      const Reconstruction& reconstruction,
                      const std::filesystem::path& image_path) {
  THROW_CHECK_GT(options.feature_patch_radius, 0);
  THROW_CHECK_LT(options.feature_peak_radius, options.feature_patch_radius);
  THROW_CHECK_GT(options.feature_patch_max_brightness, 0);
  THROW_CHECK_LT(options.feature_patch_max_brightness, 255);

  const double patch_radius = std::sqrt(2 * options.feature_patch_radius *
                                        options.feature_patch_radius);

  int total_num_descriptors = 0;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    const Camera& camera = *image.CameraPtr();

    Bitmap bitmap(camera.width, camera.height, /*as_rgb=*/true);
    bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));

    for (const auto& point2D : image.Points2D()) {
      const int x = static_cast<int>(std::round(point2D.xy(0)));
      const int y = static_cast<int>(std::round(point2D.xy(1)));
      if (x < 0 || y < 0 || x >= static_cast<int>(camera.width) ||
          y >= static_cast<int>(camera.height)) {
        continue;
      }

      std::mt19937 feature_generator(point2D.HasPoint3D()
                                         ? point2D.point3D_id
                                         : reconstruction.NumPoints3D() +
                                               (++total_num_descriptors));

      // Draw a circular patch around the feature with a unique pattern with
      // the aim of producing a unique feature descriptor. Make the pattern a
      // bit darker than the peak, so the keypoint is detected at the center.
      const int patch_minx = std::max(x - options.feature_patch_radius, 0);
      const int patch_maxx = std::min(x + options.feature_patch_radius,
                                      static_cast<int>(camera.width));
      const int patch_miny = std::max(y - options.feature_patch_radius, 0);
      const int patch_maxy = std::min(y + options.feature_patch_radius,
                                      static_cast<int>(camera.height));
      for (int py = patch_miny; py < patch_maxy; ++py) {
        for (int px = patch_minx; px < patch_maxx; ++px) {
          const double radius =
              std::sqrt((px - x) * (px - x) + (py - y) * (py - y));
          if (radius > options.feature_patch_radius) {
            continue;
          }
          // Adjust the brightness so it fades out to the edge of the patch.
          std::uniform_int_distribution<int> patch_brightness_distribution(
              0,
              (1.0 - radius / patch_radius) *
                  options.feature_patch_max_brightness);
          bitmap.SetPixel(px,
                          py,
                          BitmapColor<uint8_t>(patch_brightness_distribution(
                              feature_generator)));
        }
      }

      // Draw a small, bright peak around the feature for keypoint detection.
      const int peak_minx = std::max(x - options.feature_peak_radius, 0);
      const int peak_maxx = std::min(x + options.feature_peak_radius,
                                     static_cast<int>(camera.width));
      const int peak_miny = std::max(y - options.feature_peak_radius, 0);
      const int peak_maxy = std::min(y + options.feature_peak_radius,
                                     static_cast<int>(camera.height));
      std::uniform_int_distribution<int> peak_color_distribution(
          options.feature_patch_max_brightness, 255);
      const BitmapColor<uint8_t> peak_color(
          peak_color_distribution(feature_generator),
          peak_color_distribution(feature_generator),
          peak_color_distribution(feature_generator));
      for (int py = peak_miny; py < peak_maxy; ++py) {
        for (int px = peak_minx; px < peak_maxx; ++px) {
          bitmap.SetPixel(px, py, peak_color);
        }
      }
    }

    const std::string output_image_path =
        JoinPaths(image_path.string(), image.Name());
    if (!bitmap.Write(output_image_path)) {
      LOG(ERROR) << "Failed to write image to " << output_image_path;
    }
  }
}

}  // namespace colmap
