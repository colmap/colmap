// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/photometric_bundle_adjustment.h"

#include "colmap/estimators/alignment.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/ceres_loss_function.h"

#include <cmath>
#include <random>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Procedural multi-scale textured surface (rich low + high spatial frequencies,
// like natural scenes) evaluated at world (x, y).
double EvalPlaneTexture(const double wx, const double wy, const double phase) {
  const double v = 128.0 + 35.0 * std::sin(17.3 * wx + 11.7 * wy + phase) +
                   30.0 * std::cos(13.9 * wx - 21.1 * wy - phase) +
                   28.0 * std::sin(43.1 * wx) * std::cos(39.7 * wy) +
                   22.0 * std::sin(71.7 * wx + 63.3 * wy + 0.5 * phase) +
                   16.0 * std::cos(97.3 * wx - 89.1 * wy);
  return std::clamp(v, 5.0, 250.0);
}

struct TexturedPlane {
  Eigen::Vector3d normal;
  double d;
  double x_min, x_max, y_min, y_max;
  double phase;
};

// Raycast a scene made of two non-coplanar tilted textured planes (giving full
// 3D depth variation so focal length and distortion are uniquely observable).
GrayscaleImagePyramid RenderMultiPlaneScene(
    const Camera& camera,
    const Rigid3d& cam_from_world,
    const std::vector<TexturedPlane>& planes,
    const int num_levels) {
  const int w = static_cast<int>(camera.width);
  const int h = static_cast<int>(camera.height);
  Bitmap bitmap(w, h, /*as_rgb=*/false);
  bitmap.Fill(BitmapColor<uint8_t>(128));
  std::vector<uint8_t>& data = bitmap.RowMajorData();
  const Rigid3d world_from_cam = Inverse(cam_from_world);
  const Eigen::Vector3d C_w = world_from_cam.translation();
  const Eigen::Matrix3d R_wc = world_from_cam.rotation().toRotationMatrix();

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const auto ray2 = camera.CamFromImg(Eigen::Vector2d(x + 0.5, y + 0.5));
      if (!ray2.has_value()) continue;
      const Eigen::Vector3d dir_w =
          R_wc * Eigen::Vector3d(ray2->x(), ray2->y(), 1.0);
      double best_t = std::numeric_limits<double>::infinity();
      double best_val = 128.0;
      for (const auto& pl : planes) {
        const double denom = pl.normal.dot(dir_w);
        if (std::abs(denom) < 1e-8) continue;
        const double t = (pl.d - pl.normal.dot(C_w)) / denom;
        if (t <= 0.0 || t >= best_t) continue;
        const Eigen::Vector3d X_w = C_w + t * dir_w;
        if (X_w.x() >= pl.x_min && X_w.x() <= pl.x_max && X_w.y() >= pl.y_min &&
            X_w.y() <= pl.y_max) {
          best_t = t;
          best_val = EvalPlaneTexture(X_w.x(), X_w.y(), pl.phase);
        }
      }
      data[static_cast<size_t>(y) * w + x] =
          static_cast<uint8_t>(std::lround(best_val));
    }
  }
  return GrayscaleImagePyramid(bitmap, num_levels);
}

struct SyntheticPhotometricScene {
  Reconstruction gt;
  std::vector<TexturedPlane> planes;
  FlatHashMap<point3D_t, size_t> point_to_plane;
  std::shared_ptr<FlatHashMap<image_t, GrayscaleImagePyramid>> pyramids;
};

SyntheticPhotometricScene BuildSyntheticMultiPlaneScene(
    const CameraModelId model_id,
    const std::vector<double>& cam_params,
    const int width,
    const int height,
    const int num_views) {
  SyntheticPhotometricScene scene;
  scene.pyramids =
      std::make_shared<FlatHashMap<image_t, GrayscaleImagePyramid>>();

  // Two tilted planes at different depths (front-left at z~2.2, back at z~3.3)
  // so focal length, radial distortion, and poses have no planar degeneracy.
  const Eigen::Vector3d n1 = Eigen::Vector3d(0.22, -0.18, 1.0).normalized();
  const Eigen::Vector3d n2 = Eigen::Vector3d(-0.20, 0.15, 1.0).normalized();
  scene.planes = {
      {n1,
       n1.dot(Eigen::Vector3d(-0.35, 0.0, 2.2)),
       -1.2,
       -0.15,
       -0.9,
       0.9,
       0.0},
      {n2, n2.dot(Eigen::Vector3d(0.0, 0.0, 3.3)), -1.8, 1.8, -1.4, 1.4, 1.7},
  };

  Camera camera = Camera::CreateFromModelId(1, model_id, 400.0, width, height);
  camera.params = cam_params;
  scene.gt.AddCamera(camera);

  Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(camera.SensorId());
  scene.gt.AddRig(rig);

  for (int i = 0; i < num_views; ++i) {
    const double angle = 2.0 * M_PI * i / num_views;
    const double radius = (i == 0) ? 0.15 : 0.75;
    // Strong Z diversity avoids the planar dolly-zoom focal-length ambiguity.
    const double z_cam = -0.5 + 0.9 * (static_cast<double>(i % 3) / 2.0);
    const Eigen::Vector3d cam_center(
        radius * std::cos(angle), radius * std::sin(angle), z_cam);
    const Eigen::Vector3d z_axis =
        (Eigen::Vector3d(0.0, 0.0, 2.8) - cam_center).normalized();
    const Eigen::Vector3d x_axis =
        Eigen::Vector3d::UnitY().cross(z_axis).normalized();
    const Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();
    Eigen::Matrix3d R_cw;
    R_cw.row(0) = x_axis.transpose();
    R_cw.row(1) = y_axis.transpose();
    R_cw.row(2) = z_axis.transpose();
    const Rigid3d cam_from_world(Eigen::Quaterniond(R_cw), -R_cw * cam_center);

    const frame_t frame_id = i + 1;
    const image_t image_id = i + 1;

    Frame frame;
    frame.SetFrameId(frame_id);
    frame.SetRigId(1);
    frame.AddDataId(data_t(camera.SensorId(), image_id));
    frame.SetRigFromWorld(cam_from_world);
    scene.gt.AddFrame(frame);

    Image img;
    img.SetImageId(image_id);
    img.SetName("img_" + std::to_string(image_id) + ".png");
    img.SetCameraId(1);
    img.SetFrameId(frame_id);
    scene.gt.AddImage(img);

    scene.pyramids->emplace(
        image_id,
        RenderMultiPlaneScene(camera, cam_from_world, scene.planes, 4));
  }

  // Sample 3D points safely inside the interior of each plane (away from the
  // occlusion boundary between plane 0 and plane 1).
  std::vector<std::pair<Eigen::Vector3d, size_t>> pts3d;
  // Front plane points (x in [-0.75, -0.25], y in [-0.55, 0.55]).
  for (int gy = -2; gy <= 2; ++gy) {
    for (int gx = 0; gx < 3; ++gx) {
      const double wx = -0.75 + 0.22 * gx;
      const double wy = 0.25 * gy;
      const auto& pl = scene.planes[0];
      const double wz =
          (pl.d - pl.normal.x() * wx - pl.normal.y() * wy) / pl.normal.z();
      pts3d.emplace_back(Eigen::Vector3d(wx, wy, wz), 0);
    }
  }
  // Back plane points (x in [0.25, 0.95], y in [-0.75, 0.75]).
  for (int gy = -3; gy <= 3; ++gy) {
    for (int gx = 0; gx < 4; ++gx) {
      const double wx = 0.25 + 0.22 * gx;
      const double wy = 0.24 * gy;
      const auto& pl = scene.planes[1];
      const double wz =
          (pl.d - pl.normal.x() * wx - pl.normal.y() * wy) / pl.normal.z();
      pts3d.emplace_back(Eigen::Vector3d(wx, wy, wz), 1);
    }
  }

  std::vector<std::vector<Eigen::Vector2d>> img_pts(num_views);
  std::vector<Track> tracks(pts3d.size());
  for (size_t k = 0; k < pts3d.size(); ++k) {
    for (int i = 0; i < num_views; ++i) {
      const image_t image_id = i + 1;
      const auto xy = scene.gt.Image(image_id).ProjectPoint(pts3d[k].first);
      if (xy.has_value() && xy->x() > 16 && xy->x() < width - 16 &&
          xy->y() > 16 && xy->y() < height - 16) {
        const point2D_t p2d_idx = static_cast<point2D_t>(img_pts[i].size());
        img_pts[i].push_back(*xy);
        tracks[k].AddElement(image_id, p2d_idx);
      }
    }
  }
  for (int i = 0; i < num_views; ++i) {
    scene.gt.Image(i + 1).SetPoints2D(img_pts[i]);
  }
  for (size_t k = 0; k < pts3d.size(); ++k) {
    if (tracks[k].Length() >= 2) {
      const point3D_t id =
          scene.gt.AddPoint3D(pts3d[k].first, std::move(tracks[k]));
      scene.point_to_plane[id] = pts3d[k].second;
    }
  }
  return scene;
}

TEST(PhotometricBundleAdjustment, RecoversPerturbedPosesIntrinsicsAndPoints) {
  auto scene = BuildSyntheticMultiPlaneScene(CameraModelId::kSimpleRadial,
                                             {420.0, 256.0, 256.0, -0.08},
                                             /*width=*/512,
                                             /*height=*/512,
                                             /*num_views=*/6);

  Reconstruction rec = scene.gt;
  std::mt19937 rng(42);
  std::normal_distribution<double> pt_noise(0.0, 0.012);
  std::normal_distribution<double> trans_noise(0.0, 0.008);
  std::normal_distribution<double> rot_noise(0.0, 0.004);

  for (const point3D_t p3d_id : rec.Point3DIds()) {
    rec.Point3D(p3d_id).xyz +=
        Eigen::Vector3d(pt_noise(rng), pt_noise(rng), pt_noise(rng));
  }
  for (int i = 2; i < 6; ++i) {
    Rigid3d& rfw = rec.Frame(i + 1).RigFromWorld();
    const Eigen::Vector3d aa(rot_noise(rng), rot_noise(rng), rot_noise(rng));
    rfw.rotation() =
        (Eigen::Quaterniond(Eigen::AngleAxisd(aa.norm(), aa.normalized())) *
         rfw.rotation())
            .normalized();
    rfw.translation() +=
        Eigen::Vector3d(trans_noise(rng), trans_noise(rng), trans_noise(rng));
  }
  rec.Camera(1).params[0] += 5.0;    // focal length +5 px
  rec.Camera(1).params[3] += 0.015;  // radial distortion k1

  BundleAdjustmentOptions ba_options;
  ba_options.refine_focal_length = true;
  ba_options.refine_principal_point = false;
  ba_options.refine_extra_params = true;
  ba_options.refine_rig_from_world = true;
  ba_options.refine_points3D = true;
  ba_options.print_summary = false;
  ba_options.ceres->solver_options.max_num_iterations = 60;

  PhotometricBundleAdjustmentOptions photo_options;
  photo_options.pyramid_levels = "2,1,0";
  photo_options.geometric_reprojection_weight = 0.0;

  BundleAdjustmentConfig config;
  for (const image_t image_id : rec.RegImageIds()) {
    config.AddImage(image_id);
  }

  auto ba = CreatePhotometricBundleAdjuster(
      ba_options, photo_options, config, scene.pyramids, rec);
  auto summary = ba->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  // On 512x512 images with k1=-0.08, mipmap binomial smoothing across views
  // slightly attenuates radial curvature; still cuts k1 error by >50%.
  EXPECT_NEAR(rec.Camera(1).params[0], scene.gt.Camera(1).params[0], 1.0);
  EXPECT_NEAR(rec.Camera(1).params[3], scene.gt.Camera(1).params[3], 0.008);

  Sim3d gt_from_est;
  ASSERT_TRUE(
      AlignReconstructionsViaProjCenters(rec, scene.gt, 0.1, &gt_from_est));
  rec.Transform(gt_from_est);

  double max_trans_err = 0.0;
  double max_rot_err_deg = 0.0;
  for (int i = 0; i < 6; ++i) {
    const Rigid3d& est = rec.Frame(i + 1).RigFromWorld();
    const Rigid3d& gt_pose = scene.gt.Frame(i + 1).RigFromWorld();
    const double t_err = (est.translation() - gt_pose.translation()).norm();
    const double r_err =
        est.rotation().angularDistance(gt_pose.rotation()) * 180.0 / M_PI;
    max_trans_err = std::max(max_trans_err, t_err);
    max_rot_err_deg = std::max(max_rot_err_deg, r_err);
  }
  EXPECT_LT(max_trans_err, 0.003);
  EXPECT_LT(max_rot_err_deg, 0.15);

  double mean_plane_dist = 0.0;
  for (const point3D_t p3d_id : rec.Point3DIds()) {
    const auto& pl = scene.planes[scene.point_to_plane.at(p3d_id)];
    mean_plane_dist += std::abs(pl.normal.dot(rec.Point3D(p3d_id).xyz) - pl.d);
  }
  mean_plane_dist /= rec.NumPoints3D();
  EXPECT_LT(mean_plane_dist, 0.007);
}

TEST(PhotometricBundleAdjustment, FixedPoseTriangulationRefinement) {
  auto scene = BuildSyntheticMultiPlaneScene(CameraModelId::kThinPrismFisheye,
                                             {420.0,
                                              420.0,
                                              256.0,
                                              256.0,
                                              0.05,
                                              -0.01,
                                              0.001,
                                              -0.001,
                                              0.005,
                                              -0.005,
                                              0.0005,
                                              -0.0005},
                                             /*width=*/512,
                                             /*height=*/512,
                                             /*num_views=*/5);

  Reconstruction rec = scene.gt;
  std::mt19937 rng(7);
  std::normal_distribution<double> pt_noise(0.0, 0.025);

  double init_plane_dist = 0.0;
  for (const point3D_t p3d_id : rec.Point3DIds()) {
    rec.Point3D(p3d_id).xyz +=
        Eigen::Vector3d(pt_noise(rng), pt_noise(rng), pt_noise(rng));
    const auto& pl = scene.planes[scene.point_to_plane.at(p3d_id)];
    init_plane_dist += std::abs(pl.normal.dot(rec.Point3D(p3d_id).xyz) - pl.d);
  }
  init_plane_dist /= rec.NumPoints3D();

  BundleAdjustmentOptions ba_options;
  ba_options.refine_focal_length = false;
  ba_options.refine_principal_point = false;
  ba_options.refine_extra_params = false;
  ba_options.refine_rig_from_world = false;
  ba_options.refine_sensor_from_rig = false;
  ba_options.refine_points3D = true;
  ba_options.print_summary = false;

  PhotometricBundleAdjustmentOptions photo_options;
  photo_options.pyramid_levels = "2,1,0";
  // Keep default warmup iterations > 0 to verify warmup is skipped
  // when all camera-side blocks are already constant.

  BundleAdjustmentConfig config;
  for (const image_t image_id : rec.RegImageIds()) {
    config.AddImage(image_id);
  }

  auto ba = CreatePhotometricBundleAdjuster(
      ba_options, photo_options, config, scene.pyramids, rec);
  auto summary = ba->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  double final_plane_dist = 0.0;
  for (const point3D_t p3d_id : rec.Point3DIds()) {
    const auto& pl = scene.planes[scene.point_to_plane.at(p3d_id)];
    final_plane_dist += std::abs(pl.normal.dot(rec.Point3D(p3d_id).xyz) - pl.d);
  }
  final_plane_dist /= rec.NumPoints3D();
  EXPECT_LT(final_plane_dist, 0.15 * init_plane_dist);
  EXPECT_LT(final_plane_dist, 0.0025);
}

TEST(PhotometricBundleAdjustment, ValidationGuardsAndOptionChecks) {
  // Out-of-range patch_size rejected (2 <= patch_size <= 8).
  for (const int bad_patch : {1, 9}) {
    PhotometricBundleAdjustmentOptions bad_opts;
    bad_opts.patch_size = bad_patch;
    EXPECT_FALSE(bad_opts.Check());
  }

  // Out-of-range pyramid level rejected to prevent shift UB.
  PhotometricBundleAdjustmentOptions bad_lvl;
  bad_lvl.pyramid_levels = "31";
  EXPECT_FALSE(bad_lvl.Check());

  // Invalid max_image_size rejected (-1 or >= patch_size required).
  for (const int bad_size : {0, -2, 1, 3}) {
    PhotometricBundleAdjustmentOptions bad_max;
    bad_max.max_image_size = bad_size;
    EXPECT_FALSE(bad_max.Check());
  }
  PhotometricBundleAdjustmentOptions ok_max;
  ok_max.max_image_size = 4;
  EXPECT_TRUE(ok_max.Check());

  const auto scene = BuildSyntheticMultiPlaneScene(
      CameraModelId::kSimpleRadial, {400.0, 128.0, 128.0, 0.0}, 256, 256, 4);

  // World-fixed constant points with variable poses must fail loudly.
  {
    Reconstruction rec = scene.gt;
    BundleAdjustmentOptions ba_opts;
    ba_opts.refine_points3D = false;
    ba_opts.refine_rig_from_world = true;
    ba_opts.print_summary = false;
    PhotometricBundleAdjustmentOptions photo_opts;
    photo_opts.pyramid_levels = "0";
    photo_opts.structure_warmup_max_num_iterations = 0;
    BundleAdjustmentConfig config;
    for (const image_t image_id : rec.RegImageIds()) {
      config.AddImage(image_id);
    }
    auto ba = CreatePhotometricBundleAdjuster(
        ba_opts, photo_opts, config, scene.pyramids, rec);
    EXPECT_THROW(ba->Solve(), std::exception);
  }
}

TEST(PhotometricBundleAdjustment, DownscaledPyramidsAndTrivialZeroScaleLoss) {
  // Regression test for (a) max_image_size base downscaling in source
  // selection (scale < 1 at level 0) and (b) TRIVIAL loss with
  // loss_function_scale == 0.0 (must not cause 0/0 NaN in IRLS weights).
  const auto scene = BuildSyntheticMultiPlaneScene(
      CameraModelId::kSimpleRadial, {400.0, 256.0, 256.0, 0.0}, 512, 512, 5);

  // Build 2x base-downscaled pyramids (simulating max_image_size = 256 on
  // 512x512 images so level 0 has scale_x = scale_y = 0.5).
  auto scaled_pyramids =
      std::make_shared<FlatHashMap<image_t, GrayscaleImagePyramid>>();
  for (const auto& [image_id, pyr] : *scene.pyramids) {
    scaled_pyramids->emplace(
        image_id,
        GrayscaleImagePyramid(
            pyr.Level(0).bitmap, /*num_levels=*/4, /*max_image_size=*/256));
  }

  Reconstruction rec = scene.gt;
  std::mt19937 rng(17);
  std::normal_distribution<double> pt_noise(0.0, 0.02);
  for (const point3D_t p3d_id : rec.Point3DIds()) {
    rec.Point3D(p3d_id).xyz +=
        Eigen::Vector3d(pt_noise(rng), pt_noise(rng), pt_noise(rng));
  }

  BundleAdjustmentOptions ba_opts;
  ba_opts.refine_focal_length = false;
  ba_opts.refine_principal_point = false;
  ba_opts.refine_extra_params = false;
  ba_opts.refine_rig_from_world = false;
  ba_opts.refine_sensor_from_rig = false;
  ba_opts.refine_points3D = true;
  ba_opts.print_summary = false;

  PhotometricBundleAdjustmentOptions photo_opts;
  photo_opts.pyramid_levels = "1,0";
  photo_opts.structure_warmup_max_num_iterations = 0;
  photo_opts.loss_function_type = CeresLossFunctionType::TRIVIAL;
  photo_opts.loss_function_scale = 0.0;

  BundleAdjustmentConfig config;
  for (const image_t image_id : rec.RegImageIds()) {
    config.AddImage(image_id);
  }

  auto ba = CreatePhotometricBundleAdjuster(
      ba_opts, photo_opts, config, scaled_pyramids, rec);
  auto summary = std::dynamic_pointer_cast<PhotometricBundleAdjustmentSummary>(
      ba->Solve());
  ASSERT_NE(summary, nullptr);
  ASSERT_TRUE(summary->IsSolutionUsable());
  EXPECT_GT(summary->num_tracks_kept, 0);
  EXPECT_GT(summary->num_tracks_kept, summary->num_tracks_total / 2);
}

TEST(PhotometricBundleAdjustment, IntraFrameRigDoesNotCrash) {
  // Regression test when a rig frame only participates in intra-frame
  // photometric residuals (so its rig_from_world is absent from Ceres under
  // pure photometric BA).
  const auto base = BuildSyntheticMultiPlaneScene(
      CameraModelId::kSimpleRadial, {400.0, 128.0, 128.0, 0.0}, 256, 256, 3);

  Reconstruction rec;
  auto pyramids =
      std::make_shared<FlatHashMap<image_t, GrayscaleImagePyramid>>();

  Camera cam1 = base.gt.Camera(1);
  cam1.camera_id = 1;
  Camera cam2 = base.gt.Camera(1);
  cam2.camera_id = 2;
  rec.AddCamera(cam1);
  rec.AddCamera(cam2);

  Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(cam1.SensorId());
  const Rigid3d cam2_from_rig(Eigen::Quaterniond::Identity(),
                              Eigen::Vector3d(0.25, 0.0, 0.0));
  rig.AddSensor(cam2.SensorId(), cam2_from_rig);
  rec.AddRig(rig);

  // Frame 1 (lowest ID): only has intra-frame tracks observed by cam1 & cam2.
  // Frames 2 & 3: share inter-frame tracks observed by cam1.
  const std::vector<Rigid3d> frame_poses = {
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.0, 0.0, 0.0)),
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.4, 0.0, 0.0)),
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(-0.4, 0.1, 0.1)),
  };

  auto AddImg = [&](const frame_t fid,
                    const image_t iid,
                    const camera_t cid,
                    const Rigid3d& cfw) {
    Image img;
    img.SetImageId(iid);
    img.SetName("img_" + std::to_string(iid) + ".png");
    img.SetCameraId(cid);
    img.SetFrameId(fid);
    rec.AddImage(std::move(img));
    pyramids->emplace(iid,
                      RenderMultiPlaneScene(rec.Camera(cid),
                                            cfw,
                                            base.planes,
                                            /*num_levels=*/3));
  };

  for (int f = 0; f < 3; ++f) {
    Frame frame;
    frame.SetFrameId(f + 1);
    frame.SetRigId(1);
    frame.SetRigFromWorld(frame_poses[f]);
    frame.AddDataId(data_t(cam1.SensorId(), f * 2 + 1));
    if (f == 0) {
      frame.AddDataId(data_t(cam2.SensorId(), f * 2 + 2));
    }
    rec.AddFrame(std::move(frame));
    AddImg(f + 1, f * 2 + 1, 1, frame_poses[f]);
    if (f == 0) {
      AddImg(f + 1, f * 2 + 2, 2, cam2_from_rig * frame_poses[f]);
    }
  }

  // Add points seen ONLY inside Frame 1 (images 1 & 2) and points seen across
  // Frames 2 & 3 (images 3 & 5).
  FlatHashMap<image_t, std::vector<Eigen::Vector2d>> pts2d;
  std::vector<std::pair<Eigen::Vector3d, Track>> pending_tracks;
  const auto& pl = base.planes[1];
  for (int k = 0; k < 6; ++k) {
    const double wx = -0.3 + 0.12 * k;
    const double wy = -0.1 + 0.05 * k;
    const double wz =
        (pl.d - pl.normal.x() * wx - pl.normal.y() * wy) / pl.normal.z();
    const Eigen::Vector3d X_w(wx, wy, wz);
    auto CollectPair = [&](const image_t i1, const image_t i2) {
      const auto p1 = rec.Camera(rec.Image(i1).CameraId())
                          .ImgFromCam(rec.Image(i1).CamFromWorld() * X_w);
      const auto p2 = rec.Camera(rec.Image(i2).CameraId())
                          .ImgFromCam(rec.Image(i2).CamFromWorld() * X_w);
      if (!p1.has_value() || !p2.has_value()) return;
      const point2D_t idx1 = pts2d[i1].size();
      pts2d[i1].push_back(*p1);
      const point2D_t idx2 = pts2d[i2].size();
      pts2d[i2].push_back(*p2);
      Track track;
      track.AddElement(i1, idx1);
      track.AddElement(i2, idx2);
      pending_tracks.emplace_back(X_w, std::move(track));
    };
    CollectPair(1, 2);  // Intra-frame only on frame 1
    CollectPair(3, 5);  // Inter-frame on frames 2 & 3
  }
  for (const auto& [iid, pts] : pts2d) {
    rec.Image(iid).SetPoints2D(pts);
  }
  for (auto& [xyz, track] : pending_tracks) {
    rec.AddPoint3D(xyz, std::move(track));
  }
  ASSERT_GT(rec.NumPoints3D(), 0);

  BundleAdjustmentOptions ba_opts;
  ba_opts.refine_focal_length = false;
  ba_opts.refine_principal_point = false;
  ba_opts.refine_extra_params = false;
  ba_opts.refine_rig_from_world = true;
  ba_opts.refine_sensor_from_rig = false;
  ba_opts.refine_points3D = true;
  ba_opts.print_summary = false;
  ba_opts.ceres->solver_options.max_num_iterations = 5;

  PhotometricBundleAdjustmentOptions photo_opts;
  photo_opts.pyramid_levels = "0";
  photo_opts.structure_warmup_max_num_iterations = 0;
  // Explicitly test pure photometric BA so Frame 1's rig_from_world is absent
  // from Ceres.
  photo_opts.geometric_reprojection_weight = 0.0;

  BundleAdjustmentConfig config;
  for (const image_t image_id : rec.RegImageIds()) {
    config.AddImage(image_id);
  }

  {
    Reconstruction rec_pure = rec;
    auto ba = CreatePhotometricBundleAdjuster(
        ba_opts, photo_opts, config, pyramids, rec_pure);
    auto summary = ba->Solve();
    ASSERT_TRUE(summary->IsSolutionUsable());
  }

  // Exercise VCE geometric regularization (weight = 1.0) on a multi-camera rig
  // where Frame 1 (with Frames 2 & 3 constant) only has intra-frame
  // photometric tracks, verifying both variable and constant rig_from_world
  // handling as well as refine_sensor_from_rig = true.
  {
    Reconstruction rec_var = rec;
    BundleAdjustmentOptions rig_ba_opts = ba_opts;
    rig_ba_opts.refine_sensor_from_rig = true;
    PhotometricBundleAdjustmentOptions vce_opts = photo_opts;
    vce_opts.geometric_reprojection_weight = 1.0;

    BundleAdjustmentConfig rig_cfg;
    for (const image_t image_id : rec_var.RegImageIds()) {
      rig_cfg.AddImage(image_id);
    }
    rig_cfg.SetConstantRigFromWorldPose(2);
    rig_cfg.SetConstantRigFromWorldPose(3);

    auto ba = CreatePhotometricBundleAdjuster(
        rig_ba_opts, vce_opts, rig_cfg, pyramids, rec_var);
    ASSERT_TRUE(ba->Solve()->IsSolutionUsable());
    EXPECT_NEAR(rec_var.Frame(1).RigFromWorld().rotation().norm(), 1.0, 1e-6);
    EXPECT_NEAR(rec_var.Rig(1).SensorFromRig(cam2.SensorId()).rotation().norm(),
                1.0,
                1e-6);
  }
  {
    Reconstruction rec_const = rec;
    const Rigid3d frame1_init = rec_const.Frame(1).RigFromWorld();
    BundleAdjustmentOptions rig_ba_opts = ba_opts;
    rig_ba_opts.refine_sensor_from_rig = true;
    PhotometricBundleAdjustmentOptions vce_opts = photo_opts;
    vce_opts.geometric_reprojection_weight = 1.0;

    BundleAdjustmentConfig rig_cfg;
    for (const image_t image_id : rec_const.RegImageIds()) {
      rig_cfg.AddImage(image_id);
    }
    rig_cfg.SetConstantRigFromWorldPose(1);
    rig_cfg.SetConstantRigFromWorldPose(2);

    auto ba = CreatePhotometricBundleAdjuster(
        rig_ba_opts, vce_opts, rig_cfg, pyramids, rec_const);
    ASSERT_TRUE(ba->Solve()->IsSolutionUsable());
    EXPECT_EQ(rec_const.Frame(1).RigFromWorld().params, frame1_init.params);
  }
}

TEST(PhotometricBundleAdjustment, GeometricRegularizationAnchorsFilteredViews) {
  auto scene = BuildSyntheticMultiPlaneScene(CameraModelId::kSimpleRadial,
                                             {420.0, 256.0, 256.0, -0.05},
                                             /*width=*/512,
                                             /*height=*/512,
                                             /*num_views=*/5);

  // Replace view 5's pyramid with a flat (zero-texture) image so all
  // photometric patches involving view 5 fail the contrast filter, leaving
  // view 5 constrained only by VCE geometric reprojection regularization.
  Bitmap flat(512, 512, /*as_rgb=*/false);
  flat.Fill(BitmapColor<uint8_t>(128));
  (*scene.pyramids)[5] = GrayscaleImagePyramid(flat, /*num_levels=*/4);

  auto MakePerturbed = [&]() {
    Reconstruction rec = scene.gt;
    Rigid3d& pose5 = rec.Frame(5).RigFromWorld();
    pose5.rotation() =
        (Eigen::Quaterniond(Eigen::AngleAxisd(
             0.015, Eigen::Vector3d(0.3, -0.7, 0.6).normalized())) *
         pose5.rotation())
            .normalized();
    pose5.translation() += Eigen::Vector3d(0.025, -0.015, 0.02);
    for (const point3D_t p3d_id : rec.Point3DIds()) {
      rec.Point3D(p3d_id).xyz += Eigen::Vector3d(0.005, -0.005, 0.008);
    }
    return rec;
  };

  BundleAdjustmentOptions ba_opts;
  ba_opts.refine_focal_length = false;
  ba_opts.refine_principal_point = false;
  ba_opts.refine_extra_params = false;
  ba_opts.refine_rig_from_world = true;
  ba_opts.refine_points3D = true;
  ba_opts.print_summary = false;
  ba_opts.ceres->solver_options.max_num_iterations = 50;

  // Hold all textured views (1..4) constant so ONLY the photometrically
  // filtered-out view 5 is variable — verifying VCE regularization activates
  // even when all photometrically active views are constant.
  BundleAdjustmentConfig config;
  for (const image_t image_id : scene.gt.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);
  config.SetConstantRigFromWorldPose(3);
  config.SetConstantRigFromWorldPose(4);

  // Pure photometric BA (weight = 0) leaves the flat-image view 5 drifting.
  Reconstruction rec_pure = MakePerturbed();
  {
    PhotometricBundleAdjustmentOptions photo_opts;
    photo_opts.pyramid_levels = "1,0";
    photo_opts.geometric_reprojection_weight = 0.0;
    auto ba = CreatePhotometricBundleAdjuster(
        ba_opts, photo_opts, config, scene.pyramids, rec_pure);
    ASSERT_TRUE(ba->Solve()->IsSolutionUsable());
  }
  const double pure_err = (rec_pure.Frame(5).RigFromWorld().translation() -
                           scene.gt.Frame(5).RigFromWorld().translation())
                              .norm();
  EXPECT_GT(pure_err, 0.02);

  // Default VCE geometric regularization (weight = 1.0) recovers view 5's pose
  // from its full-track geometric observations while still refining textured
  // 3D points photometrically.
  Reconstruction rec_vce = MakePerturbed();
  {
    PhotometricBundleAdjustmentOptions photo_opts;
    photo_opts.pyramid_levels = "1,0";
    EXPECT_DOUBLE_EQ(photo_opts.geometric_reprojection_weight, 1.0);
    auto ba = CreatePhotometricBundleAdjuster(
        ba_opts, photo_opts, config, scene.pyramids, rec_vce);
    ASSERT_TRUE(ba->Solve()->IsSolutionUsable());
  }
  const Rigid3d& est5 = rec_vce.Frame(5).RigFromWorld();
  const Rigid3d& gt5 = scene.gt.Frame(5).RigFromWorld();
  EXPECT_LT((est5.translation() - gt5.translation()).norm(), 1e-3);
  EXPECT_LT(est5.rotation().angularDistance(gt5.rotation()) * 180.0 / M_PI,
            0.05);

  double mean_plane_dist = 0.0;
  for (const point3D_t p3d_id : rec_vce.Point3DIds()) {
    const auto& pl = scene.planes[scene.point_to_plane.at(p3d_id)];
    mean_plane_dist +=
        std::abs(pl.normal.dot(rec_vce.Point3D(p3d_id).xyz) - pl.d);
  }
  mean_plane_dist /= rec_vce.NumPoints3D();
  EXPECT_LT(mean_plane_dist, 0.002);

  PhotometricBundleAdjustmentOptions bad_opts;
  bad_opts.geometric_reprojection_weight = -0.1;
  EXPECT_FALSE(bad_opts.Check());
  bad_opts.geometric_reprojection_weight =
      std::numeric_limits<double>::infinity();
  EXPECT_FALSE(bad_opts.Check());
  bad_opts.geometric_reprojection_weight = 1.0;
  bad_opts.min_patch_std = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(bad_opts.Check());
}

TEST(PhotometricBundleAdjustment, FinalizeInactiveTracksVceAndPurePhotometric) {
  auto scene = BuildSyntheticMultiPlaneScene(CameraModelId::kSimpleRadial,
                                             {420.0, 256.0, 256.0, -0.05},
                                             /*width=*/512,
                                             /*height=*/512,
                                             /*num_views=*/5);

  // Pick all tracks on plane 0 and stamp a 5x5 flat gray square (radius 2 px)
  // around each of their initial projections in every view. At level 0, the
  // 4x4 patch spans [-1.5, +1.5] px (strictly inside the flat square ->
  // low-texture filtered out). At coarse level 1, the 4x4 patch spans
  // [-3, +3] px and straddles the surrounding high-contrast texture -> kept
  // at level 1!
  std::vector<point3D_t> plane0_ids;
  for (const point3D_t p3d_id : scene.gt.Point3DIds()) {
    if (scene.point_to_plane.at(p3d_id) == 0) {
      plane0_ids.push_back(p3d_id);
    }
  }
  ASSERT_FALSE(plane0_ids.empty());

  auto MakePerturbed = [&]() {
    Reconstruction rec = scene.gt;
    for (const point3D_t p3d_id : rec.Point3DIds()) {
      rec.Point3D(p3d_id).xyz += Eigen::Vector3d(0.0, 0.0, 0.01);
    }
    for (int f = 3; f <= 5; ++f) {
      rec.Frame(f).RigFromWorld().translation() +=
          Eigen::Vector3d(0.004, -0.003, 0.004);
    }
    return rec;
  };

  auto BuildMaskedPyramids = [&](const Reconstruction& ref_rec) {
    auto masked =
        std::make_shared<FlatHashMap<image_t, GrayscaleImagePyramid>>();
    for (const auto& [image_id, pyr] : *scene.pyramids) {
      Bitmap base = pyr.Level(0).bitmap.Clone();
      const Image& gt_img = scene.gt.Image(image_id);
      const Image& ref_img = ref_rec.Image(image_id);
      for (const point3D_t p3d_id : plane0_ids) {
        // Stamp around both GT and perturbed projections so small pose/depth
        // updates at level 1 still leave level 0 strictly flat while level 1
        // straddles the surrounding texture.
        for (const auto& xy :
             {gt_img.ProjectPoint(scene.gt.Point3D(p3d_id).xyz),
              ref_img.ProjectPoint(ref_rec.Point3D(p3d_id).xyz)}) {
          if (!xy.has_value()) continue;
          const int cx = static_cast<int>(std::floor(xy->x()));
          const int cy = static_cast<int>(std::floor(xy->y()));
          for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
              base.SetPixel(cx + dx, cy + dy, BitmapColor<uint8_t>(128));
            }
          }
        }
      }
      masked->emplace(image_id, GrayscaleImagePyramid(base, /*num_levels=*/4));
    }
    return masked;
  };

  BundleAdjustmentOptions ba_opts;
  ba_opts.refine_focal_length = false;
  ba_opts.refine_principal_point = false;
  ba_opts.refine_extra_params = false;
  ba_opts.refine_rig_from_world = true;
  ba_opts.refine_points3D = true;
  ba_opts.print_summary = false;
  ba_opts.ceres->solver_options.max_num_iterations = 40;

  BundleAdjustmentConfig config;
  for (const image_t image_id : scene.gt.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);

  // (a) Pure photometric BA (weight = 0) with constant poses: plane 0 tracks
  // are updated at level 1 and filtered at level 0, so FinalizeInactiveTracks
  // must revert them to their exact initial position instead of leaving them
  // at blurry coarse depth.
  {
    Reconstruction rec_pure = MakePerturbed();
    const Reconstruction rec_init = rec_pure;
    auto masked_pyramids = BuildMaskedPyramids(rec_init);
    BundleAdjustmentOptions fixed_ba_opts = ba_opts;
    fixed_ba_opts.refine_rig_from_world = false;

    PhotometricBundleAdjustmentOptions photo_opts;
    photo_opts.pyramid_levels = "1,0";
    photo_opts.geometric_reprojection_weight = 0.0;
    photo_opts.structure_warmup_max_num_iterations = 0;

    auto ba = CreatePhotometricBundleAdjuster(
        fixed_ba_opts, photo_opts, config, masked_pyramids, rec_pure);
    auto base_summary = ba->Solve();
    ASSERT_TRUE(base_summary->IsSolutionUsable());
    const auto* summary =
        dynamic_cast<const PhotometricBundleAdjustmentSummary*>(
            base_summary.get());
    ASSERT_NE(summary, nullptr);
    EXPECT_GE(summary->num_tracks_filtered_low_texture,
              static_cast<int>(plane0_ids.size()));

    for (const point3D_t p3d_id : plane0_ids) {
      EXPECT_NEAR(
          (rec_pure.Point3D(p3d_id).xyz - rec_init.Point3D(p3d_id).xyz).norm(),
          0.0,
          1e-9);
    }
  }

  // (b) Default VCE regularization (weight = 1.0) with variable poses:
  // plane 0 tracks (filtered out at level 0) must be written back from
  // geo_points3D so their reprojection error against the refined poses is tiny.
  {
    Reconstruction rec_vce = MakePerturbed();
    auto masked_pyramids = BuildMaskedPyramids(rec_vce);
    PhotometricBundleAdjustmentOptions photo_opts;
    photo_opts.pyramid_levels = "1,0";
    photo_opts.geometric_reprojection_weight = 1.0;

    auto ba = CreatePhotometricBundleAdjuster(
        ba_opts, photo_opts, config, masked_pyramids, rec_vce);
    ASSERT_TRUE(ba->Solve()->IsSolutionUsable());

    double max_reproj_err = 0.0;
    for (const point3D_t p3d_id : plane0_ids) {
      const Point3D& pt = rec_vce.Point3D(p3d_id);
      for (const auto& elem : pt.track.Elements()) {
        const Image& img = rec_vce.Image(elem.image_id);
        const auto proj = img.ProjectPoint(pt.xyz);
        ASSERT_TRUE(proj.has_value());
        max_reproj_err = std::max(
            max_reproj_err, (*proj - img.Point2D(elem.point2D_idx).xy).norm());
      }
    }
    EXPECT_LT(max_reproj_err, 0.25);
  }
}

TEST(PhotometricBundleAdjustment,
     GracefulCancellationKeepsPointsConsistentWithPoses) {
  auto scene = BuildSyntheticMultiPlaneScene(CameraModelId::kSimpleRadial,
                                             {420.0, 256.0, 256.0, -0.05},
                                             /*width=*/512,
                                             /*height=*/512,
                                             /*num_views=*/5);

  // Stamp flat gray patches around plane 0's projections so plane 0 tracks are
  // photometrically inactive (updated via FinalizeInactiveTracks from
  // geo_points3D), while plane 1 tracks stay photometrically active (updated
  // via UpdatePointsFromPlanes).
  std::vector<point3D_t> plane0_ids;
  std::vector<point3D_t> plane1_ids;
  for (const point3D_t p3d_id : scene.gt.Point3DIds()) {
    if (scene.point_to_plane.at(p3d_id) == 0) {
      plane0_ids.push_back(p3d_id);
    } else {
      plane1_ids.push_back(p3d_id);
    }
  }
  ASSERT_FALSE(plane0_ids.empty());
  ASSERT_FALSE(plane1_ids.empty());

  auto masked_pyramids =
      std::make_shared<FlatHashMap<image_t, GrayscaleImagePyramid>>();
  for (const auto& [image_id, pyr] : *scene.pyramids) {
    Bitmap base = pyr.Level(0).bitmap.Clone();
    const Image& img = scene.gt.Image(image_id);
    for (const point3D_t p3d_id : plane0_ids) {
      const auto xy = img.ProjectPoint(scene.gt.Point3D(p3d_id).xyz);
      if (!xy.has_value()) continue;
      const int cx = static_cast<int>(std::floor(xy->x()));
      const int cy = static_cast<int>(std::floor(xy->y()));
      for (int dy = -12; dy <= 12; ++dy) {
        for (int dx = -12; dx <= 12; ++dx) {
          base.SetPixel(cx + dx, cy + dy, BitmapColor<uint8_t>(128));
        }
      }
    }
    masked_pyramids->emplace(image_id,
                             GrayscaleImagePyramid(base, /*num_levels=*/4));
  }

  Reconstruction rec = scene.gt;
  for (int f = 3; f <= 5; ++f) {
    rec.Frame(f).RigFromWorld().translation() +=
        Eigen::Vector3d(0.015, -0.012, 0.012);
  }
  for (const point3D_t p3d_id : rec.Point3DIds()) {
    rec.Point3D(p3d_id).xyz += Eigen::Vector3d(0.0, 0.0, 0.02);
  }

  // Trigger cancellation during joint optimization after several iterations so
  // Ceres mutates poses/points in-place and returns USER_SUCCESS before
  // completing all pyramid levels.
  int stop_calls = 0;
  BundleAdjustmentOptions ba_opts;
  ba_opts.refine_focal_length = false;
  ba_opts.refine_principal_point = false;
  ba_opts.refine_extra_params = false;
  ba_opts.refine_rig_from_world = true;
  ba_opts.refine_points3D = true;
  ba_opts.print_summary = false;
  ba_opts.ceres->solver_options.max_num_iterations = 50;
  ba_opts.check_if_stopped = [&stop_calls]() { return ++stop_calls >= 12; };

  PhotometricBundleAdjustmentOptions photo_opts;
  photo_opts.pyramid_levels = "1,0";
  photo_opts.structure_warmup_max_num_iterations = 0;
  photo_opts.geometric_reprojection_weight = 1.0;

  BundleAdjustmentConfig config;
  for (const image_t image_id : rec.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.SetConstantRigFromWorldPose(1);
  config.SetConstantRigFromWorldPose(2);

  auto ComputePlaneDist = [&](const Reconstruction& r,
                              const std::vector<point3D_t>& ids) {
    double sum = 0.0;
    for (const point3D_t p3d_id : ids) {
      const auto& pl = scene.planes[scene.point_to_plane.at(p3d_id)];
      sum += std::abs(pl.normal.dot(r.Point3D(p3d_id).xyz) - pl.d);
    }
    return sum / ids.size();
  };
  auto ComputeMeanReprojError = [](const Reconstruction& poses_rec,
                                   const Reconstruction& points_rec,
                                   const std::vector<point3D_t>& ids) {
    double sum = 0.0;
    int count = 0;
    for (const point3D_t p3d_id : ids) {
      const Point3D& pt = points_rec.Point3D(p3d_id);
      for (const auto& elem : pt.track.Elements()) {
        const Image& img = poses_rec.Image(elem.image_id);
        const auto proj = img.ProjectPoint(pt.xyz);
        if (proj.has_value()) {
          sum += (*proj - img.Point2D(elem.point2D_idx).xy).norm();
          ++count;
        }
      }
    }
    return count > 0 ? (sum / count) : 1e9;
  };

  const Reconstruction rec_before = rec;
  const double init_plane1_dist = ComputePlaneDist(rec_before, plane1_ids);

  auto ba = CreatePhotometricBundleAdjuster(
      ba_opts, photo_opts, config, masked_pyramids, rec);
  auto summary = ba->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());
  EXPECT_EQ(summary->termination_type,
            BundleAdjustmentTerminationType::USER_SUCCESS);

  // Confirm poses were mutated, active photometric tracks (plane 1) were
  // written back from planes, and inactive tracks (plane 0) were written back
  // from geo_points3D to match the committed poses.
  double max_pose_delta = 0.0;
  for (int f = 3; f <= 5; ++f) {
    max_pose_delta = std::max(max_pose_delta,
                              (rec.Frame(f).RigFromWorld().translation() -
                               rec_before.Frame(f).RigFromWorld().translation())
                                  .norm());
  }
  EXPECT_GT(max_pose_delta, 1e-3);
  EXPECT_LT(ComputePlaneDist(rec, plane1_ids), 0.35 * init_plane1_dist);
  const double inactive_reproj_updated =
      ComputeMeanReprojError(rec, rec, plane0_ids);
  const double inactive_reproj_stale =
      ComputeMeanReprojError(rec, rec_before, plane0_ids);
  EXPECT_LT(inactive_reproj_updated, 0.25 * inactive_reproj_stale);
}

}  // namespace
}  // namespace colmap
