// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/cost_functions/photometric_error.h"

#include "colmap/scene/camera.h"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

GrayscaleImageLevel MakeSmoothTestImage(const int width, const int height) {
  GrayscaleImageLevel level;
  level.bitmap = Bitmap(width, height, /*as_rgb=*/false);
  level.scale_x = 1.0;
  level.scale_y = 1.0;
  std::vector<uint8_t>& data = level.bitmap.RowMajorData();
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const double v = 128.0 + 50.0 * std::sin(0.11 * x + 0.07 * y) +
                       40.0 * std::cos(0.05 * x - 0.13 * y) +
                       25.0 * std::sin(0.19 * x) * std::cos(0.17 * y);
      data[static_cast<size_t>(y) * width + x] =
          static_cast<uint8_t>(std::clamp(std::lround(v), 0L, 255L));
    }
  }
  return level;
}

PhotometricPatchData MakePatch(const Camera& src_cam,
                               const GrayscaleImageLevel& src_level,
                               const Eigen::Vector2d& anchor_xy,
                               const int patch_size = 4) {
  PhotometricPatchData patch;
  patch.anchor_xy = anchor_xy;
  patch.src_cam_params_init = src_cam.params;

  const double half = 0.5 * (patch_size - 1);
  std::vector<double> vals;
  for (int dy = 0; dy < patch_size; ++dy) {
    for (int dx = 0; dx < patch_size; ++dx) {
      const double px = anchor_xy.x() + (dx - half);
      const double py = anchor_xy.y() + (dy - half);
      patch.src_xs.push_back(px);
      patch.src_ys.push_back(py);
      const auto ray2 = src_cam.CamFromImg(Eigen::Vector2d(px, py));
      THROW_CHECK(ray2.has_value());
      patch.src_rays_init.push_back(ray2->homogeneous());
      const auto v = src_level.InterpolateBilinear(px, py);
      THROW_CHECK(v.has_value());
      vals.push_back(*v);
    }
  }
  double mean = 0.0;
  for (const double v : vals) mean += v;
  mean /= static_cast<double>(vals.size());
  double sq = 0.0;
  for (double& v : vals) {
    v -= mean;
    sq += v * v;
  }
  const double inv_s = 1.0 / std::sqrt(sq);
  patch.src_norm.resize(vals.size());
  for (size_t i = 0; i < vals.size(); ++i) {
    patch.src_norm[i] = vals[i] * inv_s;
  }
  return patch;
}

void VerifyAnalyticalJacobiansAgainstFiniteDiff(
    const ceres::CostFunction& cost_fn,
    std::vector<std::vector<double>> params,
    const std::vector<bool>& is_pose_block,
    const double eps = 1e-7,
    const double tol = 1e-5) {
  const int num_res = cost_fn.num_residuals();
  const auto& sizes = cost_fn.parameter_block_sizes();
  const int num_blocks = static_cast<int>(sizes.size());
  ASSERT_EQ(is_pose_block.size(), static_cast<size_t>(num_blocks));

  std::vector<double*> param_ptrs(num_blocks);
  std::vector<std::vector<double>> jac_storage(num_blocks);
  std::vector<double*> jac_ptrs(num_blocks);
  for (int b = 0; b < num_blocks; ++b) {
    param_ptrs[b] = params[b].data();
    jac_storage[b].assign(static_cast<size_t>(num_res) * sizes[b], 0.0);
    jac_ptrs[b] = jac_storage[b].data();
  }

  std::vector<double> res_base(num_res, 0.0);
  ASSERT_TRUE(
      cost_fn.Evaluate(param_ptrs.data(), res_base.data(), jac_ptrs.data()));

  std::vector<double> res_plus(num_res, 0.0);
  std::vector<double> res_minus(num_res, 0.0);
  ceres::EigenQuaternionManifold quat_manifold;

  for (int b = 0; b < num_blocks; ++b) {
    if (is_pose_block[b]) {
      ASSERT_EQ(sizes[b], 7);
      // Verify quaternion part on the SO(3) tangent space (matching Ceres's
      // EigenQuaternionManifold, since R(q)^T R(q) == I holds on ||q|| = 1).
      double plus_jac[4 * 3];
      ASSERT_TRUE(quat_manifold.PlusJacobian(params[b].data(), plus_jac));
      const std::vector<double> orig_q(params[b].begin(),
                                       params[b].begin() + 4);
      for (int t = 0; t < 3; ++t) {
        double delta_p[3] = {0.0, 0.0, 0.0};
        double delta_m[3] = {0.0, 0.0, 0.0};
        delta_p[t] = eps;
        delta_m[t] = -eps;
        ASSERT_TRUE(
            quat_manifold.Plus(orig_q.data(), delta_p, params[b].data()));
        ASSERT_TRUE(
            cost_fn.Evaluate(param_ptrs.data(), res_plus.data(), nullptr));
        ASSERT_TRUE(
            quat_manifold.Plus(orig_q.data(), delta_m, params[b].data()));
        ASSERT_TRUE(
            cost_fn.Evaluate(param_ptrs.data(), res_minus.data(), nullptr));
        std::copy(orig_q.begin(), orig_q.end(), params[b].begin());

        for (int r = 0; r < num_res; ++r) {
          const double num_j = (res_plus[r] - res_minus[r]) / (2.0 * eps);
          double ana_j = 0.0;
          for (int q = 0; q < 4; ++q) {
            ana_j += jac_storage[b][r * 7 + q] * plus_jac[q * 3 + t];
          }
          EXPECT_NEAR(ana_j, num_j, tol * std::max(1.0, std::abs(num_j)))
              << "Mismatch at pose block " << b << " rot_tangent " << t
              << " res " << r;
        }
      }
      // Verify translation part (Euclidean).
      for (int c = 4; c < 7; ++c) {
        const double orig = params[b][c];
        params[b][c] = orig + eps;
        ASSERT_TRUE(
            cost_fn.Evaluate(param_ptrs.data(), res_plus.data(), nullptr));
        params[b][c] = orig - eps;
        ASSERT_TRUE(
            cost_fn.Evaluate(param_ptrs.data(), res_minus.data(), nullptr));
        params[b][c] = orig;
        for (int r = 0; r < num_res; ++r) {
          const double num_j = (res_plus[r] - res_minus[r]) / (2.0 * eps);
          const double ana_j = jac_storage[b][r * 7 + c];
          EXPECT_NEAR(ana_j, num_j, tol * std::max(1.0, std::abs(num_j)))
              << "Mismatch at pose block " << b << " trans " << c << " res "
              << r;
        }
      }
      continue;
    }

    for (int c = 0; c < sizes[b]; ++c) {
      const double orig = params[b][c];
      const double step = eps * std::max(1.0, std::abs(orig));
      params[b][c] = orig + step;
      ASSERT_TRUE(
          cost_fn.Evaluate(param_ptrs.data(), res_plus.data(), nullptr));
      params[b][c] = orig - step;
      ASSERT_TRUE(
          cost_fn.Evaluate(param_ptrs.data(), res_minus.data(), nullptr));
      params[b][c] = orig;

      for (int r = 0; r < num_res; ++r) {
        const double num_j = (res_plus[r] - res_minus[r]) / (2.0 * step);
        const double ana_j = jac_storage[b][r * sizes[b] + c];
        EXPECT_NEAR(ana_j, num_j, tol * std::max(1.0, std::abs(num_j)))
            << "Mismatch at block " << b << " col " << c << " res " << r;
      }
    }
  }
}

TEST(PhotometricNCCCostFunction, JacobiansDistinctThinPrismCameras) {
  const auto img_src = MakeSmoothTestImage(200, 150);
  const auto img_tgt = MakeSmoothTestImage(200, 150);

  Camera src_cam = Camera::CreateFromModelId(
      1, CameraModelId::kThinPrismFisheye, 180.0, 200, 150);
  src_cam.params = {180.0,
                    179.0,
                    101.2,
                    74.8,
                    0.08,
                    -0.02,
                    0.001,
                    -0.001,
                    0.01,
                    -0.01,
                    0.0005,
                    -0.0004};
  Camera tgt_cam = Camera::CreateFromModelId(
      2, CameraModelId::kThinPrismFisheye, 185.0, 200, 150);
  tgt_cam.params = {185.0,
                    184.0,
                    99.5,
                    76.1,
                    0.06,
                    -0.015,
                    -0.001,
                    0.0008,
                    0.005,
                    -0.008,
                    -0.0003,
                    0.0006};

  const auto patch = MakePatch(src_cam, img_src, Eigen::Vector2d(103.3, 72.7));

  Rigid3d src_rfw(Eigen::Quaterniond(0.99, 0.05, -0.04, 0.03).normalized(),
                  Eigen::Vector3d(0.1, -0.05, 0.2));
  Rigid3d tgt_rfw(Eigen::Quaterniond(0.98, -0.03, 0.08, -0.02).normalized(),
                  Eigen::Vector3d(-0.25, 0.08, 0.15));
  Eigen::Vector3d n(0.03, -0.02, 0.25);

  const auto layout = photometric_internal::SourceTargetBlockLayout::Create(
      src_cam.model_id,
      tgt_cam.model_id,
      /*src_frame_id=*/1,
      /*tgt_frame_id=*/2,
      /*src_sensor_id=*/src_cam.SensorId(),
      /*tgt_sensor_id=*/tgt_cam.SensorId(),
      /*src_is_ref_sensor=*/true,
      /*tgt_is_ref_sensor=*/true,
      /*src_camera_id=*/1,
      /*tgt_camera_id=*/2);

  // Also perturb src_cam slightly away from src_cam_params_init to exercise
  // Newton undistortion + implicit function theorem!
  std::vector<double> src_cam_perturbed = src_cam.params;
  src_cam_perturbed[0] += 1.2;
  src_cam_perturbed[4] += 0.005;

  std::vector<std::vector<double>> params = {
      {n.x(), n.y(), n.z()},
      std::vector<double>(src_rfw.params.data(), src_rfw.params.data() + 7),
      std::vector<double>(tgt_rfw.params.data(), tgt_rfw.params.data() + 7),
      src_cam_perturbed,
      tgt_cam.params,
  };

  PhotometricNCCCostFunction cost(patch, img_tgt, layout, 1.0);
  VerifyAnalyticalJacobiansAgainstFiniteDiff(
      cost, params, {false, true, true, false, false});
}

TEST(PhotometricNCCCostFunction, JacobiansSharedCameraAndRigSensors) {
  const auto img_src = MakeSmoothTestImage(200, 150);
  const auto img_tgt = MakeSmoothTestImage(200, 150);

  Camera shared_cam = Camera::CreateFromModelId(
      1, CameraModelId::kSimpleRadial, 180.0, 200, 150);
  shared_cam.params = {180.0, 100.0, 75.0, -0.05};
  const auto patch =
      MakePatch(shared_cam, img_src, Eigen::Vector2d(97.4, 78.2));

  Rigid3d src_rfw(Eigen::Quaterniond(0.99, 0.02, -0.03, 0.01).normalized(),
                  Eigen::Vector3d(0.05, 0.02, 0.1));
  Rigid3d tgt_rfw(Eigen::Quaterniond(0.98, -0.04, 0.06, -0.01).normalized(),
                  Eigen::Vector3d(-0.15, 0.06, 0.08));
  Rigid3d src_sfr(Eigen::Quaterniond(0.995, 0.01, 0.02, -0.01).normalized(),
                  Eigen::Vector3d(0.04, -0.01, 0.0));
  Rigid3d tgt_sfr(Eigen::Quaterniond(0.994, -0.02, -0.01, 0.02).normalized(),
                  Eigen::Vector3d(-0.04, 0.01, 0.01));
  Eigen::Vector3d n(-0.02, 0.03, 0.22);

  // Case 1: Different frames, non-ref distinct rig sensors, shared camera.
  {
    const auto layout = photometric_internal::SourceTargetBlockLayout::Create(
        shared_cam.model_id,
        shared_cam.model_id,
        /*src_frame_id=*/1,
        /*tgt_frame_id=*/2,
        /*src_sensor_id=*/sensor_t(SensorType::CAMERA, 10),
        /*tgt_sensor_id=*/sensor_t(SensorType::CAMERA, 20),
        /*src_is_ref_sensor=*/false,
        /*tgt_is_ref_sensor=*/false,
        /*src_camera_id=*/1,
        /*tgt_camera_id=*/1);
    std::vector<std::vector<double>> params = {
        {n.x(), n.y(), n.z()},
        std::vector<double>(src_rfw.params.data(), src_rfw.params.data() + 7),
        std::vector<double>(tgt_rfw.params.data(), tgt_rfw.params.data() + 7),
        std::vector<double>(src_sfr.params.data(), src_sfr.params.data() + 7),
        std::vector<double>(tgt_sfr.params.data(), tgt_sfr.params.data() + 7),
        shared_cam.params,
    };
    PhotometricNCCCostFunction cost(patch, img_tgt, layout, 1.0);
    VerifyAnalyticalJacobiansAgainstFiniteDiff(
        cost, params, {false, true, true, true, true, false});
  }

  // Case 2: Same non-ref rig sensor across two frames, shared camera.
  {
    const auto layout = photometric_internal::SourceTargetBlockLayout::Create(
        shared_cam.model_id,
        shared_cam.model_id,
        /*src_frame_id=*/1,
        /*tgt_frame_id=*/2,
        /*src_sensor_id=*/sensor_t(SensorType::CAMERA, 10),
        /*tgt_sensor_id=*/sensor_t(SensorType::CAMERA, 10),
        /*src_is_ref_sensor=*/false,
        /*tgt_is_ref_sensor=*/false,
        /*src_camera_id=*/1,
        /*tgt_camera_id=*/1);
    std::vector<std::vector<double>> params = {
        {n.x(), n.y(), n.z()},
        std::vector<double>(src_rfw.params.data(), src_rfw.params.data() + 7),
        std::vector<double>(tgt_rfw.params.data(), tgt_rfw.params.data() + 7),
        std::vector<double>(src_sfr.params.data(), src_sfr.params.data() + 7),
        shared_cam.params,
    };
    PhotometricNCCCostFunction cost(patch, img_tgt, layout, 1.0);
    VerifyAnalyticalJacobiansAgainstFiniteDiff(
        cost, params, {false, true, true, true, false});
  }

  // Case 3: Same frame (two distinct cameras in one rig), rig_from_world
  // omitted.
  {
    const auto layout = photometric_internal::SourceTargetBlockLayout::Create(
        shared_cam.model_id,
        shared_cam.model_id,
        /*src_frame_id=*/1,
        /*tgt_frame_id=*/1,
        /*src_sensor_id=*/sensor_t(SensorType::CAMERA, 10),
        /*tgt_sensor_id=*/sensor_t(SensorType::CAMERA, 20),
        /*src_is_ref_sensor=*/false,
        /*tgt_is_ref_sensor=*/false,
        /*src_camera_id=*/1,
        /*tgt_camera_id=*/2);
    std::vector<std::vector<double>> params = {
        {n.x(), n.y(), n.z()},
        std::vector<double>(src_sfr.params.data(), src_sfr.params.data() + 7),
        std::vector<double>(tgt_sfr.params.data(), tgt_sfr.params.data() + 7),
        shared_cam.params,
        shared_cam.params,
    };
    PhotometricNCCCostFunction cost(patch, img_tgt, layout, 1.0);
    VerifyAnalyticalJacobiansAgainstFiniteDiff(
        cost, params, {false, true, true, false, false});
  }
}

TEST(PhotometricNCCCostFunction, BarrierOnInvalidDepthOrOutOfBounds) {
  const auto img = MakeSmoothTestImage(100, 100);
  Camera cam =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 100.0, 100, 100);
  const auto patch = MakePatch(cam, img, Eigen::Vector2d(50.3, 50.7));
  const auto layout =
      photometric_internal::SourceTargetBlockLayout::Create(cam.model_id,
                                                            cam.model_id,
                                                            1,
                                                            2,
                                                            cam.SensorId(),
                                                            cam.SensorId(),
                                                            true,
                                                            true,
                                                            1,
                                                            1);
  PhotometricNCCCostFunction cost(patch, img, layout, 1.0);

  Rigid3d pose1;
  Rigid3d pose2;
  // Negative depth plane -> barrier residual with ||r||^2 == 4 and zero jac.
  double n_bad[3] = {0.0, 0.0, -0.5};
  double* ptrs[4] = {
      n_bad, pose1.params.data(), pose2.params.data(), cam.params.data()};
  std::vector<double> res(16, 0.0);
  std::vector<double> j_n(16 * 3, 99.0);
  double* jptrs[4] = {j_n.data(), nullptr, nullptr, nullptr};
  ASSERT_TRUE(cost.Evaluate(ptrs, res.data(), jptrs));
  double sq = 0.0;
  for (const double r : res) sq += r * r;
  EXPECT_NEAR(sq, 4.0, 1e-12);
  for (const double v : j_n) EXPECT_DOUBLE_EQ(v, 0.0);

  // Constant-intensity target image with min_target_norm == 0.0 must still
  // return the finite barrier residual rather than dividing by zero (NaN).
  GrayscaleImageLevel flat_img;
  flat_img.bitmap = Bitmap(100, 100, /*as_rgb=*/false);
  flat_img.bitmap.Fill(BitmapColor<uint8_t>(128));
  PhotometricNCCCostFunction flat_cost(patch, flat_img, layout, 0.0);
  double n_ok[3] = {0.0, 0.0, 0.5};
  double* flat_ptrs[4] = {
      n_ok, pose1.params.data(), pose2.params.data(), cam.params.data()};
  std::fill(j_n.begin(), j_n.end(), 99.0);
  ASSERT_TRUE(flat_cost.Evaluate(flat_ptrs, res.data(), jptrs));
  double flat_sq = 0.0;
  for (const double r : res) {
    EXPECT_TRUE(std::isfinite(r));
    flat_sq += r * r;
  }
  EXPECT_NEAR(flat_sq, 4.0, 1e-12);
  for (const double v : j_n) EXPECT_DOUBLE_EQ(v, 0.0);
}

TEST(PhotometricNCCCostFunction, NonFourByFourPatchSize) {
  // patch_size = 3 -> M = 9 verifies non-default patch sizes.
  const auto img_src = MakeSmoothTestImage(200, 150);
  const auto img_tgt = MakeSmoothTestImage(200, 150);
  Camera cam = Camera::CreateFromModelId(
      1, CameraModelId::kThinPrismFisheye, 180.0, 200, 150);
  cam.params[4] = -0.04;
  cam.params[5] = 0.01;
  const auto patch =
      MakePatch(cam, img_src, Eigen::Vector2d(105.4, 72.3), /*patch_size=*/3);
  ASSERT_EQ(patch.NumSamples(), 9);

  const auto layout =
      photometric_internal::SourceTargetBlockLayout::Create(cam.model_id,
                                                            cam.model_id,
                                                            1,
                                                            2,
                                                            cam.SensorId(),
                                                            cam.SensorId(),
                                                            true,
                                                            true,
                                                            1,
                                                            1);
  Rigid3d src_rfw(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  Rigid3d tgt_rfw(
      Eigen::Quaterniond(Eigen::AngleAxisd(0.06, Eigen::Vector3d(0, 1, 0))),
      Eigen::Vector3d(0.12, -0.04, 0.02));
  const Eigen::Vector3d n(0.04, -0.02, 0.3);

  std::vector<std::vector<double>> params = {
      {n.x(), n.y(), n.z()},
      std::vector<double>(src_rfw.params.data(), src_rfw.params.data() + 7),
      std::vector<double>(tgt_rfw.params.data(), tgt_rfw.params.data() + 7),
      cam.params,
  };
  PhotometricNCCCostFunction cost(patch, img_tgt, layout, 1.0);
  VerifyAnalyticalJacobiansAgainstFiniteDiff(
      cost, params, {false, true, true, false});
}

}  // namespace
}  // namespace colmap
