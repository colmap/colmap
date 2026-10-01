// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/solvers/similarity_transform.h"

#include "colmap/optim/loransac.h"

namespace colmap {
namespace {

template <bool kEstimateScale>
inline bool EstimateRigidOrSim3d(const std::vector<Eigen::Vector3d>& src,
                                 const std::vector<Eigen::Vector3d>& tgt,
                                 Eigen::Matrix3x4d& tgt_from_src) {
  std::vector<Eigen::Matrix3x4d> models;
  SimilarityTransformEstimator<3, kEstimateScale>().Estimate(src, tgt, &models);
  if (models.empty()) {
    return false;
  }
  THROW_CHECK_EQ(models.size(), 1);
  tgt_from_src = models[0];
  return true;
}

template <bool kEstimateScale>
inline typename RANSAC<SimilarityTransformEstimator<3, kEstimateScale>>::Report
EstimateRigidOrSim3dRobust(const std::vector<Eigen::Vector3d>& src,
                           const std::vector<Eigen::Vector3d>& tgt,
                           const RANSACOptions& options,
                           Eigen::Matrix3x4d& tgt_from_src) {
  LORANSAC<SimilarityTransformEstimator<3, kEstimateScale>,
           SimilarityTransformEstimator<3, kEstimateScale>>
      ransac(options);
  auto report = ransac.Estimate(src, tgt);
  if (report.success) {
    tgt_from_src = report.model;
  }
  return report;
}

}  // namespace

bool EstimateRigid3d(const std::vector<Eigen::Vector3d>& src,
                     const std::vector<Eigen::Vector3d>& tgt,
                     Rigid3d& tgt_from_src) {
  Eigen::Matrix3x4d tgt_from_src_mat = Eigen::Matrix3x4d::Zero();
  if (!EstimateRigidOrSim3d<false>(src, tgt, tgt_from_src_mat)) {
    return false;
  }
  tgt_from_src = Rigid3d::FromMatrix(tgt_from_src_mat);
  return true;
}

typename RANSAC<SimilarityTransformEstimator<3, false>>::Report
EstimateRigid3dRobust(const std::vector<Eigen::Vector3d>& src,
                      const std::vector<Eigen::Vector3d>& tgt,
                      const RANSACOptions& options,
                      Rigid3d& tgt_from_src) {
  Eigen::Matrix3x4d tgt_from_src_mat = Eigen::Matrix3x4d::Zero();
  auto report =
      EstimateRigidOrSim3dRobust<false>(src, tgt, options, tgt_from_src_mat);
  tgt_from_src = Rigid3d::FromMatrix(tgt_from_src_mat);
  return report;
}

bool EstimateSim3d(const std::vector<Eigen::Vector3d>& src,
                   const std::vector<Eigen::Vector3d>& tgt,
                   Sim3d& tgt_from_src) {
  Eigen::Matrix3x4d tgt_from_src_mat = Eigen::Matrix3x4d::Zero();
  if (!EstimateRigidOrSim3d<true>(src, tgt, tgt_from_src_mat)) {
    return false;
  }
  tgt_from_src = Sim3d::FromMatrix(tgt_from_src_mat);
  return true;
}

typename RANSAC<SimilarityTransformEstimator<3, true>>::Report
EstimateSim3dRobust(const std::vector<Eigen::Vector3d>& src,
                    const std::vector<Eigen::Vector3d>& tgt,
                    const RANSACOptions& options,
                    Sim3d& tgt_from_src) {
  Eigen::Matrix3x4d tgt_from_src_mat = Eigen::Matrix3x4d::Zero();
  auto report =
      EstimateRigidOrSim3dRobust<true>(src, tgt, options, tgt_from_src_mat);
  if (report.success) {
    tgt_from_src = Sim3d::FromMatrix(tgt_from_src_mat);
  }
  return report;
}

}  // namespace colmap
