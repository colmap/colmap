// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/alignment.h"
#include "colmap/geometry/sim3.h"
#include "colmap/scene/reconstruction.h"

namespace colmap {

bool CompareModels(const Reconstruction& reconstruction1,
                   const Reconstruction& reconstruction2,
                   const std::string& alignment_error,
                   double min_inlier_observations,
                   double max_reproj_error,
                   double max_proj_center_error,
                   std::vector<ImageAlignmentError>& errors,
                   Sim3d& rec2_from_rec1);

int RunModelAligner(int argc, char** argv);
int RunModelAnalyzer(int argc, char** argv);
int RunModelClusterer(int argc, char** argv);
int RunModelComparer(int argc, char** argv);
int RunModelConverter(int argc, char** argv);
int RunModelCropper(int argc, char** argv);
int RunModelMerger(int argc, char** argv);
int RunModelOrientationAligner(int argc, char** argv);
int RunModelSplitter(int argc, char** argv);
int RunModelTransformer(int argc, char** argv);

}  // namespace colmap
