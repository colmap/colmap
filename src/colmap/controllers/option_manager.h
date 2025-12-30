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

#pragma once

#include "colmap/controllers/base_option_manager.h"

#include <memory>

namespace colmap {

struct ImageReaderOptions;
struct FeatureExtractionOptions;
struct FeatureMatchingOptions;
struct SiftMatchingOptions;
struct TwoViewGeometryOptions;
struct ExhaustivePairingOptions;
struct SequentialPairingOptions;
struct VocabTreePairingOptions;
struct SpatialPairingOptions;
struct TransitivePairingOptions;
struct ImportedPairingOptions;
struct ExistingMatchedPairingOptions;
struct BundleAdjustmentOptions;
struct IncrementalPipelineOptions;
struct RenderOptions;

namespace mvs {
struct PatchMatchOptions;
struct StereoFusionOptions;
struct PoissonMeshingOptions;
struct DelaunayMeshingOptions;
}  // namespace mvs

class OptionManager : public BaseOptionManager {
 public:
  explicit OptionManager(bool add_project_options = true);

  // Create "optimal" set of options for different reconstruction scenarios.
  void ModifyForIndividualData();
  void ModifyForVideoData();
  void ModifyForInternetData();

  // Create "optimal" set of options for different quality settings.
  // Note that the existing options are modified, so if your parameters are
  // already low quality, they will be further degraded.
  void ModifyForLowQuality();
  void ModifyForMediumQuality();
  void ModifyForHighQuality();
  void ModifyForExtremeQuality();

  void AddAllOptions() override;
  void AddFeatureExtractionOptions();
  void AddFeatureMatchingOptions();
  void AddTwoViewGeometryOptions();
  void AddExhaustivePairingOptions();
  void AddSequentialPairingOptions();
  void AddVocabTreePairingOptions();
  void AddSpatialPairingOptions();
  void AddTransitivePairingOptions();
  void AddImportedPairingOptions();
  void AddBundleAdjustmentOptions();
  void AddMapperOptions();
  void AddPatchMatchStereoOptions();
  void AddStereoFusionOptions();
  void AddPoissonMeshingOptions();
  void AddDelaunayMeshingOptions();
  void AddRenderOptions();

  void Reset() override;
  void ResetOptions(bool reset_paths) override;
  bool Check() override;
  bool Read(const std::string& path) override;

  std::shared_ptr<ImageReaderOptions> image_reader;
  std::shared_ptr<FeatureExtractionOptions> feature_extraction;
  std::shared_ptr<FeatureMatchingOptions> feature_matching;
  std::shared_ptr<TwoViewGeometryOptions> two_view_geometry;
  std::shared_ptr<ExhaustivePairingOptions> exhaustive_pairing;
  std::shared_ptr<SequentialPairingOptions> sequential_pairing;
  std::shared_ptr<VocabTreePairingOptions> vocab_tree_pairing;
  std::shared_ptr<SpatialPairingOptions> spatial_pairing;
  std::shared_ptr<TransitivePairingOptions> transitive_pairing;
  std::shared_ptr<ImportedPairingOptions> imported_pairing;

  std::shared_ptr<BundleAdjustmentOptions> bundle_adjustment;
  std::shared_ptr<IncrementalPipelineOptions> mapper;

  std::shared_ptr<mvs::PatchMatchOptions> patch_match_stereo;
  std::shared_ptr<mvs::StereoFusionOptions> stereo_fusion;
  std::shared_ptr<mvs::PoissonMeshingOptions> poisson_meshing;
  std::shared_ptr<mvs::DelaunayMeshingOptions> delaunay_meshing;

  std::shared_ptr<RenderOptions> render;

 protected:
  void PostParse() override;
  void PrintHelp() const override;

  std::string mapper_image_list_path_;
  std::string mapper_constant_rig_list_path_;
  std::string mapper_constant_camera_list_path_;

  bool added_feature_extraction_options_ = false;
  bool added_feature_matching_options_ = false;
  bool added_two_view_geometry_options_ = false;
  bool added_exhaustive_pairing_options_ = false;
  bool added_sequential_pairing_options_ = false;
  bool added_vocab_tree_pairing_options_ = false;
  bool added_spatial_pairing_options_ = false;
  bool added_transitive_pairing_options_ = false;
  bool added_image_pairs_pairing_options_ = false;
  bool added_ba_options_ = false;
  bool added_mapper_options_ = false;
  bool added_patch_match_stereo_options_ = false;
  bool added_stereo_fusion_options_ = false;
  bool added_poisson_meshing_options_ = false;
  bool added_delaunay_meshing_options_ = false;
  bool added_render_options_ = false;
};

}  // namespace colmap
