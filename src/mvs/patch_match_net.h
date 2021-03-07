// Copyright (c) 2021, Microsoft.
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
// Author: Antonios Matakos (anmatako-at-microsoft-dot-com)

#ifndef COLMAP_SRC_MVS_PATCH_MATCH_NET_H_
#define COLMAP_SRC_MVS_PATCH_MATCH_NET_H_

#include <memory>
#include <vector>

#include "mvs/depth_map.h"
#include "mvs/normal_map.h"
#include "mvs/patch_match.h"
#include "torch/torch.h"

namespace colmap {
namespace mvs {

class PatchMatchNet : public PatchMatch {
 public:
  PatchMatchNet(const PatchMatchOptions& options,
                const PatchMatch::Problem& problem, const int thread_index = 0);

  virtual ~PatchMatchNet() {}

  virtual void Run() override;

  inline virtual DepthMap GetDepthMap() const override { return depth_map_; }
  inline virtual ConfidenceMap GetConfidenceMap() const override {
    return confidence_map_;
  }
  inline virtual NormalMap GetNormalMap() const override {
    return NormalMap(depth_map_.GetWidth(), depth_map_.GetHeight());
  }

 private:
  void InitModule();
  void InitProblemInputs();

  const int thread_index_;
  torch::Tensor intrinsics_, extrinsics_, depth_params_;
  std::vector<torch::Tensor> images_;
  DepthMap depth_map_;
  ConfidenceMap confidence_map_;

  static std::unordered_map<int, torch::jit::Module> model_;
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_PATCH_MATCH_NET_H_
