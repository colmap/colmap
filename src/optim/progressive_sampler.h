// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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

#ifndef COLMAP_SRC_OPTIM_PROGRESSIVE_SAMPLER_H_
#define COLMAP_SRC_OPTIM_PROGRESSIVE_SAMPLER_H_

#include "optim/sampler.h"

namespace colmap {

// Random sampler for PROSAC (Progressive Sample Consensus), as described in:
//
//    "Matching with PROSAC - Progressive Sample Consensus".
//        Ondrej Chum and Matas, CVPR 2005.
//
// Note that a separate sampler should be instantiated per thread and that the
// data to be sampled from is assumed to be sorted according to the quality
// function in descending order, i.e., higher quality data is closer to the
// front of the list.
class ProgressiveSampler : public Sampler {
 public:
  explicit ProgressiveSampler(const size_t num_samples);

  void Initialize(const size_t total_num_samples) override;

  size_t MaxNumSamples() override;

  std::vector<size_t> Sample() override;

 private:
  const size_t num_samples_;
  size_t total_num_samples_;

  // The number of generated samples, i.e. the number of calls to `Sample`.
  size_t t_;
  size_t n_;

  // Variables defined in equation 3.
  double T_n_;
  double T_n_p_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_OPTIM_PROGRESSIVE_SAMPLER_H_
