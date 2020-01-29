// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#define TEST_NAME "estimators/translation_transform"
#include "util/testing.h"

#include <Eigen/Core>

#include "estimators/translation_transform.h"
#include "optim/ransac.h"
#include "util/random.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEstimate) {
  SetPRNGSeed(0);

  std::vector<Eigen::Vector2d> src;
  for (size_t i = 0; i < 100; ++i) {
    src.emplace_back(RandomReal(-1000.0, 1000.0), RandomReal(-1000.0, 1000.0));
  }

  Eigen::Vector2d translation(RandomReal(-1000.0, 1000.0),
                              RandomReal(-1000.0, 1000.0));

  std::vector<Eigen::Vector2d> dst;
  for (size_t i = 0; i < src.size(); ++i) {
    dst.push_back(src[i] + translation);
  }

  const auto estimated_translation =
      TranslationTransformEstimator<2>::Estimate(src, dst)[0];

  BOOST_CHECK_CLOSE(translation(0), estimated_translation(0), 1e-6);
  BOOST_CHECK_CLOSE(translation(1), estimated_translation(1), 1e-6);

  std::vector<double> residuals;
  TranslationTransformEstimator<2>::Residuals(src, dst, estimated_translation,
                                              &residuals);

  for (size_t i = 0; i < residuals.size(); ++i) {
    BOOST_CHECK(residuals[i] < 1e-6);
  }
}
