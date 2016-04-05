// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "estimators/translation_transform"
#include <boost/test/unit_test.hpp>

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
