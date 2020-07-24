# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

set(FOLDER_NAME "optim")

COLMAP_ADD_SOURCES(
    bundle_adjustment.h bundle_adjustment.cc
    combination_sampler.h combination_sampler.cc
    least_absolute_deviations.h least_absolute_deviations.cc
    progressive_sampler.h progressive_sampler.cc
    random_sampler.h random_sampler.cc
    sprt.h sprt.cc
    support_measurement.h support_measurement.cc
)

COLMAP_ADD_TEST(bundle_adjustment_test bundle_adjustment_test.cc)
COLMAP_ADD_TEST(combination_sampler_test combination_sampler_test.cc)
COLMAP_ADD_TEST(least_absolute_deviations_test
                least_absolute_deviations_test.cc)
COLMAP_ADD_TEST(loransac_test loransac_test.cc)
COLMAP_ADD_TEST(progressive_sampler_test progressive_sampler_test.cc)
COLMAP_ADD_TEST(random_sampler_test random_sampler_test.cc)
COLMAP_ADD_TEST(ransac_test ransac_test.cc)
COLMAP_ADD_TEST(support_measurement_test support_measurement_test.cc)
