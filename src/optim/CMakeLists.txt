set(FOLDER_NAME "optim")

COLMAP_ADD_LIBRARY(optim
    bundle_adjustment.h bundle_adjustment.cc
    combination_sampler.h combination_sampler.cc
    progressive_sampler.h progressive_sampler.cc
    random_sampler.h random_sampler.cc
    sprt.h sprt.cc
    support_measurement.h support_measurement.cc
)

COLMAP_ADD_TEST(bundle_adjustment_test bundle_adjustment_test.cc)
COLMAP_ADD_TEST(combination_sampler_test combination_sampler_test.cc)
COLMAP_ADD_TEST(loransac_test loransac_test.cc)
COLMAP_ADD_TEST(progressive_sampler_test progressive_sampler_test.cc)
COLMAP_ADD_TEST(random_sampler_test random_sampler_test.cc)
COLMAP_ADD_TEST(ransac_test ransac_test.cc)
COLMAP_ADD_TEST(support_measurement_test support_measurement_test.cc)
