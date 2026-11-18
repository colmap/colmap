// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/retrieval/global_descriptor_model.h"

#include "colmap/util/string.h"

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

#ifdef COLMAP_ONNX_ENABLED

TEST(GlobalDescriptorModel, Registry) {
  const std::vector<std::string_view> names =
      GlobalDescriptorModel::ModelNames();
  ASSERT_GE(names.size(), 2);
  EXPECT_EQ(names[0], "MixVPR");
  EXPECT_EQ(names[1], "MegaLoc");
  for (const std::string_view name : names) {
    const GlobalDescriptorModel* model = GlobalDescriptorModel::GetModel(name);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->name, std::string(name));
    EXPECT_GT(model->descriptor_dim, 0);
    ASSERT_EQ(model->expected_input_shape.size(), 4);
    EXPECT_EQ(model->expected_input_shape[1], 3);
    EXPECT_EQ(model->expected_input_shape[2], model->input_height);
    EXPECT_EQ(model->expected_input_shape[3], model->input_width);
    ASSERT_EQ(model->expected_output_shape.size(), 2);
    EXPECT_EQ(model->expected_output_shape[1], model->descriptor_dim);
    ASSERT_FALSE(model->variants.empty());
    for (const GlobalDescriptorModel::Variant& variant : model->variants) {
      // Download URIs follow the "<url>;<name>;<sha256>" format.
      const std::vector<std::string> parts = StringSplit(variant.uri, ";");
      ASSERT_EQ(parts.size(), 3) << variant.uri;
      EXPECT_EQ(parts[0].rfind("https://", 0), 0) << variant.uri;
      EXPECT_FALSE(parts[1].empty()) << variant.uri;
      EXPECT_EQ(parts[2].size(), 64) << variant.uri;
      EXPECT_EQ(parts[2].find_first_not_of("0123456789abcdef"),
                std::string::npos)
          << variant.uri;
    }
  }
}

TEST(GlobalDescriptorModel, LookupIsCaseInsensitive) {
  const GlobalDescriptorModel* model =
      GlobalDescriptorModel::GetModel("megaloc");
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->name, "MegaLoc");
  EXPECT_EQ(GlobalDescriptorModel::GetModel("MIXVPR"),
            GlobalDescriptorModel::GetModel("MixVPR"));
  EXPECT_EQ(GlobalDescriptorModel::GetModel("NetVLAD"), nullptr);
  EXPECT_EQ(GlobalDescriptorModel::GetModel(""), nullptr);
}

TEST(GlobalDescriptorModel, Variants) {
  const std::vector<std::string_view> variants =
      GlobalDescriptorModel::VariantNames("MixVPR");
  ASSERT_EQ(variants.size(), 2);
  EXPECT_EQ(variants[0], "fp16");
  EXPECT_EQ(variants[1], "fp32");
  EXPECT_TRUE(GlobalDescriptorModel::VariantNames("NetVLAD").empty());

  // An empty precision selects the default (first) variant.
  EXPECT_EQ(GlobalDescriptorModel::DefaultModelUri("MixVPR"),
            GlobalDescriptorModel::DefaultModelUri("MixVPR", "fp16"));
  EXPECT_NE(GlobalDescriptorModel::DefaultModelUri("MixVPR", "fp16"),
            GlobalDescriptorModel::DefaultModelUri("MixVPR", "fp32"));
  EXPECT_EQ(GlobalDescriptorModel::DefaultModelUri("mixvpr", "FP32"),
            GlobalDescriptorModel::DefaultModelUri("MixVPR", "fp32"));
  EXPECT_TRUE(GlobalDescriptorModel::DefaultModelUri("MixVPR", "int8").empty());
  EXPECT_TRUE(GlobalDescriptorModel::DefaultModelUri("NetVLAD").empty());

  EXPECT_TRUE(GlobalDescriptorModel::SupportsBatching("MixVPR"));
  EXPECT_TRUE(GlobalDescriptorModel::SupportsBatching("MegaLoc", "fp16"));
  EXPECT_FALSE(GlobalDescriptorModel::SupportsBatching("MixVPR", "int8"));
  EXPECT_FALSE(GlobalDescriptorModel::SupportsBatching("NetVLAD"));
}

#else

TEST(GlobalDescriptorModel, EmptyWithoutONNX) {
  EXPECT_TRUE(GlobalDescriptorModel::ModelNames().empty());
  EXPECT_EQ(GlobalDescriptorModel::GetModel("MixVPR"), nullptr);
  EXPECT_TRUE(GlobalDescriptorModel::DefaultModelUri("MixVPR").empty());
  EXPECT_FALSE(GlobalDescriptorModel::SupportsBatching("MixVPR"));
}

#endif  // COLMAP_ONNX_ENABLED

}  // namespace
}  // namespace retrieval
}  // namespace colmap
