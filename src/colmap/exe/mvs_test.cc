// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/mvs.h"

#include "colmap/util/testing.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

void ExpectStereoCommandFailure(const std::filesystem::path& workspace,
                                const std::string& format,
                                const std::string& expected_error) {
  std::vector<std::string> args = {"patch_match_stereo",
                                   "--workspace_path",
                                   workspace.string(),
                                   "--workspace_format",
                                   format,
                                   "--log_target",
                                   "stderr"};
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) argv.push_back(arg.data());
  testing::internal::CaptureStderr();
  try {
    EXPECT_EQ(RunPatchMatchStereo(static_cast<int>(argv.size()), argv.data()),
              EXIT_FAILURE);
  } catch (...) {
    testing::internal::GetCapturedStderr();
    throw;
  }
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_NE(diagnostic.find("PatchMatch stereo failed:"), std::string::npos);
  EXPECT_NE(diagnostic.find(expected_error), std::string::npos);
}

#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
TEST(PatchMatchStereoCommand, InvalidFormatReturnsFailure) {
  ExpectStereoCommandFailure(
      CreateTestDir(), "invalid", "Invalid `workspace_format`");
}

TEST(PatchMatchStereoCommand, MissingModelReturnsFailure) {
  ExpectStereoCommandFailure(
      CreateTestDir(), "COLMAP", "files do not exist at");
}
#else
TEST(PatchMatchStereoCommand, UnsupportedBuildReturnsFailure) {
  ExpectStereoCommandFailure(
      CreateTestDir(),
      "COLMAP",
      "Dense stereo reconstruction requires CUDA or HIP");
}
#endif

TEST(PatchMatchStereoCommand, LibraryEntryStillThrows) {
  try {
    RunPatchMatchStereoImpl(
        CreateTestDir(), "invalid", "option-all", mvs::PatchMatchOptions(), {});
    FAIL() << "Library entry must propagate the reconstruction error";
  } catch (const std::invalid_argument& error) {
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
    EXPECT_NE(std::string(error.what()).find("Invalid `workspace_format`"),
              std::string::npos);
#else
    EXPECT_NE(std::string(error.what())
                  .find("Dense stereo reconstruction requires CUDA or HIP"),
              std::string::npos);
#endif
  }
}

}  // namespace
}  // namespace colmap
