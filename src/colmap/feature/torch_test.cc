#include "colmap/feature/torch.h"

#include "colmap/sensor/bitmap.h"

#include <iostream>

#include <glog/logging.h>

int main(int argc, const char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: ./exe <torch-model-pth> <image1> <image2>\n";
    return -1;
  }

  colmap::Bitmap image1;
  CHECK(image1.Read(argv[2], true));
  // colmap::Bitmap image2;
  // CHECK(image2.Read(argv[3], true));

  colmap::TorchFeatureOptions extraction_options;
  extraction_options.model_script_path = argv[1];
  extraction_options.max_image_size = 500;

  auto extractor = CreateTorchFeatureExtractor(extraction_options);

  colmap::FeatureKeypoints keypoints1;
  colmap::FeatureDescriptors descriptors1;
  if (!extractor->Extract(image1, &keypoints1, &descriptors1)) {
    return -1;
  }

  // colmap::FeatureKeypoints keypoints2;
  // colmap::FeatureDescriptors descriptors2;
  // if (!colmap::ExtractCovariantFeatures(
  //         extraction_options, image2, &keypoints2, &descriptors2)) {
  //   return -1;
  // }

  // colmap::SiftMatchingOptions match_options;
  // match_options.max_distance = 100000000;
  // match_options.max_ratio = 1.;
  // colmap::FeatureMatches matches;
  // colmap::MatchSiftFeaturesCPUFLANN(
  //     match_options, descriptors1, descriptors2, &matches);

  // std::cout << keypoints1.size() << " " << keypoints2.size() << " "
  //           << matches.size() << std::endl;

  return 0;
}
