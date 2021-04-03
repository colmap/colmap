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

#include "exe/model.h"

#include "base/gps.h"
#include "base/pose.h"
#include "base/similarity_transform.h"
#include "estimators/coordinate_frame.h"
#include "util/misc.h"
#include "util/option_manager.h"

namespace colmap {
namespace {

void ReadFileCameraLocations(const std::string& ref_images_path,
                             std::vector<std::string>& ref_image_names,
                             std::vector<Eigen::Vector3d>& ref_locations) {
  const auto lines = ReadTextFileLines(ref_images_path);
  for (const auto& line : lines) {
    std::stringstream line_parser(line);
    std::string image_name;
    Eigen::Vector3d camera_position;
    line_parser >> image_name >> camera_position[0] >> camera_position[1] >>
        camera_position[2];
    ref_image_names.push_back(image_name);
    ref_locations.push_back(camera_position);
  }
}

void ReadDatabaseCameraLocations(const std::string& database_path,
                                 std::vector<std::string>& ref_image_names,
                                 std::vector<Eigen::Vector3d>& ref_locations) {
  Database database(database_path);
  auto images = database.ReadAllImages();
  std::vector<Eigen::Vector3d> gps_locations;
  GPSTransform gps_transform(GPSTransform::WGS84);
  for (const auto image : images) {
    if (image.HasTvecPrior()) {
      ref_image_names.push_back(image.Name());
      gps_locations.push_back(image.TvecPrior());
    }
  }
  ref_locations = gps_transform.EllToXYZ(gps_locations);
}

void WriteComparisonErrorsCSV(const std::string& path,
                              const std::vector<double>& rotation_errors,
                              const std::vector<double>& translation_errors,
                              const std::vector<double>& proj_center_errors) {
  CHECK_EQ(rotation_errors.size(), translation_errors.size());
  CHECK_EQ(rotation_errors.size(), proj_center_errors.size());

  std::ofstream file(path, std::ios::trunc);
  CHECK(file.is_open()) << path;

  file.precision(17);
  file << "# Model comparison pose errors: one entry per common image"
       << std::endl;
  file << "# <rotation error (deg)>, <translation error>, <proj center error>"
       << std::endl;
  for (size_t i = 0; i < rotation_errors.size(); ++i) {
    file << rotation_errors[i] << ", " << translation_errors[i] << ", "
         << proj_center_errors[i] << std::endl;
  }
}

void PrintErrorStats(std::ostream& out, std::vector<double>& vals) {
  const size_t len = vals.size();
  if (len == 0) {
    out << "Cannot extract error statistics from empty input" << std::endl;
    return;
  }
  std::sort(vals.begin(), vals.end());
  out << "Min:    " << vals.front() << std::endl;
  out << "Max:    " << vals.back() << std::endl;
  out << "Mean:   " << Mean(vals) << std::endl;
  out << "Median: " << Median(vals) << std::endl;
  out << "P90:    " << vals[size_t(0.9 * len)] << std::endl;
  out << "P99:    " << vals[size_t(0.99 * len)] << std::endl;
}

void PrintComparisonSummary(std::ostream& out,
                            std::vector<double>& rotation_errors,
                            std::vector<double>& translation_errors,
                            std::vector<double>& proj_center_errors) {
  out << "# Image pose error summary" << std::endl;
  out << std::endl << "Rotation angular errors (degrees)" << std::endl;
  PrintErrorStats(out, rotation_errors);
  out << std::endl << "Translation distance errors" << std::endl;
  PrintErrorStats(out, translation_errors);
  out << std::endl << "Projection center distance errors" << std::endl;
  PrintErrorStats(out, proj_center_errors);
}

}  // namespace

int RunModelAligner(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string database_path;
  std::string ref_images_path;
  std::string transform_path;
  std::string alignment_type = "plane";
  int min_common_images = 3;
  bool robust_alignment = true;
  bool estimate_scale = true;
  RANSACOptions ransac_options;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("database_path", &database_path);
  options.AddDefaultOption("ref_images_path", &ref_images_path);
  options.AddDefaultOption("transform_path", &transform_path);
  options.AddDefaultOption("alignment_type", &alignment_type,
                           "{plane, ecef, enu, enu-unscaled, custom}");
  options.AddDefaultOption("min_common_images", &min_common_images);
  options.AddDefaultOption("robust_alignment", &robust_alignment);
  options.AddDefaultOption("estimate_scale", &estimate_scale);
  options.AddDefaultOption("robust_alignment_max_error",
                           &ransac_options.max_error);
  options.Parse(argc, argv);

  StringToLower(&alignment_type);
  const std::unordered_set<std::string> alignment_options{
      "plane", "ecef", "enu", "enu-unscaled", "custom"};
  if (alignment_options.count(alignment_type) == 0) {
    std::cerr << "ERROR: Invalid `alignment_type` - supported values are "
                 "{'plane', 'ecef', 'enu', 'enu-unscaled', 'custom'}"
              << std::endl;
    return EXIT_FAILURE;
  }

  if (robust_alignment && ransac_options.max_error <= 0) {
    std::cout << "ERROR: You must provide a maximum alignment error > 0"
              << std::endl;
    return EXIT_FAILURE;
  }

  if (alignment_type != "plane" && database_path.empty() &&
      ref_images_path.empty()) {
    std::cerr << "ERROR: Location alignment requires either database or "
                 "location file path."
              << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<std::string> ref_image_names;
  std::vector<Eigen::Vector3d> ref_locations;
  if (!ref_images_path.empty() && database_path.empty()) {
    ReadFileCameraLocations(ref_images_path, ref_image_names, ref_locations);
  } else if (!database_path.empty() && ref_images_path.empty()) {
    ReadDatabaseCameraLocations(database_path, ref_image_names, ref_locations);
  } else {
    std::cerr << "ERROR: Use location file or database, not both" << std::endl;
    return EXIT_FAILURE;
  }

  if (alignment_type != "plane" && ref_locations.size() < min_common_images) {
    std::cout << "ERROR: Cannot align with insufficient reference locations."
              << std::endl;
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);
  SimilarityTransform3 tform;
  bool alignment_success = true;

  if (alignment_type == "plane") {
    PrintHeading2("Aligning reconstruction to principal plane");
    AlignToPrincipalPlane(&reconstruction, &tform);
  } else {
    PrintHeading2("Aligning reconstruction to ECEF");
    std::cout << StringPrintf(" => Using %d reference images",
                              ref_image_names.size())
              << std::endl;

    if (estimate_scale) {
      if (robust_alignment) {
        alignment_success = reconstruction.AlignRobust(
            ref_image_names, ref_locations, min_common_images, ransac_options,
            &tform);
      } else {
        alignment_success = reconstruction.Align(ref_image_names, ref_locations,
                                                 min_common_images, &tform);
      }
    } else {
      if (robust_alignment) {
        alignment_success = reconstruction.AlignRobust<false>(
            ref_image_names, ref_locations, min_common_images, ransac_options,
            &tform);
      } else {
        alignment_success = reconstruction.Align<false>(
            ref_image_names, ref_locations, min_common_images, &tform);
      }
    }

    std::vector<double> errors;
    errors.reserve(ref_image_names.size());

    for (size_t i = 0; i < ref_image_names.size(); ++i) {
      const Image* image = reconstruction.FindImageWithName(ref_image_names[i]);
      if (image != nullptr) {
        errors.push_back((image->ProjectionCenter() - ref_locations[i]).norm());
      }
    }
    std::cout << StringPrintf(" => Alignment error: %f (mean), %f (median)",
                              Mean(errors), Median(errors))
              << std::endl;

    if (alignment_success && StringStartsWith(alignment_type, "enu")) {
      PrintHeading2("Aligning reconstruction to ENU");
      AlignToENUPlane(&reconstruction, &tform, alignment_type == "enu-unscaled");
    }
  }

  if (alignment_success) {
    std::cout << " => Alignment succeeded" << std::endl;
    reconstruction.Write(output_path);
    if (!transform_path.empty()) {
      tform.Write(transform_path);
    }
    return EXIT_SUCCESS;
  } else {
    std::cout << " => Alignment failed" << std::endl;
    return EXIT_FAILURE;
  }
}

int RunModelAnalyzer(int argc, char** argv) {
  std::string path;

  OptionManager options;
  options.AddRequiredOption("path", &path);
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(path);

  std::cout << StringPrintf("Cameras: %d", reconstruction.NumCameras())
            << std::endl;
  std::cout << StringPrintf("Images: %d", reconstruction.NumImages())
            << std::endl;
  std::cout << StringPrintf("Registered images: %d",
                            reconstruction.NumRegImages())
            << std::endl;
  std::cout << StringPrintf("Points: %d", reconstruction.NumPoints3D())
            << std::endl;
  std::cout << StringPrintf("Observations: %d",
                            reconstruction.ComputeNumObservations())
            << std::endl;
  std::cout << StringPrintf("Mean track length: %f",
                            reconstruction.ComputeMeanTrackLength())
            << std::endl;
  std::cout << StringPrintf("Mean observations per image: %f",
                            reconstruction.ComputeMeanObservationsPerRegImage())
            << std::endl;
  std::cout << StringPrintf("Mean reprojection error: %fpx",
                            reconstruction.ComputeMeanReprojectionError())
            << std::endl;

  return EXIT_SUCCESS;
}

int RunModelComparer(int argc, char** argv) {
  std::string input_path1;
  std::string input_path2;
  std::string output_path;
  double min_inlier_observations = 0.3;
  double max_reproj_error = 8.0;

  OptionManager options;
  options.AddRequiredOption("input_path1", &input_path1);
  options.AddRequiredOption("input_path2", &input_path2);
  options.AddDefaultOption("output_path", &output_path);
  options.AddDefaultOption("min_inlier_observations", &min_inlier_observations);
  options.AddDefaultOption("max_reproj_error", &max_reproj_error);
  options.Parse(argc, argv);

  if (!output_path.empty() && !ExistsDir(output_path)) {
    std::cerr << "ERROR: Provided output path is not a valid directory"
              << std::endl;
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction1;
  reconstruction1.Read(input_path1);
  PrintHeading1("Reconstruction 1");
  std::cout << StringPrintf("Images: %d", reconstruction1.NumRegImages())
            << std::endl;
  std::cout << StringPrintf("Points: %d", reconstruction1.NumPoints3D())
            << std::endl;

  Reconstruction reconstruction2;
  reconstruction2.Read(input_path2);
  PrintHeading1("Reconstruction 2");
  std::cout << StringPrintf("Images: %d", reconstruction2.NumRegImages())
            << std::endl;
  std::cout << StringPrintf("Points: %d", reconstruction2.NumPoints3D())
            << std::endl;

  PrintHeading1("Comparing reconstructed image poses");
  const auto common_image_ids =
      reconstruction1.FindCommonRegImageIds(reconstruction2);
  std::cout << StringPrintf("Common images: %d", common_image_ids.size())
            << std::endl;

  Eigen::Matrix3x4d alignment;
  if (!ComputeAlignmentBetweenReconstructions(reconstruction2, reconstruction1,
                                              min_inlier_observations,
                                              max_reproj_error, &alignment)) {
    std::cout << "=> Reconstruction alignment failed" << std::endl;
    return EXIT_FAILURE;
  }

  const SimilarityTransform3 tform(alignment);
  std::cout << "Computed alignment transform:" << std::endl
            << tform.Matrix() << std::endl;

  const size_t num_images = common_image_ids.size();
  std::vector<double> rotation_errors(num_images, 0.0);
  std::vector<double> translation_errors(num_images, 0.0);
  std::vector<double> proj_center_errors(num_images, 0.0);
  for (size_t i = 0; i < num_images; ++i) {
    const image_t image_id = common_image_ids[i];
    const Image& image1 = reconstruction1.Image(image_id);
    Image& image2 = reconstruction2.Image(image_id);
    tform.TransformPose(&image2.Qvec(), &image2.Tvec());

    const Eigen::Vector4d normalized_qvec1 = NormalizeQuaternion(image1.Qvec());
    const Eigen::Quaterniond quat1(normalized_qvec1(0), normalized_qvec1(1),
                                   normalized_qvec1(2), normalized_qvec1(3));
    const Eigen::Vector4d normalized_qvec2 = NormalizeQuaternion(image2.Qvec());
    const Eigen::Quaterniond quat2(normalized_qvec2(0), normalized_qvec2(1),
                                   normalized_qvec2(2), normalized_qvec2(3));

    rotation_errors[i] = RadToDeg(quat1.angularDistance(quat2));
    translation_errors[i] = (image1.Tvec() - image2.Tvec()).norm();
    proj_center_errors[i] =
        (image1.ProjectionCenter() - image2.ProjectionCenter()).norm();
  }

  if (output_path.empty()) {
    PrintComparisonSummary(std::cout, rotation_errors, translation_errors,
                           proj_center_errors);
  } else {
    const std::string errors_path = JoinPaths(output_path, "errors.csv");
    WriteComparisonErrorsCSV(errors_path, rotation_errors, translation_errors,
                             proj_center_errors);
    const std::string summary_path =
        JoinPaths(output_path, "errors_summary.txt");
    std::ofstream file(summary_path, std::ios::trunc);
    CHECK(file.is_open()) << summary_path;
    PrintComparisonSummary(file, rotation_errors, translation_errors,
                           proj_center_errors);
  }

  return EXIT_SUCCESS;
}

int RunModelConverter(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string output_type;
  bool skip_distortion = false;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("output_type", &output_type,
                            "{BIN, TXT, NVM, Bundler, VRML, PLY, R3D, CAM}");
  options.AddDefaultOption("skip_distortion", &skip_distortion);
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  StringToLower(&output_type);
  if (output_type == "bin") {
    reconstruction.WriteBinary(output_path);
  } else if (output_type == "txt") {
    reconstruction.WriteText(output_path);
  } else if (output_type == "nvm") {
    reconstruction.ExportNVM(output_path, skip_distortion);
  } else if (output_type == "bundler") {
    reconstruction.ExportBundler(output_path + ".bundle.out",
                                 output_path + ".list.txt", skip_distortion);
  } else if (output_type == "r3d") {
    reconstruction.ExportRecon3D(output_path, skip_distortion);
  } else if (output_type == "cam") {
    reconstruction.ExportCam(output_path, skip_distortion);
  } else if (output_type == "ply") {
    reconstruction.ExportPLY(output_path);
  } else if (output_type == "vrml") {
    const auto base_path = output_path.substr(0, output_path.find_last_of("."));
    reconstruction.ExportVRML(base_path + ".images.wrl",
                              base_path + ".points3D.wrl", 1,
                              Eigen::Vector3d(1, 0, 0));
  } else {
    std::cerr << "ERROR: Invalid `output_type`" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int RunModelMerger(int argc, char** argv) {
  std::string input_path1;
  std::string input_path2;
  std::string output_path;
  double max_reproj_error = 64.0;

  OptionManager options;
  options.AddRequiredOption("input_path1", &input_path1);
  options.AddRequiredOption("input_path2", &input_path2);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("max_reproj_error", &max_reproj_error);
  options.Parse(argc, argv);

  Reconstruction reconstruction1;
  reconstruction1.Read(input_path1);
  PrintHeading2("Reconstruction 1");
  std::cout << StringPrintf("Images: %d", reconstruction1.NumRegImages())
            << std::endl;
  std::cout << StringPrintf("Points: %d", reconstruction1.NumPoints3D())
            << std::endl;

  Reconstruction reconstruction2;
  reconstruction2.Read(input_path2);
  PrintHeading2("Reconstruction 2");
  std::cout << StringPrintf("Images: %d", reconstruction2.NumRegImages())
            << std::endl;
  std::cout << StringPrintf("Points: %d", reconstruction2.NumPoints3D())
            << std::endl;

  PrintHeading2("Merging reconstructions");
  if (reconstruction1.Merge(reconstruction2, max_reproj_error)) {
    std::cout << "=> Merge succeeded" << std::endl;
    PrintHeading2("Merged reconstruction");
    std::cout << StringPrintf("Images: %d", reconstruction1.NumRegImages())
              << std::endl;
    std::cout << StringPrintf("Points: %d", reconstruction1.NumPoints3D())
              << std::endl;
  } else {
    std::cout << "=> Merge failed" << std::endl;
  }

  reconstruction1.Write(output_path);

  return EXIT_SUCCESS;
}

int RunModelOrientationAligner(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string method = "MANHATTAN-WORLD";

  ManhattanWorldFrameEstimationOptions frame_estimation_options;

  OptionManager options;
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("method", &method,
                           "{MANHATTAN-WORLD, IMAGE-ORIENTATION}");
  options.AddDefaultOption("max_image_size",
                           &frame_estimation_options.max_image_size);
  options.Parse(argc, argv);

  StringToLower(&method);
  if (method != "manhattan-world" && method != "image-orientation") {
    std::cout << "ERROR: Invalid `method` - supported values are "
                 "'MANHATTAN-WORLD' or 'IMAGE-ORIENTATION'."
              << std::endl;
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  PrintHeading1("Aligning Reconstruction");

  Eigen::Matrix3d tform;

  if (method == "manhattan-world") {
    const Eigen::Matrix3d frame = EstimateManhattanWorldFrame(
        frame_estimation_options, reconstruction, *options.image_path);

    if (frame.col(0).nonZeros() == 0) {
      std::cout << "Only aligning vertical axis" << std::endl;
      tform = RotationFromUnitVectors(frame.col(1), Eigen::Vector3d(0, 1, 0));
    } else if (frame.col(1).nonZeros() == 0) {
      tform = RotationFromUnitVectors(frame.col(0), Eigen::Vector3d(1, 0, 0));
      std::cout << "Only aligning horizontal axis" << std::endl;
    } else {
      tform = frame.transpose();
      std::cout << "Aligning horizontal and vertical axes" << std::endl;
    }
  } else if (method == "image-orientation") {
    const Eigen::Vector3d gravity_axis =
        EstimateGravityVectorFromImageOrientation(reconstruction);
    tform = RotationFromUnitVectors(gravity_axis, Eigen::Vector3d(0, 1, 0));
  } else {
    LOG(FATAL) << "Alignment method not supported";
  }

  std::cout << "Using the rotation matrix:" << std::endl;
  std::cout << tform << std::endl;

  reconstruction.Transform(SimilarityTransform3(
      1, RotationMatrixToQuaternion(tform), Eigen::Vector3d(0, 0, 0)));

  std::cout << "Writing aligned reconstruction..." << std::endl;
  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunModelTransformer(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string transform_path;
  bool is_inverse = false;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("transform_path", &transform_path);
  options.AddDefaultOption("is_inverse", &is_inverse);
  options.Parse(argc, argv);

  std::cout << "Reading points input: " << input_path << std::endl;
  Reconstruction recon;
  bool is_dense = false;
  if (HasFileExtension(input_path, ".ply")) {
    is_dense = true;
    recon.ImportPLY(input_path);
  } else if (ExistsDir(input_path)) {
    recon.Read(input_path);
  } else {
    std::cerr << "Invalid model input; not a PLY file or sparse reconstruction "
                 "directory."
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Reading transform input: " << transform_path << std::endl;
  SimilarityTransform3 tform(transform_path);
  if (is_inverse) {
    tform = tform.Inverse();
  }

  std::cout << "Applying transform to recon with " << recon.NumPoints3D()
            << " points" << std::endl;
  recon.Transform(tform);

  std::cout << "Writing output: " << output_path << std::endl;
  if (is_dense) {
    recon.ExportPLY(output_path);
  } else {
    recon.Write(output_path);
  }

  return EXIT_SUCCESS;
}

}  // namespace colmap
