// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/exe/model.h"

#include "colmap/estimators/coordinate_frame.h"
#include "colmap/geometry/gps.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/similarity_transform.h"
#include "colmap/util/misc.h"
#include "colmap/util/option_manager.h"
#include "colmap/util/threading.h"

namespace colmap {
namespace {

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
ComputeEqualPartsBounds(const Reconstruction& reconstruction,
                        const Eigen::Vector3i& split) {
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> bounds;
  const auto bbox = reconstruction.ComputeBoundingBox();
  const Eigen::Vector3d extent = bbox.second - bbox.first;
  const Eigen::Vector3d offset(
      extent(0) / split(0), extent(1) / split(1), extent(2) / split(2));

  for (int k = 0; k < split(2); ++k) {
    for (int j = 0; j < split(1); ++j) {
      for (int i = 0; i < split(0); ++i) {
        Eigen::Vector3d min_bound(bbox.first(0) + i * offset(0),
                                  bbox.first(1) + j * offset(1),
                                  bbox.first(2) + k * offset(2));
        bounds.emplace_back(min_bound, min_bound + offset);
      }
    }
  }

  return bounds;
}

Eigen::Vector3d TransformLatLonAltToModelCoords(
    const SimilarityTransform3& tform, double lat, double lon, double alt) {
  // Since this is intended for use in ENU aligned models we want to define the
  // altitude along the ENU frame z axis and not the Earth's radius. Thus, we
  // set the altitude to 0 when converting from LLA to ECEF and then we use the
  // altitude at the end, after scaling, to set it as the z coordinate in the
  // ENU frame.
  Eigen::Vector3d xyz = GPSTransform(GPSTransform::WGS84)
                            .EllToXYZ({Eigen::Vector3d(lat, lon, 0.0)})[0];
  tform.TransformPoint(&xyz);
  xyz(2) = tform.Scale() * alt;
  return xyz;
}

void WriteBoundingBox(const std::string& reconstruction_path,
                      const std::pair<Eigen::Vector3d, Eigen::Vector3d>& bounds,
                      const std::string& suffix = "") {
  const Eigen::Vector3d extent = bounds.second - bounds.first;
  // write axis-aligned bounding box
  {
    const std::string path =
        JoinPaths(reconstruction_path, "bbox_aligned" + suffix + ".txt");
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    // Ensure that we don't loose any precision by storing in text.
    file.precision(17);
    file << bounds.first.transpose() << std::endl;
    file << bounds.second.transpose() << std::endl;
  }
  // write oriented bounding box
  {
    const std::string path =
        JoinPaths(reconstruction_path, "bbox_oriented" + suffix + ".txt");
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    // Ensure that we don't loose any precision by storing in text.
    file.precision(17);
    const Eigen::Vector3d center = (bounds.first + bounds.second) * 0.5;
    file << center.transpose() << std::endl << std::endl;
    file << "1 0 0\n0 1 0\n0 0 1" << std::endl << std::endl;
    file << extent.transpose() << std::endl;
  }
}

std::vector<Eigen::Vector3d> ConvertCameraLocations(
    const bool ref_is_gps,
    const std::string& alignment_type,
    const std::vector<Eigen::Vector3d>& ref_locations) {
  if (ref_is_gps) {
    const GPSTransform gps_transform(GPSTransform::WGS84);
    if (alignment_type != "enu") {
      std::cout << "\nConverting Alignment Coordinates from GPS (lat/lon/alt) "
                   "to ECEF.\n";
      return gps_transform.EllToXYZ(ref_locations);
    } else {
      std::cout << "\nConverting Alignment Coordinates from GPS (lat/lon/alt) "
                   "to ENU.\n";
      return gps_transform.EllToENU(
          ref_locations, ref_locations[0](0), ref_locations[0](1));
    }
  } else {
    std::cout << "\nCartesian Alignment Coordinates extracted (MUST NOT BE "
                 "GPS coords!).\n";
    return ref_locations;
  }
}

void ReadFileCameraLocations(const std::string& ref_images_path,
                             const bool ref_is_gps,
                             const std::string& alignment_type,
                             std::vector<std::string>* ref_image_names,
                             std::vector<Eigen::Vector3d>* ref_locations) {
  for (const auto& line : ReadTextFileLines(ref_images_path)) {
    std::stringstream line_parser(line);
    std::string image_name;
    Eigen::Vector3d camera_position;
    line_parser >> image_name >> camera_position[0] >> camera_position[1] >>
        camera_position[2];
    ref_image_names->push_back(image_name);
    ref_locations->push_back(camera_position);
  }

  *ref_locations =
      ConvertCameraLocations(ref_is_gps, alignment_type, *ref_locations);
}

void ReadDatabaseCameraLocations(const std::string& database_path,
                                 const bool ref_is_gps,
                                 const std::string& alignment_type,
                                 std::vector<std::string>* ref_image_names,
                                 std::vector<Eigen::Vector3d>* ref_locations) {
  Database database(database_path);
  for (const auto& image : database.ReadAllImages()) {
    if (image.HasTvecPrior()) {
      ref_image_names->push_back(image.Name());
      ref_locations->push_back(image.TvecPrior());
    }
  }

  *ref_locations =
      ConvertCameraLocations(ref_is_gps, alignment_type, *ref_locations);
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
  PrintHeading2("Image pose error summary");
  out << std::endl << "Rotation angular errors (degrees)" << std::endl;
  PrintErrorStats(out, rotation_errors);
  out << std::endl << "Translation distance errors" << std::endl;
  PrintErrorStats(out, translation_errors);
  out << std::endl << "Projection center distance errors" << std::endl;
  PrintErrorStats(out, proj_center_errors);
}

}  // namespace

// Align given reconstruction with user provided cameras positions
// (can be used for geo-registration for instance).
// The cameras positions to be used for aligning the reconstruction
// model must be provided either by a txt file (with each line being: img_name x
// y z) or through a colmap database file containing a prior position for the
// registered images.
//
// Required Options:
// - input_path: path to initial reconstruction model
// - output_path: path to store the aligned reconstruction model
//
// Additional Options:
// - database_path: path to database file with prior positions for
// reconstruction images
// - ref_images_path: path to txt file with prior positions for reconstruction
// images (WARNING: provide only one of the above)
// - ref_is_gps: if true the prior positions are converted from GPS
// (lat/lon/alt) to ECEF or ENU
// - merge_image_and_ref_origins: if true the reconstuction will be shifted so
// that the first prior position is used for its camera position
// - transform_path: path to store the Sim3 transformation used for the
// alignment
// - alignment_type:
//    > plane: align with reconstruction principal plane
//    > ecef: align with ecef coords. (requires gps coords. or user provided
//    ecef coords.)
//    > enu: align with enu coords. (requires gps coords. or user provided enu
//    coords.)
//    > enu-plane: align to ecef and then to enu plane (requires gps
//    coords. or user provided ecef coords.)
//    > enu-plane-unscaled: same as above but do not apply the computed
//    scale when aligning the reconstruction
//    > custom: align to provided coords.
// - min_common_images: minimum number of images with prior positions to perform
// the estimate an alignment
// - estimate_scale: if true apply the computed scale when aligning the
// reconstruction
// - robust_alignment: if true use a ransac-based estimation for robust
// alignment
// - robust_alignment_max_error: ransac error to use if robust alignment is
// enabled
int RunModelAligner(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string database_path;
  std::string ref_images_path;
  bool ref_is_gps = true;
  bool merge_origins = false;
  std::string transform_path;
  std::string alignment_type = "custom";
  int min_common_images = 3;
  bool robust_alignment = true;
  bool estimate_scale = true;
  RANSACOptions ransac_options;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("database_path", &database_path);
  options.AddDefaultOption("ref_images_path", &ref_images_path);
  options.AddDefaultOption("ref_is_gps", &ref_is_gps);
  options.AddDefaultOption("merge_image_and_ref_origins", &merge_origins);
  options.AddDefaultOption("transform_path", &transform_path);
  options.AddDefaultOption(
      "alignment_type",
      &alignment_type,
      "{plane, ecef, enu, enu-plane, enu-plane-unscaled, custom}");
  options.AddDefaultOption("min_common_images", &min_common_images);
  options.AddDefaultOption("estimate_scale", &estimate_scale);
  options.AddDefaultOption("robust_alignment", &robust_alignment);
  options.AddDefaultOption("robust_alignment_max_error",
                           &ransac_options.max_error);
  options.Parse(argc, argv);

  StringToLower(&alignment_type);
  const std::unordered_set<std::string> alignment_options{
      "plane", "ecef", "enu", "enu-plane", "enu-plane-unscaled", "custom"};
  if (alignment_options.count(alignment_type) == 0) {
    std::cerr << "ERROR: Invalid `alignment_type` - supported values are "
                 "{'plane', 'ecef', 'enu', 'enu-plane', 'enu-plane-unscaled', "
                 "'custom'}"
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
    ReadFileCameraLocations(ref_images_path,
                            ref_is_gps,
                            alignment_type,
                            &ref_image_names,
                            &ref_locations);
  } else if (!database_path.empty() && ref_images_path.empty()) {
    ReadDatabaseCameraLocations(database_path,
                                ref_is_gps,
                                alignment_type,
                                &ref_image_names,
                                &ref_locations);
  } else if (alignment_type != "plane") {
    std::cerr << "ERROR: Use location file or database, not both" << std::endl;
    return EXIT_FAILURE;
  }

  if (alignment_type != "plane" &&
      static_cast<int>(ref_locations.size()) < min_common_images) {
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
    PrintHeading2("Aligning reconstruction to " + alignment_type);
    std::cout << StringPrintf(" => Using %d reference images",
                              ref_image_names.size())
              << std::endl;

    if (estimate_scale) {
      if (robust_alignment) {
        alignment_success = reconstruction.AlignRobust(ref_image_names,
                                                       ref_locations,
                                                       min_common_images,
                                                       ransac_options,
                                                       &tform);
      } else {
        alignment_success = reconstruction.Align(
            ref_image_names, ref_locations, min_common_images, &tform);
      }
    } else {
      if (robust_alignment) {
        alignment_success = reconstruction.AlignRobust<false>(ref_image_names,
                                                              ref_locations,
                                                              min_common_images,
                                                              ransac_options,
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
    std::cout << StringPrintf("=> Alignment error: %f (mean), %f (median)",
                              Mean(errors),
                              Median(errors))
              << std::endl;

    if (alignment_success && StringStartsWith(alignment_type, "enu-plane")) {
      PrintHeading2("Aligning ECEF aligned reconstruction to ENU plane");
      AlignToENUPlane(
          &reconstruction, &tform, alignment_type == "enu-plane-unscaled");
    }
  }

  if (merge_origins) {
    for (size_t i = 0; i < ref_image_names.size(); i++) {
      const Image* first_image =
          reconstruction.FindImageWithName(ref_image_names[i]);

      if (first_image != nullptr) {
        const Eigen::Vector3d& first_img_position = ref_locations[i];

        const Eigen::Vector3d trans_align =
            first_img_position - first_image->ProjectionCenter();

        const SimilarityTransform3 origin_align(
            1.0, ComposeIdentityQuaternion(), trans_align);

        std::cout << "\n Aligning Reconstruction's origin with Ref origin : "
                  << first_img_position.transpose() << "\n";

        reconstruction.Transform(origin_align);

        // Update the Sim3 transformation in case it is stored next
        tform = SimilarityTransform3(
            tform.Scale(), tform.Rotation(), tform.Translation() + trans_align);

        break;
      }
    }
  }

  if (alignment_success) {
    std::cout << "=> Alignment succeeded" << std::endl;
    reconstruction.Write(output_path);
    if (!transform_path.empty()) {
      tform.Write(transform_path);
    }
    return EXIT_SUCCESS;
  } else {
    std::cout << "=> Alignment failed" << std::endl;
    return EXIT_FAILURE;
  }
}

int RunModelAnalyzer(int argc, char** argv) {
  std::string path;
  bool verbose = false;

  OptionManager options;
  options.AddRequiredOption("path", &path);
  options.AddDefaultOption("verbose", &verbose);
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

  // verbose information
  if (verbose) {
    PrintHeading2("Cameras");
    for (const auto& camera : reconstruction.Cameras()) {
      std::cout << StringPrintf(" - Camera Id: %d, Model Name: %s, Params: %s",
                                camera.first,
                                camera.second.ModelName().c_str(),
                                camera.second.ParamsToString().c_str())
                << std::endl;
    }

    PrintHeading2("Images");
    for (const auto& image_id : reconstruction.RegImageIds()) {
      std::cout << StringPrintf(" - Registered Image Id: %d, Name: %s",
                                image_id,
                                reconstruction.Image(image_id).Name().c_str())
                << std::endl;
    }
  }

  return EXIT_SUCCESS;
}

int RunModelComparer(int argc, char** argv) {
  std::string input_path1;
  std::string input_path2;
  std::string output_path;
  std::string alignment_error = "reprojection";
  double min_inlier_observations = 0.3;
  double max_reproj_error = 8.0;
  double max_proj_center_error = 0.1;

  OptionManager options;
  options.AddRequiredOption("input_path1", &input_path1);
  options.AddRequiredOption("input_path2", &input_path2);
  options.AddDefaultOption("output_path", &output_path);
  options.AddDefaultOption(
      "alignment_error", &alignment_error, "{reprojection, proj_center}");
  options.AddDefaultOption("min_inlier_observations", &min_inlier_observations);
  options.AddDefaultOption("max_reproj_error", &max_reproj_error);
  options.AddDefaultOption("max_proj_center_error", &max_proj_center_error);
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
  bool success = false;
  if (alignment_error == "reprojection") {
    success = ComputeAlignmentBetweenReconstructions(
        reconstruction1,
        reconstruction2,
        /*min_inlier_observations=*/min_inlier_observations,
        /*max_reproj_error=*/max_reproj_error,
        &alignment);
  } else if (alignment_error == "proj_center") {
    success = ComputeAlignmentBetweenReconstructions(
        reconstruction1,
        reconstruction2,
        /*max_proj_center_error=*/max_proj_center_error,
        &alignment);
  } else {
    std::cout << "ERROR: Invalid alignment_error specified." << std::endl;
    return EXIT_FAILURE;
  }

  if (!success) {
    std::cout << "=> Reconstruction alignment failed" << std::endl;
    return EXIT_FAILURE;
  }

  const SimilarityTransform3 tform(alignment);
  std::cout << "Computed alignment transform:" << std::endl
            << tform.Matrix() << std::endl;

  const size_t num_images = common_image_ids.size();
  std::vector<double> rotation_errors(num_images,
                                      std::numeric_limits<double>::infinity());
  std::vector<double> translation_errors(
      num_images, std::numeric_limits<double>::infinity());
  std::vector<double> proj_center_errors(
      num_images, std::numeric_limits<double>::infinity());
  for (size_t i = 0; i < num_images; ++i) {
    const image_t image_id = common_image_ids[i];
    Image& image1 = reconstruction1.Image(image_id);
    tform.TransformPose(&image1.Qvec(), &image1.Tvec());
    const Image& image2 = reconstruction2.Image(image_id);

    const Eigen::Vector4d normalized_qvec1 = NormalizeQuaternion(image1.Qvec());
    const Eigen::Quaterniond quat1(normalized_qvec1(0),
                                   normalized_qvec1(1),
                                   normalized_qvec1(2),
                                   normalized_qvec1(3));
    const Eigen::Vector4d normalized_qvec2 = NormalizeQuaternion(image2.Qvec());
    const Eigen::Quaterniond quat2(normalized_qvec2(0),
                                   normalized_qvec2(1),
                                   normalized_qvec2(2),
                                   normalized_qvec2(3));

    rotation_errors[i] = RadToDeg(quat1.angularDistance(quat2));
    translation_errors[i] = (image1.Tvec() - image2.Tvec()).norm();
    proj_center_errors[i] =
        (image1.ProjectionCenter() - image2.ProjectionCenter()).norm();
  }

  PrintComparisonSummary(
      std::cout, rotation_errors, translation_errors, proj_center_errors);

  if (!output_path.empty()) {
    const std::string errors_path = JoinPaths(output_path, "errors.csv");
    WriteComparisonErrorsCSV(
        errors_path, rotation_errors, translation_errors, proj_center_errors);
    const std::string summary_path =
        JoinPaths(output_path, "errors_summary.txt");
    std::ofstream file(summary_path, std::ios::trunc);
    CHECK(file.is_open()) << summary_path;
    PrintComparisonSummary(
        file, rotation_errors, translation_errors, proj_center_errors);
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
  options.AddRequiredOption("output_type",
                            &output_type,
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
                                 output_path + ".list.txt",
                                 skip_distortion);
  } else if (output_type == "r3d") {
    reconstruction.ExportRecon3D(output_path, skip_distortion);
  } else if (output_type == "cam") {
    reconstruction.ExportCam(output_path, skip_distortion);
  } else if (output_type == "ply") {
    reconstruction.ExportPLY(output_path);
  } else if (output_type == "vrml") {
    const auto base_path = output_path.substr(0, output_path.find_last_of('.'));
    reconstruction.ExportVRML(base_path + ".images.wrl",
                              base_path + ".points3D.wrl",
                              1,
                              Eigen::Vector3d(1, 0, 0));
  } else {
    std::cerr << "ERROR: Invalid `output_type`" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int RunModelCropper(int argc, char** argv) {
  Timer timer;
  timer.Start();

  std::string input_path;
  std::string output_path;
  std::string boundary;
  std::string gps_transform_path;
  bool is_gps = false;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("boundary", &boundary);
  options.AddDefaultOption("gps_transform_path", &gps_transform_path);
  options.Parse(argc, argv);

  if (!ExistsDir(input_path)) {
    std::cerr << "ERROR: `input_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<double> boundary_elements = CSVToVector<double>(boundary);
  if (boundary_elements.size() != 2 && boundary_elements.size() != 6) {
    std::cerr << "ERROR: Invalid `boundary` - supported values are "
                 "'x1,y1,z1,x2,y2,z2' or 'p1,p2'."
              << std::endl;
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  PrintHeading2("Calculating boundary coordinates");
  std::pair<Eigen::Vector3d, Eigen::Vector3d> bounding_box;
  if (boundary_elements.size() == 6) {
    SimilarityTransform3 tform;
    if (!gps_transform_path.empty()) {
      PrintHeading2("Reading model to ECEF transform");
      is_gps = true;
      tform = SimilarityTransform3::FromFile(gps_transform_path).Inverse();
    }
    bounding_box.first =
        is_gps ? TransformLatLonAltToModelCoords(tform,
                                                 boundary_elements[0],
                                                 boundary_elements[1],
                                                 boundary_elements[2])
               : Eigen::Vector3d(boundary_elements[0],
                                 boundary_elements[1],
                                 boundary_elements[2]);
    bounding_box.second =
        is_gps ? TransformLatLonAltToModelCoords(tform,
                                                 boundary_elements[3],
                                                 boundary_elements[4],
                                                 boundary_elements[5])
               : Eigen::Vector3d(boundary_elements[3],
                                 boundary_elements[4],
                                 boundary_elements[5]);
  } else {
    bounding_box = reconstruction.ComputeBoundingBox(boundary_elements[0],
                                                     boundary_elements[1]);
  }

  PrintHeading2("Cropping reconstruction");
  reconstruction.Crop(bounding_box).Write(output_path);
  WriteBoundingBox(output_path, bounding_box);

  std::cout << "=> Cropping succeeded" << std::endl;
  timer.PrintMinutes();
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
  options.AddDefaultOption(
      "method", &method, "{MANHATTAN-WORLD, IMAGE-ORIENTATION}");
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

    if (frame.col(0).lpNorm<1>() == 0) {
      std::cout << "Only aligning vertical axis" << std::endl;
      tform = RotationFromUnitVectors(frame.col(1), Eigen::Vector3d(0, 1, 0));
    } else if (frame.col(1).lpNorm<1>() == 0) {
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

int RunModelSplitter(int argc, char** argv) {
  Timer timer;
  timer.Start();

  std::string input_path;
  std::string output_path;
  std::string split_type;
  std::string split_params;
  std::string gps_transform_path;
  int num_threads = -1;
  int min_reg_images = 10;
  int min_num_points = 100;
  double overlap_ratio = 0.0;
  double min_area_ratio = 0.0;
  bool is_gps = false;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption(
      "split_type", &split_type, "{tiles, extent, parts}");
  options.AddRequiredOption("split_params", &split_params);
  options.AddDefaultOption("gps_transform_path", &gps_transform_path);
  options.AddDefaultOption("num_threads", &num_threads);
  options.AddDefaultOption("min_reg_images", &min_reg_images);
  options.AddDefaultOption("min_num_points", &min_num_points);
  options.AddDefaultOption("overlap_ratio", &overlap_ratio);
  options.AddDefaultOption("min_area_ratio", &min_area_ratio);
  options.Parse(argc, argv);

  if (!ExistsDir(input_path)) {
    std::cerr << "ERROR: `input_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  if (overlap_ratio < 0) {
    std::cout << "WARN: Invalid `overlap_ratio`; resetting to 0" << std::endl;
    overlap_ratio = 0.0;
  }

  PrintHeading1("Splitting sparse model");
  std::cout << StringPrintf(" => Using \"%s\" split type", split_type.c_str())
            << std::endl;

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  SimilarityTransform3 tform;
  if (!gps_transform_path.empty()) {
    PrintHeading2("Reading model to ECEF transform");
    is_gps = true;
    tform = SimilarityTransform3::FromFile(gps_transform_path).Inverse();
  }
  const double scale = tform.Scale();

  // Create the necessary number of reconstructions based on the split method
  // and get the bounding boxes for each sub-reconstruction
  PrintHeading2("Computing bound_coords");
  std::vector<std::string> tile_keys;
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> exact_bounds;
  StringToLower(&split_type);
  if (split_type == "tiles") {
    std::ifstream file(split_params);
    CHECK(file.is_open()) << split_params;

    double x1, y1, z1, x2, y2, z2;
    std::string tile_key;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> bounds;
    tile_keys.clear();
    file >> tile_key >> x1 >> y1 >> z1 >> x2 >> y2 >> z2;
    while (!file.fail()) {
      tile_keys.push_back(tile_key);
      if (is_gps) {
        exact_bounds.emplace_back(
            TransformLatLonAltToModelCoords(tform, x1, y1, z1),
            TransformLatLonAltToModelCoords(tform, x2, y2, z2));
      } else {
        exact_bounds.emplace_back(Eigen::Vector3d(x1, y1, z1),
                                  Eigen::Vector3d(x2, y2, z2));
      }
      file >> tile_key >> x1 >> y1 >> z1 >> x2 >> y2 >> z2;
    }
  } else if (split_type == "extent") {
    std::vector<double> parts = CSVToVector<double>(split_params);
    Eigen::Vector3d extent(std::numeric_limits<double>::max(),
                           std::numeric_limits<double>::max(),
                           std::numeric_limits<double>::max());
    for (size_t i = 0; i < parts.size(); ++i) {
      extent(i) = parts[i] * scale;
    }

    const auto bbox = reconstruction.ComputeBoundingBox();
    const Eigen::Vector3d full_extent = bbox.second - bbox.first;
    const Eigen::Vector3i split(
        static_cast<int>(full_extent(0) / extent(0)) + 1,
        static_cast<int>(full_extent(1) / extent(1)) + 1,
        static_cast<int>(full_extent(2) / extent(2)) + 1);

    exact_bounds = ComputeEqualPartsBounds(reconstruction, split);

  } else if (split_type == "parts") {
    auto parts = CSVToVector<int>(split_params);
    Eigen::Vector3i split(1, 1, 1);
    for (size_t i = 0; i < parts.size(); ++i) {
      split(i) = parts[i];
      if (split(i) < 1) {
        std::cerr << "ERROR: Cannot split in less than 1 parts for dim " << i
                  << std::endl;
        return EXIT_FAILURE;
      }
    }
    exact_bounds = ComputeEqualPartsBounds(reconstruction, split);
  } else {
    std::cout << "ERROR: Invalid split type: " << split_type << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> bounds;
  for (const auto& bbox : exact_bounds) {
    const Eigen::Vector3d padding =
        (overlap_ratio * (bbox.second - bbox.first));
    bounds.emplace_back(bbox.first - padding, bbox.second + padding);
  }

  PrintHeading2("Applying split and writing reconstructions");
  const size_t num_parts = bounds.size();
  std::cout << StringPrintf(" => Splitting to %d parts", num_parts)
            << std::endl;

  const bool use_tile_keys = split_type == "tiles";

  auto SplitReconstruction = [&](const int idx) {
    Reconstruction tile_recon = reconstruction.Crop(bounds[idx]);
    // calculate area covered by model as proportion of box area
    auto bbox_extent = bounds[idx].second - bounds[idx].first;
    auto model_bbox = tile_recon.ComputeBoundingBox();
    auto model_extent = model_bbox.second - model_bbox.first;
    double area_ratio =
        (model_extent(0) * model_extent(1)) / (bbox_extent(0) * bbox_extent(1));
    int tile_num_points = tile_recon.NumPoints3D();

    std::string name = use_tile_keys ? tile_keys[idx] : std::to_string(idx);
    const bool include_tile =
        area_ratio >= min_area_ratio &&       //
        tile_num_points >= min_num_points &&  //
        tile_recon.NumRegImages() >= static_cast<size_t>(min_reg_images);

    if (include_tile) {
      std::cout << StringPrintf(
                       "Writing reconstruction %s with %d images, %d points, "
                       "and %.2f%% area coverage",
                       name.c_str(),
                       tile_recon.NumRegImages(),
                       tile_num_points,
                       100.0 * area_ratio)
                << std::endl;
      const std::string reconstruction_path = JoinPaths(output_path, name);
      CreateDirIfNotExists(reconstruction_path);
      tile_recon.Write(reconstruction_path);
      WriteBoundingBox(reconstruction_path, bounds[idx]);
      WriteBoundingBox(reconstruction_path, exact_bounds[idx], "_exact");

    } else {
      std::cout << StringPrintf(
                       "Skipping reconstruction %s with %d images, %d points, "
                       "and %.2f%% area coverage",
                       name.c_str(),
                       tile_recon.NumRegImages(),
                       tile_num_points,
                       100.0 * area_ratio)
                << std::endl;
    }
  };

  ThreadPool thread_pool(GetEffectiveNumThreads(num_threads));
  for (size_t idx = 0; idx < num_parts; ++idx) {
    thread_pool.AddTask(SplitReconstruction, idx);
  }
  thread_pool.Wait();

  timer.PrintMinutes();
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
  SimilarityTransform3 tform = SimilarityTransform3::FromFile(transform_path);
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
