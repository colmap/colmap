// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "base/database.h"

#include <fstream>

#include <boost/lexical_cast.hpp>

#include "util/string.h"

namespace colmap {
namespace {

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureKeypointsBlob;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptorsBlob;
typedef Eigen::Matrix<point2D_t, Eigen::Dynamic, 2, Eigen::RowMajor>
    FeatureMatchesBlob;

void SwapFeatureMatchesBlob(FeatureMatchesBlob* matches) {
  matches->col(0).swap(matches->col(1));
}

FeatureKeypointsBlob FeatureKeypointsToBlob(const FeatureKeypoints& keypoints) {
  const FeatureKeypointsBlob::Index kNumCols = 4;
  FeatureKeypointsBlob blob(keypoints.size(), kNumCols);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    blob(i, 0) = keypoints[i].x;
    blob(i, 1) = keypoints[i].y;
    blob(i, 2) = keypoints[i].scale;
    blob(i, 3) = keypoints[i].orientation;
  }
  return blob;
}

FeatureKeypoints FeatureKeypointsFromBlob(const FeatureKeypointsBlob& blob) {
  CHECK_EQ(blob.cols(), 4);
  FeatureKeypoints keypoints(static_cast<size_t>(blob.rows()));
  for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
    keypoints[i].x = blob(i, 0);
    keypoints[i].y = blob(i, 1);
    keypoints[i].scale = blob(i, 2);
    keypoints[i].orientation = blob(i, 3);
  }
  return keypoints;
}

FeatureMatchesBlob FeatureMatchesToBlob(const FeatureMatches& matches) {
  const FeatureMatchesBlob::Index kNumCols = 2;
  FeatureMatchesBlob blob(matches.size(), kNumCols);
  for (size_t i = 0; i < matches.size(); ++i) {
    blob(i, 0) = matches[i].point2D_idx1;
    blob(i, 1) = matches[i].point2D_idx2;
  }
  return blob;
}

FeatureMatches FeatureMatchesFromBlob(const FeatureMatchesBlob& blob) {
  CHECK_EQ(blob.cols(), 2);
  FeatureMatches matches(static_cast<size_t>(blob.rows()));
  for (FeatureMatchesBlob::Index i = 0; i < blob.rows(); ++i) {
    matches[i].point2D_idx1 = blob(i, 0);
    matches[i].point2D_idx2 = blob(i, 1);
  }
  return matches;
}

template <typename MatrixType>
MatrixType ReadMatrixBlob(sqlite3_stmt* sql_stmt, const int rc, const int col) {
  CHECK_GE(col, 0);

  MatrixType matrix;

  if (rc == SQLITE_ROW) {
    const size_t rows =
        static_cast<size_t>(sqlite3_column_int64(sql_stmt, col + 0));
    const size_t cols =
        static_cast<size_t>(sqlite3_column_int64(sql_stmt, col + 1));

    CHECK_GE(rows, 0);
    CHECK_GE(cols, 0);
    matrix = MatrixType(rows, cols);

    const size_t num_bytes =
        static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col + 2));
    CHECK_EQ(matrix.size() * sizeof(typename MatrixType::Scalar), num_bytes);

    memcpy(reinterpret_cast<char*>(matrix.data()),
           sqlite3_column_blob(sql_stmt, col + 2), num_bytes);
  } else {
    const typename MatrixType::Index rows =
        (MatrixType::RowsAtCompileTime == Eigen::Dynamic)
            ? 0
            : MatrixType::RowsAtCompileTime;
    const typename MatrixType::Index cols =
        (MatrixType::ColsAtCompileTime == Eigen::Dynamic)
            ? 0
            : MatrixType::ColsAtCompileTime;
    matrix = MatrixType(rows, cols);
  }

  return matrix;
}

template <typename MatrixType>
void WriteMatrixBlob(sqlite3_stmt* sql_stmt, const MatrixType& matrix,
                     const int col) {
  CHECK_GE(matrix.rows(), 0);
  CHECK_GE(matrix.cols(), 0);
  CHECK_GE(col, 0);

  const size_t num_bytes = matrix.size() * sizeof(typename MatrixType::Scalar);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, col + 0, matrix.rows()));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, col + 1, matrix.cols()));
  SQLITE3_CALL(sqlite3_bind_blob(sql_stmt, col + 2,
                                 reinterpret_cast<const char*>(matrix.data()),
                                 static_cast<int>(num_bytes), SQLITE_STATIC));
}

Camera ReadCameraRow(sqlite3_stmt* sql_stmt) {
  Camera camera;

  camera.SetCameraId(static_cast<camera_t>(sqlite3_column_int64(sql_stmt, 0)));
  camera.SetModelId(sqlite3_column_int64(sql_stmt, 1));
  camera.SetWidth(static_cast<size_t>(sqlite3_column_int64(sql_stmt, 2)));
  camera.SetHeight(static_cast<size_t>(sqlite3_column_int64(sql_stmt, 3)));

  const size_t num_params_bytes =
      static_cast<size_t>(sqlite3_column_bytes(sql_stmt, 4));
  const size_t num_params = num_params_bytes / sizeof(double);
  CHECK_EQ(num_params, camera.NumParams());
  memcpy(camera.ParamsData(), sqlite3_column_blob(sql_stmt, 4),
         num_params_bytes);

  camera.SetPriorFocalLength(sqlite3_column_int64(sql_stmt, 5) != 0);

  return camera;
}

Image ReadImageRow(sqlite3_stmt* sql_stmt) {
  Image image;
  image.SetImageId(static_cast<image_t>(sqlite3_column_int64(sql_stmt, 0)));
  image.SetName(std::string(
      reinterpret_cast<const char*>(sqlite3_column_text(sql_stmt, 1))));
  image.SetCameraId(static_cast<camera_t>(sqlite3_column_int64(sql_stmt, 2)));
  image.QvecPrior(0) = sqlite3_column_double(sql_stmt, 3);
  image.QvecPrior(1) = sqlite3_column_double(sql_stmt, 4);
  image.QvecPrior(2) = sqlite3_column_double(sql_stmt, 5);
  image.QvecPrior(3) = sqlite3_column_double(sql_stmt, 6);
  image.TvecPrior(0) = sqlite3_column_double(sql_stmt, 7);
  image.TvecPrior(1) = sqlite3_column_double(sql_stmt, 8);
  image.TvecPrior(2) = sqlite3_column_double(sql_stmt, 9);
  return image;
}

}  // namespace

const size_t Database::kMaxNumImages =
    static_cast<size_t>(std::numeric_limits<int32_t>::max());

Database::Database() : database_(nullptr) {}

Database::Database(const std::string& path) : Database() { Open(path); }

Database::~Database() { Close(); }

void Database::Open(const std::string& path) {
  Close();

  // SQLITE_OPEN_NOMUTEX specifies that the connection should not have a
  // mutex (so that we don't serialize the connection's operations).
  // Modifications to the database will still be serialized, but multiple
  // connections can read concurrently.
  SQLITE3_CALL(sqlite3_open_v2(
      path.c_str(), &database_,
      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
      nullptr));

  // Don't wait for the operating system to write the changes to disk
  SQLITE3_EXEC(database_, "PRAGMA synchronous=OFF", nullptr);

  // Use faster journaling mode
  SQLITE3_EXEC(database_, "PRAGMA journal_mode=WAL", nullptr);

  // Store temporary tables and indices in memory
  SQLITE3_EXEC(database_, "PRAGMA temp_store=MEMORY", nullptr);

  // Disabled by default
  SQLITE3_EXEC(database_, "PRAGMA foreign_keys=ON", nullptr);

  CreateTables();

  PrepareSQLStatements();
}

void Database::Close() {
  if (database_ != nullptr) {
    FinalizeSQLStatements();
    sqlite3_close_v2(database_);
    database_ = nullptr;
  }
}

bool Database::ExistsCamera(const camera_t camera_id) const {
  return ExistsRowId(sql_stmt_exists_camera_, camera_id);
}

bool Database::ExistsImage(const image_t image_id) const {
  return ExistsRowId(sql_stmt_exists_image_id_, image_id);
}

bool Database::ExistsImageWithName(std::string name) const {
  return ExistsRowString(sql_stmt_exists_image_name_, name);
}

bool Database::ExistsKeypoints(const image_t image_id) const {
  return ExistsRowId(sql_stmt_exists_keypoints_, image_id);
}

bool Database::ExistsDescriptors(const image_t image_id) const {
  return ExistsRowId(sql_stmt_exists_descriptors_, image_id);
}

bool Database::ExistsMatches(const image_t image_id1,
                             const image_t image_id2) const {
  return ExistsRowId(sql_stmt_exists_matches_,
                     ImagePairToPairId(image_id1, image_id2));
}

bool Database::ExistsInlierMatches(const image_t image_id1,
                                   const image_t image_id2) const {
  return ExistsRowId(sql_stmt_exists_inlier_matches_,
                     ImagePairToPairId(image_id1, image_id2));
}

size_t Database::NumCameras() const { return CountRows("cameras"); }

size_t Database::NumImages() const { return CountRows("images"); }

size_t Database::NumKeypoints() const { return SumColumn("rows", "keypoints"); }

size_t Database::MaxNumKeypoints() const {
  return MaxColumn("rows", "keypoints");
}

size_t Database::NumKeypointsForImage(const image_t image_id) const {
  return CountRowsForEntry(sql_stmt_num_keypoints_, image_id);
}

size_t Database::NumDescriptors() const {
  return SumColumn("rows", "descriptors");
}

size_t Database::MaxNumDescriptors() const {
  return MaxColumn("rows", "descriptors");
}

size_t Database::NumDescriptorsForImage(const image_t image_id) const {
  return CountRowsForEntry(sql_stmt_num_descriptors_, image_id);
}

size_t Database::NumMatches() const { return SumColumn("rows", "matches"); }

size_t Database::NumInlierMatches() const {
  return SumColumn("rows", "inlier_matches");
}

size_t Database::NumMatchedImagePairs() const { return CountRows("matches"); }

size_t Database::NumVerifiedImagePairs() const {
  return CountRows("inlier_matches");
}

Camera Database::ReadCamera(const camera_t camera_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_camera_, 1, camera_id));

  Camera camera;

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_camera_));
  if (rc == SQLITE_ROW) {
    camera = ReadCameraRow(sql_stmt_read_camera_);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_camera_));

  return camera;
}

std::vector<Camera> Database::ReadAllCameras() const {
  std::vector<Camera> cameras;

  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_cameras_)) == SQLITE_ROW) {
    cameras.push_back(ReadCameraRow(sql_stmt_read_cameras_));
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_cameras_));

  return cameras;
}

Image Database::ReadImage(const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_image_id_, 1, image_id));

  Image image;

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_image_id_));
  if (rc == SQLITE_ROW) {
    image = ReadImageRow(sql_stmt_read_image_id_);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_image_id_));

  return image;
}

Image Database::ReadImageWithName(const std::string& name) const {
  SQLITE3_CALL(sqlite3_bind_text(sql_stmt_read_image_name_, 1, name.c_str(),
                                 static_cast<int>(name.size()), SQLITE_STATIC));

  Image image;

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_image_name_));
  if (rc == SQLITE_ROW) {
    image = ReadImageRow(sql_stmt_read_image_name_);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_image_name_));

  return image;
}

std::vector<Image> Database::ReadAllImages() const {
  std::vector<Image> images;
  images.reserve(NumImages());

  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_images_)) == SQLITE_ROW) {
    images.push_back(ReadImageRow(sql_stmt_read_images_));
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_images_));

  return images;
}

FeatureKeypoints Database::ReadKeypoints(const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_keypoints_, 1, image_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_keypoints_));
  const FeatureKeypointsBlob blob =
      ReadMatrixBlob<FeatureKeypointsBlob>(sql_stmt_read_keypoints_, rc, 0);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_keypoints_));

  return FeatureKeypointsFromBlob(blob);
}

FeatureDescriptors Database::ReadDescriptors(const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_descriptors_, 1, image_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_descriptors_));
  const FeatureDescriptors descriptors =
      ReadMatrixBlob<FeatureDescriptors>(sql_stmt_read_descriptors_, rc, 0);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_descriptors_));

  return descriptors;
}

FeatureMatches Database::ReadMatches(image_t image_id1,
                                     image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_matches_, 1, pair_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_));
  FeatureMatchesBlob blob =
      ReadMatrixBlob<FeatureMatchesBlob>(sql_stmt_read_matches_, rc, 0);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_matches_));

  if (SwapImagePair(image_id1, image_id2)) {
    SwapFeatureMatchesBlob(&blob);
  }

  return FeatureMatchesFromBlob(blob);
}

std::vector<std::pair<image_pair_t, FeatureMatches>> Database::ReadAllMatches()
    const {
  std::vector<std::pair<image_pair_t, FeatureMatches>> all_matches;

  int rc;
  while ((rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_all_))) ==
         SQLITE_ROW) {
    const image_pair_t pair_id = static_cast<image_pair_t>(
        sqlite3_column_int64(sql_stmt_read_matches_all_, 0));
    const FeatureMatchesBlob blob =
        ReadMatrixBlob<FeatureMatchesBlob>(sql_stmt_read_matches_all_, rc, 1);
    all_matches.emplace_back(pair_id, FeatureMatchesFromBlob(blob));
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_matches_all_));

  return all_matches;
}

TwoViewGeometry Database::ReadInlierMatches(const image_t image_id1,
                                            const image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_inlier_matches_, 1, pair_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_inlier_matches_));

  TwoViewGeometry two_view_geometry;

  FeatureMatchesBlob blob =
      ReadMatrixBlob<FeatureMatchesBlob>(sql_stmt_read_inlier_matches_, rc, 0);

  two_view_geometry.config =
      static_cast<int>(sqlite3_column_int64(sql_stmt_read_inlier_matches_, 3));

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_inlier_matches_));

  if (SwapImagePair(image_id1, image_id2)) {
    SwapFeatureMatchesBlob(&blob);
  }

  two_view_geometry.inlier_matches = FeatureMatchesFromBlob(blob);

  return two_view_geometry;
}

void Database::ReadAllInlierMatches(
    std::vector<image_pair_t>* image_pair_ids,
    std::vector<TwoViewGeometry>* two_view_geometries) const {
  int rc;
  while ((rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_inlier_matches_all_))) ==
         SQLITE_ROW) {
    const image_pair_t pair_id = static_cast<image_pair_t>(
        sqlite3_column_int64(sql_stmt_read_inlier_matches_all_, 0));
    image_pair_ids->push_back(pair_id);

    TwoViewGeometry two_view_geometry;
    const FeatureMatchesBlob blob = ReadMatrixBlob<FeatureMatchesBlob>(
        sql_stmt_read_inlier_matches_all_, rc, 1);
    two_view_geometry.config = static_cast<int>(
        sqlite3_column_int64(sql_stmt_read_inlier_matches_all_, 4));
    two_view_geometry.inlier_matches = FeatureMatchesFromBlob(blob);
    two_view_geometries->push_back(two_view_geometry);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_inlier_matches_all_));
}

void Database::ReadInlierMatchesGraph(
    std::vector<std::pair<image_t, image_t>>* image_pairs,
    std::vector<int>* num_inliers) const {
  const auto num_inlier_matches = NumInlierMatches();
  image_pairs->reserve(num_inlier_matches);
  num_inliers->reserve(num_inlier_matches);

  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_inlier_matches_graph_)) ==
         SQLITE_ROW) {
    image_t image_id1;
    image_t image_id2;
    const image_pair_t pair_id = static_cast<image_pair_t>(
        sqlite3_column_int64(sql_stmt_read_inlier_matches_graph_, 0));
    PairIdToImagePair(pair_id, &image_id1, &image_id2);
    image_pairs->emplace_back(image_id1, image_id2);

    const int rows = static_cast<int>(
        sqlite3_column_int64(sql_stmt_read_inlier_matches_graph_, 1));
    num_inliers->push_back(rows);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_inlier_matches_graph_));
}

camera_t Database::WriteCamera(const Camera& camera,
                               const bool use_camera_id) const {
  if (use_camera_id) {
    CHECK(!ExistsCamera(camera.CameraId())) << "camera_id must be unique";
    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_add_camera_, 1, camera.CameraId()));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_add_camera_, 1));
  }

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 2, camera.ModelId()));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 3,
                                  static_cast<sqlite3_int64>(camera.Width())));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 4,
                                  static_cast<sqlite3_int64>(camera.Height())));

  const size_t num_params_bytes = sizeof(double) * camera.NumParams();
  SQLITE3_CALL(sqlite3_bind_blob(sql_stmt_add_camera_, 5, camera.ParamsData(),
                                 static_cast<int>(num_params_bytes),
                                 SQLITE_STATIC));

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 6,
                                  camera.HasPriorFocalLength()));

  SQLITE3_CALL(sqlite3_step(sql_stmt_add_camera_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_add_camera_));

  return static_cast<camera_t>(sqlite3_last_insert_rowid(database_));
}

image_t Database::WriteImage(const Image& image,
                             const bool use_image_id) const {
  if (use_image_id) {
    CHECK(!ExistsImage(image.ImageId())) << "image_id must be unique";
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_image_, 1, image.ImageId()));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_add_image_, 1));
  }

  SQLITE3_CALL(sqlite3_bind_text(sql_stmt_add_image_, 2, image.Name().c_str(),
                                 static_cast<int>(image.Name().size()),
                                 SQLITE_STATIC));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_image_, 3, image.CameraId()));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 4, image.QvecPrior(0)));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 5, image.QvecPrior(1)));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 6, image.QvecPrior(2)));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 7, image.QvecPrior(3)));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 8, image.TvecPrior(0)));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 9, image.TvecPrior(1)));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_add_image_, 10, image.TvecPrior(2)));

  SQLITE3_CALL(sqlite3_step(sql_stmt_add_image_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_add_image_));

  return static_cast<image_t>(sqlite3_last_insert_rowid(database_));
}

void Database::WriteKeypoints(const image_t image_id,
                              const FeatureKeypoints& keypoints) const {
  const FeatureKeypointsBlob blob = FeatureKeypointsToBlob(keypoints);

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_keypoints_, 1, image_id));
  WriteMatrixBlob(sql_stmt_write_keypoints_, blob, 2);

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_keypoints_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_keypoints_));
}

void Database::WriteDescriptors(const image_t image_id,
                                const FeatureDescriptors& descriptors) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_descriptors_, 1, image_id));
  WriteMatrixBlob(sql_stmt_write_descriptors_, descriptors, 2);

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_descriptors_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_descriptors_));
}

void Database::WriteMatches(const image_t image_id1, const image_t image_id2,
                            const FeatureMatches& matches) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_matches_, 1, pair_id));

  // Important: the swapped data must live until the query is executed.
  FeatureMatchesBlob blob = FeatureMatchesToBlob(matches);
  if (SwapImagePair(image_id1, image_id2)) {
    SwapFeatureMatchesBlob(&blob);
    WriteMatrixBlob(sql_stmt_write_matches_, blob, 2);
  } else {
    WriteMatrixBlob(sql_stmt_write_matches_, blob, 2);
  }

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_matches_));
}

void Database::WriteInlierMatches(
    const image_t image_id1, const image_t image_id2,
    const TwoViewGeometry& two_view_geometry) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_inlier_matches_, 1, pair_id));

  // Important: the swapped data must live until the query is executed.
  FeatureMatchesBlob blob =
      FeatureMatchesToBlob(two_view_geometry.inlier_matches);
  if (SwapImagePair(image_id1, image_id2)) {
    SwapFeatureMatchesBlob(&blob);
  }

  WriteMatrixBlob(sql_stmt_write_inlier_matches_, blob, 2);

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_inlier_matches_, 5,
                                  two_view_geometry.config));

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_inlier_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_inlier_matches_));
}

void Database::UpdateCamera(const Camera& camera) const {
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_update_camera_, 1, camera.ModelId()));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_camera_, 2,
                                  static_cast<sqlite3_int64>(camera.Width())));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_camera_, 3,
                                  static_cast<sqlite3_int64>(camera.Height())));

  const size_t num_params_bytes = sizeof(double) * camera.NumParams();
  SQLITE3_CALL(
      sqlite3_bind_blob(sql_stmt_update_camera_, 4, camera.ParamsData(),
                        static_cast<int>(num_params_bytes), SQLITE_STATIC));

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_camera_, 5,
                                  camera.HasPriorFocalLength()));

  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_update_camera_, 6, camera.CameraId()));

  SQLITE3_CALL(sqlite3_step(sql_stmt_update_camera_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_camera_));
}

void Database::UpdateImage(const Image& image) const {
  SQLITE3_CALL(
      sqlite3_bind_text(sql_stmt_update_image_, 1, image.Name().c_str(),
                        static_cast<int>(image.Name().size()), SQLITE_STATIC));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_image_, 2, image.CameraId()));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 3, image.QvecPrior(0)));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 4, image.QvecPrior(1)));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 5, image.QvecPrior(2)));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 6, image.QvecPrior(3)));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 7, image.TvecPrior(0)));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 8, image.TvecPrior(1)));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 9, image.TvecPrior(2)));

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_image_, 10, image.ImageId()));

  SQLITE3_CALL(sqlite3_step(sql_stmt_update_image_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_image_));
}

void Database::DeleteMatches(const image_t image_id1,
                             const image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_delete_matches_, 1,
                                  static_cast<sqlite3_int64>(pair_id)));
  SQLITE3_CALL(sqlite3_step(sql_stmt_delete_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_delete_matches_));
}

void Database::DeleteInlierMatches(const image_t image_id1,
                                   const image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_delete_inlier_matches_, 1,
                                  static_cast<sqlite3_int64>(pair_id)));
  SQLITE3_CALL(sqlite3_step(sql_stmt_delete_inlier_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_delete_inlier_matches_));
}

void Database::ClearMatches() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_matches_));
}

void Database::ClearInlierMatches() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_inlier_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_inlier_matches_));
}

void Database::BeginTransaction() const {
  SQLITE3_EXEC(database_, "BEGIN TRANSACTION", nullptr);
}

void Database::EndTransaction() const {
  SQLITE3_EXEC(database_, "END TRANSACTION", nullptr);
}

void Database::PrepareSQLStatements() {
  sql_stmts_.clear();

  std::string sql;

  //////////////////////////////////////////////////////////////////////////////
  // num_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "SELECT rows FROM keypoints WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_num_keypoints_, 0));
  sql_stmts_.push_back(sql_stmt_num_keypoints_);

  sql = "SELECT rows FROM descriptors WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_num_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_num_descriptors_);

  //////////////////////////////////////////////////////////////////////////////
  // exists_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "SELECT 1 FROM cameras WHERE camera_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_camera_, 0));
  sql_stmts_.push_back(sql_stmt_exists_camera_);

  sql = "SELECT 1 FROM images WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_image_id_, 0));
  sql_stmts_.push_back(sql_stmt_exists_image_id_);

  sql = "SELECT 1 FROM images WHERE name = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_image_name_, 0));
  sql_stmts_.push_back(sql_stmt_exists_image_name_);

  sql = "SELECT 1 FROM keypoints WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_keypoints_, 0));
  sql_stmts_.push_back(sql_stmt_exists_keypoints_);

  sql = "SELECT 1 FROM descriptors WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_exists_descriptors_);

  sql = "SELECT 1 FROM matches WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_matches_, 0));
  sql_stmts_.push_back(sql_stmt_exists_matches_);

  sql = "SELECT 1 FROM inlier_matches WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_inlier_matches_, 0));
  sql_stmts_.push_back(sql_stmt_exists_inlier_matches_);

  //////////////////////////////////////////////////////////////////////////////
  // add_*
  //////////////////////////////////////////////////////////////////////////////
  sql =
      "INSERT INTO cameras(camera_id, model, width, height, params, "
      "prior_focal_length) VALUES(?, ?, ?, ?, ?, ?);";
  SQLITE3_CALL(
      sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_add_camera_, 0));
  sql_stmts_.push_back(sql_stmt_add_camera_);

  sql =
      "INSERT INTO images(image_id, name, camera_id, prior_qw, prior_qx, "
      "prior_qy, prior_qz, prior_tx, prior_ty, prior_tz) VALUES(?, ?, ?, ?, ?, "
      "?, ?, ?, ?, ?);";
  SQLITE3_CALL(
      sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_add_image_, 0));
  sql_stmts_.push_back(sql_stmt_add_image_);

  //////////////////////////////////////////////////////////////////////////////
  // update_*
  //////////////////////////////////////////////////////////////////////////////
  sql =
      "UPDATE cameras SET model=?, width=?, height=?, params=?, "
      "prior_focal_length=? WHERE camera_id=?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_update_camera_, 0));
  sql_stmts_.push_back(sql_stmt_update_camera_);

  sql =
      "UPDATE images SET name=?, camera_id=?, prior_qw=?, prior_qx=?, "
      "prior_qy=?, prior_qz=?, prior_tx=?, prior_ty=?, prior_tz=? WHERE "
      "image_id=?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_update_image_, 0));
  sql_stmts_.push_back(sql_stmt_update_image_);

  //////////////////////////////////////////////////////////////////////////////
  // read_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "SELECT * FROM cameras WHERE camera_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_camera_, 0));
  sql_stmts_.push_back(sql_stmt_read_camera_);

  sql = "SELECT * FROM cameras;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_cameras_, 0));
  sql_stmts_.push_back(sql_stmt_read_cameras_);

  sql = "SELECT * FROM images WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_image_id_, 0));
  sql_stmts_.push_back(sql_stmt_read_image_id_);

  sql = "SELECT * FROM images WHERE name = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_image_name_, 0));
  sql_stmts_.push_back(sql_stmt_read_image_name_);

  sql = "SELECT * FROM images;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_images_, 0));
  sql_stmts_.push_back(sql_stmt_read_images_);

  sql = "SELECT rows, cols, data FROM keypoints WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_keypoints_, 0));
  sql_stmts_.push_back(sql_stmt_read_keypoints_);

  sql = "SELECT rows, cols, data FROM descriptors WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_read_descriptors_);

  sql = "SELECT rows, cols, data FROM matches WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_matches_, 0));
  sql_stmts_.push_back(sql_stmt_read_matches_);

  sql = "SELECT * FROM matches WHERE rows > 0;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_matches_all_, 0));
  sql_stmts_.push_back(sql_stmt_read_matches_all_);

  sql =
      "SELECT rows, cols, data, config FROM inlier_matches "
      "WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_inlier_matches_, 0));
  sql_stmts_.push_back(sql_stmt_read_inlier_matches_);

  sql = "SELECT * FROM inlier_matches WHERE rows > 0;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_inlier_matches_all_, 0));
  sql_stmts_.push_back(sql_stmt_read_inlier_matches_all_);

  sql = "SELECT pair_id, rows FROM inlier_matches WHERE rows > 0;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_inlier_matches_graph_, 0));
  sql_stmts_.push_back(sql_stmt_read_inlier_matches_graph_);

  //////////////////////////////////////////////////////////////////////////////
  // write_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_keypoints_, 0));
  sql_stmts_.push_back(sql_stmt_write_keypoints_);

  sql =
      "INSERT INTO descriptors(image_id, rows, cols, data) VALUES(?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_write_descriptors_);

  sql = "INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_matches_, 0));
  sql_stmts_.push_back(sql_stmt_write_matches_);

  sql =
      "INSERT INTO inlier_matches(pair_id, rows, cols, data, config) "
      "VALUES(?, ?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_inlier_matches_, 0));
  sql_stmts_.push_back(sql_stmt_write_inlier_matches_);

  //////////////////////////////////////////////////////////////////////////////
  // delete_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "DELETE FROM matches WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_delete_matches_, 0));
  sql_stmts_.push_back(sql_stmt_delete_matches_);

  sql = "DELETE FROM inlier_matches WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_delete_inlier_matches_, 0));
  sql_stmts_.push_back(sql_stmt_delete_inlier_matches_);

  //////////////////////////////////////////////////////////////////////////////
  // clear_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "DELETE FROM matches;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_clear_matches_, 0));
  sql_stmts_.push_back(sql_stmt_clear_matches_);

  sql = "DELETE FROM inlier_matches;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_clear_inlier_matches_, 0));
  sql_stmts_.push_back(sql_stmt_clear_inlier_matches_);
}

void Database::FinalizeSQLStatements() {
  for (const auto& sql_stmt : sql_stmts_) {
    SQLITE3_CALL(sqlite3_finalize(sql_stmt));
  }
}

void Database::CreateTables() const {
  CreateCameraTable();
  CreateImageTable();
  CreateKeypointsTable();
  CreateDescriptorsTable();
  CreateMatchesTable();
  CreateInlierMatchesTable();
}

void Database::CreateCameraTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS cameras"
      "   (camera_id            INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
      "    model                INTEGER                             NOT NULL,"
      "    width                INTEGER                             NOT NULL,"
      "    height               INTEGER                             NOT NULL,"
      "    params               BLOB,"
      "    prior_focal_length   INTEGER                             NOT NULL);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateImageTable() const {
  const std::string sql = StringPrintf(
      "CREATE TABLE IF NOT EXISTS images"
      "   (image_id   INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
      "    name       TEXT                                NOT NULL UNIQUE,"
      "    camera_id  INTEGER                             NOT NULL,"
      "    prior_qw   REAL,"
      "    prior_qx   REAL,"
      "    prior_qy   REAL,"
      "    prior_qz   REAL,"
      "    prior_tx   REAL,"
      "    prior_ty   REAL,"
      "    prior_tz   REAL,"
      "CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < %d),"
      "FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));"
      "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);",
      kMaxNumImages);

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateKeypointsTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS keypoints"
      "   (image_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows      INTEGER               NOT NULL,"
      "    cols      INTEGER               NOT NULL,"
      "    data      BLOB,"
      "FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateDescriptorsTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS descriptors"
      "   (image_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows      INTEGER               NOT NULL,"
      "    cols      INTEGER               NOT NULL,"
      "    data      BLOB,"
      "FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateMatchesTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS matches"
      "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows     INTEGER               NOT NULL,"
      "    cols     INTEGER               NOT NULL,"
      "    data     BLOB);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateInlierMatchesTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS inlier_matches"
      "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows     INTEGER               NOT NULL,"
      "    cols     INTEGER               NOT NULL,"
      "    data     BLOB,"
      "    config   INTEGER               NOT NULL);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

bool Database::ExistsRowId(sqlite3_stmt* sql_stmt,
                           const sqlite3_int64 row_id) const {
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt, 1, static_cast<sqlite3_int64>(row_id)));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));

  const bool exists = rc == SQLITE_ROW;

  SQLITE3_CALL(sqlite3_reset(sql_stmt));

  return exists;
}

bool Database::ExistsRowString(sqlite3_stmt* sql_stmt,
                               const std::string& row_entry) const {
  SQLITE3_CALL(sqlite3_bind_text(sql_stmt, 1, row_entry.c_str(),
                                 static_cast<int>(row_entry.size()),
                                 SQLITE_STATIC));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));

  const bool exists = rc == SQLITE_ROW;

  SQLITE3_CALL(sqlite3_reset(sql_stmt));

  return exists;
}

size_t Database::CountRows(const std::string& table) const {
  const std::string sql = "SELECT COUNT(*) FROM " + table + ";";
  sqlite3_stmt* sql_stmt;

  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt, 0));

  size_t count = 0;
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
  if (rc == SQLITE_ROW) {
    count = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
  }

  SQLITE3_CALL(sqlite3_finalize(sql_stmt));

  return count;
}

size_t Database::CountRowsForEntry(sqlite3_stmt* sql_stmt,
                                   const sqlite3_int64 row_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, 1, row_id));

  size_t count = 0;
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
  if (rc == SQLITE_ROW) {
    count = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt));

  return count;
}

size_t Database::SumColumn(const std::string& column,
                           const std::string& table) const {
  const std::string sql = "SELECT SUM(" + column + ") FROM " + table + ";";
  sqlite3_stmt* sql_stmt;

  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt, 0));

  size_t sum = 0;
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
  if (rc == SQLITE_ROW) {
    sum = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
  }

  SQLITE3_CALL(sqlite3_finalize(sql_stmt));

  return sum;
}

size_t Database::MaxColumn(const std::string& column,
                           const std::string& table) const {
  const std::string sql = "SELECT MAX(" + column + ") FROM " + table + ";";
  sqlite3_stmt* sql_stmt;

  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt, 0));

  size_t max = 0;
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
  if (rc == SQLITE_ROW) {
    max = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
  }

  SQLITE3_CALL(sqlite3_finalize(sql_stmt));

  return max;
}

DatabaseTransaction::DatabaseTransaction(Database* database)
    : database_(database), database_lock_(database->transaction_mutex_) {
  CHECK_NOTNULL(database_);
  database_->BeginTransaction();
}

DatabaseTransaction::~DatabaseTransaction() { database_->EndTransaction(); }

}  // namespace colmap
