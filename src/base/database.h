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

#ifndef COLMAP_SRC_BASE_DATABASE_H_
#define COLMAP_SRC_BASE_DATABASE_H_

#include <mutex>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "base/camera.h"
#include "base/feature.h"
#include "base/image.h"
#include "estimators/two_view_geometry.h"
#include "ext/SQLite/sqlite3.h"
#include "util/sqlite3_utils.h"
#include "util/types.h"

namespace colmap {

// Database class to read and write images, features, cameras, matches, etc.
// from a SQLite database. The class is not thread-safe and must not be accessed
// concurrently. The class is optimized for single-thread speed and for optimal
// performance, wrap multiple method calls inside a leading `BeginTransaction`
// and trailing `EndTransaction`.
class Database {
 public:
  const static int kSchemaVersion = 1;

  // The maximum number of images, that can be stored in the database.
  // This limitation arises due to the fact, that we generate unique IDs for
  // image pairs manually. Note: do not change this to
  // another type than `size_t`.
  const static size_t kMaxNumImages;

  Database();
  explicit Database(const std::string& path);
  ~Database();

  // Open and close database. The same database should not be opened
  // concurrently in multiple threads or processes.
  void Open(const std::string& path);
  void Close();

  // Check if entry already exists in database. For image pairs, the order of
  // `image_id1` and `image_id2` does not matter.
  bool ExistsCamera(const camera_t camera_id) const;
  bool ExistsImage(const image_t image_id) const;
  bool ExistsImageWithName(std::string name) const;
  bool ExistsKeypoints(const image_t image_id) const;
  bool ExistsDescriptors(const image_t image_id) const;
  bool ExistsMatches(const image_t image_id1, const image_t image_id2) const;
  bool ExistsInlierMatches(const image_t image_id1,
                           const image_t image_id2) const;

  // Number of rows in `cameras` table.
  size_t NumCameras() const;

  //  Number of rows in `images` table.
  size_t NumImages() const;

  // Sum of `rows` column in `keypoints` table, i.e. number of total keypoints.
  size_t NumKeypoints() const;

  // The number of keypoints for the image with most features.
  size_t MaxNumKeypoints() const;

  // Number of descriptors for specific image.
  size_t NumKeypointsForImage(const image_t image_id) const;

  // Sum of `rows` column in `descriptors` table,
  // i.e. number of total descriptors.
  size_t NumDescriptors() const;

  // The number of descriptors for the image with most features.
  size_t MaxNumDescriptors() const;

  // Number of descriptors for specific image.
  size_t NumDescriptorsForImage(const image_t image_id) const;

  // Sum of `rows` column in `matches` table, i.e. number of total matches.
  size_t NumMatches() const;

  // Sum of `rows` column in `inlier_matches` table,
  // i.e. number of total inlier matches.
  size_t NumInlierMatches() const;

  // Number of rows in `matches` table.
  size_t NumMatchedImagePairs() const;

  // Number of rows in `inlier_matches` table.
  size_t NumVerifiedImagePairs() const;

  // Each image pair is assigned an unique ID in the `matches` and
  // `inlier_matches` table. We intentionally avoid to store the pairs in a
  // separate table by using e.g. AUTOINCREMENT, since the overhead of querying
  // the unique pair ID is significant.
  inline static image_pair_t ImagePairToPairId(const image_t image_id1,
                                               const image_t image_id2);

  inline static void PairIdToImagePair(const image_pair_t pair_id,
                                       image_t* image_id1, image_t* image_id2);

  // Return true if image pairs should be swapped. Used to enforce a specific
  // image order to generate unique image pair identifiers independent of the
  // order in which the image identifiers are used.
  inline static bool SwapImagePair(const image_t image_id1,
                                   const image_t image_id2);

  // Read an existing entry in the database. The user is responsible for making
  // sure that the entry actually exists. For image pairs, the order of
  // `image_id1` and `image_id2` does not matter.
  Camera ReadCamera(const camera_t camera_id) const;
  std::vector<Camera> ReadAllCameras() const;

  Image ReadImage(const image_t image_id) const;
  Image ReadImageWithName(const std::string& name) const;
  std::vector<Image> ReadAllImages() const;

  FeatureKeypoints ReadKeypoints(const image_t image_id) const;
  FeatureDescriptors ReadDescriptors(const image_t image_id) const;

  FeatureMatches ReadMatches(const image_t image_id1,
                             const image_t image_id2) const;
  std::vector<std::pair<image_pair_t, FeatureMatches>> ReadAllMatches() const;

  TwoViewGeometry ReadInlierMatches(const image_t image_id1,
                                    const image_t image_id2) const;
  void ReadAllInlierMatches(
      std::vector<image_pair_t>* image_pair_ids,
      std::vector<TwoViewGeometry>* two_view_geometries) const;

  // Read all image pairs that have an entry in the `inlier_matches` table with
  // at least one inlier match and their corresponding number of inlier matches.
  void ReadInlierMatchesGraph(
      std::vector<std::pair<image_t, image_t>>* image_pairs,
      std::vector<int>* num_inliers) const;

  // Add new camera and return its database identifier. If `use_camera_id`
  // is false a new identifier is automatically generated.
  camera_t WriteCamera(const Camera& camera,
                       const bool use_camera_id = false) const;

  // Add new image and return its database identifier. If `use_image_id`
  // is false a new identifier is automatically generated.
  image_t WriteImage(const Image& image, const bool use_image_id = false) const;

  // Write a new entry in the database. The user is responsible for making sure
  // that the entry does not yet exist. For image pairs, the order of
  // `image_id1` and `image_id2` does not matter.
  void WriteKeypoints(const image_t image_id,
                      const FeatureKeypoints& keypoints) const;
  void WriteDescriptors(const image_t image_id,
                        const FeatureDescriptors& descriptors) const;
  void WriteMatches(const image_t image_id1, const image_t image_id2,
                    const FeatureMatches& matches) const;
  void WriteInlierMatches(const image_t image_id1, const image_t image_id2,
                          const TwoViewGeometry& two_view_geometry) const;

  // Update an existing camera in the database. The user is responsible for
  // making sure that the entry already exists.
  void UpdateCamera(const Camera& camera) const;

  // Update an existing image in the database. The user is responsible for
  // making sure that the entry already exists.
  void UpdateImage(const Image& image) const;

  // Delete matches of an image pair.
  void DeleteMatches(const image_t image_id1, const image_t image_id2) const;

  // Delete inlier matches of an image pair.
  void DeleteInlierMatches(const image_t image_id1,
                           const image_t image_id2) const;

  // Clear the entire matches table.
  void ClearMatches() const;

  // Clear the entire inlier matches table.
  void ClearInlierMatches() const;

 private:
  friend class DatabaseTransaction;

  // Combine multiple queries into one transaction by wrapping a code section
  // into a `BeginTransaction` and `EndTransaction`. You can create a scoped
  // transaction with `DatabaseTransaction` that ends when the transaction
  // object is destructed. Combining queries results in faster transaction time
  // due to reduced locking of the database etc.
  void BeginTransaction() const;
  void EndTransaction() const;

  // Prepare SQL statements once at construction of the database, and reuse
  // the statements for multiple queries by resetting their states.
  void PrepareSQLStatements();
  void FinalizeSQLStatements();

  // Create database tables, if not existing, called when opening a database.
  void CreateTables() const;
  void CreateCameraTable() const;
  void CreateImageTable() const;
  void CreateKeypointsTable() const;
  void CreateDescriptorsTable() const;
  void CreateMatchesTable() const;
  void CreateInlierMatchesTable() const;

  bool ExistsRowId(sqlite3_stmt* sql_stmt, const sqlite3_int64 row_id) const;
  bool ExistsRowString(sqlite3_stmt* sql_stmt,
                       const std::string& row_entry) const;

  size_t CountRows(const std::string& table) const;
  size_t CountRowsForEntry(sqlite3_stmt* sql_stmt,
                           const sqlite3_int64 row_id) const;
  size_t SumColumn(const std::string& column, const std::string& table) const;
  size_t MaxColumn(const std::string& column, const std::string& table) const;

  sqlite3* database_ = nullptr;

  // Used to ensure that only one transaction is active at the same time.
  std::mutex transaction_mutex_;

  // A collection of all `sqlite3_stmt` objects for deletion in the destructor.
  std::vector<sqlite3_stmt*> sql_stmts_;

  // num_*
  sqlite3_stmt* sql_stmt_num_keypoints_ = nullptr;
  sqlite3_stmt* sql_stmt_num_descriptors_ = nullptr;

  // exists_*
  sqlite3_stmt* sql_stmt_exists_camera_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_image_id_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_image_name_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_keypoints_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_descriptors_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_inlier_matches_ = nullptr;

  // add_*
  sqlite3_stmt* sql_stmt_add_camera_ = nullptr;
  sqlite3_stmt* sql_stmt_add_image_ = nullptr;

  // update_*
  sqlite3_stmt* sql_stmt_update_camera_ = nullptr;
  sqlite3_stmt* sql_stmt_update_image_ = nullptr;

  // read_*
  sqlite3_stmt* sql_stmt_read_camera_ = nullptr;
  sqlite3_stmt* sql_stmt_read_cameras_ = nullptr;
  sqlite3_stmt* sql_stmt_read_image_id_ = nullptr;
  sqlite3_stmt* sql_stmt_read_image_name_ = nullptr;
  sqlite3_stmt* sql_stmt_read_images_ = nullptr;
  sqlite3_stmt* sql_stmt_read_keypoints_ = nullptr;
  sqlite3_stmt* sql_stmt_read_descriptors_ = nullptr;
  sqlite3_stmt* sql_stmt_read_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_read_matches_all_ = nullptr;
  sqlite3_stmt* sql_stmt_read_inlier_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_read_inlier_matches_all_ = nullptr;
  sqlite3_stmt* sql_stmt_read_inlier_matches_graph_ = nullptr;

  // write_*
  sqlite3_stmt* sql_stmt_write_keypoints_ = nullptr;
  sqlite3_stmt* sql_stmt_write_descriptors_ = nullptr;
  sqlite3_stmt* sql_stmt_write_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_write_inlier_matches_ = nullptr;

  // delete_*
  sqlite3_stmt* sql_stmt_delete_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_delete_inlier_matches_ = nullptr;

  // clear_*
  sqlite3_stmt* sql_stmt_clear_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_inlier_matches_ = nullptr;
};

// This class automatically manages the scope of a database transaction by
// calling `BeginTransaction` and `EndTransaction` during construction and
// destruction, respectively.
class DatabaseTransaction {
 public:
  explicit DatabaseTransaction(Database* database);
  ~DatabaseTransaction();

 private:
  NON_COPYABLE(DatabaseTransaction)
  NON_MOVABLE(DatabaseTransaction)
  Database* database_;
  std::unique_lock<std::mutex> database_lock_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

image_pair_t Database::ImagePairToPairId(const image_t image_id1,
                                         const image_t image_id2) {
  CHECK_GE(image_id1, 0);
  CHECK_GE(image_id2, 0);
  CHECK_LT(image_id1, kMaxNumImages);
  CHECK_LT(image_id2, kMaxNumImages);
  if (SwapImagePair(image_id1, image_id2)) {
    return kMaxNumImages * image_id2 + image_id1;
  } else {
    return kMaxNumImages * image_id1 + image_id2;
  }
}

void Database::PairIdToImagePair(const image_pair_t pair_id, image_t* image_id1,
                                 image_t* image_id2) {
  *image_id2 = static_cast<image_t>(pair_id % kMaxNumImages);
  *image_id1 = static_cast<image_t>((pair_id - *image_id2) / kMaxNumImages);
  CHECK_GE(*image_id1, 0);
  CHECK_GE(*image_id2, 0);
  CHECK_LT(*image_id1, kMaxNumImages);
  CHECK_LT(*image_id2, kMaxNumImages);
}

// Return true if image pairs should be swapped. Used to enforce a specific
// image order to generate unique image pair identifiers independent of the
// order in which the image identifiers are used.
bool Database::SwapImagePair(const image_t image_id1, const image_t image_id2) {
  return image_id1 > image_id2;
}

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_DATABASE_H_
