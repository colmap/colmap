// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#pragma once

#include "colmap/feature/types.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/image.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/sensor/rig.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <map>
#include <mutex>
#include <vector>

#include <Eigen/Core>

namespace colmap {

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureKeypointsBlob;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptorsBlob;
typedef Eigen::Matrix<point2D_t, Eigen::Dynamic, 2, Eigen::RowMajor>
    FeatureMatchesBlob;

// Database class to read and write images, features, cameras, matches, etc.
// from a SQLite database. The class is not thread-safe and must not be accessed
// concurrently. The class is optimized for single-thread speed and for optimal
// performance, wrap multiple method calls inside a leading `BeginTransaction`
// and trailing `EndTransaction`.
class Database {
 public:
  using Options = std::map<std::string, std::string>;

  // Factory function to create a database implementation for a given path.
  // The factory should be robust to handle non-supported files and return a
  // runtime_error in that case.
  using Factory = std::function<std::shared_ptr<Database>(const std::string&,
                                                          const Options&)>;

  // Register a factory to open a database implementation. Database factories
  // are tried in reverse order of registration. In other words, later
  // registrations are tried first.
  static void Register(Factory factory);

  // Closes the database, if not closed before.
  virtual ~Database();

  // Open database and throw a runtime_error if none of the factories succeeds.
  static std::shared_ptr<Database> Open(const std::string& path,
                                        const Options& options = {});

  // Explicitly close the database before destruction.
  virtual void Close() = 0;

  // Check if entry already exists in database. For image pairs, the order of
  // `image_id1` and `image_id2` does not matter.
  virtual bool ExistsRig(rig_t rig_id) const = 0;
  virtual bool ExistsCamera(camera_t camera_id) const = 0;
  virtual bool ExistsFrame(frame_t frame_id) const = 0;
  virtual bool ExistsImage(image_t image_id) const = 0;
  virtual bool ExistsImageWithName(const std::string& name) const = 0;
  virtual bool ExistsPosePrior(image_t image_id) const = 0;
  virtual bool ExistsKeypoints(image_t image_id) const = 0;
  virtual bool ExistsDescriptors(image_t image_id) const = 0;
  virtual bool ExistsMatches(image_t image_id1, image_t image_id2) const = 0;
  virtual bool ExistsInlierMatches(image_t image_id1,
                                   image_t image_id2) const = 0;

  // Number of rows in `rigs` table.
  virtual size_t NumRigs() const = 0;

  // Number of rows in `cameras` table.
  virtual size_t NumCameras() const = 0;

  //  Number of rows in `frames` table.
  virtual size_t NumFrames() const = 0;

  //  Number of rows in `images` table.
  virtual size_t NumImages() const = 0;

  //  Number of rows in `pose_priors` table.
  virtual size_t NumPosePriors() const = 0;

  // Sum of `rows` column in `keypoints` table, i.e. number of total keypoints.
  virtual size_t NumKeypoints() const = 0;

  // The number of keypoints for the image with most features.
  virtual size_t MaxNumKeypoints() const = 0;

  // Number of descriptors for specific image.
  virtual size_t NumKeypointsForImage(image_t image_id) const = 0;

  // Sum of `rows` column in `descriptors` table,
  // i.e. number of total descriptors.
  virtual size_t NumDescriptors() const = 0;

  // The number of descriptors for the image with most features.
  virtual size_t MaxNumDescriptors() const = 0;

  // Number of descriptors for specific image.
  virtual size_t NumDescriptorsForImage(image_t image_id) const = 0;

  // Sum of `rows` column in `matches` table, i.e. number of total matches.
  virtual size_t NumMatches() const = 0;

  // Sum of `rows` column in `two_view_geometries` table,
  // i.e. number of total inlier matches.
  virtual size_t NumInlierMatches() const = 0;

  // Number of rows in `matches` table.
  virtual size_t NumMatchedImagePairs() const = 0;

  // Number of rows in `two_view_geometries` table.
  virtual size_t NumVerifiedImagePairs() const = 0;

  // Read an existing entry in the database. The user is responsible for making
  // sure that the entry actually exists. For image pairs, the order of
  // `image_id1` and `image_id2` does not matter.

  virtual Rig ReadRig(rig_t rig_id) const = 0;
  virtual std::optional<Rig> ReadRigWithSensor(sensor_t sensor_id) const = 0;
  virtual std::vector<Rig> ReadAllRigs() const = 0;

  virtual Camera ReadCamera(camera_t camera_id) const = 0;
  virtual std::vector<Camera> ReadAllCameras() const = 0;

  virtual Frame ReadFrame(frame_t frame_id) const = 0;
  virtual std::vector<Frame> ReadAllFrames() const = 0;

  virtual Image ReadImage(image_t image_id) const = 0;
  virtual std::optional<Image> ReadImageWithName(
      const std::string& name) const = 0;
  virtual std::vector<Image> ReadAllImages() const = 0;

  virtual PosePrior ReadPosePrior(image_t image_id) const = 0;

  virtual FeatureKeypointsBlob ReadKeypointsBlob(image_t image_id) const = 0;
  virtual FeatureKeypoints ReadKeypoints(image_t image_id) const = 0;
  virtual FeatureDescriptors ReadDescriptors(image_t image_id) const = 0;

  virtual FeatureMatchesBlob ReadMatchesBlob(image_t image_id1,
                                             image_t image_id2) const = 0;
  virtual FeatureMatches ReadMatches(image_t image_id1,
                                     image_t image_id2) const = 0;
  virtual std::vector<std::pair<image_pair_t, FeatureMatchesBlob>>
  ReadAllMatchesBlob() const = 0;
  virtual std::vector<std::pair<image_pair_t, FeatureMatches>> ReadAllMatches()
      const = 0;
  virtual std::vector<std::pair<image_pair_t, int>> ReadNumMatches() const = 0;

  virtual TwoViewGeometry ReadTwoViewGeometry(image_t image_id1,
                                              image_t image_id2) const = 0;
  virtual std::vector<std::pair<image_pair_t, TwoViewGeometry>>
  ReadTwoViewGeometries() const = 0;

  // Read all image pairs that have an entry in the `two_view_geometry`
  // table with at least one inlier match and their number of inlier matches.
  virtual std::vector<std::pair<image_pair_t, int>>
  ReadTwoViewGeometryNumInliers() const = 0;

  // Add new rig and return its database identifier. If `use_rig_id`
  // is false a new identifier is automatically generated.
  virtual rig_t WriteRig(const Rig& rig, bool use_rig_id = false) = 0;

  // Add new camera and return its database identifier. If `use_camera_id`
  // is false a new identifier is automatically generated.
  virtual camera_t WriteCamera(const Camera& camera,
                               bool use_camera_id = false) = 0;

  // Add new frame and return its database identifier. If `use_frame_id`
  // is false a new identifier is automatically generated.
  virtual frame_t WriteFrame(const Frame& frame, bool use_frame_id = false) = 0;

  // Add new image and return its database identifier. If `use_image_id`
  // is false a new identifier is automatically generated.
  virtual image_t WriteImage(const Image& image, bool use_image_id = false) = 0;

  // Write a new entry in the database. The user is responsible for making sure
  // that the entry does not yet exist. For image pairs, the order of
  // `image_id1` and `image_id2` does not matter.
  virtual void WritePosePrior(image_t image_id,
                              const PosePrior& pose_prior) = 0;
  virtual void WriteKeypoints(image_t image_id,
                              const FeatureKeypoints& keypoints) = 0;
  virtual void WriteKeypoints(image_t image_id,
                              const FeatureKeypointsBlob& blob) = 0;
  virtual void WriteDescriptors(image_t image_id,
                                const FeatureDescriptors& descriptors) = 0;
  virtual void WriteMatches(image_t image_id1,
                            image_t image_id2,
                            const FeatureMatches& matches) = 0;
  virtual void WriteMatches(image_t image_id1,
                            image_t image_id2,
                            const FeatureMatchesBlob& blob) = 0;
  virtual void WriteTwoViewGeometry(
      image_t image_id1,
      image_t image_id2,
      const TwoViewGeometry& two_view_geometry) = 0;

  // Update an existing rig in the database. The user is responsible for
  // making sure that the entry already exists.
  virtual void UpdateRig(const Rig& rig) = 0;

  // Update an existing camera in the database. The user is responsible for
  // making sure that the entry already exists.
  virtual void UpdateCamera(const Camera& camera) = 0;

  // Update an existing frame in the database. The user is responsible for
  // making sure that the entry already exists.
  virtual void UpdateFrame(const Frame& frame) = 0;

  // Update an existing image in the database. The user is responsible for
  // making sure that the entry already exists.
  virtual void UpdateImage(const Image& image) = 0;

  // Update an existing pose_prior in the database. The user is responsible for
  // making sure that the entry already exists.
  virtual void UpdatePosePrior(image_t image_id,
                               const PosePrior& pose_prior) = 0;

  // Update an existing image's keypoints in the database. The user is
  // responsible for making sure that the entry already exists.
  virtual void UpdateKeypoints(image_t image_id,
                               const FeatureKeypoints& keypoints) = 0;
  virtual void UpdateKeypoints(image_t image_id,
                               const FeatureKeypointsBlob& blob) = 0;

  // Delete matches of an image pair.
  virtual void DeleteMatches(image_t image_id1, image_t image_id2) = 0;

  // Delete inlier matches of an image pair.
  virtual void DeleteInlierMatches(image_t image_id1, image_t image_id2) = 0;

  // Clear all database tables
  virtual void ClearAllTables() = 0;

  // Clear the entire rigs table
  virtual void ClearRigs() = 0;

  // Clear the entire cameras table
  virtual void ClearCameras() = 0;

  // Clear the entire frames table
  virtual void ClearFrames() = 0;

  // Clear the entire images, keypoints, and descriptors tables
  virtual void ClearImages() = 0;

  // Clear the entire pose_priors table
  virtual void ClearPosePriors() = 0;

  // Clear the entire descriptors table
  virtual void ClearDescriptors() = 0;

  // Clear the entire keypoints table
  virtual void ClearKeypoints() = 0;

  // Clear the entire matches table.
  virtual void ClearMatches() = 0;

  // Clear the entire inlier matches table.
  virtual void ClearTwoViewGeometries() = 0;

  // Merge two databases into a single, new database.
  static void Merge(const Database& database1,
                    const Database& database2,
                    Database* merged_database);

  // Combine multiple queries into one transaction by wrapping a code section
  // into a `BeginTransaction` and `EndTransaction`. You can create a scoped
  // transaction with `DatabaseTransaction` that ends when the transaction
  // object is destructed. Depending on the database implementation, combining
  // queries results in faster transaction time due to reduced locking of the
  // database, etc.
  virtual void BeginTransaction() const = 0;
  virtual void EndTransaction() const = 0;

 private:
  friend class DatabaseTransaction;

  // Used to ensure that only one transaction is active at the same time.
  std::mutex transaction_mutex_;

  static std::vector<Factory> factories_;
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

}  // namespace colmap
