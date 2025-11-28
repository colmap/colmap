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

#include "colmap/scene/database.h"
#include "colmap/util/endian.h"
#include "colmap/util/string.h"

#include <sqlite3.h>

namespace colmap {
namespace {

inline int SQLite3CallHelper(int result_code,
                             const std::string& filename,
                             int line) {
  switch (result_code) {
    case SQLITE_OK:
    case SQLITE_ROW:
    case SQLITE_DONE:
      return result_code;
    default:
      LogMessageFatalThrow<std::runtime_error>(filename.c_str(), line).stream()
          << "SQLite error: " << sqlite3_errstr(result_code);
      return result_code;
  }
}

#define SQLITE3_CALL(func) SQLite3CallHelper(func, __FILE__, __LINE__)

#define SQLITE3_EXEC(database, sql, callback)                             \
  {                                                                       \
    char* err_msg = nullptr;                                              \
    const int result_code = sqlite3_exec(                                 \
        THROW_CHECK_NOTNULL(database), sql, callback, nullptr, &err_msg); \
    if (result_code != SQLITE_OK) {                                       \
      LOG(ERROR) << "SQLite error [" << __FILE__ << ", line " << __LINE__ \
                 << "]: " << err_msg;                                     \
      sqlite3_free(err_msg);                                              \
    }                                                                     \
  }

struct Sqlite3StmtContext {
  explicit Sqlite3StmtContext(sqlite3_stmt* sql_stmt)
      : sql_stmt_(THROW_CHECK_NOTNULL(sql_stmt)) {}
  ~Sqlite3StmtContext() { SQLITE3_CALL(sqlite3_reset(sql_stmt_)); }

 private:
  sqlite3_stmt* sql_stmt_;
};

void SwapFeatureMatchesBlob(FeatureMatchesBlob* matches) {
  for (Eigen::Index i = 0; i < matches->rows(); ++i) {
    std::swap((*matches)(i, 0), (*matches)(i, 1));
  }
}

FeatureKeypointsBlob FeatureKeypointsToBlob(const FeatureKeypoints& keypoints) {
  const FeatureKeypointsBlob::Index kNumCols = 6;
  FeatureKeypointsBlob blob(keypoints.size(), kNumCols);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    blob(i, 0) = keypoints[i].x;
    blob(i, 1) = keypoints[i].y;
    blob(i, 2) = keypoints[i].a11;
    blob(i, 3) = keypoints[i].a12;
    blob(i, 4) = keypoints[i].a21;
    blob(i, 5) = keypoints[i].a22;
  }
  return blob;
}

FeatureKeypoints FeatureKeypointsFromBlob(const FeatureKeypointsBlob& blob) {
  FeatureKeypoints keypoints(static_cast<size_t>(blob.rows()));
  if (blob.cols() == 2) {
    for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
      keypoints[i] = FeatureKeypoint(blob(i, 0), blob(i, 1));
    }
  } else if (blob.cols() == 4) {
    for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
      keypoints[i] =
          FeatureKeypoint(blob(i, 0), blob(i, 1), blob(i, 2), blob(i, 3));
    }
  } else if (blob.cols() == 6) {
    for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
      keypoints[i] = FeatureKeypoint(blob(i, 0),
                                     blob(i, 1),
                                     blob(i, 2),
                                     blob(i, 3),
                                     blob(i, 4),
                                     blob(i, 5));
    }
  } else {
    LOG(FATAL_THROW) << "Keypoint format not supported";
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
  THROW_CHECK_EQ(blob.cols(), 2);
  FeatureMatches matches(static_cast<size_t>(blob.rows()));
  for (FeatureMatchesBlob::Index i = 0; i < blob.rows(); ++i) {
    matches[i].point2D_idx1 = blob(i, 0);
    matches[i].point2D_idx2 = blob(i, 1);
  }
  return matches;
}

template <typename MatrixType>
MatrixType ReadStaticMatrixBlob(sqlite3_stmt* sql_stmt,
                                const int rc,
                                const int col) {
  THROW_CHECK_GE(col, 0);

  MatrixType matrix;

  if (rc == SQLITE_ROW) {
    const size_t num_bytes =
        static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col));
    if (num_bytes > 0) {
      THROW_CHECK_EQ(num_bytes,
                     matrix.size() * sizeof(typename MatrixType::Scalar));
      memcpy(reinterpret_cast<char*>(matrix.data()),
             sqlite3_column_blob(sql_stmt, col),
             num_bytes);
    } else {
      matrix = MatrixType::Zero();
    }
  } else {
    matrix = MatrixType::Zero();
  }

  return matrix;
}

template <typename MatrixType>
MatrixType ReadDynamicMatrixBlob(sqlite3_stmt* sql_stmt,
                                 const int rc,
                                 const int col) {
  THROW_CHECK_GE(col, 0);

  MatrixType matrix;

  if (rc == SQLITE_ROW) {
    const size_t rows =
        static_cast<size_t>(sqlite3_column_int64(sql_stmt, col + 0));
    const size_t cols =
        static_cast<size_t>(sqlite3_column_int64(sql_stmt, col + 1));

    THROW_CHECK_GE(rows, 0);
    THROW_CHECK_GE(cols, 0);
    matrix = MatrixType(rows, cols);

    const size_t num_bytes =
        static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col + 2));
    THROW_CHECK_EQ(matrix.size() * sizeof(typename MatrixType::Scalar),
                   num_bytes);

    memcpy(reinterpret_cast<char*>(matrix.data()),
           sqlite3_column_blob(sql_stmt, col + 2),
           num_bytes);
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
void WriteStaticMatrixBlob(sqlite3_stmt* sql_stmt,
                           const MatrixType& matrix,
                           const int col) {
  SQLITE3_CALL(sqlite3_bind_blob(
      sql_stmt,
      col,
      reinterpret_cast<const char*>(matrix.data()),
      static_cast<int>(matrix.size() * sizeof(typename MatrixType::Scalar)),
      SQLITE_STATIC));
}

template <typename MatrixType>
void WriteDynamicMatrixBlob(sqlite3_stmt* sql_stmt,
                            const MatrixType& matrix,
                            const int col) {
  THROW_CHECK_GE(matrix.rows(), 0);
  THROW_CHECK_GE(matrix.cols(), 0);
  THROW_CHECK_GE(col, 0);

  const size_t num_bytes = matrix.size() * sizeof(typename MatrixType::Scalar);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, col + 0, matrix.rows()));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, col + 1, matrix.cols()));
  SQLITE3_CALL(sqlite3_bind_blob(sql_stmt,
                                 col + 2,
                                 reinterpret_cast<const char*>(matrix.data()),
                                 static_cast<int>(num_bytes),
                                 SQLITE_STATIC));
}

std::optional<std::stringstream> BlobColumnToStringStream(
    sqlite3_stmt* sql_stmt, const int col) {
  const size_t num_bytes =
      static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col));
  if (num_bytes == 0) {
    return std::nullopt;
  }
  std::string sensor_data(num_bytes, '\0');
  std::memcpy(
      sensor_data.data(), sqlite3_column_blob(sql_stmt, col), num_bytes);
  std::stringstream stream;
  stream << sensor_data;
  return stream;
}

Rigid3d ReadRigid3dFromStringStream(std::stringstream* stream) {
  Rigid3d tform;
  tform.rotation.w() = ReadBinaryLittleEndian<double>(stream);
  tform.rotation.x() = ReadBinaryLittleEndian<double>(stream);
  tform.rotation.y() = ReadBinaryLittleEndian<double>(stream);
  tform.rotation.z() = ReadBinaryLittleEndian<double>(stream);
  tform.translation.x() = ReadBinaryLittleEndian<double>(stream);
  tform.translation.y() = ReadBinaryLittleEndian<double>(stream);
  tform.translation.z() = ReadBinaryLittleEndian<double>(stream);
  return tform;
}

void WriteRigid3dToStringStream(const Rigid3d& tform,
                                std::stringstream* stream) {
  WriteBinaryLittleEndian<double>(stream, tform.rotation.w());
  WriteBinaryLittleEndian<double>(stream, tform.rotation.x());
  WriteBinaryLittleEndian<double>(stream, tform.rotation.y());
  WriteBinaryLittleEndian<double>(stream, tform.rotation.z());
  WriteBinaryLittleEndian<double>(stream, tform.translation.x());
  WriteBinaryLittleEndian<double>(stream, tform.translation.y());
  WriteBinaryLittleEndian<double>(stream, tform.translation.z());
}

void ReadRigRows(sqlite3_stmt* sql_stmt,
                 const std::function<void(Rig)>& new_rig_callback) {
  Rig rig;
  while (SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW) {
    const rig_t rig_id = static_cast<rig_t>(sqlite3_column_int64(sql_stmt, 0));
    if (rig_id != rig.RigId()) {
      if (rig.RigId() != kInvalidRigId) {
        new_rig_callback(std::move(rig));
      }
      rig = Rig();
      rig.SetRigId(rig_id);
      sensor_t ref_sensor_id;
      ref_sensor_id.id =
          static_cast<uint32_t>(sqlite3_column_int64(sql_stmt, 1));
      ref_sensor_id.type =
          static_cast<SensorType>(sqlite3_column_int64(sql_stmt, 2));
      rig.AddRefSensor(ref_sensor_id);
    }

    if (sqlite3_column_type(sql_stmt, 3) == SQLITE_NULL) {
      // No non-reference sensors for rig.
      continue;
    }

    sensor_t sensor_id;
    sensor_id.id = static_cast<uint32_t>(sqlite3_column_int64(sql_stmt, 3));
    sensor_id.type = static_cast<SensorType>(sqlite3_column_int64(sql_stmt, 4));

    std::optional<std::stringstream> sensor_from_rig_stream =
        BlobColumnToStringStream(sql_stmt, 5);

    std::optional<Rigid3d> sensor_from_rig;
    if (sensor_from_rig_stream.has_value()) {
      sensor_from_rig =
          ReadRigid3dFromStringStream(&sensor_from_rig_stream.value());
    }

    rig.AddSensor(sensor_id, sensor_from_rig);
  }

  if (rig.RigId() != kInvalidRigId) {
    new_rig_callback(std::move(rig));
  }
}

Camera ReadCameraRow(sqlite3_stmt* sql_stmt) {
  Camera camera;

  camera.camera_id = static_cast<camera_t>(sqlite3_column_int64(sql_stmt, 0));
  camera.model_id =
      static_cast<CameraModelId>(sqlite3_column_int64(sql_stmt, 1));
  camera.width = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 2));
  camera.height = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 3));

  const size_t num_params_bytes =
      static_cast<size_t>(sqlite3_column_bytes(sql_stmt, 4));
  const size_t num_params = num_params_bytes / sizeof(double);
  THROW_CHECK_EQ(num_params, CameraModelNumParams(camera.model_id));
  camera.params.resize(num_params, 0.);
  std::memcpy(
      camera.params.data(), sqlite3_column_blob(sql_stmt, 4), num_params_bytes);

  camera.has_prior_focal_length = sqlite3_column_int64(sql_stmt, 5) != 0;

  return camera;
}

void ReadFrameRows(sqlite3_stmt* sql_stmt,
                   const std::function<void(Frame)>& new_frame_callback) {
  Frame frame;
  while (SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW) {
    const frame_t frame_id =
        static_cast<rig_t>(sqlite3_column_int64(sql_stmt, 0));
    if (frame_id != frame.FrameId()) {
      if (frame.FrameId() != kInvalidFrameId) {
        new_frame_callback(std::move(frame));
      }
      frame = Frame();
      frame.SetFrameId(frame_id);
      frame.SetRigId(static_cast<rig_t>(sqlite3_column_int64(sql_stmt, 1)));
    }

    if (sqlite3_column_type(sql_stmt, 2) == SQLITE_NULL) {
      // No data for frame.
      continue;
    }

    data_t data_id;
    data_id.id = static_cast<uint32_t>(sqlite3_column_int64(sql_stmt, 2));
    data_id.sensor_id.id =
        static_cast<uint32_t>(sqlite3_column_int64(sql_stmt, 3));
    data_id.sensor_id.type =
        static_cast<SensorType>(sqlite3_column_int64(sql_stmt, 4));
    frame.AddDataId(data_id);
  }

  if (frame.FrameId() != kInvalidFrameId) {
    new_frame_callback(std::move(frame));
  }
}

Image ReadImageRow(sqlite3_stmt* sql_stmt) {
  Image image;

  image.SetImageId(static_cast<image_t>(sqlite3_column_int64(sql_stmt, 0)));
  image.SetName(std::string(
      reinterpret_cast<const char*>(sqlite3_column_text(sql_stmt, 1))));
  image.SetCameraId(static_cast<camera_t>(sqlite3_column_int64(sql_stmt, 2)));

  return image;
}

PosePrior ReadPosePriorRow(sqlite3_stmt* sql_stmt) {
  PosePrior pose_prior;
  pose_prior.pose_prior_id =
      static_cast<pose_prior_t>(sqlite3_column_int64(sql_stmt, 0));
  pose_prior.corr_data_id.id = sqlite3_column_int64(sql_stmt, 1);
  pose_prior.corr_data_id.sensor_id.id = sqlite3_column_int64(sql_stmt, 2);
  pose_prior.corr_data_id.sensor_id.type =
      static_cast<SensorType>(sqlite3_column_int64(sql_stmt, 3));
  pose_prior.position =
      ReadStaticMatrixBlob<Eigen::Vector3d>(sql_stmt, SQLITE_ROW, 4);
  pose_prior.position_covariance =
      ReadStaticMatrixBlob<Eigen::Matrix3d>(sql_stmt, SQLITE_ROW, 5);
  pose_prior.coordinate_system = static_cast<PosePrior::CoordinateSystem>(
      sqlite3_column_int64(sql_stmt, 6));
  return pose_prior;
}

void WriteRigSensors(const rig_t rig_id,
                     const Rig& rig,
                     sqlite3_stmt* sql_stmt) {
  for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
    Sqlite3StmtContext context(sql_stmt);

    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, 1, rig_id));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt, 2, static_cast<sqlite3_int64>(sensor_id.id)));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt, 3, static_cast<sqlite3_int64>(sensor_id.type)));

    // Note that the sensor_from_rig_bytes object must outlive the call to
    // sqlite3_step.
    std::string sensor_from_rig_bytes;
    if (sensor_from_rig.has_value()) {
      std::stringstream stream;
      WriteRigid3dToStringStream(*sensor_from_rig, &stream);
      sensor_from_rig_bytes = stream.str();
      SQLITE3_CALL(
          sqlite3_bind_blob(sql_stmt,
                            4,
                            sensor_from_rig_bytes.data(),
                            static_cast<int>(sensor_from_rig_bytes.size()),
                            SQLITE_STATIC));
    } else {
      SQLITE3_CALL(sqlite3_bind_null(sql_stmt, 4));
    }

    SQLITE3_CALL(sqlite3_step(sql_stmt));
  }
}

void WriteFrameData(const frame_t frame_id,
                    const Frame& frame,
                    sqlite3_stmt* sql_stmt) {
  for (const data_t& data_id : frame.DataIds()) {
    Sqlite3StmtContext context(sql_stmt);
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, 1, frame_id));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt, 2, static_cast<sqlite3_int64>(data_id.id)));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt, 3, static_cast<sqlite3_int64>(data_id.sensor_id.id)));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt, 4, static_cast<sqlite3_int64>(data_id.sensor_id.type)));
    SQLITE3_CALL(sqlite3_step(sql_stmt));
  }
}

// TODO(jsch): Change is_deprecated_image_prior default to true after next
// version release (3.14 or 4.0) and remove the parameter in (3.15 or 4.1).
void MaybeThrowDeprecatedPosePriorError(bool is_deprecated_image_prior) {
  if (is_deprecated_image_prior) {
    throw std::runtime_error(
        "PosePrior API has changed: pose priors are now associated with "
        "frames, not images. Please update your code to use frames "
        "instead of image IDs. Data is automatically migrated upon opening a "
        "database. Update your API usage accordingly and add pose priors to "
        "frame data.");
  }
}

class SqliteDatabase : public Database {
 public:
  SqliteDatabase() : database_(nullptr) {}

  // Open and close database. The same database should not be opened
  // concurrently in multiple threads or processes.
  //
  // On Windows, the input path is converted from the local code page to UTF-8
  // for compatibility with SQLite. On POSIX platforms, the path is assumed to
  // be UTF-8.
  static std::shared_ptr<Database> Open(const std::string& path) {
    auto database = std::make_shared<SqliteDatabase>();

    // SQLITE_OPEN_NOMUTEX specifies that the connection should not have a
    // mutex (so that we don't serialize the connection's operations).
    // Modifications to the database will still be serialized, but multiple
    // connections can read concurrently.
    try {
      SQLITE3_CALL(sqlite3_open_v2(
          PlatformToUTF8(path).c_str(),
          &database->database_,
          SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
          nullptr));
    } catch (...) {
      SQLITE3_CALL(sqlite3_close_v2(database->database_));
      throw;
    }

    // Don't wait for the operating system to write the changes to disk
    SQLITE3_EXEC(database->database_, "PRAGMA synchronous=OFF", nullptr);

    // Use faster journaling mode
    SQLITE3_EXEC(database->database_, "PRAGMA journal_mode=WAL", nullptr);

    // Store temporary tables and indices in memory
    SQLITE3_EXEC(database->database_, "PRAGMA temp_store=MEMORY", nullptr);

    // Disabled by default
    SQLITE3_EXEC(database->database_, "PRAGMA foreign_keys=ON", nullptr);

    // Enable auto vacuum to reduce DB file size
    SQLITE3_EXEC(database->database_, "PRAGMA auto_vacuum=1", nullptr);

    database->CreateTables();
    database->UpdateSchema();
    database->PrepareSQLStatements();

    return database;
  }

  void CloseImpl() {
    if (database_ != nullptr) {
      FinalizeSQLStatements();
      if (database_entry_deleted_) {
        SQLITE3_EXEC(database_, "VACUUM", nullptr);
        database_entry_deleted_ = false;
      }
      SQLITE3_CALL(sqlite3_close_v2(database_));
      database_ = nullptr;
    }
  }

  ~SqliteDatabase() override { CloseImpl(); }

  void Close() override { CloseImpl(); }

  bool ExistsRig(const rig_t rig_id) const override {
    return ExistsRowId(sql_stmt_exists_rig_, rig_id);
  }

  bool ExistsCamera(const camera_t camera_id) const override {
    return ExistsRowId(sql_stmt_exists_camera_, camera_id);
  }

  bool ExistsFrame(const frame_t frame_id) const override {
    return ExistsRowId(sql_stmt_exists_frame_, frame_id);
  }

  bool ExistsImage(const image_t image_id) const override {
    return ExistsRowId(sql_stmt_exists_image_id_, image_id);
  }

  bool ExistsImageWithName(const std::string& name) const override {
    return ExistsRowString(sql_stmt_exists_image_name_, name);
  }

  bool ExistsPosePrior(const pose_prior_t pose_prior_id,
                       bool is_deprecated_image_prior) const override {
    MaybeThrowDeprecatedPosePriorError(is_deprecated_image_prior);
    return ExistsRowId(sql_stmt_exists_pose_prior_, pose_prior_id);
  }

  bool ExistsKeypoints(const image_t image_id) const override {
    return ExistsRowId(sql_stmt_exists_keypoints_, image_id);
  }

  bool ExistsDescriptors(const image_t image_id) const override {
    return ExistsRowId(sql_stmt_exists_descriptors_, image_id);
  }

  bool ExistsMatches(const image_t image_id1,
                     const image_t image_id2) const override {
    return ExistsRowId(sql_stmt_exists_matches_,
                       ImagePairToPairId(image_id1, image_id2));
  }

  bool ExistsTwoViewGeometry(const image_t image_id1,
                             const image_t image_id2) const override {
    return ExistsRowId(sql_stmt_exists_two_view_geometry_,
                       ImagePairToPairId(image_id1, image_id2));
  }

  size_t NumRigs() const override { return CountRows("rigs"); }

  size_t NumCameras() const override { return CountRows("cameras"); }

  size_t NumFrames() const override { return CountRows("frames"); }

  size_t NumImages() const override { return CountRows("images"); }

  size_t NumPosePriors() const override { return CountRows("pose_priors"); }

  size_t NumKeypoints() const override {
    return SumColumn("rows", "keypoints");
  }

  size_t MaxNumKeypoints() const override {
    return MaxColumn("rows", "keypoints");
  }

  size_t NumKeypointsForImage(const image_t image_id) const override {
    return CountRowsForEntry(sql_stmt_num_keypoints_, image_id);
  }

  size_t NumDescriptors() const override {
    return SumColumn("rows", "descriptors");
  }

  size_t MaxNumDescriptors() const override {
    return MaxColumn("rows", "descriptors");
  }

  size_t NumDescriptorsForImage(const image_t image_id) const override {
    return CountRowsForEntry(sql_stmt_num_descriptors_, image_id);
  }

  size_t NumMatches() const override { return SumColumn("rows", "matches"); }

  size_t NumInlierMatches() const override {
    return SumColumn("rows", "two_view_geometries");
  }

  size_t NumMatchedImagePairs() const override { return CountRows("matches"); }

  size_t NumVerifiedImagePairs() const override {
    return CountRows("two_view_geometries");
  }

  Rig ReadRig(const rig_t rig_id) const override {
    Sqlite3StmtContext context(sql_stmt_read_rig_);

    Rig rig;

    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_rig_, 1, rig_id));
    ReadRigRows(sql_stmt_read_rig_,
                [&rig](Rig new_rig) { rig = std::move(new_rig); });

    return rig;
  }

  std::optional<Rig> ReadRigWithSensor(sensor_t sensor_id) const override {
    auto find_rig_id_with_sensor = [&sensor_id](sqlite3_stmt* sql_stmt) {
      Sqlite3StmtContext context(sql_stmt);
      SQLITE3_CALL(sqlite3_bind_int64(
          sql_stmt, 1, static_cast<sqlite3_int64>(sensor_id.id)));
      SQLITE3_CALL(sqlite3_bind_int64(
          sql_stmt, 2, static_cast<sqlite3_int64>(sensor_id.type)));
      std::optional<rig_t> rig_id;
      if (SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW) {
        rig_id = static_cast<rig_t>(sqlite3_column_int64(sql_stmt, 0));
      }
      return rig_id;
    };

    if (const std::optional<rig_t> rig_id =
            find_rig_id_with_sensor(sql_stmt_read_rig_with_sensor_);
        rig_id.has_value()) {
      return ReadRig(*rig_id);
    }

    if (const std::optional<rig_t> rig_id =
            find_rig_id_with_sensor(sql_stmt_read_rig_with_ref_sensor_);
        rig_id.has_value()) {
      return ReadRig(*rig_id);
    }

    return std::nullopt;
  }

  std::vector<Rig> ReadAllRigs() const override {
    Sqlite3StmtContext context(sql_stmt_read_rigs_);

    std::vector<Rig> rigs;

    ReadRigRows(sql_stmt_read_rigs_,
                [&rigs](Rig new_rig) { rigs.push_back(std::move(new_rig)); });

    return rigs;
  }

  Camera ReadCamera(const camera_t camera_id) const override {
    Sqlite3StmtContext context(sql_stmt_read_camera_);

    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_camera_, 1, camera_id));

    Camera camera;

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_camera_));
    if (rc == SQLITE_ROW) {
      camera = ReadCameraRow(sql_stmt_read_camera_);
    }

    return camera;
  }

  std::vector<Camera> ReadAllCameras() const override {
    Sqlite3StmtContext context(sql_stmt_read_cameras_);

    std::vector<Camera> cameras;

    while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_cameras_)) == SQLITE_ROW) {
      cameras.push_back(ReadCameraRow(sql_stmt_read_cameras_));
    }

    return cameras;
  }

  Frame ReadFrame(const frame_t frame_id) const override {
    Sqlite3StmtContext context(sql_stmt_read_frame_);

    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_frame_, 1, frame_id));

    Frame frame;
    ReadFrameRows(sql_stmt_read_frame_, [&frame](Frame new_frame) {
      THROW_CHECK_EQ(frame.FrameId(), kInvalidFrameId);
      frame = std::move(new_frame);
    });

    return frame;
  }

  std::vector<Frame> ReadAllFrames() const override {
    Sqlite3StmtContext context(sql_stmt_read_frames_);

    std::vector<Frame> frames;

    ReadFrameRows(sql_stmt_read_frames_, [&frames](Frame new_frame) {
      frames.push_back(std::move(new_frame));
    });

    return frames;
  }

  Image ReadImage(const image_t image_id) const override {
    Sqlite3StmtContext context(sql_stmt_read_image_id_);

    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_image_id_, 1, image_id));

    Image image;

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_image_id_));
    if (rc == SQLITE_ROW) {
      image = ReadImageRow(sql_stmt_read_image_id_);
    }

    return image;
  }

  std::optional<Image> ReadImageWithName(
      const std::string& name) const override {
    Sqlite3StmtContext context(sql_stmt_read_image_with_name_);

    SQLITE3_CALL(sqlite3_bind_text(sql_stmt_read_image_with_name_,
                                   1,
                                   name.c_str(),
                                   static_cast<int>(name.size()),
                                   SQLITE_STATIC));

    std::optional<Image> image;
    if (SQLITE3_CALL(sqlite3_step(sql_stmt_read_image_with_name_)) ==
        SQLITE_ROW) {
      image = ReadImageRow(sql_stmt_read_image_with_name_);
    }

    return image;
  }

  std::vector<Image> ReadAllImages() const override {
    Sqlite3StmtContext context(sql_stmt_read_images_);

    std::vector<Image> images;
    images.reserve(NumImages());

    while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_images_)) == SQLITE_ROW) {
      images.push_back(ReadImageRow(sql_stmt_read_images_));
    }

    return images;
  }

  PosePrior ReadPosePrior(const pose_prior_t pose_prior_id,
                          bool is_deprecated_image_prior) const override {
    MaybeThrowDeprecatedPosePriorError(is_deprecated_image_prior);

    Sqlite3StmtContext context(sql_stmt_read_pose_prior_);

    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_read_pose_prior_, 1, pose_prior_id));

    PosePrior pose_prior;
    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_pose_prior_));
    if (rc == SQLITE_ROW) {
      pose_prior = ReadPosePriorRow(sql_stmt_read_pose_prior_);
    }
    return pose_prior;
  }

  std::vector<PosePrior> ReadAllPosePriors() const override {
    Sqlite3StmtContext context(sql_stmt_read_pose_priors_);

    std::vector<PosePrior> pose_priors;
    while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_pose_priors_)) ==
           SQLITE_ROW) {
      pose_priors.push_back(ReadPosePriorRow(sql_stmt_read_pose_priors_));
    }

    return pose_priors;
  }

  FeatureKeypointsBlob ReadKeypointsBlob(
      const image_t image_id) const override {
    Sqlite3StmtContext context(sql_stmt_read_keypoints_);

    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_keypoints_, 1, image_id));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_keypoints_));
    FeatureKeypointsBlob blob = ReadDynamicMatrixBlob<FeatureKeypointsBlob>(
        sql_stmt_read_keypoints_, rc, 0);

    return blob;
  }

  FeatureKeypoints ReadKeypoints(const image_t image_id) const override {
    return FeatureKeypointsFromBlob(ReadKeypointsBlob(image_id));
  }

  FeatureDescriptors ReadDescriptors(const image_t image_id) const override {
    Sqlite3StmtContext context(sql_stmt_read_descriptors_);

    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_descriptors_, 1, image_id));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_descriptors_));
    FeatureDescriptors descriptors = ReadDynamicMatrixBlob<FeatureDescriptors>(
        sql_stmt_read_descriptors_, rc, 0);

    return descriptors;
  }

  FeatureMatchesBlob ReadMatchesBlob(image_t image_id1,
                                     image_t image_id2) const override {
    Sqlite3StmtContext context(sql_stmt_read_matches_);

    const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_matches_, 1, pair_id));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_));
    FeatureMatchesBlob blob = ReadDynamicMatrixBlob<FeatureMatchesBlob>(
        sql_stmt_read_matches_, rc, 0);

    if (SwapImagePair(image_id1, image_id2)) {
      SwapFeatureMatchesBlob(&blob);
    }
    return blob;
  }

  FeatureMatches ReadMatches(image_t image_id1,
                             image_t image_id2) const override {
    return FeatureMatchesFromBlob(ReadMatchesBlob(image_id1, image_id2));
  }

  std::vector<std::pair<image_pair_t, FeatureMatchesBlob>> ReadAllMatchesBlob()
      const override {
    Sqlite3StmtContext context(sql_stmt_read_matches_all_);

    std::vector<std::pair<image_pair_t, FeatureMatchesBlob>> all_matches;

    int rc;
    while ((rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_all_))) ==
           SQLITE_ROW) {
      const image_pair_t pair_id = static_cast<image_pair_t>(
          sqlite3_column_int64(sql_stmt_read_matches_all_, 0));
      all_matches.emplace_back(pair_id,
                               ReadDynamicMatrixBlob<FeatureMatchesBlob>(
                                   sql_stmt_read_matches_all_, rc, 1));
    }

    return all_matches;
  }

  std::vector<std::pair<image_pair_t, FeatureMatches>> ReadAllMatches()
      const override {
    Sqlite3StmtContext context(sql_stmt_read_matches_all_);

    std::vector<std::pair<image_pair_t, FeatureMatches>> all_matches;

    int rc;
    while ((rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_all_))) ==
           SQLITE_ROW) {
      const image_pair_t pair_id = static_cast<image_pair_t>(
          sqlite3_column_int64(sql_stmt_read_matches_all_, 0));
      const FeatureMatchesBlob blob = ReadDynamicMatrixBlob<FeatureMatchesBlob>(
          sql_stmt_read_matches_all_, rc, 1);
      all_matches.emplace_back(pair_id, FeatureMatchesFromBlob(blob));
    }

    return all_matches;
  }

  std::vector<std::pair<image_pair_t, int>> ReadNumMatches() const override {
    Sqlite3StmtContext context(sql_stmt_read_num_matches_);

    std::vector<std::pair<image_pair_t, int>> num_matches;
    while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_num_matches_)) ==
           SQLITE_ROW) {
      const image_pair_t pair_id = static_cast<image_pair_t>(
          sqlite3_column_int64(sql_stmt_read_num_matches_, 0));

      const int rows =
          static_cast<int>(sqlite3_column_int64(sql_stmt_read_num_matches_, 1));
      num_matches.emplace_back(pair_id, rows);
    }

    return num_matches;
  }

  TwoViewGeometry ReadTwoViewGeometry(const image_t image_id1,
                                      const image_t image_id2) const override {
    Sqlite3StmtContext context(sql_stmt_read_two_view_geometry_);

    const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_read_two_view_geometry_, 1, pair_id));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_two_view_geometry_));

    TwoViewGeometry two_view_geometry;

    FeatureMatchesBlob blob = ReadDynamicMatrixBlob<FeatureMatchesBlob>(
        sql_stmt_read_two_view_geometry_, rc, 0);

    two_view_geometry.config = static_cast<int>(
        sqlite3_column_int64(sql_stmt_read_two_view_geometry_, 3));

    two_view_geometry.F = ReadStaticMatrixBlob<Eigen::Matrix3d>(
        sql_stmt_read_two_view_geometry_, rc, 4);
    two_view_geometry.E = ReadStaticMatrixBlob<Eigen::Matrix3d>(
        sql_stmt_read_two_view_geometry_, rc, 5);
    two_view_geometry.H = ReadStaticMatrixBlob<Eigen::Matrix3d>(
        sql_stmt_read_two_view_geometry_, rc, 6);
    const Eigen::Vector4d quat_wxyz = ReadStaticMatrixBlob<Eigen::Vector4d>(
        sql_stmt_read_two_view_geometry_, rc, 7);
    two_view_geometry.cam2_from_cam1.rotation = Eigen::Quaterniond(
        quat_wxyz(0), quat_wxyz(1), quat_wxyz(2), quat_wxyz(3));
    two_view_geometry.cam2_from_cam1.translation =
        ReadStaticMatrixBlob<Eigen::Vector3d>(
            sql_stmt_read_two_view_geometry_, rc, 8);

    two_view_geometry.inlier_matches = FeatureMatchesFromBlob(blob);
    two_view_geometry.F.transposeInPlace();
    two_view_geometry.E.transposeInPlace();
    two_view_geometry.H.transposeInPlace();

    if (SwapImagePair(image_id1, image_id2)) {
      two_view_geometry.Invert();
    }

    return two_view_geometry;
  }

  std::vector<std::pair<image_pair_t, TwoViewGeometry>> ReadTwoViewGeometries()
      const override {
    Sqlite3StmtContext context(sql_stmt_read_two_view_geometries_);

    std::vector<std::pair<image_pair_t, TwoViewGeometry>>
        all_two_view_geometries;

    int rc;
    while ((rc = SQLITE3_CALL(sqlite3_step(
                sql_stmt_read_two_view_geometries_))) == SQLITE_ROW) {
      const image_pair_t pair_id = static_cast<image_pair_t>(
          sqlite3_column_int64(sql_stmt_read_two_view_geometries_, 0));

      TwoViewGeometry two_view_geometry;

      const FeatureMatchesBlob blob = ReadDynamicMatrixBlob<FeatureMatchesBlob>(
          sql_stmt_read_two_view_geometries_, rc, 1);
      two_view_geometry.inlier_matches = FeatureMatchesFromBlob(blob);

      two_view_geometry.config = static_cast<int>(
          sqlite3_column_int64(sql_stmt_read_two_view_geometries_, 4));

      two_view_geometry.F = ReadStaticMatrixBlob<Eigen::Matrix3d>(
          sql_stmt_read_two_view_geometries_, rc, 5);
      two_view_geometry.E = ReadStaticMatrixBlob<Eigen::Matrix3d>(
          sql_stmt_read_two_view_geometries_, rc, 6);
      two_view_geometry.H = ReadStaticMatrixBlob<Eigen::Matrix3d>(
          sql_stmt_read_two_view_geometries_, rc, 7);
      const Eigen::Vector4d quat_wxyz = ReadStaticMatrixBlob<Eigen::Vector4d>(
          sql_stmt_read_two_view_geometries_, rc, 8);
      two_view_geometry.cam2_from_cam1.rotation = Eigen::Quaterniond(
          quat_wxyz(0), quat_wxyz(1), quat_wxyz(2), quat_wxyz(3));
      two_view_geometry.cam2_from_cam1.translation =
          ReadStaticMatrixBlob<Eigen::Vector3d>(
              sql_stmt_read_two_view_geometries_, rc, 9);

      two_view_geometry.F.transposeInPlace();
      two_view_geometry.E.transposeInPlace();
      two_view_geometry.H.transposeInPlace();

      all_two_view_geometries.emplace_back(pair_id,
                                           std::move(two_view_geometry));
    }

    return all_two_view_geometries;
  }

  std::vector<std::pair<image_pair_t, int>> ReadTwoViewGeometryNumInliers()
      const override {
    Sqlite3StmtContext context(sql_stmt_read_two_view_geometry_num_inliers_);

    std::vector<std::pair<image_pair_t, int>> num_inliers;
    while (SQLITE3_CALL(sqlite3_step(
               sql_stmt_read_two_view_geometry_num_inliers_)) == SQLITE_ROW) {
      const image_pair_t pair_id =
          static_cast<image_pair_t>(sqlite3_column_int64(
              sql_stmt_read_two_view_geometry_num_inliers_, 0));

      const int rows = static_cast<int>(sqlite3_column_int64(
          sql_stmt_read_two_view_geometry_num_inliers_, 1));
      num_inliers.emplace_back(pair_id, rows);
    }

    return num_inliers;
  }

  rig_t WriteRig(const Rig& rig, const bool use_rig_id) override {
    THROW_CHECK(rig.NumSensors() > 0) << "Rig must have at least one sensor";

    Sqlite3StmtContext context(sql_stmt_write_rig_);

    if (use_rig_id) {
      THROW_CHECK(!ExistsRig(rig.RigId())) << "rig_id must be unique";
      SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_rig_, 1, rig.RigId()));
    } else {
      SQLITE3_CALL(sqlite3_bind_null(sql_stmt_write_rig_, 1));
    }

    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_write_rig_,
                           2,
                           static_cast<sqlite3_int64>(rig.RefSensorId().id)));
    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_write_rig_,
                           3,
                           static_cast<sqlite3_int64>(rig.RefSensorId().type)));

    SQLITE3_CALL(sqlite3_step(sql_stmt_write_rig_));

    const rig_t rig_id = static_cast<rig_t>(
        sqlite3_last_insert_rowid(THROW_CHECK_NOTNULL(database_)));

    WriteRigSensors(rig_id, rig, sql_stmt_write_rig_sensor_);

    return rig_id;
  }

  camera_t WriteCamera(const Camera& camera,
                       const bool use_camera_id) override {
    Sqlite3StmtContext context(sql_stmt_write_camera_);

    if (use_camera_id) {
      THROW_CHECK(!ExistsCamera(camera.camera_id))
          << "camera_id must be unique";
      SQLITE3_CALL(
          sqlite3_bind_int64(sql_stmt_write_camera_, 1, camera.camera_id));
    } else {
      SQLITE3_CALL(sqlite3_bind_null(sql_stmt_write_camera_, 1));
    }

    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_write_camera_,
                           2,
                           static_cast<sqlite3_int64>(camera.model_id)));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_write_camera_, 3, static_cast<sqlite3_int64>(camera.width)));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_write_camera_, 4, static_cast<sqlite3_int64>(camera.height)));

    const size_t num_params_bytes = sizeof(double) * camera.params.size();
    SQLITE3_CALL(sqlite3_bind_blob(sql_stmt_write_camera_,
                                   5,
                                   camera.params.data(),
                                   static_cast<int>(num_params_bytes),
                                   SQLITE_STATIC));

    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_write_camera_, 6, camera.has_prior_focal_length));

    SQLITE3_CALL(sqlite3_step(sql_stmt_write_camera_));

    return static_cast<camera_t>(
        sqlite3_last_insert_rowid(THROW_CHECK_NOTNULL(database_)));
  }

  frame_t WriteFrame(const Frame& frame, const bool use_frame_id) override {
    Sqlite3StmtContext context(sql_stmt_write_frame_);

    if (use_frame_id) {
      THROW_CHECK(!ExistsFrame(frame.FrameId())) << "frame_id must be unique";
      SQLITE3_CALL(
          sqlite3_bind_int64(sql_stmt_write_frame_, 1, frame.FrameId()));
    } else {
      SQLITE3_CALL(sqlite3_bind_null(sql_stmt_write_frame_, 1));
    }

    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_frame_, 2, frame.RigId()));

    SQLITE3_CALL(sqlite3_step(sql_stmt_write_frame_));

    const frame_t frame_id = static_cast<frame_t>(
        sqlite3_last_insert_rowid(THROW_CHECK_NOTNULL(database_)));

    WriteFrameData(frame_id, frame, sql_stmt_write_frame_data_);

    return frame_id;
  }

  image_t WriteImage(const Image& image, const bool use_image_id) override {
    Sqlite3StmtContext context(sql_stmt_write_image_);

    if (use_image_id) {
      THROW_CHECK(!ExistsImage(image.ImageId())) << "image_id must be unique";
      SQLITE3_CALL(
          sqlite3_bind_int64(sql_stmt_write_image_, 1, image.ImageId()));
    } else {
      SQLITE3_CALL(sqlite3_bind_null(sql_stmt_write_image_, 1));
    }

    SQLITE3_CALL(sqlite3_bind_text(sql_stmt_write_image_,
                                   2,
                                   image.Name().c_str(),
                                   static_cast<int>(image.Name().size()),
                                   SQLITE_STATIC));
    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_write_image_, 3, image.CameraId()));

    SQLITE3_CALL(sqlite3_step(sql_stmt_write_image_));

    return static_cast<image_t>(
        sqlite3_last_insert_rowid(THROW_CHECK_NOTNULL(database_)));
  }

  pose_prior_t WritePosePrior(const PosePrior& pose_prior,
                              bool use_pose_prior_id) override {
    Sqlite3StmtContext context(sql_stmt_write_pose_prior_);

    if (use_pose_prior_id) {
      THROW_CHECK(!ExistsPosePrior(pose_prior.pose_prior_id,
                                   /*is_deprecated_image_prior=*/false))
          << "pose_prior_id must be unique";
      SQLITE3_CALL(sqlite3_bind_int64(
          sql_stmt_write_pose_prior_, 1, pose_prior.pose_prior_id));
    } else {
      SQLITE3_CALL(sqlite3_bind_null(sql_stmt_write_pose_prior_, 1));
    }

    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_write_pose_prior_, 2, pose_prior.corr_data_id.id));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_write_pose_prior_, 3, pose_prior.corr_data_id.sensor_id.id));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_write_pose_prior_,
        4,
        static_cast<sqlite3_int64>(pose_prior.corr_data_id.sensor_id.type)));
    WriteStaticMatrixBlob(sql_stmt_write_pose_prior_, pose_prior.position, 5);
    WriteStaticMatrixBlob(
        sql_stmt_write_pose_prior_, pose_prior.position_covariance, 6);
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_write_pose_prior_,
        7,
        static_cast<sqlite3_int64>(pose_prior.coordinate_system)));
    SQLITE3_CALL(sqlite3_step(sql_stmt_write_pose_prior_));

    return static_cast<image_t>(
        sqlite3_last_insert_rowid(THROW_CHECK_NOTNULL(database_)));
  }

  void WriteKeypoints(const image_t image_id,
                      const FeatureKeypoints& keypoints) override {
    WriteKeypoints(image_id, FeatureKeypointsToBlob(keypoints));
  }

  void WriteKeypoints(const image_t image_id,
                      const FeatureKeypointsBlob& blob) override {
    Sqlite3StmtContext context(sql_stmt_write_keypoints_);

    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_keypoints_, 1, image_id));
    WriteDynamicMatrixBlob(sql_stmt_write_keypoints_, blob, 2);

    SQLITE3_CALL(sqlite3_step(sql_stmt_write_keypoints_));
  }

  void WriteDescriptors(const image_t image_id,
                        const FeatureDescriptors& descriptors) override {
    Sqlite3StmtContext context(sql_stmt_write_descriptors_);

    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_descriptors_, 1, image_id));
    WriteDynamicMatrixBlob(sql_stmt_write_descriptors_, descriptors, 2);

    SQLITE3_CALL(sqlite3_step(sql_stmt_write_descriptors_));
  }

  void WriteMatches(const image_t image_id1,
                    const image_t image_id2,
                    const FeatureMatches& matches) override {
    WriteMatches(image_id1, image_id2, FeatureMatchesToBlob(matches));
  }

  void WriteMatches(const image_t image_id1,
                    const image_t image_id2,
                    const FeatureMatchesBlob& blob) override {
    Sqlite3StmtContext context(sql_stmt_write_matches_);

    const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_matches_, 1, pair_id));

    // Important: the swapped data must live until the query is executed.
    FeatureMatchesBlob swapped_blob;
    if (SwapImagePair(image_id1, image_id2)) {
      swapped_blob = blob;
      SwapFeatureMatchesBlob(&swapped_blob);
      WriteDynamicMatrixBlob(sql_stmt_write_matches_, swapped_blob, 2);
    } else {
      WriteDynamicMatrixBlob(sql_stmt_write_matches_, blob, 2);
    }

    SQLITE3_CALL(sqlite3_step(sql_stmt_write_matches_));
  }

  void WriteTwoViewGeometry(const image_t image_id1,
                            const image_t image_id2,
                            const TwoViewGeometry& two_view_geometry) override {
    THROW_CHECK(!ExistsTwoViewGeometry(image_id1, image_id2))
        << "Two view geometry between image " << image_id1 << " and "
        << image_id2 << " already exists.";
    Sqlite3StmtContext context(sql_stmt_write_two_view_geometry_);

    const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_write_two_view_geometry_, 1, pair_id));

    const TwoViewGeometry* two_view_geometry_ptr = &two_view_geometry;

    // Invert the two-view geometry if the image pair has to be swapped.
    std::unique_ptr<TwoViewGeometry> swapped_two_view_geometry;
    if (SwapImagePair(image_id1, image_id2)) {
      swapped_two_view_geometry = std::make_unique<TwoViewGeometry>();
      *swapped_two_view_geometry = two_view_geometry;
      swapped_two_view_geometry->Invert();
      two_view_geometry_ptr = swapped_two_view_geometry.get();
    }

    const FeatureMatchesBlob inlier_matches =
        FeatureMatchesToBlob(two_view_geometry_ptr->inlier_matches);
    WriteDynamicMatrixBlob(
        sql_stmt_write_two_view_geometry_, inlier_matches, 2);

    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_write_two_view_geometry_, 5, two_view_geometry_ptr->config));

    // Transpose the matrices to obtain row-major data layout.
    // Important: Do not move these objects inside the if-statement, because
    // the objects must live until `sqlite3_step` is called on the statement.
    const Eigen::Matrix3d Ft = two_view_geometry_ptr->F.transpose();
    const Eigen::Matrix3d Et = two_view_geometry_ptr->E.transpose();
    const Eigen::Matrix3d Ht = two_view_geometry_ptr->H.transpose();
    const Eigen::Vector4d quat_wxyz(
        two_view_geometry_ptr->cam2_from_cam1.rotation.w(),
        two_view_geometry_ptr->cam2_from_cam1.rotation.x(),
        two_view_geometry_ptr->cam2_from_cam1.rotation.y(),
        two_view_geometry_ptr->cam2_from_cam1.rotation.z());

    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, Ft, 6);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, Et, 7);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, Ht, 8);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, quat_wxyz, 9);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_,
                          two_view_geometry_ptr->cam2_from_cam1.translation,
                          10);
    SQLITE3_CALL(sqlite3_step(sql_stmt_write_two_view_geometry_));
  }

  void UpdateRig(const Rig& rig) override {
    // Update rig.
    {
      Sqlite3StmtContext context(sql_stmt_update_rig_);
      SQLITE3_CALL(
          sqlite3_bind_int64(sql_stmt_update_rig_,
                             1,
                             static_cast<sqlite3_int64>(rig.RefSensorId().id)));
      SQLITE3_CALL(sqlite3_bind_int64(
          sql_stmt_update_rig_,
          2,
          static_cast<sqlite3_int64>(rig.RefSensorId().type)));
      SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_rig_, 3, rig.RigId()));
      SQLITE3_CALL(sqlite3_step(sql_stmt_update_rig_));
    }

    // Clear the rig sensors.
    {
      Sqlite3StmtContext context(sql_stmt_delete_rig_sensors_);
      SQLITE3_CALL(
          sqlite3_bind_int64(sql_stmt_delete_rig_sensors_, 1, rig.RigId()));
      SQLITE3_CALL(sqlite3_step(sql_stmt_delete_rig_sensors_));
    }

    // Write the updated rig sensors.
    WriteRigSensors(rig.RigId(), rig, sql_stmt_write_rig_sensor_);
  }

  void UpdateCamera(const Camera& camera) override {
    Sqlite3StmtContext context(sql_stmt_update_camera_);

    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_update_camera_,
                           1,
                           static_cast<sqlite3_int64>(camera.model_id)));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_update_camera_, 2, static_cast<sqlite3_int64>(camera.width)));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_update_camera_, 3, static_cast<sqlite3_int64>(camera.height)));

    const size_t num_params_bytes = sizeof(double) * camera.params.size();
    SQLITE3_CALL(sqlite3_bind_blob(sql_stmt_update_camera_,
                                   4,
                                   camera.params.data(),
                                   static_cast<int>(num_params_bytes),
                                   SQLITE_STATIC));

    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_update_camera_, 5, camera.has_prior_focal_length));

    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_update_camera_, 6, camera.camera_id));

    SQLITE3_CALL(sqlite3_step(sql_stmt_update_camera_));
  }

  void UpdateFrame(const Frame& frame) override {
    // Update frame.
    {
      Sqlite3StmtContext context(sql_stmt_update_frame_);
      SQLITE3_CALL(
          sqlite3_bind_int64(sql_stmt_update_frame_, 1, frame.RigId()));
      SQLITE3_CALL(
          sqlite3_bind_int64(sql_stmt_update_frame_, 2, frame.FrameId()));
      SQLITE3_CALL(sqlite3_step(sql_stmt_update_frame_));
    }

    // Clear the frame data.
    {
      Sqlite3StmtContext context(sql_stmt_delete_frame_data_);
      SQLITE3_CALL(
          sqlite3_bind_int64(sql_stmt_delete_frame_data_, 1, frame.FrameId()));
      SQLITE3_CALL(sqlite3_step(sql_stmt_delete_frame_data_));
    }

    // Write the updated frame data.
    WriteFrameData(frame.FrameId(), frame, sql_stmt_write_frame_data_);
  }

  void UpdateImage(const Image& image) override {
    Sqlite3StmtContext context(sql_stmt_update_image_);

    SQLITE3_CALL(sqlite3_bind_text(sql_stmt_update_image_,
                                   1,
                                   image.Name().c_str(),
                                   static_cast<int>(image.Name().size()),
                                   SQLITE_STATIC));
    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_update_image_, 2, image.CameraId()));
    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_update_image_, 3, image.ImageId()));

    SQLITE3_CALL(sqlite3_step(sql_stmt_update_image_));
  }

  void UpdatePosePrior(const PosePrior& pose_prior) override {
    Sqlite3StmtContext context(sql_stmt_update_pose_prior_);

    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_update_pose_prior_, 1, pose_prior.corr_data_id.id));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_update_pose_prior_, 2, pose_prior.corr_data_id.sensor_id.id));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_update_pose_prior_,
        3,
        static_cast<sqlite3_int64>(pose_prior.corr_data_id.sensor_id.type)));
    WriteStaticMatrixBlob(sql_stmt_update_pose_prior_, pose_prior.position, 4);
    WriteStaticMatrixBlob(
        sql_stmt_update_pose_prior_, pose_prior.position_covariance, 5);
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_update_pose_prior_,
        6,
        static_cast<sqlite3_int64>(pose_prior.coordinate_system)));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_update_pose_prior_, 7, pose_prior.pose_prior_id));

    SQLITE3_CALL(sqlite3_step(sql_stmt_update_pose_prior_));
  }

  void UpdateKeypoints(image_t image_id,
                       const FeatureKeypoints& keypoints) override {
    UpdateKeypoints(image_id, FeatureKeypointsToBlob(keypoints));
  }

  void UpdateKeypoints(image_t image_id,
                       const FeatureKeypointsBlob& blob) override {
    Sqlite3StmtContext context(sql_stmt_update_keypoints_);

    WriteDynamicMatrixBlob(sql_stmt_update_keypoints_, blob, 1);
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_keypoints_, 4, image_id));

    SQLITE3_CALL(sqlite3_step(sql_stmt_update_keypoints_));
  }

  void UpdateTwoViewGeometry(
      const image_t image_id1,
      const image_t image_id2,
      const TwoViewGeometry& two_view_geometry) override {
    // Do nothing if the image pair does not exist, to align with the UPDATE
    // behavior in SQL.
    if (ExistsTwoViewGeometry(image_id1, image_id2)) {
      DeleteTwoViewGeometry(image_id1, image_id2);
      WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
    }
  }

  void DeleteMatches(const image_t image_id1,
                     const image_t image_id2) override {
    Sqlite3StmtContext context(sql_stmt_delete_matches_);

    const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt_delete_matches_, 1, static_cast<sqlite3_int64>(pair_id)));
    SQLITE3_CALL(sqlite3_step(sql_stmt_delete_matches_));
    database_entry_deleted_ = true;
  }

  void DeleteTwoViewGeometry(const image_t image_id1,
                             const image_t image_id2) override {
    Sqlite3StmtContext context(sql_stmt_delete_two_view_geometry_);

    const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_delete_two_view_geometry_,
                                    1,
                                    static_cast<sqlite3_int64>(pair_id)));
    SQLITE3_CALL(sqlite3_step(sql_stmt_delete_two_view_geometry_));
    database_entry_deleted_ = true;
  }

  void DeleteInlierMatches(const image_t image_id1,
                           const image_t image_id2) override {
    if (!ExistsTwoViewGeometry(image_id1, image_id2)) {
      return;
    }
    TwoViewGeometry geom = ReadTwoViewGeometry(image_id1, image_id2);
    geom.inlier_matches.clear();
    UpdateTwoViewGeometry(image_id1, image_id2, geom);
  }

  void ClearAllTables() override {
    ClearMatches();
    ClearTwoViewGeometries();
    ClearDescriptors();
    ClearKeypoints();
    ClearPosePriors();
    ClearFrames();
    ClearImages();
    ClearRigs();
    ClearCameras();
  }

  void ClearRigs() override {
    Sqlite3StmtContext context(sql_stmt_clear_rigs_);
    SQLITE3_CALL(sqlite3_step(sql_stmt_clear_rigs_));
    database_entry_deleted_ = true;
  }

  void ClearCameras() override {
    Sqlite3StmtContext context(sql_stmt_clear_cameras_);
    SQLITE3_CALL(sqlite3_step(sql_stmt_clear_cameras_));
    database_entry_deleted_ = true;
  }

  void ClearFrames() override {
    Sqlite3StmtContext context(sql_stmt_clear_frames_);
    SQLITE3_CALL(sqlite3_step(sql_stmt_clear_frames_));
    database_entry_deleted_ = true;
  }

  void ClearImages() override {
    Sqlite3StmtContext context(sql_stmt_clear_images_);
    SQLITE3_CALL(sqlite3_step(sql_stmt_clear_images_));
    database_entry_deleted_ = true;
  }

  void ClearPosePriors() override {
    Sqlite3StmtContext context(sql_stmt_clear_pose_priors_);
    SQLITE3_CALL(sqlite3_step(sql_stmt_clear_pose_priors_));
    database_entry_deleted_ = true;
  }

  void ClearDescriptors() override {
    Sqlite3StmtContext context(sql_stmt_clear_descriptors_);
    SQLITE3_CALL(sqlite3_step(sql_stmt_clear_descriptors_));
    database_entry_deleted_ = true;
  }

  void ClearKeypoints() override {
    Sqlite3StmtContext context(sql_stmt_clear_keypoints_);
    SQLITE3_CALL(sqlite3_step(sql_stmt_clear_keypoints_));
    database_entry_deleted_ = true;
  }

  void ClearMatches() override {
    Sqlite3StmtContext context(sql_stmt_clear_matches_);
    SQLITE3_CALL(sqlite3_step(sql_stmt_clear_matches_));
    database_entry_deleted_ = true;
  }

  void ClearTwoViewGeometries() override {
    Sqlite3StmtContext context(sql_stmt_clear_two_view_geometries_);
    SQLITE3_CALL(sqlite3_step(sql_stmt_clear_two_view_geometries_));
    database_entry_deleted_ = true;
  }

  void BeginTransaction() const override {
    SQLITE3_EXEC(THROW_CHECK_NOTNULL(database_), "BEGIN TRANSACTION", nullptr);
  }

  void EndTransaction() const override {
    SQLITE3_EXEC(THROW_CHECK_NOTNULL(database_), "END TRANSACTION", nullptr);
  }

  void PrepareSQLStatements() {
    sql_stmts_.clear();

    auto prepare_sql_stmt = [this](const std::string& sql,
                                   sqlite3_stmt** sql_stmt) {
      THROW_CHECK_NOTNULL(database_);
      VLOG(3) << "Preparing SQL statement: " << sql;
      SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, sql_stmt, 0));
      sql_stmts_.push_back(sql_stmt);
    };

    //////////////////////////////////////////////////////////////////////////////
    // num_*
    //////////////////////////////////////////////////////////////////////////////
    prepare_sql_stmt("SELECT rows FROM keypoints WHERE image_id = ?;",
                     &sql_stmt_num_keypoints_);
    prepare_sql_stmt("SELECT rows FROM descriptors WHERE image_id = ?;",
                     &sql_stmt_num_descriptors_);

    //////////////////////////////////////////////////////////////////////////////
    // exists_*
    //////////////////////////////////////////////////////////////////////////////
    prepare_sql_stmt("SELECT 1 FROM rigs WHERE rig_id = ?;",
                     &sql_stmt_exists_rig_);
    prepare_sql_stmt("SELECT 1 FROM cameras WHERE camera_id = ?;",
                     &sql_stmt_exists_camera_);
    prepare_sql_stmt("SELECT 1 FROM frames WHERE frame_id = ?;",
                     &sql_stmt_exists_frame_);
    prepare_sql_stmt("SELECT 1 FROM images WHERE image_id = ?;",
                     &sql_stmt_exists_image_id_);
    prepare_sql_stmt("SELECT 1 FROM images WHERE name = ?;",
                     &sql_stmt_exists_image_name_);
    prepare_sql_stmt("SELECT 1 FROM pose_priors WHERE pose_prior_id = ?;",
                     &sql_stmt_exists_pose_prior_);
    prepare_sql_stmt("SELECT 1 FROM keypoints WHERE image_id = ?;",
                     &sql_stmt_exists_keypoints_);
    prepare_sql_stmt("SELECT 1 FROM descriptors WHERE image_id = ?;",
                     &sql_stmt_exists_descriptors_);
    prepare_sql_stmt("SELECT 1 FROM matches WHERE pair_id = ?;",
                     &sql_stmt_exists_matches_);
    prepare_sql_stmt("SELECT 1 FROM two_view_geometries WHERE pair_id = ?;",
                     &sql_stmt_exists_two_view_geometry_);

    //////////////////////////////////////////////////////////////////////////////
    // update_*
    //////////////////////////////////////////////////////////////////////////////
    prepare_sql_stmt(
        "UPDATE rigs SET ref_sensor_id=?, ref_sensor_type=? WHERE rig_id=?;",
        &sql_stmt_update_rig_);
    prepare_sql_stmt(
        "UPDATE cameras SET model=?, width=?, height=?, params=?, "
        "prior_focal_length=? WHERE camera_id=?;",
        &sql_stmt_update_camera_);
    prepare_sql_stmt("UPDATE frames SET rig_id=? WHERE frame_id=?;",
                     &sql_stmt_update_frame_);
    prepare_sql_stmt("UPDATE images SET name=?, camera_id=? WHERE image_id=?;",
                     &sql_stmt_update_image_);
    prepare_sql_stmt(
        "UPDATE pose_priors SET corr_data_id=?, corr_sensor_id=?, "
        "corr_sensor_type=?, position=?, position_covariance=?, "
        "coordinate_system=? WHERE pose_prior_id=?;",
        &sql_stmt_update_pose_prior_);
    prepare_sql_stmt(
        "UPDATE keypoints SET rows=?, cols=?, data=? WHERE image_id=?;",
        &sql_stmt_update_keypoints_);

    //////////////////////////////////////////////////////////////////////////////
    // read_*
    //////////////////////////////////////////////////////////////////////////////

    prepare_sql_stmt(
        "SELECT rigs.rig_id, rigs.ref_sensor_id, rigs.ref_sensor_type, "
        "rig_sensors.sensor_id, rig_sensors.sensor_type, "
        "rig_sensors.sensor_from_rig FROM rigs "
        "LEFT OUTER JOIN rig_sensors ON rigs.rig_id = rig_sensors.rig_id "
        "ORDER BY rigs.rig_id;",
        &sql_stmt_read_rigs_);
    prepare_sql_stmt(
        "SELECT rigs.rig_id, rigs.ref_sensor_id, rigs.ref_sensor_type, "
        "rig_sensors.sensor_id, rig_sensors.sensor_type, "
        "rig_sensors.sensor_from_rig FROM rigs "
        "LEFT OUTER JOIN rig_sensors ON rigs.rig_id = rig_sensors.rig_id "
        "WHERE rigs.rig_id = ? "
        "ORDER BY rigs.rig_id;",
        &sql_stmt_read_rig_);
    prepare_sql_stmt(
        "SELECT rig_id FROM rig_sensors WHERE sensor_id = ? AND sensor_type = "
        "?;",
        &sql_stmt_read_rig_with_sensor_);
    prepare_sql_stmt(
        "SELECT rig_id FROM rigs "
        "WHERE ref_sensor_id = ? AND ref_sensor_type = ?;",
        &sql_stmt_read_rig_with_ref_sensor_);
    prepare_sql_stmt("SELECT * FROM cameras;", &sql_stmt_read_cameras_);
    prepare_sql_stmt("SELECT * FROM cameras WHERE camera_id = ?;",
                     &sql_stmt_read_camera_);
    prepare_sql_stmt(
        "SELECT frames.frame_id, frames.rig_id, frame_data.data_id, "
        "frame_data.sensor_id, frame_data.sensor_type FROM frames "
        "LEFT OUTER JOIN frame_data ON frames.frame_id = frame_data.frame_id "
        "ORDER BY frames.frame_id;",
        &sql_stmt_read_frames_);
    prepare_sql_stmt(
        "SELECT frames.frame_id, frames.rig_id, frame_data.data_id, "
        "frame_data.sensor_id, frame_data.sensor_type FROM frames "
        "LEFT OUTER JOIN frame_data ON frames.frame_id = frame_data.frame_id "
        "WHERE frames.frame_id = ? "
        "ORDER BY frames.frame_id;",
        &sql_stmt_read_frame_);
    prepare_sql_stmt("SELECT * FROM images WHERE image_id = ?;",
                     &sql_stmt_read_image_id_);
    prepare_sql_stmt("SELECT * FROM images;", &sql_stmt_read_images_);
    prepare_sql_stmt("SELECT * FROM images WHERE name = ?;",
                     &sql_stmt_read_image_with_name_);
    prepare_sql_stmt(
        "SELECT pose_prior_id, corr_data_id, corr_sensor_id, corr_sensor_type, "
        "position, position_covariance, coordinate_system FROM pose_priors "
        "WHERE pose_prior_id = ?;",
        &sql_stmt_read_pose_prior_);
    prepare_sql_stmt(
        "SELECT pose_prior_id, corr_data_id, corr_sensor_id, corr_sensor_type, "
        "position, position_covariance, coordinate_system FROM pose_priors;",
        &sql_stmt_read_pose_priors_);
    prepare_sql_stmt(
        "SELECT rows, cols, data FROM keypoints WHERE image_id = ?;",
        &sql_stmt_read_keypoints_);
    prepare_sql_stmt(
        "SELECT rows, cols, data FROM descriptors WHERE image_id = ?;",
        &sql_stmt_read_descriptors_);
    prepare_sql_stmt("SELECT rows, cols, data FROM matches WHERE pair_id = ?;",
                     &sql_stmt_read_matches_);
    prepare_sql_stmt("SELECT * FROM matches WHERE rows > 0;",
                     &sql_stmt_read_matches_all_);
    prepare_sql_stmt("SELECT pair_id, rows FROM matches WHERE rows > 0;",
                     &sql_stmt_read_num_matches_);
    prepare_sql_stmt(
        "SELECT rows, cols, data, config, F, E, H, qvec, tvec FROM "
        "two_view_geometries WHERE pair_id = ?;",
        &sql_stmt_read_two_view_geometry_);
    prepare_sql_stmt("SELECT * FROM two_view_geometries WHERE rows > 0;",
                     &sql_stmt_read_two_view_geometries_);
    prepare_sql_stmt(
        "SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0;",
        &sql_stmt_read_two_view_geometry_num_inliers_);

    //////////////////////////////////////////////////////////////////////////////
    // write_*
    //////////////////////////////////////////////////////////////////////////////
    prepare_sql_stmt(
        "INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) "
        "VALUES(?, ?, ?);",
        &sql_stmt_write_rig_);
    prepare_sql_stmt(
        "INSERT INTO rig_sensors(rig_id, sensor_id, sensor_type, "
        "sensor_from_rig) VALUES(?, ?, ?, ?);",
        &sql_stmt_write_rig_sensor_);
    prepare_sql_stmt(
        "INSERT INTO cameras(camera_id, model, width, height, params, "
        "prior_focal_length) VALUES(?, ?, ?, ?, ?, ?);",
        &sql_stmt_write_camera_);
    prepare_sql_stmt("INSERT INTO frames(frame_id, rig_id) VALUES(?, ?);",
                     &sql_stmt_write_frame_);
    prepare_sql_stmt(
        "INSERT INTO frame_data(frame_id, data_id, sensor_id, sensor_type) "
        "VALUES(?, ?, ?, ?);",
        &sql_stmt_write_frame_data_);
    prepare_sql_stmt(
        "INSERT INTO images(image_id, name, camera_id) VALUES(?, ?, ?);",
        &sql_stmt_write_image_);
    prepare_sql_stmt(
        "INSERT INTO pose_priors(pose_prior_id, corr_data_id, corr_sensor_id, "
        "corr_sensor_type, position, position_covariance, coordinate_system) "
        "VALUES(?, ?, ?, ?, ?, ?, ?);",
        &sql_stmt_write_pose_prior_);
    prepare_sql_stmt(
        "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);",
        &sql_stmt_write_keypoints_);
    prepare_sql_stmt(
        "INSERT INTO descriptors(image_id, rows, cols, data) VALUES(?, ?, ?, "
        "?);",
        &sql_stmt_write_descriptors_);
    prepare_sql_stmt(
        "INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, "
        "?, ?);",
        &sql_stmt_write_matches_);
    prepare_sql_stmt(
        "INSERT INTO two_view_geometries(pair_id, rows, cols, data, config, F, "
        "E, H, qvec, tvec) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
        &sql_stmt_write_two_view_geometry_);

    //////////////////////////////////////////////////////////////////////////////
    // delete_*
    //////////////////////////////////////////////////////////////////////////////
    prepare_sql_stmt("DELETE FROM rig_sensors WHERE rig_id = ?;",
                     &sql_stmt_delete_rig_sensors_);
    prepare_sql_stmt("DELETE FROM frame_data WHERE frame_id = ?;",
                     &sql_stmt_delete_frame_data_);
    prepare_sql_stmt("DELETE FROM matches WHERE pair_id = ?;",
                     &sql_stmt_delete_matches_);
    prepare_sql_stmt("DELETE FROM two_view_geometries WHERE pair_id = ?;",
                     &sql_stmt_delete_two_view_geometry_);

    //////////////////////////////////////////////////////////////////////////////
    // clear_*
    //////////////////////////////////////////////////////////////////////////////
    prepare_sql_stmt("DELETE FROM rigs; DELETE FROM rig_sensors;",
                     &sql_stmt_clear_rigs_);
    prepare_sql_stmt("DELETE FROM cameras;", &sql_stmt_clear_cameras_);
    prepare_sql_stmt("DELETE FROM frames; DELETE FROM frame_data;",
                     &sql_stmt_clear_frames_);
    prepare_sql_stmt("DELETE FROM images;", &sql_stmt_clear_images_);
    prepare_sql_stmt("DELETE FROM pose_priors;", &sql_stmt_clear_pose_priors_);
    prepare_sql_stmt("DELETE FROM keypoints;", &sql_stmt_clear_keypoints_);
    prepare_sql_stmt("DELETE FROM descriptors;", &sql_stmt_clear_descriptors_);
    prepare_sql_stmt("DELETE FROM matches;", &sql_stmt_clear_matches_);
    prepare_sql_stmt("DELETE FROM two_view_geometries;",
                     &sql_stmt_clear_two_view_geometries_);
  }

  void FinalizeSQLStatements() {
    for (sqlite3_stmt** sql_stmt : sql_stmts_) {
      SQLITE3_CALL(sqlite3_finalize(*sql_stmt));
      *sql_stmt = nullptr;
    }
  }

  void CreateTables() const {
    CreateRigTable();
    CreateRigSensorsTable();
    CreateCameraTable();
    CreateFrameTable();
    CreateFrameDataTable();
    CreateImageTable();
    CreatePosePriorTable();
    CreateKeypointsTable();
    CreateDescriptorsTable();
    CreateMatchesTable();
    CreateTwoViewGeometriesTable();
  }

  void CreateRigTable() const {
    const std::string sql =
        "CREATE TABLE IF NOT EXISTS rigs"
        "   (rig_id               INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
        "    ref_sensor_id        INTEGER                             NOT NULL,"
        "    ref_sensor_type      INTEGER                             NOT "
        "NULL);"
        "CREATE UNIQUE INDEX IF NOT EXISTS rig_ref_sensor_assignment ON "
        "   rigs(ref_sensor_id, ref_sensor_type);";

    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
  }

  void CreateRigSensorsTable() const {
    const std::string sql =
        "CREATE TABLE IF NOT EXISTS rig_sensors"
        "   (rig_id               INTEGER                             NOT NULL,"
        "    sensor_id            INTEGER                             NOT NULL,"
        "    sensor_type          INTEGER                             NOT NULL,"
        "    sensor_from_rig      BLOB,"
        "FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);"
        "CREATE UNIQUE INDEX IF NOT EXISTS rig_sensor_assignment ON "
        "   rig_sensors(sensor_id, sensor_type);";

    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
  }

  void CreateCameraTable() const {
    const std::string sql =
        "CREATE TABLE IF NOT EXISTS cameras"
        "   (camera_id            INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
        "    model                INTEGER                             NOT NULL,"
        "    width                INTEGER                             NOT NULL,"
        "    height               INTEGER                             NOT NULL,"
        "    params               BLOB,"
        "    prior_focal_length   INTEGER                             NOT "
        "NULL);";

    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
  }

  void CreateFrameTable() const {
    const std::string sql =
        "CREATE TABLE IF NOT EXISTS frames"
        "   (frame_id             INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
        "    rig_id               INTEGER                             NOT NULL,"
        "    FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);";

    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
  }

  void CreateFrameDataTable() const {
    const std::string sql =
        "CREATE TABLE IF NOT EXISTS frame_data"
        "   (frame_id             INTEGER                             NOT NULL,"
        "    data_id              INTEGER                             NOT NULL,"
        "    sensor_id            INTEGER                             NOT NULL,"
        "    sensor_type          INTEGER                             NOT NULL,"
        "    FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE "
        "CASCADE);"
        "CREATE UNIQUE INDEX IF NOT EXISTS frame_sensor_assignment ON "
        "   frame_data(data_id, sensor_type);";

    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
  }

  void CreateImageTable() const {
    const std::string sql = StringPrintf(
        "CREATE TABLE IF NOT EXISTS images"
        "   (image_id   INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
        "    name       TEXT                                NOT NULL UNIQUE,"
        "    camera_id  INTEGER                             NOT NULL,"
        "    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < %d),"
        "    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));"
        "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);",
        kMaxNumImages);

    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
  }

  void CreatePosePriorTable() const {
    const std::string sql =
        "CREATE TABLE IF NOT EXISTS pose_priors"
        "   (pose_prior_id              INTEGER  PRIMARY KEY  NOT NULL,"
        "    corr_data_id               INTEGER               NOT NULL,"
        "    corr_sensor_id             INTEGER               NOT NULL,"
        "    corr_sensor_type           INTEGER               NOT NULL,"
        "    position                   BLOB,"
        "    position_covariance        BLOB,"
        "    coordinate_system          INTEGER               NOT NULL);";

    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
  }

  void CreateKeypointsTable() const {
    const std::string sql =
        "CREATE TABLE IF NOT EXISTS keypoints"
        "   (image_id  INTEGER  PRIMARY KEY  NOT NULL,"
        "    rows      INTEGER               NOT NULL,"
        "    cols      INTEGER               NOT NULL,"
        "    data      BLOB,"
        "    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE "
        "CASCADE);";

    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
  }

  void CreateDescriptorsTable() const {
    const std::string sql =
        "CREATE TABLE IF NOT EXISTS descriptors"
        "   (image_id  INTEGER  PRIMARY KEY  NOT NULL,"
        "    rows      INTEGER               NOT NULL,"
        "    cols      INTEGER               NOT NULL,"
        "    data      BLOB,"
        "    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE "
        "CASCADE);";

    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
  }

  void CreateMatchesTable() const {
    const std::string sql =
        "CREATE TABLE IF NOT EXISTS matches"
        "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
        "    rows     INTEGER               NOT NULL,"
        "    cols     INTEGER               NOT NULL,"
        "    data     BLOB);";

    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
  }

  void CreateTwoViewGeometriesTable() const {
    if (ExistsTable("inlier_matches")) {
      SQLITE3_EXEC(database_,
                   "ALTER TABLE inlier_matches RENAME TO two_view_geometries;",
                   nullptr);
    } else {
      const std::string sql =
          "CREATE TABLE IF NOT EXISTS two_view_geometries"
          "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
          "    rows     INTEGER               NOT NULL,"
          "    cols     INTEGER               NOT NULL,"
          "    data     BLOB,"
          "    config   INTEGER               NOT NULL,"
          "    F        BLOB,"
          "    E        BLOB,"
          "    H        BLOB,"
          "    qvec     BLOB,"
          "    tvec     BLOB);";
      SQLITE3_EXEC(database_, sql.c_str(), nullptr);
    }
  }

  void UpdateSchema() {
    if (!ExistsColumn("two_view_geometries", "F")) {
      SQLITE3_EXEC(database_,
                   "ALTER TABLE two_view_geometries ADD COLUMN F BLOB;",
                   nullptr);
    }

    if (!ExistsColumn("two_view_geometries", "E")) {
      SQLITE3_EXEC(database_,
                   "ALTER TABLE two_view_geometries ADD COLUMN E BLOB;",
                   nullptr);
    }

    if (!ExistsColumn("two_view_geometries", "H")) {
      SQLITE3_EXEC(database_,
                   "ALTER TABLE two_view_geometries ADD COLUMN H BLOB;",
                   nullptr);
    }

    if (!ExistsColumn("two_view_geometries", "qvec")) {
      SQLITE3_EXEC(database_,
                   "ALTER TABLE two_view_geometries ADD COLUMN qvec BLOB;",
                   nullptr);
    }

    if (!ExistsColumn("two_view_geometries", "tvec")) {
      SQLITE3_EXEC(database_,
                   "ALTER TABLE two_view_geometries ADD COLUMN tvec BLOB;",
                   nullptr);
    }

    if (!ExistsColumn("pose_priors", "position_covariance")) {
      // Create position_covariance matrix column
      SQLITE3_EXEC(
          database_,
          "ALTER TABLE pose_priors ADD COLUMN position_covariance BLOB "
          "DEFAULT NULL;",
          nullptr);

      // Set position_covariance column to NaN matrices
      const std::string update_sql =
          "UPDATE pose_priors SET position_covariance = ?;";
      sqlite3_stmt* update_stmt;
      SQLITE3_CALL(sqlite3_prepare_v2(
          database_, update_sql.c_str(), -1, &update_stmt, 0));
      WriteStaticMatrixBlob(update_stmt, PosePrior().position_covariance, 1);
      SQLITE3_CALL(sqlite3_step(update_stmt));
      SQLITE3_CALL(sqlite3_finalize(update_stmt));
    }

    if (ExistsColumn("pose_priors", "image_id") &&
        !ExistsColumn("pose_priors", "pose_prior_id")) {
      SQLITE3_EXEC(
          database_,
          "ALTER TABLE pose_priors RENAME COLUMN image_id TO pose_prior_id;"
          "ALTER TABLE pose_priors ADD COLUMN corr_data_id INTEGER NOT NULL;"
          "ALTER TABLE pose_priors ADD COLUMN corr_sensor_id INTEGER NOT NULL;"
          "ALTER TABLE pose_priors ADD COLUMN corr_sensor_type INTEGER NOT "
          "NULL;",
          nullptr);

      // Migrate existing data to frame_data table.
      for (Frame& frame : ReadAllFrames()) {
        for (const auto& data_id : frame.ImageIds()) {
          // Note that in the old schema pose_prior_id == image_id.
          if (ExistsPosePrior(data_id.id,
                              /*is_deprecated_image_prior=*/false)) {
            PosePrior pose_prior =
                ReadPosePrior(data_id.id, /*is_deprecated_image_prior=*/false);
            pose_prior.corr_data_id = data_id;
            UpdatePosePrior(pose_prior);
          }
        }
        UpdateFrame(frame);
      }
    }

    // Update user version number.
    std::unique_lock<std::mutex> lock(update_schema_mutex_);
    const std::string update_user_version_sql =
        StringPrintf("PRAGMA user_version = 3900;");
    SQLITE3_EXEC(database_, update_user_version_sql.c_str(), nullptr);
  }

  bool ExistsTable(const std::string& table_name) const {
    const std::string sql =
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?;";

    sqlite3_stmt* sql_stmt;
    SQLITE3_CALL(sqlite3_prepare_v2(
        THROW_CHECK_NOTNULL(database_), sql.c_str(), -1, &sql_stmt, 0));

    SQLITE3_CALL(sqlite3_bind_text(sql_stmt,
                                   1,
                                   table_name.c_str(),
                                   static_cast<int>(table_name.size()),
                                   SQLITE_STATIC));

    const bool exists = SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW;

    SQLITE3_CALL(sqlite3_finalize(sql_stmt));

    return exists;
  }

  bool ExistsColumn(const std::string& table_name,
                    const std::string& column_name) const {
    const std::string sql =
        StringPrintf("PRAGMA table_info(%s);", table_name.c_str());

    sqlite3_stmt* sql_stmt;
    SQLITE3_CALL(sqlite3_prepare_v2(
        THROW_CHECK_NOTNULL(database_), sql.c_str(), -1, &sql_stmt, 0));

    bool exists_column = false;
    while (SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW) {
      const std::string result =
          reinterpret_cast<const char*>(sqlite3_column_text(sql_stmt, 1));
      if (column_name == result) {
        exists_column = true;
        break;
      }
    }

    SQLITE3_CALL(sqlite3_finalize(sql_stmt));

    return exists_column;
  }

  bool ExistsRowId(sqlite3_stmt* sql_stmt, const sqlite3_int64 row_id) const {
    Sqlite3StmtContext context(sql_stmt);
    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt, 1, static_cast<sqlite3_int64>(row_id)));

    return SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW;
  }

  bool ExistsRowString(sqlite3_stmt* sql_stmt,
                       const std::string& row_entry) const {
    Sqlite3StmtContext context(sql_stmt);
    SQLITE3_CALL(sqlite3_bind_text(sql_stmt,
                                   1,
                                   row_entry.c_str(),
                                   static_cast<int>(row_entry.size()),
                                   SQLITE_STATIC));
    return SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW;
  }

  size_t CountRows(const std::string& table) const {
    const std::string sql =
        StringPrintf("SELECT COUNT(*) FROM %s;", table.c_str());

    sqlite3_stmt* sql_stmt;
    SQLITE3_CALL(sqlite3_prepare_v2(
        THROW_CHECK_NOTNULL(database_), sql.c_str(), -1, &sql_stmt, 0));

    size_t count = 0;
    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
    if (rc == SQLITE_ROW) {
      count = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
    }

    SQLITE3_CALL(sqlite3_finalize(sql_stmt));

    return count;
  }

  size_t CountRowsForEntry(sqlite3_stmt* sql_stmt,
                           const sqlite3_int64 row_id) const {
    Sqlite3StmtContext context(sql_stmt);
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, 1, row_id));
    if (SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW) {
      return static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
    }
    return 0;
  }

  size_t SumColumn(const std::string& column, const std::string& table) const {
    const std::string sql =
        StringPrintf("SELECT SUM(%s) FROM %s;", column.c_str(), table.c_str());

    sqlite3_stmt* sql_stmt;
    SQLITE3_CALL(sqlite3_prepare_v2(
        THROW_CHECK_NOTNULL(database_), sql.c_str(), -1, &sql_stmt, 0));

    size_t sum = 0;
    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
    if (rc == SQLITE_ROW) {
      sum = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
    }

    SQLITE3_CALL(sqlite3_finalize(sql_stmt));

    return sum;
  }

  size_t MaxColumn(const std::string& column, const std::string& table) const {
    const std::string sql =
        StringPrintf("SELECT MAX(%s) FROM %s;", column.c_str(), table.c_str());

    sqlite3_stmt* sql_stmt;
    SQLITE3_CALL(sqlite3_prepare_v2(
        THROW_CHECK_NOTNULL(database_), sql.c_str(), -1, &sql_stmt, 0));

    size_t max = 0;
    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
    if (rc == SQLITE_ROW) {
      max = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
    }

    SQLITE3_CALL(sqlite3_finalize(sql_stmt));

    return max;
  }

  sqlite3* database_ = nullptr;

  // Check if elements got removed from the database to only apply
  // the VACUUM command in such case
  mutable bool database_entry_deleted_ = false;

  // Ensure that only one database object at a time updates the schema of a
  // database. Since the schema is updated every time a database is opened, this
  // is to ensure that there are no race conditions ("database locked" error
  // messages) when the user actually only intends to read from the database,
  // which requires to open it.
  static std::mutex update_schema_mutex_;

  // A collection of all `sqlite3_stmt` objects for deletion in the destructor.
  std::vector<sqlite3_stmt**> sql_stmts_;

  // num_*
  sqlite3_stmt* sql_stmt_num_keypoints_ = nullptr;
  sqlite3_stmt* sql_stmt_num_descriptors_ = nullptr;

  // exists_*
  sqlite3_stmt* sql_stmt_exists_rig_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_camera_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_frame_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_image_id_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_image_name_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_pose_prior_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_keypoints_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_descriptors_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_exists_two_view_geometry_ = nullptr;

  // update_*
  sqlite3_stmt* sql_stmt_update_rig_ = nullptr;
  sqlite3_stmt* sql_stmt_update_camera_ = nullptr;
  sqlite3_stmt* sql_stmt_update_frame_ = nullptr;
  sqlite3_stmt* sql_stmt_update_image_ = nullptr;
  sqlite3_stmt* sql_stmt_update_pose_prior_ = nullptr;
  sqlite3_stmt* sql_stmt_update_keypoints_ = nullptr;

  // read_*
  sqlite3_stmt* sql_stmt_read_rig_ = nullptr;
  sqlite3_stmt* sql_stmt_read_rigs_ = nullptr;
  sqlite3_stmt* sql_stmt_read_rig_with_sensor_ = nullptr;
  sqlite3_stmt* sql_stmt_read_rig_with_ref_sensor_ = nullptr;
  sqlite3_stmt* sql_stmt_read_camera_ = nullptr;
  sqlite3_stmt* sql_stmt_read_cameras_ = nullptr;
  sqlite3_stmt* sql_stmt_read_frame_ = nullptr;
  sqlite3_stmt* sql_stmt_read_frames_ = nullptr;
  sqlite3_stmt* sql_stmt_read_image_id_ = nullptr;
  sqlite3_stmt* sql_stmt_read_image_with_name_ = nullptr;
  sqlite3_stmt* sql_stmt_read_images_ = nullptr;
  sqlite3_stmt* sql_stmt_read_pose_prior_ = nullptr;
  sqlite3_stmt* sql_stmt_read_pose_priors_ = nullptr;
  sqlite3_stmt* sql_stmt_read_keypoints_ = nullptr;
  sqlite3_stmt* sql_stmt_read_descriptors_ = nullptr;
  sqlite3_stmt* sql_stmt_read_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_read_matches_all_ = nullptr;
  sqlite3_stmt* sql_stmt_read_num_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_read_two_view_geometry_ = nullptr;
  sqlite3_stmt* sql_stmt_read_two_view_geometries_ = nullptr;
  sqlite3_stmt* sql_stmt_read_two_view_geometry_num_inliers_ = nullptr;

  // write_*
  sqlite3_stmt* sql_stmt_write_rig_ = nullptr;
  sqlite3_stmt* sql_stmt_write_rig_sensor_ = nullptr;
  sqlite3_stmt* sql_stmt_write_camera_ = nullptr;
  sqlite3_stmt* sql_stmt_write_frame_ = nullptr;
  sqlite3_stmt* sql_stmt_write_frame_data_ = nullptr;
  sqlite3_stmt* sql_stmt_write_image_ = nullptr;
  sqlite3_stmt* sql_stmt_write_pose_prior_ = nullptr;
  sqlite3_stmt* sql_stmt_write_keypoints_ = nullptr;
  sqlite3_stmt* sql_stmt_write_descriptors_ = nullptr;
  sqlite3_stmt* sql_stmt_write_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_write_two_view_geometry_ = nullptr;

  // delete_*
  sqlite3_stmt* sql_stmt_delete_rig_sensors_ = nullptr;
  sqlite3_stmt* sql_stmt_delete_frame_data_ = nullptr;
  sqlite3_stmt* sql_stmt_delete_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_delete_two_view_geometry_ = nullptr;

  // clear_*
  sqlite3_stmt* sql_stmt_clear_rigs_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_cameras_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_frames_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_images_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_pose_priors_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_descriptors_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_keypoints_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_matches_ = nullptr;
  sqlite3_stmt* sql_stmt_clear_two_view_geometries_ = nullptr;
};

std::mutex SqliteDatabase::update_schema_mutex_;

}  // namespace

std::shared_ptr<Database> OpenSqliteDatabase(const std::string& path) {
  return SqliteDatabase::Open(path);
}

}  // namespace colmap
