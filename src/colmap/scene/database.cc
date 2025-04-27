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
#include "colmap/util/sqlite3_utils.h"
#include "colmap/util/string.h"
#include "colmap/util/version.h"

#include <fstream>
#include <memory>

namespace colmap {
namespace {

void SwapFeatureMatchesBlob(FeatureMatchesBlob* matches) {
  matches->col(0).swap(matches->col(1));
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

Rig ReadRigRow(sqlite3_stmt* sql_stmt) {
  Rig rig;

  rig.SetRigId(static_cast<rig_t>(sqlite3_column_int64(sql_stmt, 0)));

  sensor_t ref_sensor_id;
  ref_sensor_id.id = static_cast<uint32_t>(sqlite3_column_int64(sql_stmt, 1));
  ref_sensor_id.type =
      static_cast<SensorType>(sqlite3_column_int64(sql_stmt, 2));
  rig.AddRefSensor(ref_sensor_id);

  return rig;
}

void ReadRigSensorRow(sqlite3_stmt* sql_stmt, Rig* rig) {
  sensor_t sensor_id;
  sensor_id.id = static_cast<uint32_t>(sqlite3_column_int64(sql_stmt, 0));
  sensor_id.type = static_cast<SensorType>(sqlite3_column_int64(sql_stmt, 1));

  std::optional<std::stringstream> sensor_from_rig_stream =
      BlobColumnToStringStream(sql_stmt, 2);

  std::optional<Rigid3d> sensor_from_rig;
  if (sensor_from_rig_stream.has_value()) {
    std::stringstream* sensor_from_rig_stream_ptr =
        &sensor_from_rig_stream.value();
    sensor_from_rig = Rigid3d();
    sensor_from_rig->rotation.w() =
        ReadBinaryLittleEndian<double>(sensor_from_rig_stream_ptr);
    sensor_from_rig->rotation.x() =
        ReadBinaryLittleEndian<double>(sensor_from_rig_stream_ptr);
    sensor_from_rig->rotation.y() =
        ReadBinaryLittleEndian<double>(sensor_from_rig_stream_ptr);
    sensor_from_rig->rotation.z() =
        ReadBinaryLittleEndian<double>(sensor_from_rig_stream_ptr);
    sensor_from_rig->translation.x() =
        ReadBinaryLittleEndian<double>(sensor_from_rig_stream_ptr);
    sensor_from_rig->translation.y() =
        ReadBinaryLittleEndian<double>(sensor_from_rig_stream_ptr);
    sensor_from_rig->translation.z() =
        ReadBinaryLittleEndian<double>(sensor_from_rig_stream_ptr);
  }

  rig->AddSensor(sensor_id, sensor_from_rig);
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

Frame ReadFrameRow(sqlite3_stmt* sql_stmt) {
  Frame frame;
  frame.SetFrameId(static_cast<rig_t>(sqlite3_column_int64(sql_stmt, 0)));
  frame.SetRigId(static_cast<rig_t>(sqlite3_column_int64(sql_stmt, 1)));
  return frame;
}

void ReadFrameData(sqlite3_stmt* sql_stmt, Frame* frame) {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, 1, frame->FrameId()));
  while (SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW) {
    data_t data_id;
    data_id.id = static_cast<uint32_t>(sqlite3_column_int64(sql_stmt, 0));
    data_id.sensor_id.id =
        static_cast<uint32_t>(sqlite3_column_int64(sql_stmt, 1));
    data_id.sensor_id.type =
        static_cast<SensorType>(sqlite3_column_int64(sql_stmt, 2));
    frame->AddDataId(data_id);
  }
  SQLITE3_CALL(sqlite3_reset(sql_stmt));
}

Image ReadImageRow(sqlite3_stmt* sql_stmt) {
  Image image;

  image.SetImageId(static_cast<image_t>(sqlite3_column_int64(sql_stmt, 0)));
  image.SetName(std::string(
      reinterpret_cast<const char*>(sqlite3_column_text(sql_stmt, 1))));
  image.SetCameraId(static_cast<camera_t>(sqlite3_column_int64(sql_stmt, 2)));

  return image;
}

void WriteRigSensors(const rig_t rig_id,
                     const Rig& rig,
                     sqlite3_stmt* sql_stmt) {
  for (const auto& [sensor_id, sensor_from_rig] : rig.Sensors()) {
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, 1, rig_id));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt, 2, static_cast<sqlite3_int64>(sensor_id.id)));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt, 3, static_cast<sqlite3_int64>(sensor_id.type)));

    // Note that the sensor_from_rig_bytes object must outlive the call to
    // sqlite3_step.
    std::string sensor_from_rig_bytes;
    if (sensor_from_rig.has_value()) {
      std::ostringstream stream;
      WriteBinaryLittleEndian<double>(&stream, sensor_from_rig->rotation.w());
      WriteBinaryLittleEndian<double>(&stream, sensor_from_rig->rotation.x());
      WriteBinaryLittleEndian<double>(&stream, sensor_from_rig->rotation.y());
      WriteBinaryLittleEndian<double>(&stream, sensor_from_rig->rotation.z());
      WriteBinaryLittleEndian<double>(&stream,
                                      sensor_from_rig->translation.x());
      WriteBinaryLittleEndian<double>(&stream,
                                      sensor_from_rig->translation.y());
      WriteBinaryLittleEndian<double>(&stream,
                                      sensor_from_rig->translation.z());
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
    SQLITE3_CALL(sqlite3_reset(sql_stmt));
  }
}

void WriteFrameData(const frame_t frame_id,
                    const Frame& frame,
                    sqlite3_stmt* sql_stmt) {
  for (const data_t& data_id : frame.DataIds()) {
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, 1, frame_id));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt, 2, static_cast<sqlite3_int64>(data_id.id)));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt, 3, static_cast<sqlite3_int64>(data_id.sensor_id.id)));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt, 4, static_cast<sqlite3_int64>(data_id.sensor_id.type)));
    SQLITE3_CALL(sqlite3_step(sql_stmt));
    SQLITE3_CALL(sqlite3_reset(sql_stmt));
  }
}

}  // namespace

const size_t Database::kMaxNumImages =
    static_cast<size_t>(std::numeric_limits<int32_t>::max());

const std::string Database::kInMemoryDatabasePath = ":memory:";

std::mutex Database::update_schema_mutex_;

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
      path.c_str(),
      &database_,
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

  // Enable auto vacuum to reduce DB file size
  SQLITE3_EXEC(database_, "PRAGMA auto_vacuum=1", nullptr);

  CreateTables();
  UpdateSchema();
  PrepareSQLStatements();
}

void Database::Close() {
  if (database_ != nullptr) {
    FinalizeSQLStatements();
    if (database_entry_deleted_) {
      SQLITE3_EXEC(database_, "VACUUM", nullptr);
      database_entry_deleted_ = false;
    }
    sqlite3_close_v2(database_);
    database_ = nullptr;
  }
}

bool Database::ExistsRig(const rig_t rig_id) const {
  return ExistsRowId(sql_stmt_exists_rig_, rig_id);
}

bool Database::ExistsCamera(const camera_t camera_id) const {
  return ExistsRowId(sql_stmt_exists_camera_, camera_id);
}

bool Database::ExistsFrame(const frame_t frame_id) const {
  return ExistsRowId(sql_stmt_exists_frame_, frame_id);
}

bool Database::ExistsImage(const image_t image_id) const {
  return ExistsRowId(sql_stmt_exists_image_id_, image_id);
}

bool Database::ExistsImageWithName(const std::string& name) const {
  return ExistsRowString(sql_stmt_exists_image_name_, name);
}

bool Database::ExistsPosePrior(const image_t image_id) const {
  return ExistsRowId(sql_stmt_exists_pose_prior_, image_id);
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
  return ExistsRowId(sql_stmt_exists_two_view_geometry_,
                     ImagePairToPairId(image_id1, image_id2));
}

size_t Database::NumRigs() const { return CountRows("rigs"); }

size_t Database::NumCameras() const { return CountRows("cameras"); }

size_t Database::NumFrames() const { return CountRows("frames"); }

size_t Database::NumImages() const { return CountRows("images"); }

size_t Database::NumPosePriors() const { return CountRows("pose_priors"); }

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
  return SumColumn("rows", "two_view_geometries");
}

size_t Database::NumMatchedImagePairs() const { return CountRows("matches"); }

size_t Database::NumVerifiedImagePairs() const {
  return CountRows("two_view_geometries");
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

Frame Database::ReadFrame(const frame_t frame_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_frame_, 1, frame_id));

  Frame frame;
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_frame_));
  if (rc == SQLITE_ROW) {
    frame = ReadFrameRow(sql_stmt_read_frame_);
  }
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_frame_));

  ReadFrameData(sql_stmt_read_frame_data_, &frame);

  return frame;
}

std::vector<Frame> Database::ReadAllFrames() const {
  std::vector<Frame> frames;

  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_frames_)) == SQLITE_ROW) {
    frames.push_back(ReadFrameRow(sql_stmt_read_frames_));
  }
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_frames_));

  for (Frame& frame : frames) {
    ReadFrameData(sql_stmt_read_frame_data_, &frame);
  }

  return frames;
}

Rig Database::ReadRig(const rig_t rig_id) const {
  Rig rig;

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_rig_, 1, rig_id));
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_rig_));
  if (rc == SQLITE_ROW) {
    rig = ReadRigRow(sql_stmt_read_rig_);
  }
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_rig_));

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_rig_sensors_, 1, rig_id));
  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_rig_sensors_)) == SQLITE_ROW) {
    ReadRigSensorRow(sql_stmt_read_rig_sensors_, &rig);
  }
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_rig_sensors_));

  return rig;
}

std::optional<Rig> Database::ReadRigWithSensor(sensor_t sensor_id) const {
  auto find_rig_id_with_sensor = [&sensor_id](sqlite3_stmt* sql_stmt) {
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt, 1, static_cast<sqlite3_int64>(sensor_id.id)));
    SQLITE3_CALL(sqlite3_bind_int64(
        sql_stmt, 2, static_cast<sqlite3_int64>(sensor_id.type)));
    std::optional<rig_t> rig_id;
    if (SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW) {
      rig_id = static_cast<rig_t>(sqlite3_column_int64(sql_stmt, 0));
    }
    SQLITE3_CALL(sqlite3_reset(sql_stmt));
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

std::vector<Rig> Database::ReadAllRigs() const {
  std::vector<Rig> rigs;
  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_rigs_)) == SQLITE_ROW) {
    rigs.push_back(ReadRigRow(sql_stmt_read_rigs_));
  }
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_rigs_));

  for (Rig& rig : rigs) {
    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_read_rig_sensors_, 1, rig.RigId()));
    while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_rig_sensors_)) ==
           SQLITE_ROW) {
      ReadRigSensorRow(sql_stmt_read_rig_sensors_, &rig);
    }
    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_rig_sensors_));
  }

  return rigs;
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

std::optional<Image> Database::ReadImageWithName(
    const std::string& name) const {
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

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_image_with_name_));

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

PosePrior Database::ReadPosePrior(const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_pose_prior_, 1, image_id));
  PosePrior prior;
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_pose_prior_));
  if (rc == SQLITE_ROW) {
    prior.position =
        ReadStaticMatrixBlob<Eigen::Vector3d>(sql_stmt_read_pose_prior_, rc, 1);
    prior.coordinate_system = static_cast<PosePrior::CoordinateSystem>(
        sqlite3_column_int64(sql_stmt_read_pose_prior_, 2));
    prior.position_covariance =
        ReadStaticMatrixBlob<Eigen::Matrix3d>(sql_stmt_read_pose_prior_, rc, 3);
  }
  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_pose_prior_));
  return prior;
}

FeatureKeypointsBlob Database::ReadKeypointsBlob(const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_keypoints_, 1, image_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_keypoints_));
  FeatureKeypointsBlob blob = ReadDynamicMatrixBlob<FeatureKeypointsBlob>(
      sql_stmt_read_keypoints_, rc, 0);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_keypoints_));
  return blob;
}

FeatureKeypoints Database::ReadKeypoints(const image_t image_id) const {
  return FeatureKeypointsFromBlob(ReadKeypointsBlob(image_id));
}

FeatureDescriptors Database::ReadDescriptors(const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_descriptors_, 1, image_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_descriptors_));
  FeatureDescriptors descriptors = ReadDynamicMatrixBlob<FeatureDescriptors>(
      sql_stmt_read_descriptors_, rc, 0);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_descriptors_));

  return descriptors;
}

FeatureMatchesBlob Database::ReadMatchesBlob(image_t image_id1,
                                             image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_matches_, 1, pair_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_));
  FeatureMatchesBlob blob =
      ReadDynamicMatrixBlob<FeatureMatchesBlob>(sql_stmt_read_matches_, rc, 0);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_matches_));

  if (SwapImagePair(image_id1, image_id2)) {
    SwapFeatureMatchesBlob(&blob);
  }
  return blob;
}

FeatureMatches Database::ReadMatches(image_t image_id1,
                                     image_t image_id2) const {
  return FeatureMatchesFromBlob(ReadMatchesBlob(image_id1, image_id2));
}

std::vector<std::pair<image_pair_t, FeatureMatchesBlob>>
Database::ReadAllMatchesBlob() const {
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

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_matches_all_));

  return all_matches;
}

std::vector<std::pair<image_pair_t, FeatureMatches>> Database::ReadAllMatches()
    const {
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

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_matches_all_));

  return all_matches;
}

TwoViewGeometry Database::ReadTwoViewGeometry(const image_t image_id1,
                                              const image_t image_id2) const {
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

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_two_view_geometry_));

  two_view_geometry.inlier_matches = FeatureMatchesFromBlob(blob);
  two_view_geometry.F.transposeInPlace();
  two_view_geometry.E.transposeInPlace();
  two_view_geometry.H.transposeInPlace();

  if (SwapImagePair(image_id1, image_id2)) {
    two_view_geometry.Invert();
  }

  return two_view_geometry;
}

std::vector<std::pair<image_pair_t, TwoViewGeometry>>
Database::ReadTwoViewGeometries() const {
  std::vector<std::pair<image_pair_t, TwoViewGeometry>> all_two_view_geometries;

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

    all_two_view_geometries.emplace_back(pair_id, std::move(two_view_geometry));
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_two_view_geometries_));

  return all_two_view_geometries;
}

std::vector<std::pair<image_pair_t, int>>
Database::ReadTwoViewGeometryNumInliers() const {
  std::vector<std::pair<image_pair_t, int>> num_inliers;
  while (SQLITE3_CALL(sqlite3_step(
             sql_stmt_read_two_view_geometry_num_inliers_)) == SQLITE_ROW) {
    const image_pair_t pair_id = static_cast<image_pair_t>(
        sqlite3_column_int64(sql_stmt_read_two_view_geometry_num_inliers_, 0));

    const int rows = static_cast<int>(
        sqlite3_column_int64(sql_stmt_read_two_view_geometry_num_inliers_, 1));
    num_inliers.emplace_back(pair_id, rows);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_two_view_geometry_num_inliers_));

  return num_inliers;
}

rig_t Database::WriteRig(const Rig& rig, const bool use_rig_id) const {
  THROW_CHECK(rig.NumSensors() > 0) << "Rig must have at least one sensor";

  if (use_rig_id) {
    THROW_CHECK(!ExistsRig(rig.RigId())) << "rig_id must be unique";
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_rig_, 1, rig.RigId()));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_add_rig_, 1));
  }

  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_add_rig_, 2, static_cast<sqlite3_int64>(rig.RefSensorId().id)));
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_add_rig_,
                         3,
                         static_cast<sqlite3_int64>(rig.RefSensorId().type)));

  SQLITE3_CALL(sqlite3_step(sql_stmt_add_rig_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_add_rig_));

  const rig_t rig_id = static_cast<rig_t>(sqlite3_last_insert_rowid(database_));

  WriteRigSensors(rig_id, rig, sql_stmt_add_rig_sensor_);

  return rig_id;
}

camera_t Database::WriteCamera(const Camera& camera,
                               const bool use_camera_id) const {
  if (use_camera_id) {
    THROW_CHECK(!ExistsCamera(camera.camera_id)) << "camera_id must be unique";
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 1, camera.camera_id));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_add_camera_, 1));
  }

  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_add_camera_, 2, static_cast<sqlite3_int64>(camera.model_id)));
  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_add_camera_, 3, static_cast<sqlite3_int64>(camera.width)));
  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_add_camera_, 4, static_cast<sqlite3_int64>(camera.height)));

  const size_t num_params_bytes = sizeof(double) * camera.params.size();
  SQLITE3_CALL(sqlite3_bind_blob(sql_stmt_add_camera_,
                                 5,
                                 camera.params.data(),
                                 static_cast<int>(num_params_bytes),
                                 SQLITE_STATIC));

  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_add_camera_, 6, camera.has_prior_focal_length));

  SQLITE3_CALL(sqlite3_step(sql_stmt_add_camera_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_add_camera_));

  return static_cast<camera_t>(sqlite3_last_insert_rowid(database_));
}

frame_t Database::WriteFrame(const Frame& frame,
                             const bool use_frame_id) const {
  if (use_frame_id) {
    THROW_CHECK(!ExistsFrame(frame.FrameId())) << "frame_id must be unique";
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_frame_, 1, frame.FrameId()));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_add_frame_, 1));
  }

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_frame_, 2, frame.RigId()));

  SQLITE3_CALL(sqlite3_step(sql_stmt_add_frame_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_add_frame_));

  const frame_t frame_id =
      static_cast<frame_t>(sqlite3_last_insert_rowid(database_));

  WriteFrameData(frame_id, frame, sql_stmt_add_frame_data_);

  return frame_id;
}

image_t Database::WriteImage(const Image& image,
                             const bool use_image_id) const {
  if (use_image_id) {
    THROW_CHECK(!ExistsImage(image.ImageId())) << "image_id must be unique";
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_image_, 1, image.ImageId()));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_add_image_, 1));
  }

  SQLITE3_CALL(sqlite3_bind_text(sql_stmt_add_image_,
                                 2,
                                 image.Name().c_str(),
                                 static_cast<int>(image.Name().size()),
                                 SQLITE_STATIC));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_image_, 3, image.CameraId()));

  SQLITE3_CALL(sqlite3_step(sql_stmt_add_image_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_add_image_));

  return static_cast<image_t>(sqlite3_last_insert_rowid(database_));
}

void Database::WritePosePrior(const image_t image_id,
                              const PosePrior& pose_prior) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_pose_prior_, 1, image_id));
  WriteStaticMatrixBlob(sql_stmt_write_pose_prior_, pose_prior.position, 2);
  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_write_pose_prior_,
      3,
      static_cast<sqlite3_int64>(pose_prior.coordinate_system)));
  WriteStaticMatrixBlob(
      sql_stmt_write_pose_prior_, pose_prior.position_covariance, 4);
  SQLITE3_CALL(sqlite3_step(sql_stmt_write_pose_prior_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_pose_prior_));
}

void Database::WriteKeypoints(const image_t image_id,
                              const FeatureKeypoints& keypoints) const {
  WriteKeypoints(image_id, FeatureKeypointsToBlob(keypoints));
}

void Database::WriteKeypoints(const image_t image_id,
                              const FeatureKeypointsBlob& blob) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_keypoints_, 1, image_id));
  WriteDynamicMatrixBlob(sql_stmt_write_keypoints_, blob, 2);

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_keypoints_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_keypoints_));
}

void Database::WriteDescriptors(const image_t image_id,
                                const FeatureDescriptors& descriptors) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_descriptors_, 1, image_id));
  WriteDynamicMatrixBlob(sql_stmt_write_descriptors_, descriptors, 2);

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_descriptors_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_descriptors_));
}

void Database::WriteMatches(const image_t image_id1,
                            const image_t image_id2,
                            const FeatureMatches& matches) const {
  WriteMatches(image_id1, image_id2, FeatureMatchesToBlob(matches));
}

void Database::WriteMatches(const image_t image_id1,
                            const image_t image_id2,
                            const FeatureMatchesBlob& blob) const {
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
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_matches_));
}

void Database::WriteTwoViewGeometry(
    const image_t image_id1,
    const image_t image_id2,
    const TwoViewGeometry& two_view_geometry) const {
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
  WriteDynamicMatrixBlob(sql_stmt_write_two_view_geometry_, inlier_matches, 2);

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

  if (two_view_geometry_ptr->inlier_matches.size() > 0) {
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, Ft, 6);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, Et, 7);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, Ht, 8);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_, quat_wxyz, 9);
    WriteStaticMatrixBlob(sql_stmt_write_two_view_geometry_,
                          two_view_geometry_ptr->cam2_from_cam1.translation,
                          10);
  } else {
    WriteStaticMatrixBlob(
        sql_stmt_write_two_view_geometry_, Eigen::MatrixXd(0, 0), 6);
    WriteStaticMatrixBlob(
        sql_stmt_write_two_view_geometry_, Eigen::MatrixXd(0, 0), 7);
    WriteStaticMatrixBlob(
        sql_stmt_write_two_view_geometry_, Eigen::MatrixXd(0, 0), 8);
    WriteStaticMatrixBlob(
        sql_stmt_write_two_view_geometry_, Eigen::MatrixXd(0, 0), 9);
    WriteStaticMatrixBlob(
        sql_stmt_write_two_view_geometry_, Eigen::MatrixXd(0, 0), 10);
  }

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_two_view_geometry_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_two_view_geometry_));
}

void Database::UpdateRig(const Rig& rig) const {
  // Update rig.
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_update_rig_,
                         1,
                         static_cast<sqlite3_int64>(rig.RefSensorId().id)));
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_update_rig_,
                         2,
                         static_cast<sqlite3_int64>(rig.RefSensorId().type)));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_rig_, 3, rig.RigId()));
  SQLITE3_CALL(sqlite3_step(sql_stmt_update_rig_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_rig_));

  // Clear the rig sensors.
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_delete_rig_sensors_, 1, rig.RigId()));
  SQLITE3_CALL(sqlite3_step(sql_stmt_delete_rig_sensors_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_delete_rig_sensors_));
  // Write the updated rig sensors.
  WriteRigSensors(rig.RigId(), rig, sql_stmt_add_rig_sensor_);
}

void Database::UpdateCamera(const Camera& camera) const {
  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_update_camera_, 1, static_cast<sqlite3_int64>(camera.model_id)));
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
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_camera_));
}

void Database::UpdateFrame(const Frame& frame) const {
  // Update frame.
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_frame_, 1, frame.RigId()));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_frame_, 2, frame.FrameId()));
  SQLITE3_CALL(sqlite3_step(sql_stmt_update_frame_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_frame_));

  // Clear the frame data.
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_delete_frame_data_, 1, frame.FrameId()));
  SQLITE3_CALL(sqlite3_step(sql_stmt_delete_frame_data_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_delete_frame_data_));
  // Write the updated frame data.
  WriteFrameData(frame.FrameId(), frame, sql_stmt_add_frame_data_);
}

void Database::UpdateImage(const Image& image) const {
  SQLITE3_CALL(sqlite3_bind_text(sql_stmt_update_image_,
                                 1,
                                 image.Name().c_str(),
                                 static_cast<int>(image.Name().size()),
                                 SQLITE_STATIC));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_image_, 2, image.CameraId()));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_image_, 3, image.ImageId()));

  SQLITE3_CALL(sqlite3_step(sql_stmt_update_image_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_image_));
}

void Database::UpdatePosePrior(image_t image_id,
                               const PosePrior& pose_prior) const {
  WriteStaticMatrixBlob(sql_stmt_update_pose_prior_, pose_prior.position, 1);
  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_update_pose_prior_,
      2,
      static_cast<sqlite3_int64>(pose_prior.coordinate_system)));
  WriteStaticMatrixBlob(
      sql_stmt_update_pose_prior_, pose_prior.position_covariance, 3);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_pose_prior_, 4, image_id));

  SQLITE3_CALL(sqlite3_step(sql_stmt_update_pose_prior_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_pose_prior_));
}

void Database::DeleteMatches(const image_t image_id1,
                             const image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(
      sql_stmt_delete_matches_, 1, static_cast<sqlite3_int64>(pair_id)));
  SQLITE3_CALL(sqlite3_step(sql_stmt_delete_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_delete_matches_));
  database_entry_deleted_ = true;
}

void Database::DeleteInlierMatches(const image_t image_id1,
                                   const image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_delete_two_view_geometry_,
                                  1,
                                  static_cast<sqlite3_int64>(pair_id)));
  SQLITE3_CALL(sqlite3_step(sql_stmt_delete_two_view_geometry_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_delete_two_view_geometry_));
  database_entry_deleted_ = true;
}

void Database::ClearAllTables() const {
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

void Database::ClearRigs() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_rigs_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_rigs_));
  database_entry_deleted_ = true;
}

void Database::ClearCameras() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_cameras_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_cameras_));
  database_entry_deleted_ = true;
}

void Database::ClearFrames() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_frames_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_frames_));
  database_entry_deleted_ = true;
}

void Database::ClearImages() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_images_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_images_));
  database_entry_deleted_ = true;
}

void Database::ClearPosePriors() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_pose_priors_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_pose_priors_));
  database_entry_deleted_ = true;
}

void Database::ClearDescriptors() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_descriptors_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_descriptors_));
  database_entry_deleted_ = true;
}

void Database::ClearKeypoints() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_keypoints_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_keypoints_));
  database_entry_deleted_ = true;
}

void Database::ClearMatches() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_matches_));
  database_entry_deleted_ = true;
}

void Database::ClearTwoViewGeometries() const {
  SQLITE3_CALL(sqlite3_step(sql_stmt_clear_two_view_geometries_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_clear_two_view_geometries_));
  database_entry_deleted_ = true;
}

void Database::Merge(const Database& database1,
                     const Database& database2,
                     Database* merged_database) {
  // Merge the cameras.

  std::unordered_map<camera_t, camera_t> new_camera_ids1;
  for (const auto& camera : database1.ReadAllCameras()) {
    const camera_t new_camera_id = merged_database->WriteCamera(camera);
    new_camera_ids1.emplace(camera.camera_id, new_camera_id);
  }

  std::unordered_map<camera_t, camera_t> new_camera_ids2;
  for (const auto& camera : database2.ReadAllCameras()) {
    const camera_t new_camera_id = merged_database->WriteCamera(camera);
    new_camera_ids2.emplace(camera.camera_id, new_camera_id);
  }

  // Merge the rigs.

  auto update_rig =
      [](const Rig& rig,
         const std::unordered_map<camera_t, camera_t>& new_camera_ids) {
        if (rig.NumSensors() == 0) {
          return rig;
        }
        Rig updated_rig;
        updated_rig.SetRigId(rig.RigId());
        sensor_t ref_sensor_id = rig.RefSensorId();
        if (ref_sensor_id.type == SensorType::CAMERA) {
          ref_sensor_id.id = new_camera_ids.at(ref_sensor_id.id);
        }
        updated_rig.AddRefSensor(ref_sensor_id);
        for (auto [sensor_id, sensor_from_rig] : rig.Sensors()) {
          sensor_t updated_sensor_id = sensor_id;
          if (sensor_id.type == SensorType::CAMERA) {
            updated_sensor_id.id = new_camera_ids.at(sensor_id.id);
          }
          updated_rig.AddSensor(updated_sensor_id, sensor_from_rig);
        }
        return updated_rig;
      };

  std::unordered_map<rig_t, rig_t> new_rig_ids1;
  for (auto& rig : database1.ReadAllRigs()) {
    const rig_t new_rig_id =
        merged_database->WriteRig(update_rig(rig, new_camera_ids1));
    new_rig_ids1.emplace(rig.RigId(), new_rig_id);
  }

  std::unordered_map<rig_t, rig_t> new_rig_ids2;
  for (auto& rig : database2.ReadAllRigs()) {
    const rig_t new_rig_id =
        merged_database->WriteRig(update_rig(rig, new_camera_ids2));
    new_rig_ids2.emplace(rig.RigId(), new_rig_id);
  }

  // Merge the images.

  std::unordered_map<image_t, image_t> new_image_ids1;
  for (auto& image : database1.ReadAllImages()) {
    image.SetCameraId(new_camera_ids1.at(image.CameraId()));
    THROW_CHECK(!merged_database->ExistsImageWithName(image.Name()))
        << "The two databases must not contain images with the same name, but "
           "the there are images with name "
        << image.Name() << " in both databases";
    const image_t new_image_id = merged_database->WriteImage(image);
    new_image_ids1.emplace(image.ImageId(), new_image_id);
    const auto keypoints = database1.ReadKeypoints(image.ImageId());
    const auto descriptors = database1.ReadDescriptors(image.ImageId());
    merged_database->WriteKeypoints(new_image_id, keypoints);
    merged_database->WriteDescriptors(new_image_id, descriptors);
    if (database1.ExistsPosePrior(image.ImageId())) {
      merged_database->WritePosePrior(new_image_id,
                                      database1.ReadPosePrior(image.ImageId()));
    }
  }

  std::unordered_map<image_t, image_t> new_image_ids2;
  for (auto& image : database2.ReadAllImages()) {
    image.SetCameraId(new_camera_ids2.at(image.CameraId()));
    THROW_CHECK(!merged_database->ExistsImageWithName(image.Name()))
        << "The two databases must not contain images with the same name, but "
           "the there are images with name "
        << image.Name() << " in both databases";
    const image_t new_image_id = merged_database->WriteImage(image);
    new_image_ids2.emplace(image.ImageId(), new_image_id);
    const auto keypoints = database2.ReadKeypoints(image.ImageId());
    const auto descriptors = database2.ReadDescriptors(image.ImageId());
    merged_database->WriteKeypoints(new_image_id, keypoints);
    merged_database->WriteDescriptors(new_image_id, descriptors);
    if (database2.ExistsPosePrior(image.ImageId())) {
      merged_database->WritePosePrior(new_image_id,
                                      database2.ReadPosePrior(image.ImageId()));
    }
  }

  // Merge the frames.

  auto update_frame =
      [](const Frame& frame,
         const std::unordered_map<camera_t, camera_t>& new_camera_ids,
         const std::unordered_map<image_t, image_t>& new_image_ids) {
        Frame updated_frame;
        updated_frame.SetFrameId(frame.FrameId());
        updated_frame.SetRigId(frame.RigId());
        for (data_t data_id : frame.DataIds()) {
          if (data_id.sensor_id.type == SensorType::CAMERA) {
            data_id.id = new_image_ids.at(data_id.id);
            data_id.sensor_id.id = new_camera_ids.at(data_id.sensor_id.id);
            updated_frame.AddDataId(data_id);
          } else {
            std::ostringstream error;
            error << "Data type not supported: " << data_id.sensor_id.type;
            throw std::runtime_error(error.str());
          }
        }
        return updated_frame;
      };

  for (Frame& frame : database1.ReadAllFrames()) {
    merged_database->WriteFrame(
        update_frame(frame, new_camera_ids1, new_image_ids1));
  }

  for (Frame& frame : database2.ReadAllFrames()) {
    merged_database->WriteFrame(
        update_frame(frame, new_camera_ids2, new_image_ids2));
  }

  // Merge the matches.

  for (const auto& matches : database1.ReadAllMatches()) {
    const auto image_pair = Database::PairIdToImagePair(matches.first);

    const image_t new_image_id1 = new_image_ids1.at(image_pair.first);
    const image_t new_image_id2 = new_image_ids1.at(image_pair.second);

    merged_database->WriteMatches(new_image_id1, new_image_id2, matches.second);
  }

  for (const auto& matches : database2.ReadAllMatches()) {
    const auto image_pair = Database::PairIdToImagePair(matches.first);

    const image_t new_image_id1 = new_image_ids2.at(image_pair.first);
    const image_t new_image_id2 = new_image_ids2.at(image_pair.second);

    merged_database->WriteMatches(new_image_id1, new_image_id2, matches.second);
  }

  // Merge the two-view geometries.

  for (const auto& [pair_id, two_view_geometry] :
       database1.ReadTwoViewGeometries()) {
    const auto image_pair = Database::PairIdToImagePair(pair_id);

    const image_t new_image_id1 = new_image_ids1.at(image_pair.first);
    const image_t new_image_id2 = new_image_ids1.at(image_pair.second);

    merged_database->WriteTwoViewGeometry(
        new_image_id1, new_image_id2, two_view_geometry);
  }

  for (const auto& [pair_id, two_view_geometry] :
       database2.ReadTwoViewGeometries()) {
    const auto image_pair = Database::PairIdToImagePair(pair_id);

    const image_t new_image_id1 = new_image_ids2.at(image_pair.first);
    const image_t new_image_id2 = new_image_ids2.at(image_pair.second);

    merged_database->WriteTwoViewGeometry(
        new_image_id1, new_image_id2, two_view_geometry);
  }
}

void Database::BeginTransaction() const {
  SQLITE3_EXEC(database_, "BEGIN TRANSACTION", nullptr);
}

void Database::EndTransaction() const {
  SQLITE3_EXEC(database_, "END TRANSACTION", nullptr);
}

void Database::PrepareSQLStatements() {
  sql_stmts_.clear();

  auto prepare_sql_stmt = [this](const std::string_view sql,
                                 sqlite3_stmt** sql_stmt) {
    VLOG(3) << "Preparing SQL statement: " << sql;
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.data(), -1, sql_stmt, 0));
    sql_stmts_.push_back(*sql_stmt);
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
  prepare_sql_stmt("SELECT 1 FROM pose_priors WHERE image_id = ?;",
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
  // add_*
  //////////////////////////////////////////////////////////////////////////////
  prepare_sql_stmt(
      "INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) "
      "VALUES(?, ?, ?);",
      &sql_stmt_add_rig_);
  prepare_sql_stmt(
      "INSERT INTO rig_sensors(rig_id, sensor_id, sensor_type, "
      "sensor_from_rig) VALUES(?, ?, ?, ?);",
      &sql_stmt_add_rig_sensor_);
  prepare_sql_stmt(
      "INSERT INTO cameras(camera_id, model, width, height, params, "
      "prior_focal_length) VALUES(?, ?, ?, ?, ?, ?);",
      &sql_stmt_add_camera_);
  prepare_sql_stmt("INSERT INTO frames(frame_id, rig_id) VALUES(?, ?);",
                   &sql_stmt_add_frame_);
  prepare_sql_stmt(
      "INSERT INTO frame_data(frame_id, data_id, sensor_id, sensor_type) "
      "VALUES(?, ?, ?, ?);",
      &sql_stmt_add_frame_data_);
  prepare_sql_stmt(
      "INSERT INTO images(image_id, name, camera_id) VALUES(?, ?, ?);",
      &sql_stmt_add_image_);

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
      "UPDATE pose_priors SET position=?, coordinate_system=?, "
      "position_covariance=? WHERE image_id=?;",
      &sql_stmt_update_pose_prior_);

  //////////////////////////////////////////////////////////////////////////////
  // read_*
  //////////////////////////////////////////////////////////////////////////////
  prepare_sql_stmt("SELECT * FROM rigs;", &sql_stmt_read_rigs_);
  prepare_sql_stmt("SELECT * FROM rigs WHERE rig_id = ?;", &sql_stmt_read_rig_);
  prepare_sql_stmt(
      "SELECT sensor_id, sensor_type, sensor_from_rig "
      "FROM rig_sensors WHERE rig_id = ?;",
      &sql_stmt_read_rig_sensors_);
  prepare_sql_stmt(
      "SELECT rig_id FROM rig_sensors WHERE sensor_id = ? AND sensor_type = ?;",
      &sql_stmt_read_rig_with_sensor_);
  prepare_sql_stmt(
      "SELECT rig_id FROM rigs "
      "WHERE ref_sensor_id = ? AND ref_sensor_type = ?;",
      &sql_stmt_read_rig_with_ref_sensor_);
  prepare_sql_stmt("SELECT * FROM cameras;", &sql_stmt_read_cameras_);
  prepare_sql_stmt("SELECT * FROM cameras WHERE camera_id = ?;",
                   &sql_stmt_read_camera_);
  prepare_sql_stmt("SELECT * FROM frames;", &sql_stmt_read_frames_);
  prepare_sql_stmt("SELECT * FROM frames WHERE frame_id = ?;",
                   &sql_stmt_read_frame_);
  prepare_sql_stmt(
      "SELECT data_id, sensor_id, sensor_type "
      "FROM frame_data WHERE frame_id = ?;",
      &sql_stmt_read_frame_data_);
  prepare_sql_stmt("SELECT * FROM images WHERE image_id = ?;",
                   &sql_stmt_read_image_id_);
  prepare_sql_stmt("SELECT * FROM images;", &sql_stmt_read_images_);
  prepare_sql_stmt("SELECT * FROM images WHERE name = ?;",
                   &sql_stmt_read_image_with_name_);
  prepare_sql_stmt("SELECT * FROM pose_priors WHERE image_id = ?;",
                   &sql_stmt_read_pose_prior_);
  prepare_sql_stmt("SELECT rows, cols, data FROM keypoints WHERE image_id = ?;",
                   &sql_stmt_read_keypoints_);
  prepare_sql_stmt(
      "SELECT rows, cols, data FROM descriptors WHERE image_id = ?;",
      &sql_stmt_read_descriptors_);
  prepare_sql_stmt("SELECT rows, cols, data FROM matches WHERE pair_id = ?;",
                   &sql_stmt_read_matches_);
  prepare_sql_stmt("SELECT * FROM matches WHERE rows > 0;",
                   &sql_stmt_read_matches_all_);
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
      "INSERT INTO pose_priors(image_id, position, coordinate_system, "
      "position_covariance) VALUES(?, ?, ?, ?);",
      &sql_stmt_write_pose_prior_);
  prepare_sql_stmt(
      "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);",
      &sql_stmt_write_keypoints_);
  prepare_sql_stmt(
      "INSERT INTO descriptors(image_id, rows, cols, data) VALUES(?, ?, ?, ?);",
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

void Database::FinalizeSQLStatements() {
  for (const auto& sql_stmt : sql_stmts_) {
    SQLITE3_CALL(sqlite3_finalize(sql_stmt));
  }
}

void Database::CreateTables() const {
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

void Database::CreateRigTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS rigs"
      "   (rig_id               INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
      "    ref_sensor_id        INTEGER                             NOT NULL,"
      "    ref_sensor_type      INTEGER                             NOT NULL);"
      "CREATE UNIQUE INDEX IF NOT EXISTS rig_ref_sensor_assignment ON "
      "   rigs(ref_sensor_id, ref_sensor_type);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateRigSensorsTable() const {
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

void Database::CreateFrameTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS frames"
      "   (frame_id             INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
      "    rig_id               INTEGER                             NOT NULL,"
      "    FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateFrameDataTable() const {
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

void Database::CreateImageTable() const {
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

void Database::CreatePosePriorTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS pose_priors"
      "   (image_id                   INTEGER  PRIMARY KEY  NOT NULL,"
      "    position                   BLOB,"
      "    coordinate_system          INTEGER               NOT NULL,"
      "    position_covariance        BLOB,"
      "    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE "
      "CASCADE);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateKeypointsTable() const {
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

void Database::CreateDescriptorsTable() const {
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

void Database::CreateMatchesTable() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS matches"
      "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows     INTEGER               NOT NULL,"
      "    cols     INTEGER               NOT NULL,"
      "    data     BLOB);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateTwoViewGeometriesTable() const {
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

void Database::UpdateSchema() const {
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
    SQLITE3_EXEC(database_,
                 "ALTER TABLE pose_priors ADD COLUMN position_covariance BLOB "
                 "DEFAULT NULL;",
                 nullptr);

    // Set position_covariance column to NaN matrices
    const std::string update_sql =
        "UPDATE pose_priors SET position_covariance = ?;";
    sqlite3_stmt* update_stmt;
    SQLITE3_CALL(
        sqlite3_prepare_v2(database_, update_sql.c_str(), -1, &update_stmt, 0));
    WriteStaticMatrixBlob(update_stmt, PosePrior().position_covariance, 1);
    SQLITE3_CALL(sqlite3_step(update_stmt));
    SQLITE3_CALL(sqlite3_finalize(update_stmt));
  }

  // Update user version number.
  std::unique_lock<std::mutex> lock(update_schema_mutex_);
  const std::string update_user_version_sql =
      StringPrintf("PRAGMA user_version = 3900;");
  SQLITE3_EXEC(database_, update_user_version_sql.c_str(), nullptr);
}

bool Database::ExistsTable(const std::string& table_name) const {
  const std::string sql =
      "SELECT name FROM sqlite_master WHERE type='table' AND name = ?;";

  sqlite3_stmt* sql_stmt;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt, 0));

  SQLITE3_CALL(sqlite3_bind_text(sql_stmt,
                                 1,
                                 table_name.c_str(),
                                 static_cast<int>(table_name.size()),
                                 SQLITE_STATIC));

  const bool exists = SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW;

  SQLITE3_CALL(sqlite3_finalize(sql_stmt));

  return exists;
}

bool Database::ExistsColumn(const std::string& table_name,
                            const std::string& column_name) const {
  const std::string sql =
      StringPrintf("PRAGMA table_info(%s);", table_name.c_str());

  sqlite3_stmt* sql_stmt;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt, 0));

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

bool Database::ExistsRowId(sqlite3_stmt* sql_stmt,
                           const sqlite3_int64 row_id) const {
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt, 1, static_cast<sqlite3_int64>(row_id)));

  const bool exists = SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW;

  SQLITE3_CALL(sqlite3_reset(sql_stmt));

  return exists;
}

bool Database::ExistsRowString(sqlite3_stmt* sql_stmt,
                               const std::string& row_entry) const {
  SQLITE3_CALL(sqlite3_bind_text(sql_stmt,
                                 1,
                                 row_entry.c_str(),
                                 static_cast<int>(row_entry.size()),
                                 SQLITE_STATIC));

  const bool exists = SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW;

  SQLITE3_CALL(sqlite3_reset(sql_stmt));

  return exists;
}

size_t Database::CountRows(const std::string& table) const {
  const std::string sql =
      StringPrintf("SELECT COUNT(*) FROM %s;", table.c_str());

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
  const std::string sql =
      StringPrintf("SELECT SUM(%s) FROM %s;", column.c_str(), table.c_str());

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
  const std::string sql =
      StringPrintf("SELECT MAX(%s) FROM %s;", column.c_str(), table.c_str());

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
  THROW_CHECK_NOTNULL(database_);
  database_->BeginTransaction();
}

DatabaseTransaction::~DatabaseTransaction() { database_->EndTransaction(); }

}  // namespace colmap
