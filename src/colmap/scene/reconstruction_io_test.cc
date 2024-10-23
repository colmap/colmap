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

#include "colmap/scene/reconstruction_io.h"

#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

namespace colmap {
namespace {

struct ReaderWriter {
  virtual ~ReaderWriter() = default;

  virtual void ReadCameras(Reconstruction& reconstruction) = 0;
  virtual void ReadImages(Reconstruction& reconstruction) = 0;
  virtual void ReadPoints3D(Reconstruction& reconstruction) = 0;

  virtual void WriteCameras(const Reconstruction& reconstruction) = 0;
  virtual void WriteImages(const Reconstruction& reconstruction) = 0;
  virtual void WritePoints3D(const Reconstruction& reconstruction) = 0;

  virtual std::string CamerasStr() const = 0;
  virtual std::string ImagesStr() const = 0;
  virtual std::string Points3DStr() const = 0;
};

struct ReaderWriterTextStringStream : public ReaderWriter {
  void ReadCameras(Reconstruction& reconstruction) override {
    ReadCamerasText(reconstruction, cameras_stream_);
  }
  void ReadImages(Reconstruction& reconstruction) override {
    ReadImagesText(reconstruction, images_stream_);
  }
  void ReadPoints3D(Reconstruction& reconstruction) override {
    ReadPoints3DText(reconstruction, points3D_stream_);
  }

  void WriteCameras(const Reconstruction& reconstruction) override {
    WriteCamerasText(reconstruction, cameras_stream_);
  }
  void WriteImages(const Reconstruction& reconstruction) override {
    WriteImagesText(reconstruction, images_stream_);
  }
  void WritePoints3D(const Reconstruction& reconstruction) override {
    WritePoints3DText(reconstruction, points3D_stream_);
  }

  virtual std::string CamerasStr() const override {
    return cameras_stream_.str();
  }
  virtual std::string ImagesStr() const override {
    return images_stream_.str();
  }
  virtual std::string Points3DStr() const override {
    return points3D_stream_.str();
  }

 private:
  std::stringstream cameras_stream_;
  std::stringstream images_stream_;
  std::stringstream points3D_stream_;
};

struct ReaderWriterBinaryStringStream : public ReaderWriter {
  void ReadCameras(Reconstruction& reconstruction) override {
    ReadCamerasBinary(reconstruction, cameras_stream_);
  }
  void ReadImages(Reconstruction& reconstruction) override {
    ReadImagesBinary(reconstruction, images_stream_);
  }
  void ReadPoints3D(Reconstruction& reconstruction) override {
    ReadPoints3DBinary(reconstruction, points3D_stream_);
  }

  void WriteCameras(const Reconstruction& reconstruction) override {
    WriteCamerasBinary(reconstruction, cameras_stream_);
  }
  void WriteImages(const Reconstruction& reconstruction) override {
    WriteImagesBinary(reconstruction, images_stream_);
  }
  void WritePoints3D(const Reconstruction& reconstruction) override {
    WritePoints3DBinary(reconstruction, points3D_stream_);
  }

  virtual std::string CamerasStr() const override {
    return cameras_stream_.str();
  }
  virtual std::string ImagesStr() const override {
    return images_stream_.str();
  }
  virtual std::string Points3DStr() const override {
    return points3D_stream_.str();
  }

 private:
  std::stringstream cameras_stream_;
  std::stringstream images_stream_;
  std::stringstream points3D_stream_;
};

std::string ReadFileAsString(const std::string& path) {
  std::fstream file(path);
  THROW_CHECK(file.good());
  std::stringstream buf;
  buf << file.rdbuf();
  return buf.str();
}

struct ReaderWriterFileStream : public ReaderWriter {
  explicit ReaderWriterFileStream(const std::string& ext)
      : test_dir_(CreateTestDir()),
        cameras_path_((test_dir_ / ("cameras." + ext)).string()),
        images_path_((test_dir_ / ("images." + ext)).string()),
        points3D_path_((test_dir_ / ("points3D." + ext)).string()) {}

  virtual std::string CamerasStr() const override {
    return ReadFileAsString(cameras_path_);
  }
  virtual std::string ImagesStr() const override {
    return ReadFileAsString(images_path_);
  }
  virtual std::string Points3DStr() const override {
    return ReadFileAsString(points3D_path_);
  }

 protected:
  const std::filesystem::path test_dir_;
  const std::string cameras_path_;
  const std::string images_path_;
  const std::string points3D_path_;
};

struct ReaderWriterTextFileStream : public ReaderWriterFileStream {
  ReaderWriterTextFileStream() : ReaderWriterFileStream("txt") {}

  void ReadCameras(Reconstruction& reconstruction) override {
    ReadCamerasText(reconstruction, cameras_path_);
  }
  void ReadImages(Reconstruction& reconstruction) override {
    ReadImagesText(reconstruction, images_path_);
  }
  void ReadPoints3D(Reconstruction& reconstruction) override {
    ReadPoints3DText(reconstruction, points3D_path_);
  }

  void WriteCameras(const Reconstruction& reconstruction) override {
    WriteCamerasText(reconstruction, cameras_path_);
  }
  void WriteImages(const Reconstruction& reconstruction) override {
    WriteImagesText(reconstruction, images_path_);
  }
  void WritePoints3D(const Reconstruction& reconstruction) override {
    WritePoints3DText(reconstruction, points3D_path_);
  }
};

struct ReaderWriterBinaryFileStream : public ReaderWriterFileStream {
  ReaderWriterBinaryFileStream() : ReaderWriterFileStream("bin") {}

  void ReadCameras(Reconstruction& reconstruction) override {
    ReadCamerasBinary(reconstruction, cameras_path_);
  }
  void ReadImages(Reconstruction& reconstruction) override {
    ReadImagesBinary(reconstruction, images_path_);
  }
  void ReadPoints3D(Reconstruction& reconstruction) override {
    ReadPoints3DBinary(reconstruction, points3D_path_);
  }

  void WriteCameras(const Reconstruction& reconstruction) override {
    WriteCamerasBinary(reconstruction, cameras_path_);
  }
  void WriteImages(const Reconstruction& reconstruction) override {
    WriteImagesBinary(reconstruction, images_path_);
  }
  void WritePoints3D(const Reconstruction& reconstruction) override {
    WritePoints3DBinary(reconstruction, points3D_path_);
  }
};

class ParameterizedReaderWriterTests
    : public ::testing::TestWithParam<
          std::function<std::unique_ptr<ReaderWriter>()>> {};

TEST_P(ParameterizedReaderWriterTests, Roundtrip) {
  std::unique_ptr<ReaderWriter> reader_writer = GetParam()();

  Reconstruction orig;
  SyntheticDatasetOptions options;
  options.num_cameras = 11;
  options.num_images = 43;
  options.num_points3D = 321;
  SynthesizeDataset(options, &orig);

  reader_writer->WriteCameras(orig);
  EXPECT_FALSE(reader_writer->CamerasStr().empty());

  reader_writer->WriteImages(orig);
  EXPECT_FALSE(reader_writer->ImagesStr().empty());

  reader_writer->WritePoints3D(orig);
  EXPECT_FALSE(reader_writer->Points3DStr().empty());

  Reconstruction test;
  reader_writer->ReadCameras(test);
  EXPECT_EQ(orig.Cameras(), test.Cameras());
  reader_writer->ReadImages(test);
  EXPECT_EQ(orig.Images(), test.Images());
  reader_writer->ReadPoints3D(test);
  EXPECT_EQ(orig.Points3D(), test.Points3D());
}

INSTANTIATE_TEST_SUITE_P(
    ReaderWriterTests,
    ParameterizedReaderWriterTests,
    ::testing::Values(
        []() { return std::make_unique<ReaderWriterTextStringStream>(); },
        []() { return std::make_unique<ReaderWriterBinaryStringStream>(); },
        []() { return std::make_unique<ReaderWriterTextFileStream>(); },
        []() { return std::make_unique<ReaderWriterBinaryFileStream>(); }));

}  // namespace
}  // namespace colmap
