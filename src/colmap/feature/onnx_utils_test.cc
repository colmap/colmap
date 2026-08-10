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
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/feature/onnx_utils.h"

#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <cstddef>
#include <filesystem>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

std::filesystem::path WriteIdentityModel() {
  // ONNX IR v8, opset 13: float input [1, 2] -> Identity -> float output
  // [1, 2]. Keeping the serialized protobuf inline makes this test hermetic.
  constexpr char kIdentityModel[] = {
      0x08, 0x08, 0x3a, 0x5e, 0x0a, 0x19, 0x0a, 0x05, 0x69, 0x6e, 0x70,
      0x75, 0x74, 0x12, 0x06, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x22,
      0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x12, 0x0e,
      0x69, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x5f, 0x67, 0x72,
      0x61, 0x70, 0x68, 0x5a, 0x17, 0x0a, 0x05, 0x69, 0x6e, 0x70, 0x75,
      0x74, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x01, 0x12, 0x08, 0x0a, 0x02,
      0x08, 0x01, 0x0a, 0x02, 0x08, 0x02, 0x62, 0x18, 0x0a, 0x06, 0x6f,
      0x75, 0x74, 0x70, 0x75, 0x74, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x01,
      0x12, 0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x02, 0x42,
      0x02, 0x10, 0x0d};

  const std::filesystem::path model_path = CreateTestDir() / "identity.onnx";
  WriteBinaryBlob(model_path,
                  {kIdentityModel, sizeof(kIdentityModel) / sizeof(char)});
  return model_path;
}

template <typename T>
Ort::Value MakeTensor(std::vector<T>* data,
                      const std::vector<int64_t>& shape) {
  return Ort::Value::CreateTensor<T>(
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                 OrtMemType::OrtMemTypeCPU),
      data->data(),
      data->size(),
      shape.data(),
      shape.size());
}

class ErrorLogSink : public google::LogSink {
 public:
  ErrorLogSink() { google::AddLogSink(this); }
  ~ErrorLogSink() override { google::RemoveLogSink(this); }

  void send(google::LogSeverity severity,
            const char*,
            const char*,
            int,
            const struct ::tm*,
            const char*,
            size_t) override {
    if (severity >= google::GLOG_ERROR) {
      ++num_error_logs_;
    }
  }

  size_t NumErrorLogs() const { return num_error_logs_; }

 private:
  size_t num_error_logs_ = 0;
};

TEST(ONNXModelTest, MetadataAndInference) {
  ONNXModel model(WriteIdentityModel().string(),
                  /*num_threads=*/1,
                  /*use_gpu=*/false,
                  /*gpu_index=*/"-1");

  ASSERT_EQ(model.input_names().size(), 1);
  EXPECT_STREQ(model.input_names()[0], "input");
  ASSERT_EQ(model.input_shapes().size(), 1);
  EXPECT_EQ(model.input_shapes()[0], (std::vector<int64_t>{1, 2}));
  ASSERT_EQ(model.output_names().size(), 1);
  EXPECT_STREQ(model.output_names()[0], "output");
  ASSERT_EQ(model.output_shapes().size(), 1);
  EXPECT_EQ(model.output_shapes()[0], (std::vector<int64_t>{1, 2}));

  std::vector<float> input_data{1.25f, -2.5f};
  const std::vector<int64_t> input_shape{1, 2};
  std::vector<Ort::Value> inputs;
  inputs.push_back(MakeTensor(&input_data, input_shape));
  const std::vector<Ort::Value> outputs = model.Run(inputs);

  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].GetTensorTypeAndShapeInfo().GetShape(), input_shape);
  const float* output_data = outputs[0].GetTensorData<float>();
  EXPECT_FLOAT_EQ(output_data[0], input_data[0]);
  EXPECT_FLOAT_EQ(output_data[1], input_data[1]);
}

TEST(ONNXModelTest, InvalidInputTypeRethrows) {
  ONNXModel model(WriteIdentityModel().string(),
                  /*num_threads=*/1,
                  /*use_gpu=*/false,
                  /*gpu_index=*/"-1");

  std::vector<int64_t> input_data{1, 2};
  const std::vector<int64_t> input_shape{1, 2};
  std::vector<Ort::Value> inputs;
  inputs.push_back(MakeTensor(&input_data, input_shape));
  EXPECT_THROW(model.Run(inputs), Ort::Exception);
}

TEST(ONNXModelTest, CapabilityProbeRethrowsWithoutErrorLog) {
  const std::filesystem::path model_path = CreateTestDir() / "invalid.onnx";
  constexpr char kInvalidModel[] = "not an ONNX model";
  WriteBinaryBlob(model_path, {kInvalidModel, sizeof(kInvalidModel)});

  ErrorLogSink log_sink;
  EXPECT_THROW(ONNXModel(model_path.string(),
                         /*num_threads=*/1,
                         /*use_gpu=*/false,
                         /*gpu_index=*/"-1",
                         /*is_capability_probe=*/true),
               Ort::Exception);
  EXPECT_EQ(log_sink.NumErrorLogs(), 0);
}

}  // namespace
}  // namespace colmap
