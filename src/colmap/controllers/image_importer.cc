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

#include "colmap/controllers/image_importer.h"

#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

namespace colmap {
namespace {

// Image importer class to import images into the database.
class ImageImporterController : public Thread {
 public:
  ImageImporterController(const std::string& database_path,
                          const ImageReaderOptions& reader_options)
      : database_(database_path), image_reader_(reader_options, &database_) {}

 private:
  void Run() override {
    PrintHeading1("Image import");
    Timer run_timer;
    run_timer.Start();

    while (image_reader_.NextIndex() < image_reader_.NumImages()) {
      if (IsStopped()) {
        break;
      }

      LOG(INFO) << StringPrintf("Processed file [%d/%d]",
                                image_reader_.NextIndex(),
                                image_reader_.NumImages());

      Rig rig;
      Camera camera;
      Image image;
      PosePrior pose_prior;
      Bitmap bitmap;  // Bitmap is read but not used after import
      ImageReader::Status status =
          image_reader_.Next(&rig, &camera, &image, &pose_prior, &bitmap);

      LOG(INFO) << StringPrintf("  Name:            %s", image.Name().c_str());
      if (status != ImageReader::Status::SUCCESS) {
        LOG(ERROR) << image.Name() << " "
                   << ImageReader::StatusToString(status);
        continue;
      }
      LOG(INFO) << StringPrintf(
          "  Dimensions:      %d x %d", camera.width, camera.height);
      LOG(INFO) << StringPrintf("  Camera:          #%d - %s",
                                camera.camera_id,
                                camera.ModelName().c_str());
      LOG(INFO) << StringPrintf(
          "  Focal Length:    %.2fpx%s",
          camera.MeanFocalLength(),
          camera.has_prior_focal_length ? " (Prior)" : "");

      DatabaseTransaction database_transaction(&database_);

      if (image.ImageId() == kInvalidImageId) {
        image.SetImageId(database_.WriteImage(image));
        if (pose_prior.IsValid()) {
          LOG(INFO) << StringPrintf(
              "  GPS:             LAT=%.3f, LON=%.3f, ALT=%.3f",
              pose_prior.position.x(),
              pose_prior.position.y(),
              pose_prior.position.z());
          database_.WritePosePrior(image.ImageId(), pose_prior);
        }
        Frame frame;
        frame.SetRigId(rig.RigId());
        frame.AddDataId(image.DataId());
        database_.WriteFrame(frame);
      }
    }

    run_timer.PrintMinutes();
  }

  Database database_;
  ImageReader image_reader_;
};

}  // namespace

std::unique_ptr<Thread> CreateImageImporterController(
    const std::string& database_path,
    const ImageReaderOptions& reader_options) {
  return std::make_unique<ImageImporterController>(database_path,
                                                   reader_options);
}

}  // namespace colmap