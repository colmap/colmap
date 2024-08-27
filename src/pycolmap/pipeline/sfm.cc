#include "colmap/exe/sfm.h"

#include "colmap/controllers/bundle_adjustment.h"
#include "colmap/controllers/incremental_mapper.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/models.h"
#include "colmap/util/misc.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <memory>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

std::shared_ptr<Reconstruction> TriangulatePoints(
    const std::shared_ptr<Reconstruction>& reconstruction,
    const std::string& database_path,
    const std::string& image_path,
    const std::string& output_path,
    const bool clear_points,
    const IncrementalPipelineOptions& options,
    const bool refine_intrinsics) {
  THROW_CHECK_FILE_EXISTS(database_path);
  THROW_CHECK_DIR_EXISTS(image_path);
  CreateDirIfNotExists(output_path);

  py::gil_scoped_release release;
  RunPointTriangulatorImpl(reconstruction,
                           database_path,
                           image_path,
                           output_path,
                           options,
                           clear_points,
                           refine_intrinsics);
  return reconstruction;
}

std::map<size_t, std::shared_ptr<Reconstruction>> IncrementalMapping(
    const std::string& database_path,
    const std::string& image_path,
    const std::string& output_path,
    const IncrementalPipelineOptions& options,
    const std::string& input_path,
    std::function<void()> initial_image_pair_callback,
    std::function<void()> next_image_callback) {
  THROW_CHECK_FILE_EXISTS(database_path);
  THROW_CHECK_DIR_EXISTS(image_path);
  CreateDirIfNotExists(output_path);

  py::gil_scoped_release release;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  if (input_path != "") {
    reconstruction_manager->Read(input_path);
  }
  auto options_ = std::make_shared<IncrementalPipelineOptions>(options);
  IncrementalPipeline mapper(
      options_, image_path, database_path, reconstruction_manager);

  PyInterrupt py_interrupt(1.0);  // Check for interrupts every second
  mapper.AddCallback(
      IncrementalPipeline::NEXT_IMAGE_REG_CALLBACK,
      [&py_interrupt, next_image_callback = std::move(next_image_callback)]() {
        if (py_interrupt.Raised()) {
          throw py::error_already_set();
        }
        if (next_image_callback) {
          next_image_callback();
        }
      });
  if (initial_image_pair_callback) {
    mapper.AddCallback(IncrementalPipeline::INITIAL_IMAGE_PAIR_REG_CALLBACK,
                       std::move(initial_image_pair_callback));
  }

  mapper.Run();

  reconstruction_manager->Write(output_path);
  std::map<size_t, std::shared_ptr<Reconstruction>> reconstructions;
  for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
    reconstructions[i] = reconstruction_manager->Get(i);
  }
  return reconstructions;
}

void BundleAdjustment(const std::shared_ptr<Reconstruction>& reconstruction,
                      const BundleAdjustmentOptions& options) {
  py::gil_scoped_release release;
  OptionManager option_manager;
  *option_manager.bundle_adjustment = options;
  BundleAdjustmentController controller(option_manager, reconstruction);
  controller.Run();
}

void BindSfM(py::module& m) {
  m.def("triangulate_points",
        &TriangulatePoints,
        "reconstruction"_a,
        "database_path"_a,
        "image_path"_a,
        "output_path"_a,
        "clear_points"_a = true,
        py::arg_v("options",
                  IncrementalPipelineOptions(),
                  "IncrementalPipelineOptions()"),
        "refine_intrinsics"_a = false,
        "Triangulate 3D points from known camera poses");

  m.def("incremental_mapping",
        &IncrementalMapping,
        "database_path"_a,
        "image_path"_a,
        "output_path"_a,
        py::arg_v("options",
                  IncrementalPipelineOptions(),
                  "IncrementalPipelineOptions()"),
        "input_path"_a = py::str(""),
        "initial_image_pair_callback"_a = py::none(),
        "next_image_callback"_a = py::none(),
        "Recover 3D points and unknown camera poses");

  m.def("bundle_adjustment",
        &BundleAdjustment,
        "reconstruction"_a,
        py::arg_v(
            "options", BundleAdjustmentOptions(), "BundleAdjustmentOptions()"),
        "Jointly refine 3D points and camera poses");
}
