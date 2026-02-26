#include "colmap/retrieval/visual_index.h"

#include "colmap/feature/types.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <filesystem>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using VisualIndex = retrieval::VisualIndex;
using ImageScore = retrieval::ImageScore;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

class PyVisualIndexImpl : public VisualIndex, py::trampoline_self_life_support {
 public:
  size_t NumVisualWords() const override {
    PYBIND11_OVERRIDE_PURE(size_t, VisualIndex, NumVisualWords);
  }

  size_t NumImages() const override {
    PYBIND11_OVERRIDE_PURE(size_t, VisualIndex, NumImages);
  }

  int DescDim() const override {
    PYBIND11_OVERRIDE_PURE(int, VisualIndex, DescDim);
  }

  int EmbeddingDim() const override {
    PYBIND11_OVERRIDE_PURE(int, VisualIndex, EmbeddingDim);
  }

  FeatureExtractorType FeatureType() const override {
    PYBIND11_OVERRIDE_PURE(FeatureExtractorType, VisualIndex, FeatureType);
  }

  void Add(const IndexOptions& options,
           int image_id,
           const FeatureKeypoints& keypoints,
           const FeatureDescriptorsFloat& descriptors) override {
    PYBIND11_OVERRIDE_PURE(
        void, VisualIndex, Add, options, image_id, keypoints, descriptors);
  }

  bool IsImageIndexed(int image_id) const override {
    PYBIND11_OVERRIDE_PURE(bool, VisualIndex, IsImageIndexed, image_id);
  }

  void Query(const QueryOptions& options,
             const FeatureDescriptorsFloat& descriptors,
             std::vector<ImageScore>* image_scores) const override {
    PYBIND11_OVERRIDE_PURE(
        void, VisualIndex, Query, options, descriptors, image_scores);
  }

  void Query(const QueryOptions& options,
             const FeatureKeypoints& keypoints,
             const FeatureDescriptorsFloat& descriptors,
             std::vector<ImageScore>* image_scores) const override {
    PYBIND11_OVERRIDE_PURE(
        void, VisualIndex, Query, keypoints, descriptors, image_scores);
  }

  void Prepare() override {
    PYBIND11_OVERRIDE_PURE(void, VisualIndex, Prepare);
  }

  void Build(const BuildOptions& options,
             const FeatureDescriptorsFloat& descriptors) override {
    PYBIND11_OVERRIDE_PURE(void, VisualIndex, Build, options, descriptors);
  }

  void Write(const std::filesystem::path& path) const override {
    PYBIND11_OVERRIDE_PURE(void, VisualIndex, Read);
  }

 protected:
  void ReadFromFaiss(const std::filesystem::path& path,
                     long offset,
                     FeatureExtractorType feature_type) override {
    PYBIND11_OVERRIDE_PURE(
        void, VisualIndex, ReadFromFaiss, path, offset, feature_type);
  }
};

}  // namespace

void BindVisualIndex(py::module& m) {
  auto PyImageScore = py::classh<ImageScore>(m, "ImageScore")
                          .def(py::init<>())
                          .def_readonly("image_id", &ImageScore::image_id)
                          .def_readonly("score", &ImageScore::score);
  MakeDataclass(PyImageScore);

  py::classh<VisualIndex, PyVisualIndexImpl> PyVisualIndex(m, "VisualIndex");

  auto PyIndexOptions =
      py::classh<VisualIndex::IndexOptions>(PyVisualIndex, "IndexOptions")
          .def(py::init<>())
          .def_readwrite("num_neighbors",
                         &VisualIndex::IndexOptions::num_neighbors)
          .def_readwrite("num_checks", &VisualIndex::IndexOptions::num_checks)
          .def_readwrite("num_threads",
                         &VisualIndex::IndexOptions::num_threads);
  MakeDataclass(PyIndexOptions);

  auto PyQueryOptions =
      py::classh<VisualIndex::QueryOptions>(PyVisualIndex, "QueryOptions")
          .def(py::init<>())
          .def_readwrite("max_num_images",
                         &VisualIndex::QueryOptions::max_num_images)
          .def_readwrite("num_neighbors",
                         &VisualIndex::QueryOptions::num_neighbors)
          .def_readwrite("num_checks", &VisualIndex::QueryOptions::num_checks)
          .def_readwrite("num_threads", &VisualIndex::QueryOptions::num_threads)
          .def_readwrite(
              "num_images_after_verification",
              &VisualIndex::QueryOptions::num_images_after_verification);
  MakeDataclass(PyQueryOptions);

  auto PyBuildOptions =
      py::classh<VisualIndex::BuildOptions>(PyVisualIndex, "BuildOptions")
          .def(py::init<>())
          .def_readwrite("num_visual_words",
                         &VisualIndex::BuildOptions::num_visual_words)
          .def_readwrite("num_iterations",
                         &VisualIndex::BuildOptions::num_iterations)
          .def_readwrite("num_rounds", &VisualIndex::BuildOptions::num_rounds)
          .def_readwrite("num_checks", &VisualIndex::BuildOptions::num_checks)
          .def_readwrite("num_threads",
                         &VisualIndex::BuildOptions::num_threads);
  MakeDataclass(PyBuildOptions);

  PyVisualIndex.def(py::init<>())
      .def_static("create", &VisualIndex::Create)
      .def("add",
           static_cast<void (VisualIndex::*)(
               const typename VisualIndex::IndexOptions&,
               int,
               const FeatureKeypoints&,
               const FeatureDescriptorsFloat&)>(&VisualIndex::Add),
           py::call_guard<py::gil_scoped_release>())
      .def("is_image_indexed", &VisualIndex::IsImageIndexed)
      .def("num_visual_words", &VisualIndex::NumVisualWords)
      .def("num_images", &VisualIndex::NumImages)
      .def("desc_dim", &VisualIndex::DescDim)
      .def("embedding_dim", &VisualIndex::EmbeddingDim)
      .def("feature_type", &VisualIndex::FeatureType)
      .def("query",
           static_cast<void (VisualIndex::*)(
               const typename VisualIndex::QueryOptions&,
               const FeatureDescriptorsFloat&,
               std::vector<ImageScore>*) const>(&VisualIndex::Query),
           py::call_guard<py::gil_scoped_release>())
      .def("query",
           static_cast<void (VisualIndex::*)(
               const typename VisualIndex::QueryOptions&,
               const FeatureKeypoints&,
               const FeatureDescriptorsFloat&,
               std::vector<ImageScore>*) const>(&VisualIndex::Query),
           py::call_guard<py::gil_scoped_release>())
      .def("prepare",
           &VisualIndex::Prepare,
           py::call_guard<py::gil_scoped_release>())
      .def("build",
           &VisualIndex::Build,
           py::call_guard<py::gil_scoped_release>())
      .def_static(
          "read", &VisualIndex::Read, py::call_guard<py::gil_scoped_release>())
      .def("write",
           &VisualIndex::Write,
           py::call_guard<py::gil_scoped_release>())
      .def("__repr__", [](const VisualIndex& self) {
        std::ostringstream ss;
        ss << self;
        return ss.str();
      });
}
