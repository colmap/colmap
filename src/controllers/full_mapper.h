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

#ifndef COLMAP_SRC_CONTROLLERS_FULL_MAPPER_H_
#define COLMAP_SRC_CONTROLLERS_FULL_MAPPER_H_

#include <string>

#include "base/reconstruction_manager.h"
#include "util/threading.h"

namespace colmap {

class FullMapperController : public Thread {
 public:
  enum class DataType {
    VIDEO,
    DSLR,
    INTERNET
  };

  struct Options {
    // The path to the workspace folder in which all results are stored.
    std::string workspace_path;

    // The path to the image folder which are used as input.
    std::string image_path;

    // The path to the vocabulary tree for feature matching.
    std::string vocab_tree_path;

    // The type of input data used to choose optimal mapper settings.
    DataType data_type;

    // Whether to use the GPU in feature extraction and matching.
    bool use_gpu = true;

    // Whether to use OpenGL in GPU-based feature extraction and matching.
    bool use_opengl = true;

    // Whether to perform sparse mapping.
    bool sparse = true;

    // Whether to perform dense mapping.
    bool dense = true;
  };

  FullMapperController(const Options& options);

 private:
  void Run();
  void RunFeatureExtraction();
  void RunFeatureMatching();
  void RunSparseMapper();
  void RunDenseMapper();

  const Options options_;
  OptionManager option_manager_;
  ReconstructionManager reconstruction_manager_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_CONTROLLERS_FULL_MAPPER_H_
