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

#include "colmap/mvs/poisson_meshing.h"

#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"

#include "thirdparty/PoissonRecon/PoissonRecon.h"
#include "thirdparty/PoissonRecon/SurfaceTrimmer.h"

#include <fstream>
#include <vector>

#include <omp.h>

namespace colmap {
namespace mvs {

bool PoissonMeshingOptions::Check() const {
  CHECK_OPTION_GE(point_weight, 0);
  CHECK_OPTION_GT(depth, 0);
  CHECK_OPTION_GE(trim, 0);
  CHECK_OPTION_GE(num_threads, -1);
  CHECK_OPTION_NE(num_threads, 0);
  return true;
}

bool PoissonMeshing(const PoissonMeshingOptions& options,
                    const std::filesystem::path& input_path,
                    const std::filesystem::path& output_path) {
  THROW_CHECK(options.Check());
  THROW_CHECK_HAS_FILE_EXTENSION(input_path, ".ply");
  THROW_CHECK_FILE_EXISTS(input_path);
  THROW_CHECK_HAS_FILE_EXTENSION(output_path, ".ply");
  THROW_CHECK_PATH_OPEN(output_path);

  bool success = true;

#pragma omp parallel num_threads(1)
  {
    omp_set_num_threads(GetEffectiveNumThreads(options.num_threads));
#ifdef _MSC_VER
    omp_set_nested(1);
#else
    omp_set_max_active_levels(1);
#endif

    std::vector<std::string> args;

    args.push_back("./poisson_recon");

    args.push_back("--in");
    args.push_back(input_path.string());

    args.push_back("--out");
    args.push_back(output_path.string());

    args.push_back("--pointWeight");
    args.push_back(std::to_string(options.point_weight));

    args.push_back("--depth");
    args.push_back(std::to_string(options.depth));

    // Full depth cannot exceed system depth.
    if (options.depth < 5) {
      args.push_back("--fullDepth");
      args.push_back(std::to_string(options.depth));
    }

    if (options.color) {
      args.push_back("--colors");
    }

    if (options.num_threads > 0) {
      args.push_back("--parallel");
      args.push_back("0");
    }

    if (options.trim > 0) {
      args.push_back("--density");
    }

    std::vector<const char*> args_cstr;
    args_cstr.reserve(args.size());
    for (const auto& arg : args) {
      args_cstr.push_back(arg.c_str());
    }

    if (RunPoissonRecon(args_cstr.size(),
                        const_cast<char**>(args_cstr.data())) != EXIT_SUCCESS) {
      success = false;
    }

    if (success && options.trim != 0) {
      args.clear();
      args_cstr.clear();

      args.push_back("./surface_trimmer");

      args.push_back("--in");
      args.push_back(output_path.string());

      args.push_back("--out");
      args.push_back(output_path.string());

      args.push_back("--trim");
      args.push_back(std::to_string(options.trim));

      args_cstr.reserve(args.size());
      for (const auto& arg : args) {
        args_cstr.push_back(arg.c_str());
      }

      if (RunSurfaceTrimmer(args_cstr.size(),
                            const_cast<char**>(args_cstr.data())) !=
          EXIT_SUCCESS) {
        success = false;
      }
    }
  }

  return success;
}

}  // namespace mvs
}  // namespace colmap
