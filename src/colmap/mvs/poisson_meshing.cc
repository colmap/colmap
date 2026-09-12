// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/poisson_meshing.h"

#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

#include "thirdparty/PoissonRecon/MultiThreading.h"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "thirdparty/PoissonRecon/PoissonRecon.h"
#include "thirdparty/PoissonRecon/SurfaceTrimmer.h"

#include <vector>

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

  const int num_effective_threads = GetEffectiveNumThreads(options.num_threads);

  // Configure PoissonRecon's internal thread pool directly, since it uses its
  // own threading mechanism that is not controlled by OMP settings.
  PoissonRecon::ThreadPool::SetNumThreads(num_effective_threads);
  if (num_effective_threads > 1) {
#ifdef _OPENMP
    PoissonRecon::ThreadPool::ParallelizationType =
        PoissonRecon::ThreadPool::OPEN_MP;
#else
    PoissonRecon::ThreadPool::ParallelizationType =
        PoissonRecon::ThreadPool::ASYNC;
#endif
  } else {
    PoissonRecon::ThreadPool::ParallelizationType =
        PoissonRecon::ThreadPool::NONE;
  }

  try {
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
  } catch (const std::exception& e) {
    LOG(WARNING) << "PoissonRecon failed with exception: " << e.what();
    success = false;
  }

  return success;
}

}  // namespace mvs
}  // namespace colmap
