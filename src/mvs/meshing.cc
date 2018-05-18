// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//     * Neither the name of the ETH Zurich and UNC Chapel Hill nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

#include "mvs/meshing.h"

#include <vector>

#include "PoissonRecon/PoissonRecon.h"
#include "PoissonRecon/SurfaceTrimmer.h"
#include "util/logging.h"

namespace colmap {
namespace mvs {

bool PoissonReconstructionOptions::Check() const {
  CHECK_OPTION_GE(point_weight, 0);
  CHECK_OPTION_GT(depth, 0);
  CHECK_OPTION_GE(color, 0);
  CHECK_OPTION_GE(trim, 0);
  CHECK_OPTION_GE(num_threads, -1);
  CHECK_OPTION_NE(num_threads, 0);
  return true;
}

bool PoissonReconstruction(const PoissonReconstructionOptions& options,
                           const std::string& input_path,
                           const std::string& output_path) {
  CHECK(options.Check());

  std::vector<std::string> args;

  args.push_back("./binary");

  args.push_back("--in");
  args.push_back(input_path);

  args.push_back("--out");
  args.push_back(output_path);

  args.push_back("--pointWeight");
  args.push_back(std::to_string(options.point_weight));

  args.push_back("--depth");
  args.push_back(std::to_string(options.depth));

  if (options.color > 0) {
    args.push_back("--color");
    args.push_back(std::to_string(options.color));
  }

#ifdef OPENMP_ENABLED
  if (options.num_threads > 0) {
    args.push_back("--threads");
    args.push_back(std::to_string(options.num_threads));
  }
#endif  // OPENMP_ENABLED

  if (options.trim > 0) {
    args.push_back("--density");
  }

  std::vector<const char*> args_cstr;
  args_cstr.reserve(args.size());
  for (const auto& arg : args) {
    args_cstr.push_back(arg.c_str());
  }

  if (PoissonRecon(args_cstr.size(), const_cast<char**>(args_cstr.data())) !=
      EXIT_SUCCESS) {
    return false;
  }

  if (options.trim == 0) {
    return true;
  }

  args.clear();
  args_cstr.clear();

  args.push_back("./binary");

  args.push_back("--in");
  args.push_back(output_path);

  args.push_back("--out");
  args.push_back(output_path);

  args.push_back("--trim");
  args.push_back(std::to_string(options.trim));

  args_cstr.reserve(args.size());
  for (const auto& arg : args) {
    args_cstr.push_back(arg.c_str());
  }

  return SurfaceTrimmer(args_cstr.size(),
                        const_cast<char**>(args_cstr.data())) == EXIT_SUCCESS;
}

}  // namespace mvs
}  // namespace colmap
