// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#include "base/database.h"
#include "base/vocabulary_tree.h"
#include "util/logging.h"
#include "util/option_manager.h"

using namespace colmap;

namespace config = boost::program_options;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string vocab_tree_path;
  int depth;
  int branching_factor;
  int restarts;

  OptionManager options;
  options.AddDatabaseOptions();
  options.desc->add_options()(
      "vocab_tree_path",
      config::value<std::string>(&vocab_tree_path)->required());
  options.desc->add_options()("depth",
                              config::value<int>(&depth)->default_value(2));
  options.desc->add_options()(
      "branching_factor",
      config::value<int>(&branching_factor)->default_value(100));
  options.desc->add_options()("restarts",
                              config::value<int>(&restarts)->default_value(3));
  options.AddMatchOptions();

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  Database database;
  database.Open(*options.database_path);

  auto vocab_tree =
      VocabularyTree::Build(database, depth, branching_factor, restarts);

  vocab_tree.Write(vocab_tree_path);

  return EXIT_SUCCESS;
}
