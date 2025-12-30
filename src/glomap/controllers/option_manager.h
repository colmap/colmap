#pragma once

#include "colmap/controllers/base_option_manager.h"

#include <memory>

namespace glomap {

struct GlobalMapperOptions;
struct GravityRefinerOptions;

class OptionManager : public colmap::BaseOptionManager {
 public:
  explicit OptionManager(bool add_project_options = true);

  void AddAllOptions() override;
  void AddGlobalMapperOptions();
  void AddGravityRefinerOptions();

  void Reset() override;
  void ResetOptions(bool reset_paths) override;

  std::shared_ptr<GlobalMapperOptions> mapper;
  std::shared_ptr<GravityRefinerOptions> gravity_refiner;

 private:
  bool added_global_mapper_options_ = false;
  bool added_gravity_refiner_options_ = false;
};

}  // namespace glomap
