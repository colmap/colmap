#include "solver.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "caspar_mappings.h"
#include "kernel_PinholeExtraCalib_alpha_denumerator_or_beta_nummerator.h"
#include "kernel_PinholeExtraCalib_alpha_numerator_denominator.h"
#include "kernel_PinholeExtraCalib_normalize.h"
#include "kernel_PinholeExtraCalib_pred_decrease_times_two.h"
#include "kernel_PinholeExtraCalib_retract.h"
#include "kernel_PinholeExtraCalib_start_w.h"
#include "kernel_PinholeExtraCalib_start_w_contribute.h"
#include "kernel_PinholeExtraCalib_update_Mp.h"
#include "kernel_PinholeExtraCalib_update_p.h"
#include "kernel_PinholeExtraCalib_update_r.h"
#include "kernel_PinholeExtraCalib_update_r_first.h"
#include "kernel_PinholeExtraCalib_update_step.h"
#include "kernel_PinholeExtraCalib_update_step_first.h"
#include "kernel_PinholeFocal_alpha_denumerator_or_beta_nummerator.h"
#include "kernel_PinholeFocal_alpha_numerator_denominator.h"
#include "kernel_PinholeFocal_normalize.h"
#include "kernel_PinholeFocal_pred_decrease_times_two.h"
#include "kernel_PinholeFocal_retract.h"
#include "kernel_PinholeFocal_start_w.h"
#include "kernel_PinholeFocal_start_w_contribute.h"
#include "kernel_PinholeFocal_update_Mp.h"
#include "kernel_PinholeFocal_update_p.h"
#include "kernel_PinholeFocal_update_r.h"
#include "kernel_PinholeFocal_update_r_first.h"
#include "kernel_PinholeFocal_update_step.h"
#include "kernel_PinholeFocal_update_step_first.h"
#include "kernel_Point_alpha_denumerator_or_beta_nummerator.h"
#include "kernel_Point_alpha_numerator_denominator.h"
#include "kernel_Point_normalize.h"
#include "kernel_Point_pred_decrease_times_two.h"
#include "kernel_Point_retract.h"
#include "kernel_Point_start_w.h"
#include "kernel_Point_start_w_contribute.h"
#include "kernel_Point_update_Mp.h"
#include "kernel_Point_update_p.h"
#include "kernel_Point_update_r.h"
#include "kernel_Point_update_r_first.h"
#include "kernel_Point_update_step.h"
#include "kernel_Point_update_step_first.h"
#include "kernel_Pose_alpha_denumerator_or_beta_nummerator.h"
#include "kernel_Pose_alpha_numerator_denominator.h"
#include "kernel_Pose_normalize.h"
#include "kernel_Pose_pred_decrease_times_two.h"
#include "kernel_Pose_retract.h"
#include "kernel_Pose_start_w.h"
#include "kernel_Pose_start_w_contribute.h"
#include "kernel_Pose_update_Mp.h"
#include "kernel_Pose_update_p.h"
#include "kernel_Pose_update_r.h"
#include "kernel_Pose_update_r_first.h"
#include "kernel_Pose_update_step.h"
#include "kernel_Pose_update_step_first.h"
#include "kernel_SimpleRadialExtraCalib_alpha_denumerator_or_beta_nummerator.h"
#include "kernel_SimpleRadialExtraCalib_alpha_numerator_denominator.h"
#include "kernel_SimpleRadialExtraCalib_normalize.h"
#include "kernel_SimpleRadialExtraCalib_pred_decrease_times_two.h"
#include "kernel_SimpleRadialExtraCalib_retract.h"
#include "kernel_SimpleRadialExtraCalib_start_w.h"
#include "kernel_SimpleRadialExtraCalib_start_w_contribute.h"
#include "kernel_SimpleRadialExtraCalib_update_Mp.h"
#include "kernel_SimpleRadialExtraCalib_update_p.h"
#include "kernel_SimpleRadialExtraCalib_update_r.h"
#include "kernel_SimpleRadialExtraCalib_update_r_first.h"
#include "kernel_SimpleRadialExtraCalib_update_step.h"
#include "kernel_SimpleRadialExtraCalib_update_step_first.h"
#include "kernel_SimpleRadialFocal_alpha_denumerator_or_beta_nummerator.h"
#include "kernel_SimpleRadialFocal_alpha_numerator_denominator.h"
#include "kernel_SimpleRadialFocal_normalize.h"
#include "kernel_SimpleRadialFocal_pred_decrease_times_two.h"
#include "kernel_SimpleRadialFocal_retract.h"
#include "kernel_SimpleRadialFocal_start_w.h"
#include "kernel_SimpleRadialFocal_start_w_contribute.h"
#include "kernel_SimpleRadialFocal_update_Mp.h"
#include "kernel_SimpleRadialFocal_update_p.h"
#include "kernel_SimpleRadialFocal_update_r.h"
#include "kernel_SimpleRadialFocal_update_r_first.h"
#include "kernel_SimpleRadialFocal_update_step.h"
#include "kernel_SimpleRadialFocal_update_step_first.h"
#include "kernel_pinhole_fixed_extra_calib_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_extra_calib_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_extra_calib_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_extra_calib_fixed_point_score.h"
#include "kernel_pinhole_fixed_extra_calib_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_extra_calib_res_jac.h"
#include "kernel_pinhole_fixed_extra_calib_res_jac_first.h"
#include "kernel_pinhole_fixed_extra_calib_score.h"
#include "kernel_pinhole_fixed_focal_fixed_extra_calib_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_focal_fixed_extra_calib_fixed_point_score.h"
#include "kernel_pinhole_fixed_focal_fixed_extra_calib_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_focal_fixed_extra_calib_res_jac.h"
#include "kernel_pinhole_fixed_focal_fixed_extra_calib_res_jac_first.h"
#include "kernel_pinhole_fixed_focal_fixed_extra_calib_score.h"
#include "kernel_pinhole_fixed_focal_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_focal_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_focal_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_focal_fixed_point_score.h"
#include "kernel_pinhole_fixed_focal_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_focal_res_jac.h"
#include "kernel_pinhole_fixed_focal_res_jac_first.h"
#include "kernel_pinhole_fixed_focal_score.h"
#include "kernel_pinhole_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_point_score.h"
#include "kernel_pinhole_fixed_pose_fixed_extra_calib_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_fixed_extra_calib_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_pose_fixed_extra_calib_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_fixed_extra_calib_fixed_point_score.h"
#include "kernel_pinhole_fixed_pose_fixed_extra_calib_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_fixed_extra_calib_res_jac.h"
#include "kernel_pinhole_fixed_pose_fixed_extra_calib_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_fixed_extra_calib_score.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_res_jac.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_score.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_fixed_point_score.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_res_jac.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_score.h"
#include "kernel_pinhole_fixed_pose_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_pose_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_fixed_point_score.h"
#include "kernel_pinhole_fixed_pose_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_res_jac.h"
#include "kernel_pinhole_fixed_pose_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_score.h"
#include "kernel_pinhole_jtjnjtr_direct.h"
#include "kernel_pinhole_res_jac.h"
#include "kernel_pinhole_res_jac_first.h"
#include "kernel_pinhole_score.h"
#include "kernel_simple_radial_fixed_extra_calib_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_extra_calib_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_extra_calib_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_extra_calib_fixed_point_score.h"
#include "kernel_simple_radial_fixed_extra_calib_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_extra_calib_res_jac.h"
#include "kernel_simple_radial_fixed_extra_calib_res_jac_first.h"
#include "kernel_simple_radial_fixed_extra_calib_score.h"
#include "kernel_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_score.h"
#include "kernel_simple_radial_fixed_focal_fixed_extra_calib_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_focal_fixed_extra_calib_res_jac.h"
#include "kernel_simple_radial_fixed_focal_fixed_extra_calib_res_jac_first.h"
#include "kernel_simple_radial_fixed_focal_fixed_extra_calib_score.h"
#include "kernel_simple_radial_fixed_focal_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_focal_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_focal_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_focal_fixed_point_score.h"
#include "kernel_simple_radial_fixed_focal_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_focal_res_jac.h"
#include "kernel_simple_radial_fixed_focal_res_jac_first.h"
#include "kernel_simple_radial_fixed_focal_score.h"
#include "kernel_simple_radial_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_point_score.h"
#include "kernel_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_score.h"
#include "kernel_simple_radial_fixed_pose_fixed_extra_calib_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_fixed_extra_calib_res_jac.h"
#include "kernel_simple_radial_fixed_pose_fixed_extra_calib_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_fixed_extra_calib_score.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_res_jac.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_score.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_fixed_point_score.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_res_jac.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_score.h"
#include "kernel_simple_radial_fixed_pose_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_pose_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_fixed_point_score.h"
#include "kernel_simple_radial_fixed_pose_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_res_jac.h"
#include "kernel_simple_radial_fixed_pose_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_score.h"
#include "kernel_simple_radial_jtjnjtr_direct.h"
#include "kernel_simple_radial_res_jac.h"
#include "kernel_simple_radial_res_jac_first.h"
#include "kernel_simple_radial_score.h"
#include "shared_indices.h"
#include "solver_tools.h"
#include "sort_indices.h"

namespace {

void make_aligned(size_t& offset, size_t alignment_bytes) {
  offset = ((offset + alignment_bytes - 1) / alignment_bytes) * alignment_bytes;
}

template <typename T>
void increment_offset(size_t& offset,
                      size_t num_elements,
                      size_t alignment_elements) {
  make_aligned(offset, alignment_elements * sizeof(T));
  offset += num_elements * sizeof(T);
}

template <typename T>
T* assign_and_increment(uint8_t* origin_ptr,
                        size_t& offset,
                        size_t num_elements,
                        size_t alignment_elements) {
  make_aligned(offset, alignment_elements * sizeof(T));
  size_t old_offset = offset;
  offset += num_elements * sizeof(T);
  return reinterpret_cast<T*>(origin_ptr + old_offset);
}

}  // namespace

namespace caspar {

GraphSolver::GraphSolver(
    const SolverParams<double>& params,
    size_t PinholeExtraCalib_num_max,
    size_t PinholeFocal_num_max,
    size_t Point_num_max,
    size_t Pose_num_max,
    size_t SimpleRadialExtraCalib_num_max,
    size_t SimpleRadialFocal_num_max,
    size_t simple_radial_num_max,
    size_t simple_radial_fixed_pose_num_max,
    size_t simple_radial_fixed_focal_num_max,
    size_t simple_radial_fixed_extra_calib_num_max,
    size_t simple_radial_fixed_point_num_max,
    size_t simple_radial_fixed_pose_fixed_focal_num_max,
    size_t simple_radial_fixed_pose_fixed_extra_calib_num_max,
    size_t simple_radial_fixed_pose_fixed_point_num_max,
    size_t simple_radial_fixed_focal_fixed_extra_calib_num_max,
    size_t simple_radial_fixed_focal_fixed_point_num_max,
    size_t simple_radial_fixed_extra_calib_fixed_point_num_max,
    size_t simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max,
    size_t simple_radial_fixed_pose_fixed_focal_fixed_point_num_max,
    size_t simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max,
    size_t simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max,
    size_t pinhole_num_max,
    size_t pinhole_fixed_pose_num_max,
    size_t pinhole_fixed_focal_num_max,
    size_t pinhole_fixed_extra_calib_num_max,
    size_t pinhole_fixed_point_num_max,
    size_t pinhole_fixed_pose_fixed_focal_num_max,
    size_t pinhole_fixed_pose_fixed_extra_calib_num_max,
    size_t pinhole_fixed_pose_fixed_point_num_max,
    size_t pinhole_fixed_focal_fixed_extra_calib_num_max,
    size_t pinhole_fixed_focal_fixed_point_num_max,
    size_t pinhole_fixed_extra_calib_fixed_point_num_max,
    size_t pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max,
    size_t pinhole_fixed_pose_fixed_focal_fixed_point_num_max,
    size_t pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max,
    size_t pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max)
    : params_(params),
      PinholeExtraCalib_num_(PinholeExtraCalib_num_max),
      PinholeExtraCalib_num_max_(PinholeExtraCalib_num_max),
      PinholeFocal_num_(PinholeFocal_num_max),
      PinholeFocal_num_max_(PinholeFocal_num_max),
      Point_num_(Point_num_max),
      Point_num_max_(Point_num_max),
      Pose_num_(Pose_num_max),
      Pose_num_max_(Pose_num_max),
      SimpleRadialExtraCalib_num_(SimpleRadialExtraCalib_num_max),
      SimpleRadialExtraCalib_num_max_(SimpleRadialExtraCalib_num_max),
      SimpleRadialFocal_num_(SimpleRadialFocal_num_max),
      SimpleRadialFocal_num_max_(SimpleRadialFocal_num_max),
      simple_radial_num_(simple_radial_num_max),
      simple_radial_num_max_(simple_radial_num_max),
      simple_radial_fixed_pose_num_(simple_radial_fixed_pose_num_max),
      simple_radial_fixed_pose_num_max_(simple_radial_fixed_pose_num_max),
      simple_radial_fixed_focal_num_(simple_radial_fixed_focal_num_max),
      simple_radial_fixed_focal_num_max_(simple_radial_fixed_focal_num_max),
      simple_radial_fixed_extra_calib_num_(
          simple_radial_fixed_extra_calib_num_max),
      simple_radial_fixed_extra_calib_num_max_(
          simple_radial_fixed_extra_calib_num_max),
      simple_radial_fixed_point_num_(simple_radial_fixed_point_num_max),
      simple_radial_fixed_point_num_max_(simple_radial_fixed_point_num_max),
      simple_radial_fixed_pose_fixed_focal_num_(
          simple_radial_fixed_pose_fixed_focal_num_max),
      simple_radial_fixed_pose_fixed_focal_num_max_(
          simple_radial_fixed_pose_fixed_focal_num_max),
      simple_radial_fixed_pose_fixed_extra_calib_num_(
          simple_radial_fixed_pose_fixed_extra_calib_num_max),
      simple_radial_fixed_pose_fixed_extra_calib_num_max_(
          simple_radial_fixed_pose_fixed_extra_calib_num_max),
      simple_radial_fixed_pose_fixed_point_num_(
          simple_radial_fixed_pose_fixed_point_num_max),
      simple_radial_fixed_pose_fixed_point_num_max_(
          simple_radial_fixed_pose_fixed_point_num_max),
      simple_radial_fixed_focal_fixed_extra_calib_num_(
          simple_radial_fixed_focal_fixed_extra_calib_num_max),
      simple_radial_fixed_focal_fixed_extra_calib_num_max_(
          simple_radial_fixed_focal_fixed_extra_calib_num_max),
      simple_radial_fixed_focal_fixed_point_num_(
          simple_radial_fixed_focal_fixed_point_num_max),
      simple_radial_fixed_focal_fixed_point_num_max_(
          simple_radial_fixed_focal_fixed_point_num_max),
      simple_radial_fixed_extra_calib_fixed_point_num_(
          simple_radial_fixed_extra_calib_fixed_point_num_max),
      simple_radial_fixed_extra_calib_fixed_point_num_max_(
          simple_radial_fixed_extra_calib_fixed_point_num_max),
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_(
          simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max),
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_(
          simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max),
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_(
          simple_radial_fixed_pose_fixed_focal_fixed_point_num_max),
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_(
          simple_radial_fixed_pose_fixed_focal_fixed_point_num_max),
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_(
          simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max),
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_(
          simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max),
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_(
          simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max),
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_(
          simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max),
      pinhole_num_(pinhole_num_max),
      pinhole_num_max_(pinhole_num_max),
      pinhole_fixed_pose_num_(pinhole_fixed_pose_num_max),
      pinhole_fixed_pose_num_max_(pinhole_fixed_pose_num_max),
      pinhole_fixed_focal_num_(pinhole_fixed_focal_num_max),
      pinhole_fixed_focal_num_max_(pinhole_fixed_focal_num_max),
      pinhole_fixed_extra_calib_num_(pinhole_fixed_extra_calib_num_max),
      pinhole_fixed_extra_calib_num_max_(pinhole_fixed_extra_calib_num_max),
      pinhole_fixed_point_num_(pinhole_fixed_point_num_max),
      pinhole_fixed_point_num_max_(pinhole_fixed_point_num_max),
      pinhole_fixed_pose_fixed_focal_num_(
          pinhole_fixed_pose_fixed_focal_num_max),
      pinhole_fixed_pose_fixed_focal_num_max_(
          pinhole_fixed_pose_fixed_focal_num_max),
      pinhole_fixed_pose_fixed_extra_calib_num_(
          pinhole_fixed_pose_fixed_extra_calib_num_max),
      pinhole_fixed_pose_fixed_extra_calib_num_max_(
          pinhole_fixed_pose_fixed_extra_calib_num_max),
      pinhole_fixed_pose_fixed_point_num_(
          pinhole_fixed_pose_fixed_point_num_max),
      pinhole_fixed_pose_fixed_point_num_max_(
          pinhole_fixed_pose_fixed_point_num_max),
      pinhole_fixed_focal_fixed_extra_calib_num_(
          pinhole_fixed_focal_fixed_extra_calib_num_max),
      pinhole_fixed_focal_fixed_extra_calib_num_max_(
          pinhole_fixed_focal_fixed_extra_calib_num_max),
      pinhole_fixed_focal_fixed_point_num_(
          pinhole_fixed_focal_fixed_point_num_max),
      pinhole_fixed_focal_fixed_point_num_max_(
          pinhole_fixed_focal_fixed_point_num_max),
      pinhole_fixed_extra_calib_fixed_point_num_(
          pinhole_fixed_extra_calib_fixed_point_num_max),
      pinhole_fixed_extra_calib_fixed_point_num_max_(
          pinhole_fixed_extra_calib_fixed_point_num_max),
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_(
          pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max),
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_(
          pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max),
      pinhole_fixed_pose_fixed_focal_fixed_point_num_(
          pinhole_fixed_pose_fixed_focal_fixed_point_num_max),
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_(
          pinhole_fixed_pose_fixed_focal_fixed_point_num_max),
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_(
          pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max),
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_(
          pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max),
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_(
          pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max),
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_(
          pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max) {
  indices_valid_ = false;
  if (params.pcg_rel_error_exit <= 0.0f) {
    throw std::runtime_error("params.pcg_rel_error_exit must be positive");
  }
  if (params.diag_init < 0.0f) {
    throw std::runtime_error("params.diag_init must be positive");
  }
  allocation_size_ = get_nbytes();
  cudaMalloc(&origin_ptr_, allocation_size_);

  size_t offset = 0;
  marker__start_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeExtraCalib__storage_current_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeExtraCalib_num_, 4);
  nodes__PinholeExtraCalib__storage_check_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeExtraCalib_num_, 4);
  nodes__PinholeExtraCalib__storage_new_best_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeExtraCalib_num_, 4);
  nodes__PinholeFocal__storage_current_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholeFocal__storage_check_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholeFocal__storage_new_best_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__Point__storage_current_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__storage_check_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__storage_new_best_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Pose__storage_current_ =
      assign_and_increment<float>(origin_ptr_, offset, 8 * Pose_num_, 4);
  nodes__Pose__storage_check_ =
      assign_and_increment<float>(origin_ptr_, offset, 8 * Pose_num_, 4);
  nodes__Pose__storage_new_best_ =
      assign_and_increment<float>(origin_ptr_, offset, 8 * Pose_num_, 4);
  nodes__SimpleRadialExtraCalib__storage_current_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialExtraCalib_num_, 4);
  nodes__SimpleRadialExtraCalib__storage_check_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialExtraCalib_num_, 4);
  nodes__SimpleRadialExtraCalib__storage_new_best_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 4 * SimpleRadialExtraCalib_num_, 4);
  nodes__SimpleRadialFocal__storage_current_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * SimpleRadialFocal_num_, 4);
  nodes__SimpleRadialFocal__storage_check_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * SimpleRadialFocal_num_, 4);
  nodes__SimpleRadialFocal__storage_new_best_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * SimpleRadialFocal_num_, 4);
  facs__simple_radial__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_num_, 4);
  facs__simple_radial__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_num_, 4);
  facs__simple_radial__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_num_, 4);
  facs__simple_radial__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_num_, 4);
  facs__simple_radial__args__pixel__data_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_num_, 4);
  facs__simple_radial_fixed_pose__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_pose__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_pose__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_pose__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_pose__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 8 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_focal__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_focal_num_, 4);
  facs__simple_radial_fixed_focal__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_focal_num_, 4);
  facs__simple_radial_fixed_focal__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_focal_num_, 4);
  facs__simple_radial_fixed_focal__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * simple_radial_fixed_focal_num_, 4);
  facs__simple_radial_fixed_focal__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 1 * simple_radial_fixed_focal_num_, 4);
  facs__simple_radial_fixed_extra_calib__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_extra_calib_num_, 4);
  facs__simple_radial_fixed_extra_calib__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_extra_calib_num_, 4);
  facs__simple_radial_fixed_extra_calib__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_extra_calib_num_, 4);
  facs__simple_radial_fixed_extra_calib__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * simple_radial_fixed_extra_calib_num_, 4);
  facs__simple_radial_fixed_extra_calib__args__extra_calib__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 4 * simple_radial_fixed_extra_calib_num_, 4);
  facs__simple_radial_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 4 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_pose_fixed_focal__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_focal_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_focal_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal__args__pixel__data_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * simple_radial_fixed_pose_fixed_focal_num_,
                                  4);
  facs__simple_radial_fixed_pose_fixed_focal__args__pose__data_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  8 * simple_radial_fixed_pose_fixed_focal_num_,
                                  4);
  facs__simple_radial_fixed_pose_fixed_focal__args__focal__data_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  1 * simple_radial_fixed_pose_fixed_focal_num_,
                                  4);
  facs__simple_radial_fixed_pose_fixed_extra_calib__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_extra_calib__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_extra_calib__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_extra_calib__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_fixed_pose_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_extra_calib__args__extra_calib__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_pose_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_point__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_point__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * simple_radial_fixed_pose_fixed_point_num_,
                                  4);
  facs__simple_radial_fixed_pose_fixed_point__args__pose__data_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  8 * simple_radial_fixed_pose_fixed_point_num_,
                                  4);
  facs__simple_radial_fixed_pose_fixed_point__args__point__data_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  4 * simple_radial_fixed_pose_fixed_point_num_,
                                  4);
  facs__simple_radial_fixed_focal_fixed_extra_calib__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_extra_calib__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_extra_calib__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_extra_calib__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_extra_calib__args__extra_calib__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_point__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_point__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_extra_calib_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_extra_calib_fixed_point__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_extra_calib_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_extra_calib_fixed_point__args__extra_calib__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_extra_calib_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__extra_calib__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__extra_calib__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__extra_calib__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole__args__pose__idx_shared_ = assign_and_increment<SharedIndex>(
      origin_ptr_, offset, 1 * pinhole_num_, 4);
  facs__pinhole__args__focal__idx_shared_ = assign_and_increment<SharedIndex>(
      origin_ptr_, offset, 1 * pinhole_num_, 4);
  facs__pinhole__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_num_, 4);
  facs__pinhole__args__point__idx_shared_ = assign_and_increment<SharedIndex>(
      origin_ptr_, offset, 1 * pinhole_num_, 4);
  facs__pinhole__args__pixel__data_ =
      assign_and_increment<float>(origin_ptr_, offset, 2 * pinhole_num_, 4);
  facs__pinhole_fixed_pose__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__pixel__data_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__pose__data_ = assign_and_increment<float>(
      origin_ptr_, offset, 8 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_focal__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_focal_num_, 4);
  facs__pinhole_fixed_focal__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_focal_num_, 4);
  facs__pinhole_fixed_focal__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_focal_num_, 4);
  facs__pinhole_fixed_focal__args__pixel__data_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_focal_num_, 4);
  facs__pinhole_fixed_focal__args__focal__data_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_focal_num_, 4);
  facs__pinhole_fixed_extra_calib__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_extra_calib_num_, 4);
  facs__pinhole_fixed_extra_calib__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_extra_calib_num_, 4);
  facs__pinhole_fixed_extra_calib__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_extra_calib_num_, 4);
  facs__pinhole_fixed_extra_calib__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_fixed_extra_calib_num_, 4);
  facs__pinhole_fixed_extra_calib__args__extra_calib__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_fixed_extra_calib_num_, 4);
  facs__pinhole_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__pixel__data_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__point__data_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_focal__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_pose_fixed_focal_num_, 4);
  facs__pinhole_fixed_pose_fixed_focal__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_pose_fixed_focal_num_, 4);
  facs__pinhole_fixed_pose_fixed_focal__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_focal_num_, 4);
  facs__pinhole_fixed_pose_fixed_focal__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 8 * pinhole_fixed_pose_fixed_focal_num_, 4);
  facs__pinhole_fixed_pose_fixed_focal__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_focal_num_, 4);
  facs__pinhole_fixed_pose_fixed_extra_calib__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_pose_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_pose_fixed_extra_calib__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_pose_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_pose_fixed_extra_calib__args__pixel__data_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * pinhole_fixed_pose_fixed_extra_calib_num_,
                                  4);
  facs__pinhole_fixed_pose_fixed_extra_calib__args__pose__data_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  8 * pinhole_fixed_pose_fixed_extra_calib_num_,
                                  4);
  facs__pinhole_fixed_pose_fixed_extra_calib__args__extra_calib__data_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * pinhole_fixed_pose_fixed_extra_calib_num_,
                                  4);
  facs__pinhole_fixed_pose_fixed_point__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 8 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 4 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_focal_fixed_extra_calib__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_focal_fixed_extra_calib__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_focal_fixed_extra_calib__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_focal_fixed_extra_calib__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_focal_fixed_extra_calib__args__extra_calib__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_focal_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_focal_fixed_point_num_, 4);
  facs__pinhole_fixed_focal_fixed_point__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_focal_fixed_point_num_, 4);
  facs__pinhole_fixed_focal_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_fixed_focal_fixed_point_num_, 4);
  facs__pinhole_fixed_focal_fixed_point__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_fixed_focal_fixed_point_num_, 4);
  facs__pinhole_fixed_focal_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 4 * pinhole_fixed_focal_fixed_point_num_, 4);
  facs__pinhole_fixed_extra_calib_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_extra_calib_fixed_point__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_extra_calib_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_extra_calib_fixed_point__args__extra_calib__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_extra_calib_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * pinhole_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__extra_calib__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__extra_calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * pinhole_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__extra_calib__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__extra_calib__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  marker__scratch_inout_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  facs__simple_radial__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_num_, 4);
  facs__simple_radial_fixed_pose__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_focal__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_focal_num_, 4);
  facs__simple_radial_fixed_extra_calib__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_extra_calib_num_, 4);
  facs__simple_radial_fixed_point__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_pose_fixed_focal__res_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * simple_radial_fixed_pose_fixed_focal_num_,
                                  4);
  facs__simple_radial_fixed_pose_fixed_extra_calib__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_point__res_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * simple_radial_fixed_pose_fixed_point_num_,
                                  4);
  facs__simple_radial_fixed_focal_fixed_extra_calib__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_extra_calib_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole__res_ =
      assign_and_increment<float>(origin_ptr_, offset, 2 * pinhole_num_, 4);
  facs__pinhole_fixed_pose__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_focal__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_focal_num_, 4);
  facs__pinhole_fixed_extra_calib__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_extra_calib_num_, 4);
  facs__pinhole_fixed_point__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_focal__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_focal_num_, 4);
  facs__pinhole_fixed_pose_fixed_extra_calib__res_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * pinhole_fixed_pose_fixed_extra_calib_num_,
                                  4);
  facs__pinhole_fixed_pose_fixed_point__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_focal_fixed_extra_calib__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_focal_fixed_point__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_focal_fixed_point_num_, 4);
  facs__pinhole_fixed_extra_calib_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial__args__pose__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 12 * simple_radial_num_, 4);
  facs__simple_radial__args__focal__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_num_, 4);
  facs__simple_radial__args__extra_calib__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_num_, 4);
  facs__simple_radial__args__point__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * simple_radial_num_, 4);
  facs__simple_radial_fixed_pose__args__focal__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_pose__args__extra_calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_pose__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 6 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_focal__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 12 * simple_radial_fixed_focal_num_, 4);
  facs__simple_radial_fixed_focal__args__extra_calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * simple_radial_fixed_focal_num_, 4);
  facs__simple_radial_fixed_focal__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 6 * simple_radial_fixed_focal_num_, 4);
  facs__simple_radial_fixed_extra_calib__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 12 * simple_radial_fixed_extra_calib_num_, 4);
  facs__simple_radial_fixed_extra_calib__args__focal__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * simple_radial_fixed_extra_calib_num_, 4);
  facs__simple_radial_fixed_extra_calib__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 6 * simple_radial_fixed_extra_calib_num_, 4);
  facs__simple_radial_fixed_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 12 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__focal__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__extra_calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_pose_fixed_focal__args__extra_calib__jac_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * simple_radial_fixed_pose_fixed_focal_num_,
                                  4);
  facs__simple_radial_fixed_pose_fixed_focal__args__point__jac_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  6 * simple_radial_fixed_pose_fixed_focal_num_,
                                  4);
  facs__simple_radial_fixed_pose_fixed_extra_calib__args__focal__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_extra_calib__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          6 * simple_radial_fixed_pose_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_point__args__focal__jac_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * simple_radial_fixed_pose_fixed_point_num_,
                                  4);
  facs__simple_radial_fixed_pose_fixed_point__args__extra_calib__jac_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * simple_radial_fixed_pose_fixed_point_num_,
                                  4);
  facs__simple_radial_fixed_focal_fixed_extra_calib__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          12 * simple_radial_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_extra_calib__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          6 * simple_radial_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          12 * simple_radial_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_point__args__extra_calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_extra_calib_fixed_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          12 * simple_radial_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_extra_calib_fixed_point__args__focal__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          6 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__extra_calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__focal__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          12 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole__args__pose__jac_ =
      assign_and_increment<float>(origin_ptr_, offset, 10 * pinhole_num_, 4);
  facs__pinhole__args__focal__jac_ =
      assign_and_increment<float>(origin_ptr_, offset, 2 * pinhole_num_, 4);
  facs__pinhole__args__extra_calib__jac_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * pinhole_num_, 4);
  facs__pinhole__args__point__jac_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * pinhole_num_, 4);
  facs__pinhole_fixed_pose__args__focal__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__extra_calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 0 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__point__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_focal__args__pose__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 10 * pinhole_fixed_focal_num_, 4);
  facs__pinhole_fixed_focal__args__extra_calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 0 * pinhole_fixed_focal_num_, 4);
  facs__pinhole_fixed_focal__args__point__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * pinhole_fixed_focal_num_, 4);
  facs__pinhole_fixed_extra_calib__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 10 * pinhole_fixed_extra_calib_num_, 4);
  facs__pinhole_fixed_extra_calib__args__focal__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_fixed_extra_calib_num_, 4);
  facs__pinhole_fixed_extra_calib__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 6 * pinhole_fixed_extra_calib_num_, 4);
  facs__pinhole_fixed_point__args__pose__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 10 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__focal__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__extra_calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 0 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_focal__args__extra_calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 0 * pinhole_fixed_pose_fixed_focal_num_, 4);
  facs__pinhole_fixed_pose_fixed_focal__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 6 * pinhole_fixed_pose_fixed_focal_num_, 4);
  facs__pinhole_fixed_pose_fixed_extra_calib__args__focal__jac_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * pinhole_fixed_pose_fixed_extra_calib_num_,
                                  4);
  facs__pinhole_fixed_pose_fixed_extra_calib__args__point__jac_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  6 * pinhole_fixed_pose_fixed_extra_calib_num_,
                                  4);
  facs__pinhole_fixed_pose_fixed_point__args__focal__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__args__extra_calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 0 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_focal_fixed_extra_calib__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          10 * pinhole_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_focal_fixed_extra_calib__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          6 * pinhole_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_focal_fixed_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 10 * pinhole_fixed_focal_fixed_point_num_, 4);
  facs__pinhole_fixed_focal_fixed_point__args__extra_calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 0 * pinhole_fixed_focal_fixed_point_num_, 4);
  facs__pinhole_fixed_extra_calib_fixed_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          10 * pinhole_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_extra_calib_fixed_point__args__focal__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          6 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__extra_calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          0 * pinhole_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__focal__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          10 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  nodes__PinholeExtraCalib__z_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeExtraCalib_num_, 4);
  nodes__PinholeExtraCalib__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocal__z_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholeFocal__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__z_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Pose__z_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__Pose__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialExtraCalib__z_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialExtraCalib_num_, 4);
  nodes__SimpleRadialExtraCalib__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialFocal__z_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * SimpleRadialFocal_num_, 4);
  nodes__SimpleRadialFocal__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeExtraCalib__p_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeExtraCalib_num_, 4);
  nodes__PinholeExtraCalib__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocal__p_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholeFocal__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__p_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Pose__p_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__Pose__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialExtraCalib__p_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialExtraCalib_num_, 4);
  nodes__SimpleRadialExtraCalib__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialFocal__p_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * SimpleRadialFocal_num_, 4);
  nodes__SimpleRadialFocal__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeExtraCalib__step_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeExtraCalib_num_, 4);
  nodes__PinholeExtraCalib__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocal__step_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholeFocal__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__step_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Pose__step_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__Pose__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialExtraCalib__step_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialExtraCalib_num_, 4);
  nodes__SimpleRadialExtraCalib__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialFocal__step_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * SimpleRadialFocal_num_, 4);
  nodes__SimpleRadialFocal__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  marker__w_start_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeExtraCalib__w_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeExtraCalib_num_, 4);
  nodes__PinholeFocal__w_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__Point__w_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Pose__w_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__SimpleRadialExtraCalib__w_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialExtraCalib_num_, 4);
  nodes__SimpleRadialFocal__w_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * SimpleRadialFocal_num_, 4);
  marker__w_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 1);
  marker__r_0_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeExtraCalib__r_0_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeExtraCalib_num_, 4);
  nodes__PinholeFocal__r_0_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__Point__r_0_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Pose__r_0_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__SimpleRadialExtraCalib__r_0_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialExtraCalib_num_, 4);
  nodes__SimpleRadialFocal__r_0_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * SimpleRadialFocal_num_, 4);
  marker__r_0_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  marker__r_k_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeExtraCalib__r_k_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeExtraCalib_num_, 4);
  nodes__PinholeFocal__r_k_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__Point__r_k_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Pose__r_k_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__SimpleRadialExtraCalib__r_k_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialExtraCalib_num_, 4);
  nodes__SimpleRadialFocal__r_k_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * SimpleRadialFocal_num_, 4);
  marker__r_k_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  marker__Mp_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeExtraCalib__Mp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeExtraCalib_num_, 4);
  nodes__PinholeFocal__Mp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__Point__Mp_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Pose__Mp_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__SimpleRadialExtraCalib__Mp_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialExtraCalib_num_, 4);
  nodes__SimpleRadialFocal__Mp_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * SimpleRadialFocal_num_, 4);
  marker__Mp_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  marker__precond_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeExtraCalib__precond_diag_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeExtraCalib_num_, 4);
  nodes__PinholeExtraCalib__precond_tril_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * PinholeExtraCalib_num_, 4);
  nodes__PinholeFocal__precond_diag_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholeFocal__precond_tril_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * PinholeFocal_num_, 4);
  nodes__Point__precond_diag_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__precond_tril_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Pose__precond_diag_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__Pose__precond_tril_ =
      assign_and_increment<float>(origin_ptr_, offset, 16 * Pose_num_, 4);
  nodes__SimpleRadialExtraCalib__precond_diag_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialExtraCalib_num_, 4);
  nodes__SimpleRadialExtraCalib__precond_tril_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialExtraCalib_num_, 4);
  nodes__SimpleRadialFocal__precond_diag_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * SimpleRadialFocal_num_, 4);
  nodes__SimpleRadialFocal__precond_tril_ = assign_and_increment<float>(
      origin_ptr_, offset, 0 * SimpleRadialFocal_num_, 4);
  marker__precond_end_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 1);
  marker__jp_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  facs__simple_radial__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_num_, 4);
  facs__simple_radial_fixed_pose__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_focal__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_focal_num_, 4);
  facs__simple_radial_fixed_extra_calib__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_extra_calib_num_, 4);
  facs__simple_radial_fixed_point__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_pose_fixed_focal__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_pose_fixed_focal_num_, 4);
  facs__simple_radial_fixed_pose_fixed_extra_calib__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_point__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_pose_fixed_point_num_, 4);
  facs__simple_radial_fixed_focal_fixed_extra_calib__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_extra_calib_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole__jp_ =
      assign_and_increment<float>(origin_ptr_, offset, 2 * pinhole_num_, 4);
  facs__pinhole_fixed_pose__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_focal__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_focal_num_, 4);
  facs__pinhole_fixed_extra_calib__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_extra_calib_num_, 4);
  facs__pinhole_fixed_point__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_focal__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_focal_num_, 4);
  facs__pinhole_fixed_pose_fixed_extra_calib__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_extra_calib_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_focal_fixed_extra_calib__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_focal_fixed_point__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_focal_fixed_point_num_, 4);
  facs__pinhole_fixed_extra_calib_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_,
          4);
  marker__jp_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 1);
  solver__current_diag_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__alpha_numerator_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__alpha_denumerator_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__alpha_ = assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__neg_alpha_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__beta_numerator_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__beta_ = assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__r_0_norm2_tot_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__r_kp1_norm2_tot_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__pred_decrease_tot_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__res_tot_ = assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);

  scratch_inout_size_ = offset;  // sorting, sum,
}

GraphSolver::~GraphSolver() { cudaFree(origin_ptr_); }

void GraphSolver::set_params(const SolverParams<double>& params) {
  this->params_ = params;
}

size_t GraphSolver::get_allocation_size() { return allocation_size_; }

SolveResult GraphSolver::solve(bool print_progress, bool verbose_logging) {
  SolveResult result;
  result.exit_reason = ExitReason::MAX_ITERATIONS;
  float score_best;
  float score_best_pcg;
  float diag = params_.diag_init;
  cudaMemcpy(
      solver__current_diag_, &diag, sizeof(float), cudaMemcpyHostToDevice);

  float up_scale = params_.diag_scaling_up;
  float quality;

  std::chrono::time_point<std::chrono::steady_clock> t0 =
      std::chrono::steady_clock::now();
  std::chrono::time_point<std::chrono::steady_clock> t_prev = t0;
  score_best = do_res_jac_first();
  result.initial_score = score_best;
  if (print_progress) {
    printf("                                 score_init: % .6e\n", score_best);
  }

  for (solver_iter_ = 0; solver_iter_ < params_.solver_iter_max;
       solver_iter_++) {
    if (solver_iter_ != 0 && solver_iter_ < params_.solver_iter_max - 1) {
      do_res_jac();
    }
    score_best_pcg = score_best;
    for (pcg_iter_ = 0; pcg_iter_ < params_.pcg_iter_max; pcg_iter_++) {
      do_normalize();

      if (pcg_iter_ == 0) {
        copy(marker__r_k_start_, marker__r_k_end_, marker__w_start_);
        do_jtjp_direct();
        do_alpha_first();
        do_update_step_first();
        do_update_r_first();
      } else {
        do_beta();
        do_update_p();
        do_update_Mp();
        do_jtjp_direct();
        do_alpha();
        do_update_step();
        do_update_r();
      }
      if (params_.pcg_rel_decrease_min != -1.0f ||
          params_.pcg_rel_score_exit != -1.0f) {
        float score_new_pcg = do_retract_score();
        if (!(score_new_pcg <= score_best_pcg * params_.pcg_rel_decrease_min)) {
          break;
        }
        std::swap(nodes__PinholeExtraCalib__storage_check_,
                  nodes__PinholeExtraCalib__storage_new_best_);
        std::swap(nodes__PinholeFocal__storage_check_,
                  nodes__PinholeFocal__storage_new_best_);
        std::swap(nodes__Point__storage_check_,
                  nodes__Point__storage_new_best_);
        std::swap(nodes__Pose__storage_check_, nodes__Pose__storage_new_best_);
        std::swap(nodes__SimpleRadialExtraCalib__storage_check_,
                  nodes__SimpleRadialExtraCalib__storage_new_best_);
        std::swap(nodes__SimpleRadialFocal__storage_check_,
                  nodes__SimpleRadialFocal__storage_new_best_);
        score_best_pcg = score_new_pcg;
        if (params_.pcg_rel_score_exit != -1.0f &&
            score_best_pcg < score_best * params_.pcg_rel_score_exit) {
          break;
        }
      }
      if (pcg_r_kp1_norm2_ < pcg_r_0_norm2_ * params_.pcg_rel_error_exit) {
        break;
      }
    }
    pcg_iter_ = std::min(pcg_iter_, params_.pcg_iter_max - 1);

    if (params_.pcg_rel_decrease_min == -1.0f &&
        params_.pcg_rel_score_exit == -1.0f) {
      score_best_pcg = do_retract_score();
      std::swap(nodes__PinholeExtraCalib__storage_check_,
                nodes__PinholeExtraCalib__storage_new_best_);
      std::swap(nodes__PinholeFocal__storage_check_,
                nodes__PinholeFocal__storage_new_best_);
      std::swap(nodes__Point__storage_check_, nodes__Point__storage_new_best_);
      std::swap(nodes__Pose__storage_check_, nodes__Pose__storage_new_best_);
      std::swap(nodes__SimpleRadialExtraCalib__storage_check_,
                nodes__SimpleRadialExtraCalib__storage_new_best_);
      std::swap(nodes__SimpleRadialFocal__storage_check_,
                nodes__SimpleRadialFocal__storage_new_best_);
    }

    const float diag_current = diag;
    bool step_accepted = false;
    if (score_best_pcg < score_best * params_.solver_rel_decrease_min) {
      step_accepted = true;
      quality = (score_best - score_best_pcg) / get_pred_decrease();
      const float quality_tmp = 2 * quality - 1;
      float scale = std::max(params_.diag_scaling_down,
                             1.0f - quality_tmp * quality_tmp * quality_tmp);
      diag = std::max(params_.diag_min, diag * scale);
      cudaMemcpy(
          solver__current_diag_, &diag, sizeof(float), cudaMemcpyHostToDevice);
      up_scale = params_.diag_scaling_up;
      score_best = score_best_pcg;
      std::swap(nodes__PinholeExtraCalib__storage_current_,
                nodes__PinholeExtraCalib__storage_new_best_);
      std::swap(nodes__PinholeFocal__storage_current_,
                nodes__PinholeFocal__storage_new_best_);
      std::swap(nodes__Point__storage_current_,
                nodes__Point__storage_new_best_);
      std::swap(nodes__Pose__storage_current_, nodes__Pose__storage_new_best_);
      std::swap(nodes__SimpleRadialExtraCalib__storage_current_,
                nodes__SimpleRadialExtraCalib__storage_new_best_);
      std::swap(nodes__SimpleRadialFocal__storage_current_,
                nodes__SimpleRadialFocal__storage_new_best_);

    } else {
      quality = 0.0f;
      diag = diag * up_scale;
      if (diag > params_.diag_exit_value) {
        result.exit_reason = ExitReason::CONVERGED_DIAG_EXIT;
        break;
      }
      cudaMemcpy(
          solver__current_diag_, &diag, sizeof(float), cudaMemcpyHostToDevice);
      up_scale *= 2;
    }
    const auto t_now = std::chrono::steady_clock::now();
    const double dt_inc = std::chrono::duration<double>(t_now - t_prev).count();
    const double dt_tot = std::chrono::duration<double>(t_now - t0).count();

    if (verbose_logging) {
      IterationData iter_data;
      iter_data.solver_iter = solver_iter_;
      iter_data.pcg_iter = pcg_iter_;
      iter_data.score_current = score_best_pcg;
      iter_data.score_best = score_best;
      iter_data.step_quality = quality;
      iter_data.diag = diag_current;
      iter_data.dt_inc = dt_inc;
      iter_data.dt_tot = dt_tot;
      iter_data.step_accepted = step_accepted;
      result.iterations.push_back(iter_data);
    }

    if (print_progress) {
      printf("solver_iter: % 3d  ", solver_iter_);
      printf("pcg_iter: % 3d  ", pcg_iter_);
      printf("score_current: % 13.6e  ", score_best_pcg);
      printf("score_best: % 13.6e  ", score_best);
      printf("step_quality: % 7.3f  ", quality);
      printf("diag: % 6.3e  ", diag_current);
      printf("dt_inc: % 10.6f  ", dt_inc);
      printf("dt_tot: % 10.6f  ", dt_tot);
      t_prev = t_now;
      printf("\n");
    }
    if (score_best <= params_.score_exit_value) {
      result.exit_reason = ExitReason::CONVERGED_SCORE_THRESHOLD;
      break;
    }
  }

  const auto t_final = std::chrono::steady_clock::now();
  result.final_score = score_best;
  result.iteration_count = solver_iter_;
  result.runtime = std::chrono::duration<double>(t_final - t0).count();
  return result;
}

float GraphSolver::do_res_jac_first() {
  zero(solver__res_tot_, solver__res_tot_ + 1);
  zero(marker__r_0_start_, marker__precond_end_);

  simple_radial_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial__args__pose__idx_shared_,
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial__args__focal__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial__args__extra_calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial__args__point__idx_shared_,
      facs__simple_radial__args__pixel__data_,
      simple_radial_num_max_,

      facs__simple_radial__res_,
      simple_radial_num_,
      solver__res_tot_,
      facs__simple_radial__args__pose__jac_,
      simple_radial_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial__args__focal__jac_,
      simple_radial_num_,
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      facs__simple_radial__args__extra_calib__jac_,
      simple_radial_num_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      facs__simple_radial__args__point__jac_,
      simple_radial_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_num_);

  simple_radial_fixed_pose_res_jac_first(
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_pose__args__focal__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_pose__args__extra_calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose__args__point__idx_shared_,
      facs__simple_radial_fixed_pose__args__pixel__data_,
      simple_radial_fixed_pose_num_max_,
      facs__simple_radial_fixed_pose__args__pose__data_,
      simple_radial_fixed_pose_num_max_,

      facs__simple_radial_fixed_pose__res_,
      simple_radial_fixed_pose_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_pose__args__focal__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_pose__args__extra_calib__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      facs__simple_radial_fixed_pose__args__point__jac_,
      simple_radial_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_num_);

  simple_radial_fixed_focal_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_focal__args__pose__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_focal__args__extra_calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_focal__args__point__idx_shared_,
      facs__simple_radial_fixed_focal__args__pixel__data_,
      simple_radial_fixed_focal_num_max_,
      facs__simple_radial_fixed_focal__args__focal__data_,
      simple_radial_fixed_focal_num_max_,

      facs__simple_radial_fixed_focal__res_,
      simple_radial_fixed_focal_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_focal__args__pose__jac_,
      simple_radial_fixed_focal_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_focal__args__extra_calib__jac_,
      simple_radial_fixed_focal_num_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      facs__simple_radial_fixed_focal__args__point__jac_,
      simple_radial_fixed_focal_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_focal_num_);

  simple_radial_fixed_extra_calib_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_extra_calib__args__pose__idx_shared_,
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_extra_calib__args__focal__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_extra_calib_num_max_,

      facs__simple_radial_fixed_extra_calib__res_,
      simple_radial_fixed_extra_calib_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_extra_calib__args__pose__jac_,
      simple_radial_fixed_extra_calib_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_extra_calib__args__focal__jac_,
      simple_radial_fixed_extra_calib_num_,
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_extra_calib__args__point__jac_,
      simple_radial_fixed_extra_calib_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_extra_calib_num_);

  simple_radial_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_point__args__focal__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_point__args__pixel__data_,
      simple_radial_fixed_point_num_max_,
      facs__simple_radial_fixed_point__args__point__data_,
      simple_radial_fixed_point_num_max_,

      facs__simple_radial_fixed_point__res_,
      simple_radial_fixed_point_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_point__args__pose__jac_,
      simple_radial_fixed_point_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_point__args__focal__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_point__args__extra_calib__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      simple_radial_fixed_point_num_);

  simple_radial_fixed_pose_fixed_focal_res_jac_first(
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal__args__extra_calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,

      facs__simple_radial_fixed_pose_fixed_focal__res_,
      simple_radial_fixed_pose_fixed_focal_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_pose_fixed_focal__args__extra_calib__jac_,
      simple_radial_fixed_pose_fixed_focal_num_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      facs__simple_radial_fixed_pose_fixed_focal__args__point__jac_,
      simple_radial_fixed_pose_fixed_focal_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_fixed_focal_num_);

  simple_radial_fixed_pose_fixed_extra_calib_res_jac_first(
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__focal__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__pose__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,

      facs__simple_radial_fixed_pose_fixed_extra_calib__res_,
      simple_radial_fixed_pose_fixed_extra_calib_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__focal__jac_,
      simple_radial_fixed_pose_fixed_extra_calib_num_,
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__point__jac_,
      simple_radial_fixed_pose_fixed_extra_calib_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_fixed_extra_calib_num_);

  simple_radial_fixed_pose_fixed_point_res_jac_first(
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__focal__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_point__res_,
      simple_radial_fixed_pose_fixed_point_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_pose_fixed_point__args__focal__jac_,
      simple_radial_fixed_pose_fixed_point_num_,
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_pose_fixed_point__args__extra_calib__jac_,
      simple_radial_fixed_pose_fixed_point_num_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      simple_radial_fixed_pose_fixed_point_num_);

  simple_radial_fixed_focal_fixed_extra_calib_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__pose__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__focal__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,

      facs__simple_radial_fixed_focal_fixed_extra_calib__res_,
      simple_radial_fixed_focal_fixed_extra_calib_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__pose__jac_,
      simple_radial_fixed_focal_fixed_extra_calib_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__point__jac_,
      simple_radial_fixed_focal_fixed_extra_calib_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_focal_fixed_extra_calib_num_);

  simple_radial_fixed_focal_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_focal_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_focal_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_point__args__focal__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_point__args__point__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,

      facs__simple_radial_fixed_focal_fixed_point__res_,
      simple_radial_fixed_focal_fixed_point_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_focal_fixed_point__args__pose__jac_,
      simple_radial_fixed_focal_fixed_point_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_focal_fixed_point__args__extra_calib__jac_,
      simple_radial_fixed_focal_fixed_point_num_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      simple_radial_fixed_focal_fixed_point_num_);

  simple_radial_fixed_extra_calib_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,

      facs__simple_radial_fixed_extra_calib_fixed_point__res_,
      simple_radial_fixed_extra_calib_fixed_point_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__pose__jac_,
      simple_radial_fixed_extra_calib_fixed_point_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__focal__jac_,
      simple_radial_fixed_extra_calib_fixed_point_num_,
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      simple_radial_fixed_extra_calib_fixed_point_num_);

  simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_res_jac_first(
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,

      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__res_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
      solver__res_tot_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_);

  simple_radial_fixed_pose_fixed_focal_fixed_point_res_jac_first(
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__res_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_,
      solver__res_tot_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_);

  simple_radial_fixed_pose_fixed_extra_calib_fixed_point_res_jac_first(
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__res_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
      solver__res_tot_,
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_);

  simple_radial_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__focal__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,

      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__res_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
      solver__res_tot_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_);

  pinhole_res_jac_first(nodes__Pose__storage_current_,
                        Pose_num_max_,
                        facs__pinhole__args__pose__idx_shared_,
                        nodes__PinholeFocal__storage_current_,
                        PinholeFocal_num_max_,
                        facs__pinhole__args__focal__idx_shared_,
                        nodes__PinholeExtraCalib__storage_current_,
                        PinholeExtraCalib_num_max_,
                        facs__pinhole__args__extra_calib__idx_shared_,
                        nodes__Point__storage_current_,
                        Point_num_max_,
                        facs__pinhole__args__point__idx_shared_,
                        facs__pinhole__args__pixel__data_,
                        pinhole_num_max_,

                        facs__pinhole__res_,
                        pinhole_num_,
                        solver__res_tot_,
                        facs__pinhole__args__pose__jac_,
                        pinhole_num_,
                        nodes__Pose__r_k_,
                        Pose_num_,
                        nodes__Pose__precond_diag_,
                        Pose_num_,
                        nodes__Pose__precond_tril_,
                        Pose_num_,
                        facs__pinhole__args__focal__jac_,
                        pinhole_num_,
                        nodes__PinholeFocal__r_k_,
                        PinholeFocal_num_,
                        nodes__PinholeFocal__precond_diag_,
                        PinholeFocal_num_,
                        nodes__PinholeFocal__precond_tril_,
                        PinholeFocal_num_,
                        facs__pinhole__args__extra_calib__jac_,
                        pinhole_num_,
                        nodes__PinholeExtraCalib__r_k_,
                        PinholeExtraCalib_num_,
                        nodes__PinholeExtraCalib__precond_diag_,
                        PinholeExtraCalib_num_,
                        nodes__PinholeExtraCalib__precond_tril_,
                        PinholeExtraCalib_num_,
                        facs__pinhole__args__point__jac_,
                        pinhole_num_,
                        nodes__Point__r_k_,
                        Point_num_,
                        nodes__Point__precond_diag_,
                        Point_num_,
                        nodes__Point__precond_tril_,
                        Point_num_,
                        pinhole_num_);

  pinhole_fixed_pose_res_jac_first(
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_pose__args__focal__idx_shared_,
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_pose__args__extra_calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose__args__point__idx_shared_,
      facs__pinhole_fixed_pose__args__pixel__data_,
      pinhole_fixed_pose_num_max_,
      facs__pinhole_fixed_pose__args__pose__data_,
      pinhole_fixed_pose_num_max_,

      facs__pinhole_fixed_pose__res_,
      pinhole_fixed_pose_num_,
      solver__res_tot_,
      facs__pinhole_fixed_pose__args__focal__jac_,
      pinhole_fixed_pose_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      facs__pinhole_fixed_pose__args__extra_calib__jac_,
      pinhole_fixed_pose_num_,
      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_tril_,
      PinholeExtraCalib_num_,
      facs__pinhole_fixed_pose__args__point__jac_,
      pinhole_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_num_);

  pinhole_fixed_focal_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_focal__args__pose__idx_shared_,
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_focal__args__extra_calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_focal__args__point__idx_shared_,
      facs__pinhole_fixed_focal__args__pixel__data_,
      pinhole_fixed_focal_num_max_,
      facs__pinhole_fixed_focal__args__focal__data_,
      pinhole_fixed_focal_num_max_,

      facs__pinhole_fixed_focal__res_,
      pinhole_fixed_focal_num_,
      solver__res_tot_,
      facs__pinhole_fixed_focal__args__pose__jac_,
      pinhole_fixed_focal_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_focal__args__extra_calib__jac_,
      pinhole_fixed_focal_num_,
      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_tril_,
      PinholeExtraCalib_num_,
      facs__pinhole_fixed_focal__args__point__jac_,
      pinhole_fixed_focal_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_focal_num_);

  pinhole_fixed_extra_calib_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_extra_calib__args__pose__idx_shared_,
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_extra_calib__args__focal__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_extra_calib_num_max_,

      facs__pinhole_fixed_extra_calib__res_,
      pinhole_fixed_extra_calib_num_,
      solver__res_tot_,
      facs__pinhole_fixed_extra_calib__args__pose__jac_,
      pinhole_fixed_extra_calib_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_extra_calib__args__focal__jac_,
      pinhole_fixed_extra_calib_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      facs__pinhole_fixed_extra_calib__args__point__jac_,
      pinhole_fixed_extra_calib_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_extra_calib_num_);

  pinhole_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_point__args__pose__idx_shared_,
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_point__args__focal__idx_shared_,
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_point__args__pixel__data_,
      pinhole_fixed_point_num_max_,
      facs__pinhole_fixed_point__args__point__data_,
      pinhole_fixed_point_num_max_,

      facs__pinhole_fixed_point__res_,
      pinhole_fixed_point_num_,
      solver__res_tot_,
      facs__pinhole_fixed_point__args__pose__jac_,
      pinhole_fixed_point_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_point__args__focal__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      facs__pinhole_fixed_point__args__extra_calib__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_tril_,
      PinholeExtraCalib_num_,
      pinhole_fixed_point_num_);

  pinhole_fixed_pose_fixed_focal_res_jac_first(
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal__args__extra_calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,
      facs__pinhole_fixed_pose_fixed_focal__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,
      facs__pinhole_fixed_pose_fixed_focal__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,

      facs__pinhole_fixed_pose_fixed_focal__res_,
      pinhole_fixed_pose_fixed_focal_num_,
      solver__res_tot_,
      facs__pinhole_fixed_pose_fixed_focal__args__extra_calib__jac_,
      pinhole_fixed_pose_fixed_focal_num_,
      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_tril_,
      PinholeExtraCalib_num_,
      facs__pinhole_fixed_pose_fixed_focal__args__point__jac_,
      pinhole_fixed_pose_fixed_focal_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_fixed_focal_num_);

  pinhole_fixed_pose_fixed_extra_calib_res_jac_first(
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__focal__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__pose__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,

      facs__pinhole_fixed_pose_fixed_extra_calib__res_,
      pinhole_fixed_pose_fixed_extra_calib_num_,
      solver__res_tot_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__focal__jac_,
      pinhole_fixed_pose_fixed_extra_calib_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__point__jac_,
      pinhole_fixed_pose_fixed_extra_calib_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_fixed_extra_calib_num_);

  pinhole_fixed_pose_fixed_point_res_jac_first(
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__focal__idx_shared_,
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_point_num_max_,

      facs__pinhole_fixed_pose_fixed_point__res_,
      pinhole_fixed_pose_fixed_point_num_,
      solver__res_tot_,
      facs__pinhole_fixed_pose_fixed_point__args__focal__jac_,
      pinhole_fixed_pose_fixed_point_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      facs__pinhole_fixed_pose_fixed_point__args__extra_calib__jac_,
      pinhole_fixed_pose_fixed_point_num_,
      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_tril_,
      PinholeExtraCalib_num_,
      pinhole_fixed_pose_fixed_point_num_);

  pinhole_fixed_focal_fixed_extra_calib_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__pose__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__focal__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,

      facs__pinhole_fixed_focal_fixed_extra_calib__res_,
      pinhole_fixed_focal_fixed_extra_calib_num_,
      solver__res_tot_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__pose__jac_,
      pinhole_fixed_focal_fixed_extra_calib_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__point__jac_,
      pinhole_fixed_focal_fixed_extra_calib_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_focal_fixed_extra_calib_num_);

  pinhole_fixed_focal_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_focal_fixed_point__args__pose__idx_shared_,
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_point__args__focal__data_,
      pinhole_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_point__args__point__data_,
      pinhole_fixed_focal_fixed_point_num_max_,

      facs__pinhole_fixed_focal_fixed_point__res_,
      pinhole_fixed_focal_fixed_point_num_,
      solver__res_tot_,
      facs__pinhole_fixed_focal_fixed_point__args__pose__jac_,
      pinhole_fixed_focal_fixed_point_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_focal_fixed_point__args__extra_calib__jac_,
      pinhole_fixed_focal_fixed_point_num_,
      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_tril_,
      PinholeExtraCalib_num_,
      pinhole_fixed_focal_fixed_point_num_);

  pinhole_fixed_extra_calib_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,

      facs__pinhole_fixed_extra_calib_fixed_point__res_,
      pinhole_fixed_extra_calib_fixed_point_num_,
      solver__res_tot_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__pose__jac_,
      pinhole_fixed_extra_calib_fixed_point_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__focal__jac_,
      pinhole_fixed_extra_calib_fixed_point_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      pinhole_fixed_extra_calib_fixed_point_num_);

  pinhole_fixed_pose_fixed_focal_fixed_extra_calib_res_jac_first(
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,

      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__res_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_,
      solver__res_tot_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_);

  pinhole_fixed_pose_fixed_focal_fixed_point_res_jac_first(
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,

      facs__pinhole_fixed_pose_fixed_focal_fixed_point__res_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_,
      solver__res_tot_,
      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_tril_,
      PinholeExtraCalib_num_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_);

  pinhole_fixed_pose_fixed_extra_calib_fixed_point_res_jac_first(
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,

      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__res_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_,
      solver__res_tot_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_);

  pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__focal__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,

      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__res_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_,
      solver__res_tot_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_);
  copy(marker__r_k_start_, marker__r_k_end_, marker__r_0_start_);
  copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
  return 0.5 * read_cumem(solver__res_tot_);
}
void GraphSolver::do_res_jac() {
  zero(solver__res_tot_, solver__res_tot_ + 1);
  zero(marker__r_0_start_, marker__precond_end_);

  simple_radial_res_jac(nodes__Pose__storage_current_,
                        Pose_num_max_,
                        facs__simple_radial__args__pose__idx_shared_,
                        nodes__SimpleRadialFocal__storage_current_,
                        SimpleRadialFocal_num_max_,
                        facs__simple_radial__args__focal__idx_shared_,
                        nodes__SimpleRadialExtraCalib__storage_current_,
                        SimpleRadialExtraCalib_num_max_,
                        facs__simple_radial__args__extra_calib__idx_shared_,
                        nodes__Point__storage_current_,
                        Point_num_max_,
                        facs__simple_radial__args__point__idx_shared_,
                        facs__simple_radial__args__pixel__data_,
                        simple_radial_num_max_,

                        facs__simple_radial__res_,
                        simple_radial_num_,

                        facs__simple_radial__args__pose__jac_,
                        simple_radial_num_,
                        nodes__Pose__r_k_,
                        Pose_num_,
                        nodes__Pose__precond_diag_,
                        Pose_num_,
                        nodes__Pose__precond_tril_,
                        Pose_num_,
                        facs__simple_radial__args__focal__jac_,
                        simple_radial_num_,
                        nodes__SimpleRadialFocal__r_k_,
                        SimpleRadialFocal_num_,
                        nodes__SimpleRadialFocal__precond_diag_,
                        SimpleRadialFocal_num_,
                        nodes__SimpleRadialFocal__precond_tril_,
                        SimpleRadialFocal_num_,
                        facs__simple_radial__args__extra_calib__jac_,
                        simple_radial_num_,
                        nodes__SimpleRadialExtraCalib__r_k_,
                        SimpleRadialExtraCalib_num_,
                        nodes__SimpleRadialExtraCalib__precond_diag_,
                        SimpleRadialExtraCalib_num_,
                        nodes__SimpleRadialExtraCalib__precond_tril_,
                        SimpleRadialExtraCalib_num_,
                        facs__simple_radial__args__point__jac_,
                        simple_radial_num_,
                        nodes__Point__r_k_,
                        Point_num_,
                        nodes__Point__precond_diag_,
                        Point_num_,
                        nodes__Point__precond_tril_,
                        Point_num_,
                        simple_radial_num_);

  simple_radial_fixed_pose_res_jac(
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_pose__args__focal__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_pose__args__extra_calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose__args__point__idx_shared_,
      facs__simple_radial_fixed_pose__args__pixel__data_,
      simple_radial_fixed_pose_num_max_,
      facs__simple_radial_fixed_pose__args__pose__data_,
      simple_radial_fixed_pose_num_max_,

      facs__simple_radial_fixed_pose__res_,
      simple_radial_fixed_pose_num_,

      facs__simple_radial_fixed_pose__args__focal__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_pose__args__extra_calib__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      facs__simple_radial_fixed_pose__args__point__jac_,
      simple_radial_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_num_);

  simple_radial_fixed_focal_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_focal__args__pose__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_focal__args__extra_calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_focal__args__point__idx_shared_,
      facs__simple_radial_fixed_focal__args__pixel__data_,
      simple_radial_fixed_focal_num_max_,
      facs__simple_radial_fixed_focal__args__focal__data_,
      simple_radial_fixed_focal_num_max_,

      facs__simple_radial_fixed_focal__res_,
      simple_radial_fixed_focal_num_,

      facs__simple_radial_fixed_focal__args__pose__jac_,
      simple_radial_fixed_focal_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_focal__args__extra_calib__jac_,
      simple_radial_fixed_focal_num_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      facs__simple_radial_fixed_focal__args__point__jac_,
      simple_radial_fixed_focal_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_focal_num_);

  simple_radial_fixed_extra_calib_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_extra_calib__args__pose__idx_shared_,
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_extra_calib__args__focal__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_extra_calib_num_max_,

      facs__simple_radial_fixed_extra_calib__res_,
      simple_radial_fixed_extra_calib_num_,

      facs__simple_radial_fixed_extra_calib__args__pose__jac_,
      simple_radial_fixed_extra_calib_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_extra_calib__args__focal__jac_,
      simple_radial_fixed_extra_calib_num_,
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_extra_calib__args__point__jac_,
      simple_radial_fixed_extra_calib_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_extra_calib_num_);

  simple_radial_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_point__args__focal__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_point__args__pixel__data_,
      simple_radial_fixed_point_num_max_,
      facs__simple_radial_fixed_point__args__point__data_,
      simple_radial_fixed_point_num_max_,

      facs__simple_radial_fixed_point__res_,
      simple_radial_fixed_point_num_,

      facs__simple_radial_fixed_point__args__pose__jac_,
      simple_radial_fixed_point_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_point__args__focal__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_point__args__extra_calib__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      simple_radial_fixed_point_num_);

  simple_radial_fixed_pose_fixed_focal_res_jac(
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal__args__extra_calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,

      facs__simple_radial_fixed_pose_fixed_focal__res_,
      simple_radial_fixed_pose_fixed_focal_num_,

      facs__simple_radial_fixed_pose_fixed_focal__args__extra_calib__jac_,
      simple_radial_fixed_pose_fixed_focal_num_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      facs__simple_radial_fixed_pose_fixed_focal__args__point__jac_,
      simple_radial_fixed_pose_fixed_focal_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_fixed_focal_num_);

  simple_radial_fixed_pose_fixed_extra_calib_res_jac(
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__focal__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__pose__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,

      facs__simple_radial_fixed_pose_fixed_extra_calib__res_,
      simple_radial_fixed_pose_fixed_extra_calib_num_,

      facs__simple_radial_fixed_pose_fixed_extra_calib__args__focal__jac_,
      simple_radial_fixed_pose_fixed_extra_calib_num_,
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__point__jac_,
      simple_radial_fixed_pose_fixed_extra_calib_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_fixed_extra_calib_num_);

  simple_radial_fixed_pose_fixed_point_res_jac(
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__focal__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_point__res_,
      simple_radial_fixed_pose_fixed_point_num_,

      facs__simple_radial_fixed_pose_fixed_point__args__focal__jac_,
      simple_radial_fixed_pose_fixed_point_num_,
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_pose_fixed_point__args__extra_calib__jac_,
      simple_radial_fixed_pose_fixed_point_num_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      simple_radial_fixed_pose_fixed_point_num_);

  simple_radial_fixed_focal_fixed_extra_calib_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__pose__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__focal__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,

      facs__simple_radial_fixed_focal_fixed_extra_calib__res_,
      simple_radial_fixed_focal_fixed_extra_calib_num_,

      facs__simple_radial_fixed_focal_fixed_extra_calib__args__pose__jac_,
      simple_radial_fixed_focal_fixed_extra_calib_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__point__jac_,
      simple_radial_fixed_focal_fixed_extra_calib_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_focal_fixed_extra_calib_num_);

  simple_radial_fixed_focal_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_focal_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_focal_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_point__args__focal__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_point__args__point__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,

      facs__simple_radial_fixed_focal_fixed_point__res_,
      simple_radial_fixed_focal_fixed_point_num_,

      facs__simple_radial_fixed_focal_fixed_point__args__pose__jac_,
      simple_radial_fixed_focal_fixed_point_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_focal_fixed_point__args__extra_calib__jac_,
      simple_radial_fixed_focal_fixed_point_num_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      simple_radial_fixed_focal_fixed_point_num_);

  simple_radial_fixed_extra_calib_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,

      facs__simple_radial_fixed_extra_calib_fixed_point__res_,
      simple_radial_fixed_extra_calib_fixed_point_num_,

      facs__simple_radial_fixed_extra_calib_fixed_point__args__pose__jac_,
      simple_radial_fixed_extra_calib_fixed_point_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__focal__jac_,
      simple_radial_fixed_extra_calib_fixed_point_num_,
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      simple_radial_fixed_extra_calib_fixed_point_num_);

  simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_res_jac(
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,

      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__res_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,

      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_);

  simple_radial_fixed_pose_fixed_focal_fixed_point_res_jac(
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__res_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_,

      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_tril_,
      SimpleRadialExtraCalib_num_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_);

  simple_radial_fixed_pose_fixed_extra_calib_fixed_point_res_jac(
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__res_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,

      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_tril_,
      SimpleRadialFocal_num_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_);

  simple_radial_fixed_focal_fixed_extra_calib_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__focal__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,

      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__res_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,

      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_);

  pinhole_res_jac(nodes__Pose__storage_current_,
                  Pose_num_max_,
                  facs__pinhole__args__pose__idx_shared_,
                  nodes__PinholeFocal__storage_current_,
                  PinholeFocal_num_max_,
                  facs__pinhole__args__focal__idx_shared_,
                  nodes__PinholeExtraCalib__storage_current_,
                  PinholeExtraCalib_num_max_,
                  facs__pinhole__args__extra_calib__idx_shared_,
                  nodes__Point__storage_current_,
                  Point_num_max_,
                  facs__pinhole__args__point__idx_shared_,
                  facs__pinhole__args__pixel__data_,
                  pinhole_num_max_,

                  facs__pinhole__res_,
                  pinhole_num_,

                  facs__pinhole__args__pose__jac_,
                  pinhole_num_,
                  nodes__Pose__r_k_,
                  Pose_num_,
                  nodes__Pose__precond_diag_,
                  Pose_num_,
                  nodes__Pose__precond_tril_,
                  Pose_num_,
                  facs__pinhole__args__focal__jac_,
                  pinhole_num_,
                  nodes__PinholeFocal__r_k_,
                  PinholeFocal_num_,
                  nodes__PinholeFocal__precond_diag_,
                  PinholeFocal_num_,
                  nodes__PinholeFocal__precond_tril_,
                  PinholeFocal_num_,
                  facs__pinhole__args__extra_calib__jac_,
                  pinhole_num_,
                  nodes__PinholeExtraCalib__r_k_,
                  PinholeExtraCalib_num_,
                  nodes__PinholeExtraCalib__precond_diag_,
                  PinholeExtraCalib_num_,
                  nodes__PinholeExtraCalib__precond_tril_,
                  PinholeExtraCalib_num_,
                  facs__pinhole__args__point__jac_,
                  pinhole_num_,
                  nodes__Point__r_k_,
                  Point_num_,
                  nodes__Point__precond_diag_,
                  Point_num_,
                  nodes__Point__precond_tril_,
                  Point_num_,
                  pinhole_num_);

  pinhole_fixed_pose_res_jac(
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_pose__args__focal__idx_shared_,
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_pose__args__extra_calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose__args__point__idx_shared_,
      facs__pinhole_fixed_pose__args__pixel__data_,
      pinhole_fixed_pose_num_max_,
      facs__pinhole_fixed_pose__args__pose__data_,
      pinhole_fixed_pose_num_max_,

      facs__pinhole_fixed_pose__res_,
      pinhole_fixed_pose_num_,

      facs__pinhole_fixed_pose__args__focal__jac_,
      pinhole_fixed_pose_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      facs__pinhole_fixed_pose__args__extra_calib__jac_,
      pinhole_fixed_pose_num_,
      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_tril_,
      PinholeExtraCalib_num_,
      facs__pinhole_fixed_pose__args__point__jac_,
      pinhole_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_num_);

  pinhole_fixed_focal_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_focal__args__pose__idx_shared_,
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_focal__args__extra_calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_focal__args__point__idx_shared_,
      facs__pinhole_fixed_focal__args__pixel__data_,
      pinhole_fixed_focal_num_max_,
      facs__pinhole_fixed_focal__args__focal__data_,
      pinhole_fixed_focal_num_max_,

      facs__pinhole_fixed_focal__res_,
      pinhole_fixed_focal_num_,

      facs__pinhole_fixed_focal__args__pose__jac_,
      pinhole_fixed_focal_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_focal__args__extra_calib__jac_,
      pinhole_fixed_focal_num_,
      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_tril_,
      PinholeExtraCalib_num_,
      facs__pinhole_fixed_focal__args__point__jac_,
      pinhole_fixed_focal_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_focal_num_);

  pinhole_fixed_extra_calib_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_extra_calib__args__pose__idx_shared_,
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_extra_calib__args__focal__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_extra_calib_num_max_,

      facs__pinhole_fixed_extra_calib__res_,
      pinhole_fixed_extra_calib_num_,

      facs__pinhole_fixed_extra_calib__args__pose__jac_,
      pinhole_fixed_extra_calib_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_extra_calib__args__focal__jac_,
      pinhole_fixed_extra_calib_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      facs__pinhole_fixed_extra_calib__args__point__jac_,
      pinhole_fixed_extra_calib_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_extra_calib_num_);

  pinhole_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_point__args__pose__idx_shared_,
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_point__args__focal__idx_shared_,
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_point__args__pixel__data_,
      pinhole_fixed_point_num_max_,
      facs__pinhole_fixed_point__args__point__data_,
      pinhole_fixed_point_num_max_,

      facs__pinhole_fixed_point__res_,
      pinhole_fixed_point_num_,

      facs__pinhole_fixed_point__args__pose__jac_,
      pinhole_fixed_point_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_point__args__focal__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      facs__pinhole_fixed_point__args__extra_calib__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_tril_,
      PinholeExtraCalib_num_,
      pinhole_fixed_point_num_);

  pinhole_fixed_pose_fixed_focal_res_jac(
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal__args__extra_calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,
      facs__pinhole_fixed_pose_fixed_focal__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,
      facs__pinhole_fixed_pose_fixed_focal__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,

      facs__pinhole_fixed_pose_fixed_focal__res_,
      pinhole_fixed_pose_fixed_focal_num_,

      facs__pinhole_fixed_pose_fixed_focal__args__extra_calib__jac_,
      pinhole_fixed_pose_fixed_focal_num_,
      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_tril_,
      PinholeExtraCalib_num_,
      facs__pinhole_fixed_pose_fixed_focal__args__point__jac_,
      pinhole_fixed_pose_fixed_focal_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_fixed_focal_num_);

  pinhole_fixed_pose_fixed_extra_calib_res_jac(
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__focal__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__pose__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,

      facs__pinhole_fixed_pose_fixed_extra_calib__res_,
      pinhole_fixed_pose_fixed_extra_calib_num_,

      facs__pinhole_fixed_pose_fixed_extra_calib__args__focal__jac_,
      pinhole_fixed_pose_fixed_extra_calib_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__point__jac_,
      pinhole_fixed_pose_fixed_extra_calib_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_fixed_extra_calib_num_);

  pinhole_fixed_pose_fixed_point_res_jac(
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__focal__idx_shared_,
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_point_num_max_,

      facs__pinhole_fixed_pose_fixed_point__res_,
      pinhole_fixed_pose_fixed_point_num_,

      facs__pinhole_fixed_pose_fixed_point__args__focal__jac_,
      pinhole_fixed_pose_fixed_point_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      facs__pinhole_fixed_pose_fixed_point__args__extra_calib__jac_,
      pinhole_fixed_pose_fixed_point_num_,
      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_tril_,
      PinholeExtraCalib_num_,
      pinhole_fixed_pose_fixed_point_num_);

  pinhole_fixed_focal_fixed_extra_calib_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__pose__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__focal__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,

      facs__pinhole_fixed_focal_fixed_extra_calib__res_,
      pinhole_fixed_focal_fixed_extra_calib_num_,

      facs__pinhole_fixed_focal_fixed_extra_calib__args__pose__jac_,
      pinhole_fixed_focal_fixed_extra_calib_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__point__jac_,
      pinhole_fixed_focal_fixed_extra_calib_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_focal_fixed_extra_calib_num_);

  pinhole_fixed_focal_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_focal_fixed_point__args__pose__idx_shared_,
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_point__args__focal__data_,
      pinhole_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_point__args__point__data_,
      pinhole_fixed_focal_fixed_point_num_max_,

      facs__pinhole_fixed_focal_fixed_point__res_,
      pinhole_fixed_focal_fixed_point_num_,

      facs__pinhole_fixed_focal_fixed_point__args__pose__jac_,
      pinhole_fixed_focal_fixed_point_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_focal_fixed_point__args__extra_calib__jac_,
      pinhole_fixed_focal_fixed_point_num_,
      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_tril_,
      PinholeExtraCalib_num_,
      pinhole_fixed_focal_fixed_point_num_);

  pinhole_fixed_extra_calib_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,

      facs__pinhole_fixed_extra_calib_fixed_point__res_,
      pinhole_fixed_extra_calib_fixed_point_num_,

      facs__pinhole_fixed_extra_calib_fixed_point__args__pose__jac_,
      pinhole_fixed_extra_calib_fixed_point_num_,
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__focal__jac_,
      pinhole_fixed_extra_calib_fixed_point_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      pinhole_fixed_extra_calib_fixed_point_num_);

  pinhole_fixed_pose_fixed_focal_fixed_extra_calib_res_jac(
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,

      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__res_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_,

      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_);

  pinhole_fixed_pose_fixed_focal_fixed_point_res_jac(
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,

      facs__pinhole_fixed_pose_fixed_focal_fixed_point__res_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_,

      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_tril_,
      PinholeExtraCalib_num_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_);

  pinhole_fixed_pose_fixed_extra_calib_fixed_point_res_jac(
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,

      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__res_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_,

      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_);

  pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__focal__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,

      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__res_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_,

      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_);
  copy(marker__r_k_start_, marker__r_k_end_, marker__r_0_start_);
  copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
}

void GraphSolver::do_normalize() {
  float* r_k;
  float* z;
  z = pcg_iter_ == 0 ? nodes__PinholeExtraCalib__p_
                     : nodes__PinholeExtraCalib__z_;
  PinholeExtraCalib_normalize(nodes__PinholeExtraCalib__precond_diag_,
                              PinholeExtraCalib_num_,
                              nodes__PinholeExtraCalib__precond_tril_,
                              PinholeExtraCalib_num_,
                              nodes__PinholeExtraCalib__r_k_,
                              PinholeExtraCalib_num_,
                              solver__current_diag_,
                              z,
                              PinholeExtraCalib_num_,
                              PinholeExtraCalib_num_);
  z = pcg_iter_ == 0 ? nodes__PinholeFocal__p_ : nodes__PinholeFocal__z_;
  PinholeFocal_normalize(nodes__PinholeFocal__precond_diag_,
                         PinholeFocal_num_,
                         nodes__PinholeFocal__precond_tril_,
                         PinholeFocal_num_,
                         nodes__PinholeFocal__r_k_,
                         PinholeFocal_num_,
                         solver__current_diag_,
                         z,
                         PinholeFocal_num_,
                         PinholeFocal_num_);
  z = pcg_iter_ == 0 ? nodes__Point__p_ : nodes__Point__z_;
  Point_normalize(nodes__Point__precond_diag_,
                  Point_num_,
                  nodes__Point__precond_tril_,
                  Point_num_,
                  nodes__Point__r_k_,
                  Point_num_,
                  solver__current_diag_,
                  z,
                  Point_num_,
                  Point_num_);
  z = pcg_iter_ == 0 ? nodes__Pose__p_ : nodes__Pose__z_;
  Pose_normalize(nodes__Pose__precond_diag_,
                 Pose_num_,
                 nodes__Pose__precond_tril_,
                 Pose_num_,
                 nodes__Pose__r_k_,
                 Pose_num_,
                 solver__current_diag_,
                 z,
                 Pose_num_,
                 Pose_num_);
  z = pcg_iter_ == 0 ? nodes__SimpleRadialExtraCalib__p_
                     : nodes__SimpleRadialExtraCalib__z_;
  SimpleRadialExtraCalib_normalize(nodes__SimpleRadialExtraCalib__precond_diag_,
                                   SimpleRadialExtraCalib_num_,
                                   nodes__SimpleRadialExtraCalib__precond_tril_,
                                   SimpleRadialExtraCalib_num_,
                                   nodes__SimpleRadialExtraCalib__r_k_,
                                   SimpleRadialExtraCalib_num_,
                                   solver__current_diag_,
                                   z,
                                   SimpleRadialExtraCalib_num_,
                                   SimpleRadialExtraCalib_num_);
  z = pcg_iter_ == 0 ? nodes__SimpleRadialFocal__p_
                     : nodes__SimpleRadialFocal__z_;
  SimpleRadialFocal_normalize(nodes__SimpleRadialFocal__precond_diag_,
                              SimpleRadialFocal_num_,
                              nodes__SimpleRadialFocal__precond_tril_,
                              SimpleRadialFocal_num_,
                              nodes__SimpleRadialFocal__r_k_,
                              SimpleRadialFocal_num_,
                              solver__current_diag_,
                              z,
                              SimpleRadialFocal_num_,
                              SimpleRadialFocal_num_);
}

void GraphSolver::do_update_Mp() {
  PinholeExtraCalib_update_Mp(nodes__PinholeExtraCalib__r_k_,
                              PinholeExtraCalib_num_,
                              nodes__PinholeExtraCalib__Mp_,
                              PinholeExtraCalib_num_,
                              solver__beta_,
                              nodes__PinholeExtraCalib__Mp_,
                              PinholeExtraCalib_num_,
                              nodes__PinholeExtraCalib__w_,
                              PinholeExtraCalib_num_,
                              PinholeExtraCalib_num_);
  PinholeFocal_update_Mp(nodes__PinholeFocal__r_k_,
                         PinholeFocal_num_,
                         nodes__PinholeFocal__Mp_,
                         PinholeFocal_num_,
                         solver__beta_,
                         nodes__PinholeFocal__Mp_,
                         PinholeFocal_num_,
                         nodes__PinholeFocal__w_,
                         PinholeFocal_num_,
                         PinholeFocal_num_);
  Point_update_Mp(nodes__Point__r_k_,
                  Point_num_,
                  nodes__Point__Mp_,
                  Point_num_,
                  solver__beta_,
                  nodes__Point__Mp_,
                  Point_num_,
                  nodes__Point__w_,
                  Point_num_,
                  Point_num_);
  Pose_update_Mp(nodes__Pose__r_k_,
                 Pose_num_,
                 nodes__Pose__Mp_,
                 Pose_num_,
                 solver__beta_,
                 nodes__Pose__Mp_,
                 Pose_num_,
                 nodes__Pose__w_,
                 Pose_num_,
                 Pose_num_);
  SimpleRadialExtraCalib_update_Mp(nodes__SimpleRadialExtraCalib__r_k_,
                                   SimpleRadialExtraCalib_num_,
                                   nodes__SimpleRadialExtraCalib__Mp_,
                                   SimpleRadialExtraCalib_num_,
                                   solver__beta_,
                                   nodes__SimpleRadialExtraCalib__Mp_,
                                   SimpleRadialExtraCalib_num_,
                                   nodes__SimpleRadialExtraCalib__w_,
                                   SimpleRadialExtraCalib_num_,
                                   SimpleRadialExtraCalib_num_);
  SimpleRadialFocal_update_Mp(nodes__SimpleRadialFocal__r_k_,
                              SimpleRadialFocal_num_,
                              nodes__SimpleRadialFocal__Mp_,
                              SimpleRadialFocal_num_,
                              solver__beta_,
                              nodes__SimpleRadialFocal__Mp_,
                              SimpleRadialFocal_num_,
                              nodes__SimpleRadialFocal__w_,
                              SimpleRadialFocal_num_,
                              SimpleRadialFocal_num_);
}

void GraphSolver::do_jtjp_direct() {
  simple_radial_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__simple_radial__args__pose__idx_shared_,
      facs__simple_radial__args__pose__jac_,
      simple_radial_num_,
      nodes__SimpleRadialFocal__p_,
      SimpleRadialFocal_num_,
      facs__simple_radial__args__focal__idx_shared_,
      facs__simple_radial__args__focal__jac_,
      simple_radial_num_,
      nodes__SimpleRadialExtraCalib__p_,
      SimpleRadialExtraCalib_num_,
      facs__simple_radial__args__extra_calib__idx_shared_,
      facs__simple_radial__args__extra_calib__jac_,
      simple_radial_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial__args__point__idx_shared_,
      facs__simple_radial__args__point__jac_,
      simple_radial_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__SimpleRadialFocal__w_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialExtraCalib__w_,
      SimpleRadialExtraCalib_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_num_);
  simple_radial_fixed_pose_jtjnjtr_direct(
      nodes__SimpleRadialFocal__p_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_pose__args__focal__idx_shared_,
      facs__simple_radial_fixed_pose__args__focal__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialExtraCalib__p_,
      SimpleRadialExtraCalib_num_,
      facs__simple_radial_fixed_pose__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_pose__args__extra_calib__jac_,
      simple_radial_fixed_pose_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_fixed_pose__args__point__idx_shared_,
      facs__simple_radial_fixed_pose__args__point__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialFocal__w_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialExtraCalib__w_,
      SimpleRadialExtraCalib_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_pose_num_);
  simple_radial_fixed_focal_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__simple_radial_fixed_focal__args__pose__idx_shared_,
      facs__simple_radial_fixed_focal__args__pose__jac_,
      simple_radial_fixed_focal_num_,
      nodes__SimpleRadialExtraCalib__p_,
      SimpleRadialExtraCalib_num_,
      facs__simple_radial_fixed_focal__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_focal__args__extra_calib__jac_,
      simple_radial_fixed_focal_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_fixed_focal__args__point__idx_shared_,
      facs__simple_radial_fixed_focal__args__point__jac_,
      simple_radial_fixed_focal_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__SimpleRadialExtraCalib__w_,
      SimpleRadialExtraCalib_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_focal_num_);
  simple_radial_fixed_extra_calib_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__simple_radial_fixed_extra_calib__args__pose__idx_shared_,
      facs__simple_radial_fixed_extra_calib__args__pose__jac_,
      simple_radial_fixed_extra_calib_num_,
      nodes__SimpleRadialFocal__p_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_extra_calib__args__focal__idx_shared_,
      facs__simple_radial_fixed_extra_calib__args__focal__jac_,
      simple_radial_fixed_extra_calib_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_extra_calib__args__point__jac_,
      simple_radial_fixed_extra_calib_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__SimpleRadialFocal__w_,
      SimpleRadialFocal_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_extra_calib_num_);
  simple_radial_fixed_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__simple_radial_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_point__args__pose__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialFocal__p_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_point__args__focal__idx_shared_,
      facs__simple_radial_fixed_point__args__focal__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialExtraCalib__p_,
      SimpleRadialExtraCalib_num_,
      facs__simple_radial_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_point__args__extra_calib__jac_,
      simple_radial_fixed_point_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__SimpleRadialFocal__w_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialExtraCalib__w_,
      SimpleRadialExtraCalib_num_,
      simple_radial_fixed_point_num_);
  simple_radial_fixed_pose_fixed_focal_jtjnjtr_direct(
      nodes__SimpleRadialExtraCalib__p_,
      SimpleRadialExtraCalib_num_,
      facs__simple_radial_fixed_pose_fixed_focal__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal__args__extra_calib__jac_,
      simple_radial_fixed_pose_fixed_focal_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_fixed_pose_fixed_focal__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal__args__point__jac_,
      simple_radial_fixed_pose_fixed_focal_num_,
      nodes__SimpleRadialExtraCalib__w_,
      SimpleRadialExtraCalib_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_pose_fixed_focal_num_);
  simple_radial_fixed_pose_fixed_extra_calib_jtjnjtr_direct(
      nodes__SimpleRadialFocal__p_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__focal__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__focal__jac_,
      simple_radial_fixed_pose_fixed_extra_calib_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__point__jac_,
      simple_radial_fixed_pose_fixed_extra_calib_num_,
      nodes__SimpleRadialFocal__w_,
      SimpleRadialFocal_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_pose_fixed_extra_calib_num_);
  simple_radial_fixed_pose_fixed_point_jtjnjtr_direct(
      nodes__SimpleRadialFocal__p_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_pose_fixed_point__args__focal__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_point__args__focal__jac_,
      simple_radial_fixed_pose_fixed_point_num_,
      nodes__SimpleRadialExtraCalib__p_,
      SimpleRadialExtraCalib_num_,
      facs__simple_radial_fixed_pose_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_point__args__extra_calib__jac_,
      simple_radial_fixed_pose_fixed_point_num_,
      nodes__SimpleRadialFocal__w_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialExtraCalib__w_,
      SimpleRadialExtraCalib_num_,
      simple_radial_fixed_pose_fixed_point_num_);
  simple_radial_fixed_focal_fixed_extra_calib_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__pose__idx_shared_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__pose__jac_,
      simple_radial_fixed_focal_fixed_extra_calib_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__point__jac_,
      simple_radial_fixed_focal_fixed_extra_calib_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_focal_fixed_extra_calib_num_);
  simple_radial_fixed_focal_fixed_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__simple_radial_fixed_focal_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_focal_fixed_point__args__pose__jac_,
      simple_radial_fixed_focal_fixed_point_num_,
      nodes__SimpleRadialExtraCalib__p_,
      SimpleRadialExtraCalib_num_,
      facs__simple_radial_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_focal_fixed_point__args__extra_calib__jac_,
      simple_radial_fixed_focal_fixed_point_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__SimpleRadialExtraCalib__w_,
      SimpleRadialExtraCalib_num_,
      simple_radial_fixed_focal_fixed_point_num_);
  simple_radial_fixed_extra_calib_fixed_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__pose__jac_,
      simple_radial_fixed_extra_calib_fixed_point_num_,
      nodes__SimpleRadialFocal__p_,
      SimpleRadialFocal_num_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__focal__jac_,
      simple_radial_fixed_extra_calib_fixed_point_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__SimpleRadialFocal__w_,
      SimpleRadialFocal_num_,
      simple_radial_fixed_extra_calib_fixed_point_num_);
  pinhole_jtjnjtr_direct(nodes__Pose__p_,
                         Pose_num_,
                         facs__pinhole__args__pose__idx_shared_,
                         facs__pinhole__args__pose__jac_,
                         pinhole_num_,
                         nodes__PinholeFocal__p_,
                         PinholeFocal_num_,
                         facs__pinhole__args__focal__idx_shared_,
                         facs__pinhole__args__focal__jac_,
                         pinhole_num_,
                         nodes__PinholeExtraCalib__p_,
                         PinholeExtraCalib_num_,
                         facs__pinhole__args__extra_calib__idx_shared_,
                         facs__pinhole__args__extra_calib__jac_,
                         pinhole_num_,
                         nodes__Point__p_,
                         Point_num_,
                         facs__pinhole__args__point__idx_shared_,
                         facs__pinhole__args__point__jac_,
                         pinhole_num_,
                         nodes__Pose__w_,
                         Pose_num_,
                         nodes__PinholeFocal__w_,
                         PinholeFocal_num_,
                         nodes__PinholeExtraCalib__w_,
                         PinholeExtraCalib_num_,
                         nodes__Point__w_,
                         Point_num_,
                         pinhole_num_);
  pinhole_fixed_pose_jtjnjtr_direct(
      nodes__PinholeFocal__p_,
      PinholeFocal_num_,
      facs__pinhole_fixed_pose__args__focal__idx_shared_,
      facs__pinhole_fixed_pose__args__focal__jac_,
      pinhole_fixed_pose_num_,
      nodes__PinholeExtraCalib__p_,
      PinholeExtraCalib_num_,
      facs__pinhole_fixed_pose__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_pose__args__extra_calib__jac_,
      pinhole_fixed_pose_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_fixed_pose__args__point__idx_shared_,
      facs__pinhole_fixed_pose__args__point__jac_,
      pinhole_fixed_pose_num_,
      nodes__PinholeFocal__w_,
      PinholeFocal_num_,
      nodes__PinholeExtraCalib__w_,
      PinholeExtraCalib_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_pose_num_);
  pinhole_fixed_focal_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__pinhole_fixed_focal__args__pose__idx_shared_,
      facs__pinhole_fixed_focal__args__pose__jac_,
      pinhole_fixed_focal_num_,
      nodes__PinholeExtraCalib__p_,
      PinholeExtraCalib_num_,
      facs__pinhole_fixed_focal__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_focal__args__extra_calib__jac_,
      pinhole_fixed_focal_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_fixed_focal__args__point__idx_shared_,
      facs__pinhole_fixed_focal__args__point__jac_,
      pinhole_fixed_focal_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__PinholeExtraCalib__w_,
      PinholeExtraCalib_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_focal_num_);
  pinhole_fixed_extra_calib_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__pinhole_fixed_extra_calib__args__pose__idx_shared_,
      facs__pinhole_fixed_extra_calib__args__pose__jac_,
      pinhole_fixed_extra_calib_num_,
      nodes__PinholeFocal__p_,
      PinholeFocal_num_,
      facs__pinhole_fixed_extra_calib__args__focal__idx_shared_,
      facs__pinhole_fixed_extra_calib__args__focal__jac_,
      pinhole_fixed_extra_calib_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_extra_calib__args__point__jac_,
      pinhole_fixed_extra_calib_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__PinholeFocal__w_,
      PinholeFocal_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_extra_calib_num_);
  pinhole_fixed_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__pinhole_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_point__args__pose__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholeFocal__p_,
      PinholeFocal_num_,
      facs__pinhole_fixed_point__args__focal__idx_shared_,
      facs__pinhole_fixed_point__args__focal__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholeExtraCalib__p_,
      PinholeExtraCalib_num_,
      facs__pinhole_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_point__args__extra_calib__jac_,
      pinhole_fixed_point_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__PinholeFocal__w_,
      PinholeFocal_num_,
      nodes__PinholeExtraCalib__w_,
      PinholeExtraCalib_num_,
      pinhole_fixed_point_num_);
  pinhole_fixed_pose_fixed_focal_jtjnjtr_direct(
      nodes__PinholeExtraCalib__p_,
      PinholeExtraCalib_num_,
      facs__pinhole_fixed_pose_fixed_focal__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal__args__extra_calib__jac_,
      pinhole_fixed_pose_fixed_focal_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_fixed_pose_fixed_focal__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal__args__point__jac_,
      pinhole_fixed_pose_fixed_focal_num_,
      nodes__PinholeExtraCalib__w_,
      PinholeExtraCalib_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_pose_fixed_focal_num_);
  pinhole_fixed_pose_fixed_extra_calib_jtjnjtr_direct(
      nodes__PinholeFocal__p_,
      PinholeFocal_num_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__focal__idx_shared_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__focal__jac_,
      pinhole_fixed_pose_fixed_extra_calib_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__point__jac_,
      pinhole_fixed_pose_fixed_extra_calib_num_,
      nodes__PinholeFocal__w_,
      PinholeFocal_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_pose_fixed_extra_calib_num_);
  pinhole_fixed_pose_fixed_point_jtjnjtr_direct(
      nodes__PinholeFocal__p_,
      PinholeFocal_num_,
      facs__pinhole_fixed_pose_fixed_point__args__focal__idx_shared_,
      facs__pinhole_fixed_pose_fixed_point__args__focal__jac_,
      pinhole_fixed_pose_fixed_point_num_,
      nodes__PinholeExtraCalib__p_,
      PinholeExtraCalib_num_,
      facs__pinhole_fixed_pose_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_pose_fixed_point__args__extra_calib__jac_,
      pinhole_fixed_pose_fixed_point_num_,
      nodes__PinholeFocal__w_,
      PinholeFocal_num_,
      nodes__PinholeExtraCalib__w_,
      PinholeExtraCalib_num_,
      pinhole_fixed_pose_fixed_point_num_);
  pinhole_fixed_focal_fixed_extra_calib_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__pose__idx_shared_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__pose__jac_,
      pinhole_fixed_focal_fixed_extra_calib_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__point__jac_,
      pinhole_fixed_focal_fixed_extra_calib_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_focal_fixed_extra_calib_num_);
  pinhole_fixed_focal_fixed_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__pinhole_fixed_focal_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_focal_fixed_point__args__pose__jac_,
      pinhole_fixed_focal_fixed_point_num_,
      nodes__PinholeExtraCalib__p_,
      PinholeExtraCalib_num_,
      facs__pinhole_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_focal_fixed_point__args__extra_calib__jac_,
      pinhole_fixed_focal_fixed_point_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__PinholeExtraCalib__w_,
      PinholeExtraCalib_num_,
      pinhole_fixed_focal_fixed_point_num_);
  pinhole_fixed_extra_calib_fixed_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__pose__jac_,
      pinhole_fixed_extra_calib_fixed_point_num_,
      nodes__PinholeFocal__p_,
      PinholeFocal_num_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__focal__jac_,
      pinhole_fixed_extra_calib_fixed_point_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__PinholeFocal__w_,
      PinholeFocal_num_,
      pinhole_fixed_extra_calib_fixed_point_num_);
}

void GraphSolver::do_alpha_first() {
  zero(solver__alpha_numerator_, solver__alpha_denumerator_ + 1);
  float* p_kp1;
  float* r_k;
  PinholeExtraCalib_alpha_numerator_denominator(nodes__PinholeExtraCalib__p_,
                                                PinholeExtraCalib_num_,
                                                nodes__PinholeExtraCalib__r_k_,
                                                PinholeExtraCalib_num_,
                                                nodes__PinholeExtraCalib__w_,
                                                PinholeExtraCalib_num_,
                                                solver__alpha_numerator_,
                                                solver__alpha_denumerator_,
                                                PinholeExtraCalib_num_);
  PinholeFocal_alpha_numerator_denominator(nodes__PinholeFocal__p_,
                                           PinholeFocal_num_,
                                           nodes__PinholeFocal__r_k_,
                                           PinholeFocal_num_,
                                           nodes__PinholeFocal__w_,
                                           PinholeFocal_num_,
                                           solver__alpha_numerator_,
                                           solver__alpha_denumerator_,
                                           PinholeFocal_num_);
  Point_alpha_numerator_denominator(nodes__Point__p_,
                                    Point_num_,
                                    nodes__Point__r_k_,
                                    Point_num_,
                                    nodes__Point__w_,
                                    Point_num_,
                                    solver__alpha_numerator_,
                                    solver__alpha_denumerator_,
                                    Point_num_);
  Pose_alpha_numerator_denominator(nodes__Pose__p_,
                                   Pose_num_,
                                   nodes__Pose__r_k_,
                                   Pose_num_,
                                   nodes__Pose__w_,
                                   Pose_num_,
                                   solver__alpha_numerator_,
                                   solver__alpha_denumerator_,
                                   Pose_num_);
  SimpleRadialExtraCalib_alpha_numerator_denominator(
      nodes__SimpleRadialExtraCalib__p_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__w_,
      SimpleRadialExtraCalib_num_,
      solver__alpha_numerator_,
      solver__alpha_denumerator_,
      SimpleRadialExtraCalib_num_);
  SimpleRadialFocal_alpha_numerator_denominator(nodes__SimpleRadialFocal__p_,
                                                SimpleRadialFocal_num_,
                                                nodes__SimpleRadialFocal__r_k_,
                                                SimpleRadialFocal_num_,
                                                nodes__SimpleRadialFocal__w_,
                                                SimpleRadialFocal_num_,
                                                solver__alpha_numerator_,
                                                solver__alpha_denumerator_,
                                                SimpleRadialFocal_num_);

  alpha_from_num_denum(solver__alpha_numerator_,
                       solver__alpha_denumerator_,
                       solver__alpha_,
                       solver__neg_alpha_);
}

void GraphSolver::do_alpha() {
  zero(solver__alpha_denumerator_, solver__alpha_denumerator_ + 1);
  PinholeExtraCalib_alpha_denumerator_or_beta_nummerator(
      nodes__PinholeExtraCalib__p_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__w_,
      PinholeExtraCalib_num_,
      solver__alpha_denumerator_,
      PinholeExtraCalib_num_);
  PinholeFocal_alpha_denumerator_or_beta_nummerator(nodes__PinholeFocal__p_,
                                                    PinholeFocal_num_,
                                                    nodes__PinholeFocal__w_,
                                                    PinholeFocal_num_,
                                                    solver__alpha_denumerator_,
                                                    PinholeFocal_num_);
  Point_alpha_denumerator_or_beta_nummerator(nodes__Point__p_,
                                             Point_num_,
                                             nodes__Point__w_,
                                             Point_num_,
                                             solver__alpha_denumerator_,
                                             Point_num_);
  Pose_alpha_denumerator_or_beta_nummerator(nodes__Pose__p_,
                                            Pose_num_,
                                            nodes__Pose__w_,
                                            Pose_num_,
                                            solver__alpha_denumerator_,
                                            Pose_num_);
  SimpleRadialExtraCalib_alpha_denumerator_or_beta_nummerator(
      nodes__SimpleRadialExtraCalib__p_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__w_,
      SimpleRadialExtraCalib_num_,
      solver__alpha_denumerator_,
      SimpleRadialExtraCalib_num_);
  SimpleRadialFocal_alpha_denumerator_or_beta_nummerator(
      nodes__SimpleRadialFocal__p_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__w_,
      SimpleRadialFocal_num_,
      solver__alpha_denumerator_,
      SimpleRadialFocal_num_);

  alpha_from_num_denum(solver__beta_numerator_,
                       solver__alpha_denumerator_,
                       solver__alpha_,
                       solver__neg_alpha_);
}

void GraphSolver::do_update_step_first() {
  PinholeExtraCalib_update_step_first(nodes__PinholeExtraCalib__p_,
                                      PinholeExtraCalib_num_,
                                      solver__alpha_,
                                      nodes__PinholeExtraCalib__step_,
                                      PinholeExtraCalib_num_,
                                      PinholeExtraCalib_num_);
  PinholeFocal_update_step_first(nodes__PinholeFocal__p_,
                                 PinholeFocal_num_,
                                 solver__alpha_,
                                 nodes__PinholeFocal__step_,
                                 PinholeFocal_num_,
                                 PinholeFocal_num_);
  Point_update_step_first(nodes__Point__p_,
                          Point_num_,
                          solver__alpha_,
                          nodes__Point__step_,
                          Point_num_,
                          Point_num_);
  Pose_update_step_first(nodes__Pose__p_,
                         Pose_num_,
                         solver__alpha_,
                         nodes__Pose__step_,
                         Pose_num_,
                         Pose_num_);
  SimpleRadialExtraCalib_update_step_first(nodes__SimpleRadialExtraCalib__p_,
                                           SimpleRadialExtraCalib_num_,
                                           solver__alpha_,
                                           nodes__SimpleRadialExtraCalib__step_,
                                           SimpleRadialExtraCalib_num_,
                                           SimpleRadialExtraCalib_num_);
  SimpleRadialFocal_update_step_first(nodes__SimpleRadialFocal__p_,
                                      SimpleRadialFocal_num_,
                                      solver__alpha_,
                                      nodes__SimpleRadialFocal__step_,
                                      SimpleRadialFocal_num_,
                                      SimpleRadialFocal_num_);
}

void GraphSolver::do_update_step() {
  PinholeExtraCalib_update_step(nodes__PinholeExtraCalib__step_,
                                PinholeExtraCalib_num_,
                                nodes__PinholeExtraCalib__p_,
                                PinholeExtraCalib_num_,
                                solver__alpha_,
                                nodes__PinholeExtraCalib__step_,
                                PinholeExtraCalib_num_,
                                PinholeExtraCalib_num_);
  PinholeFocal_update_step(nodes__PinholeFocal__step_,
                           PinholeFocal_num_,
                           nodes__PinholeFocal__p_,
                           PinholeFocal_num_,
                           solver__alpha_,
                           nodes__PinholeFocal__step_,
                           PinholeFocal_num_,
                           PinholeFocal_num_);
  Point_update_step(nodes__Point__step_,
                    Point_num_,
                    nodes__Point__p_,
                    Point_num_,
                    solver__alpha_,
                    nodes__Point__step_,
                    Point_num_,
                    Point_num_);
  Pose_update_step(nodes__Pose__step_,
                   Pose_num_,
                   nodes__Pose__p_,
                   Pose_num_,
                   solver__alpha_,
                   nodes__Pose__step_,
                   Pose_num_,
                   Pose_num_);
  SimpleRadialExtraCalib_update_step(nodes__SimpleRadialExtraCalib__step_,
                                     SimpleRadialExtraCalib_num_,
                                     nodes__SimpleRadialExtraCalib__p_,
                                     SimpleRadialExtraCalib_num_,
                                     solver__alpha_,
                                     nodes__SimpleRadialExtraCalib__step_,
                                     SimpleRadialExtraCalib_num_,
                                     SimpleRadialExtraCalib_num_);
  SimpleRadialFocal_update_step(nodes__SimpleRadialFocal__step_,
                                SimpleRadialFocal_num_,
                                nodes__SimpleRadialFocal__p_,
                                SimpleRadialFocal_num_,
                                solver__alpha_,
                                nodes__SimpleRadialFocal__step_,
                                SimpleRadialFocal_num_,
                                SimpleRadialFocal_num_);
}

void GraphSolver::do_update_r_first() {
  zero(solver__r_0_norm2_tot_, solver__r_0_norm2_tot_ + 1);

  PinholeExtraCalib_update_r_first(nodes__PinholeExtraCalib__r_k_,
                                   PinholeExtraCalib_num_,
                                   nodes__PinholeExtraCalib__w_,
                                   PinholeExtraCalib_num_,
                                   solver__neg_alpha_,
                                   nodes__PinholeExtraCalib__r_k_,
                                   PinholeExtraCalib_num_,
                                   solver__r_0_norm2_tot_,
                                   solver__r_kp1_norm2_tot_,
                                   PinholeExtraCalib_num_);

  PinholeFocal_update_r_first(nodes__PinholeFocal__r_k_,
                              PinholeFocal_num_,
                              nodes__PinholeFocal__w_,
                              PinholeFocal_num_,
                              solver__neg_alpha_,
                              nodes__PinholeFocal__r_k_,
                              PinholeFocal_num_,
                              solver__r_0_norm2_tot_,
                              solver__r_kp1_norm2_tot_,
                              PinholeFocal_num_);

  Point_update_r_first(nodes__Point__r_k_,
                       Point_num_,
                       nodes__Point__w_,
                       Point_num_,
                       solver__neg_alpha_,
                       nodes__Point__r_k_,
                       Point_num_,
                       solver__r_0_norm2_tot_,
                       solver__r_kp1_norm2_tot_,
                       Point_num_);

  Pose_update_r_first(nodes__Pose__r_k_,
                      Pose_num_,
                      nodes__Pose__w_,
                      Pose_num_,
                      solver__neg_alpha_,
                      nodes__Pose__r_k_,
                      Pose_num_,
                      solver__r_0_norm2_tot_,
                      solver__r_kp1_norm2_tot_,
                      Pose_num_);

  SimpleRadialExtraCalib_update_r_first(nodes__SimpleRadialExtraCalib__r_k_,
                                        SimpleRadialExtraCalib_num_,
                                        nodes__SimpleRadialExtraCalib__w_,
                                        SimpleRadialExtraCalib_num_,
                                        solver__neg_alpha_,
                                        nodes__SimpleRadialExtraCalib__r_k_,
                                        SimpleRadialExtraCalib_num_,
                                        solver__r_0_norm2_tot_,
                                        solver__r_kp1_norm2_tot_,
                                        SimpleRadialExtraCalib_num_);

  SimpleRadialFocal_update_r_first(nodes__SimpleRadialFocal__r_k_,
                                   SimpleRadialFocal_num_,
                                   nodes__SimpleRadialFocal__w_,
                                   SimpleRadialFocal_num_,
                                   solver__neg_alpha_,
                                   nodes__SimpleRadialFocal__r_k_,
                                   SimpleRadialFocal_num_,
                                   solver__r_0_norm2_tot_,
                                   solver__r_kp1_norm2_tot_,
                                   SimpleRadialFocal_num_);

  pcg_r_0_norm2_ = read_cumem(solver__r_0_norm2_tot_);
  pcg_r_kp1_norm2_ = read_cumem(solver__r_kp1_norm2_tot_);
}

void GraphSolver::do_update_r() {
  zero(solver__r_kp1_norm2_tot_, solver__r_kp1_norm2_tot_ + 1);

  PinholeExtraCalib_update_r(nodes__PinholeExtraCalib__r_k_,
                             PinholeExtraCalib_num_,
                             nodes__PinholeExtraCalib__w_,
                             PinholeExtraCalib_num_,
                             solver__neg_alpha_,
                             nodes__PinholeExtraCalib__r_k_,
                             PinholeExtraCalib_num_,
                             solver__r_kp1_norm2_tot_,
                             PinholeExtraCalib_num_);
  PinholeFocal_update_r(nodes__PinholeFocal__r_k_,
                        PinholeFocal_num_,
                        nodes__PinholeFocal__w_,
                        PinholeFocal_num_,
                        solver__neg_alpha_,
                        nodes__PinholeFocal__r_k_,
                        PinholeFocal_num_,
                        solver__r_kp1_norm2_tot_,
                        PinholeFocal_num_);
  Point_update_r(nodes__Point__r_k_,
                 Point_num_,
                 nodes__Point__w_,
                 Point_num_,
                 solver__neg_alpha_,
                 nodes__Point__r_k_,
                 Point_num_,
                 solver__r_kp1_norm2_tot_,
                 Point_num_);
  Pose_update_r(nodes__Pose__r_k_,
                Pose_num_,
                nodes__Pose__w_,
                Pose_num_,
                solver__neg_alpha_,
                nodes__Pose__r_k_,
                Pose_num_,
                solver__r_kp1_norm2_tot_,
                Pose_num_);
  SimpleRadialExtraCalib_update_r(nodes__SimpleRadialExtraCalib__r_k_,
                                  SimpleRadialExtraCalib_num_,
                                  nodes__SimpleRadialExtraCalib__w_,
                                  SimpleRadialExtraCalib_num_,
                                  solver__neg_alpha_,
                                  nodes__SimpleRadialExtraCalib__r_k_,
                                  SimpleRadialExtraCalib_num_,
                                  solver__r_kp1_norm2_tot_,
                                  SimpleRadialExtraCalib_num_);
  SimpleRadialFocal_update_r(nodes__SimpleRadialFocal__r_k_,
                             SimpleRadialFocal_num_,
                             nodes__SimpleRadialFocal__w_,
                             SimpleRadialFocal_num_,
                             solver__neg_alpha_,
                             nodes__SimpleRadialFocal__r_k_,
                             SimpleRadialFocal_num_,
                             solver__r_kp1_norm2_tot_,
                             SimpleRadialFocal_num_);
  pcg_r_kp1_norm2_ = read_cumem(solver__r_kp1_norm2_tot_);
}

float GraphSolver::do_retract_score() {
  PinholeExtraCalib_retract(nodes__PinholeExtraCalib__storage_current_,
                            PinholeExtraCalib_num_max_,
                            nodes__PinholeExtraCalib__step_,
                            PinholeExtraCalib_num_,
                            nodes__PinholeExtraCalib__storage_check_,
                            PinholeExtraCalib_num_max_,
                            PinholeExtraCalib_num_);
  PinholeFocal_retract(nodes__PinholeFocal__storage_current_,
                       PinholeFocal_num_max_,
                       nodes__PinholeFocal__step_,
                       PinholeFocal_num_,
                       nodes__PinholeFocal__storage_check_,
                       PinholeFocal_num_max_,
                       PinholeFocal_num_);
  Point_retract(nodes__Point__storage_current_,
                Point_num_max_,
                nodes__Point__step_,
                Point_num_,
                nodes__Point__storage_check_,
                Point_num_max_,
                Point_num_);
  Pose_retract(nodes__Pose__storage_current_,
               Pose_num_max_,
               nodes__Pose__step_,
               Pose_num_,
               nodes__Pose__storage_check_,
               Pose_num_max_,
               Pose_num_);
  SimpleRadialExtraCalib_retract(
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      nodes__SimpleRadialExtraCalib__step_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__storage_check_,
      SimpleRadialExtraCalib_num_max_,
      SimpleRadialExtraCalib_num_);
  SimpleRadialFocal_retract(nodes__SimpleRadialFocal__storage_current_,
                            SimpleRadialFocal_num_max_,
                            nodes__SimpleRadialFocal__step_,
                            SimpleRadialFocal_num_,
                            nodes__SimpleRadialFocal__storage_check_,
                            SimpleRadialFocal_num_max_,
                            SimpleRadialFocal_num_);

  zero(solver__res_tot_, solver__res_tot_ + 1);
  simple_radial_score(nodes__Pose__storage_check_,
                      Pose_num_max_,
                      facs__simple_radial__args__pose__idx_shared_,
                      nodes__SimpleRadialFocal__storage_check_,
                      SimpleRadialFocal_num_max_,
                      facs__simple_radial__args__focal__idx_shared_,
                      nodes__SimpleRadialExtraCalib__storage_check_,
                      SimpleRadialExtraCalib_num_max_,
                      facs__simple_radial__args__extra_calib__idx_shared_,
                      nodes__Point__storage_check_,
                      Point_num_max_,
                      facs__simple_radial__args__point__idx_shared_,
                      facs__simple_radial__args__pixel__data_,
                      simple_radial_num_max_,
                      solver__res_tot_,
                      simple_radial_num_);
  simple_radial_fixed_pose_score(
      nodes__SimpleRadialFocal__storage_check_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_pose__args__focal__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_check_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_pose__args__extra_calib__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_pose__args__point__idx_shared_,
      facs__simple_radial_fixed_pose__args__pixel__data_,
      simple_radial_fixed_pose_num_max_,
      facs__simple_radial_fixed_pose__args__pose__data_,
      simple_radial_fixed_pose_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_num_);
  simple_radial_fixed_focal_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__simple_radial_fixed_focal__args__pose__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_check_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_focal__args__extra_calib__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_focal__args__point__idx_shared_,
      facs__simple_radial_fixed_focal__args__pixel__data_,
      simple_radial_fixed_focal_num_max_,
      facs__simple_radial_fixed_focal__args__focal__data_,
      simple_radial_fixed_focal_num_max_,
      solver__res_tot_,
      simple_radial_fixed_focal_num_);
  simple_radial_fixed_extra_calib_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__simple_radial_fixed_extra_calib__args__pose__idx_shared_,
      nodes__SimpleRadialFocal__storage_check_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_extra_calib__args__focal__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_extra_calib_num_max_,
      solver__res_tot_,
      simple_radial_fixed_extra_calib_num_);
  simple_radial_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__simple_radial_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocal__storage_check_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_point__args__focal__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_check_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_point__args__pixel__data_,
      simple_radial_fixed_point_num_max_,
      facs__simple_radial_fixed_point__args__point__data_,
      simple_radial_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_point_num_);
  simple_radial_fixed_pose_fixed_focal_score(
      nodes__SimpleRadialExtraCalib__storage_check_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal__args__extra_calib__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_fixed_focal_num_);
  simple_radial_fixed_pose_fixed_extra_calib_score(
      nodes__SimpleRadialFocal__storage_check_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__focal__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__pose__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_fixed_extra_calib_num_);
  simple_radial_fixed_pose_fixed_point_score(
      nodes__SimpleRadialFocal__storage_check_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__focal__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_check_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_fixed_point_num_);
  simple_radial_fixed_focal_fixed_extra_calib_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__pose__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__focal__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,
      solver__res_tot_,
      simple_radial_fixed_focal_fixed_extra_calib_num_);
  simple_radial_fixed_focal_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__simple_radial_fixed_focal_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialExtraCalib__storage_check_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_focal_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_point__args__focal__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_point__args__point__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_focal_fixed_point_num_);
  simple_radial_fixed_extra_calib_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocal__storage_check_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_extra_calib_fixed_point_num_);
  simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_score(
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_);
  simple_radial_fixed_pose_fixed_focal_fixed_point_score(
      nodes__SimpleRadialExtraCalib__storage_check_,
      SimpleRadialExtraCalib_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_);
  simple_radial_fixed_pose_fixed_extra_calib_fixed_point_score(
      nodes__SimpleRadialFocal__storage_check_,
      SimpleRadialFocal_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_);
  simple_radial_fixed_focal_fixed_extra_calib_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__focal__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_);
  pinhole_score(nodes__Pose__storage_check_,
                Pose_num_max_,
                facs__pinhole__args__pose__idx_shared_,
                nodes__PinholeFocal__storage_check_,
                PinholeFocal_num_max_,
                facs__pinhole__args__focal__idx_shared_,
                nodes__PinholeExtraCalib__storage_check_,
                PinholeExtraCalib_num_max_,
                facs__pinhole__args__extra_calib__idx_shared_,
                nodes__Point__storage_check_,
                Point_num_max_,
                facs__pinhole__args__point__idx_shared_,
                facs__pinhole__args__pixel__data_,
                pinhole_num_max_,
                solver__res_tot_,
                pinhole_num_);
  pinhole_fixed_pose_score(
      nodes__PinholeFocal__storage_check_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_pose__args__focal__idx_shared_,
      nodes__PinholeExtraCalib__storage_check_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_pose__args__extra_calib__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_fixed_pose__args__point__idx_shared_,
      facs__pinhole_fixed_pose__args__pixel__data_,
      pinhole_fixed_pose_num_max_,
      facs__pinhole_fixed_pose__args__pose__data_,
      pinhole_fixed_pose_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_num_);
  pinhole_fixed_focal_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__pinhole_fixed_focal__args__pose__idx_shared_,
      nodes__PinholeExtraCalib__storage_check_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_focal__args__extra_calib__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_fixed_focal__args__point__idx_shared_,
      facs__pinhole_fixed_focal__args__pixel__data_,
      pinhole_fixed_focal_num_max_,
      facs__pinhole_fixed_focal__args__focal__data_,
      pinhole_fixed_focal_num_max_,
      solver__res_tot_,
      pinhole_fixed_focal_num_);
  pinhole_fixed_extra_calib_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__pinhole_fixed_extra_calib__args__pose__idx_shared_,
      nodes__PinholeFocal__storage_check_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_extra_calib__args__focal__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_extra_calib_num_max_,
      solver__res_tot_,
      pinhole_fixed_extra_calib_num_);
  pinhole_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__pinhole_fixed_point__args__pose__idx_shared_,
      nodes__PinholeFocal__storage_check_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_point__args__focal__idx_shared_,
      nodes__PinholeExtraCalib__storage_check_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_point__args__pixel__data_,
      pinhole_fixed_point_num_max_,
      facs__pinhole_fixed_point__args__point__data_,
      pinhole_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_point_num_);
  pinhole_fixed_pose_fixed_focal_score(
      nodes__PinholeExtraCalib__storage_check_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal__args__extra_calib__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,
      facs__pinhole_fixed_pose_fixed_focal__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,
      facs__pinhole_fixed_pose_fixed_focal__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_fixed_focal_num_);
  pinhole_fixed_pose_fixed_extra_calib_score(
      nodes__PinholeFocal__storage_check_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__focal__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__pose__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_fixed_extra_calib_num_);
  pinhole_fixed_pose_fixed_point_score(
      nodes__PinholeFocal__storage_check_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__focal__idx_shared_,
      nodes__PinholeExtraCalib__storage_check_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_fixed_point_num_);
  pinhole_fixed_focal_fixed_extra_calib_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__pose__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__focal__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,
      solver__res_tot_,
      pinhole_fixed_focal_fixed_extra_calib_num_);
  pinhole_fixed_focal_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__pinhole_fixed_focal_fixed_point__args__pose__idx_shared_,
      nodes__PinholeExtraCalib__storage_check_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_point__args__focal__data_,
      pinhole_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_point__args__point__data_,
      pinhole_fixed_focal_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_focal_fixed_point_num_);
  pinhole_fixed_extra_calib_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      nodes__PinholeFocal__storage_check_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_extra_calib_fixed_point_num_);
  pinhole_fixed_pose_fixed_focal_fixed_extra_calib_score(
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_);
  pinhole_fixed_pose_fixed_focal_fixed_point_score(
      nodes__PinholeExtraCalib__storage_check_,
      PinholeExtraCalib_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_);
  pinhole_fixed_pose_fixed_extra_calib_fixed_point_score(
      nodes__PinholeFocal__storage_check_,
      PinholeFocal_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_);
  pinhole_fixed_focal_fixed_extra_calib_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__focal__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_);
  return 0.5 * read_cumem(solver__res_tot_);
}

void GraphSolver::do_beta() {
  zero(solver__beta_numerator_, solver__beta_numerator_ + 1);

  PinholeExtraCalib_alpha_denumerator_or_beta_nummerator(
      nodes__PinholeExtraCalib__r_k_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__z_,
      PinholeExtraCalib_num_,
      solver__beta_numerator_,
      PinholeExtraCalib_num_);

  PinholeFocal_alpha_denumerator_or_beta_nummerator(nodes__PinholeFocal__r_k_,
                                                    PinholeFocal_num_,
                                                    nodes__PinholeFocal__z_,
                                                    PinholeFocal_num_,
                                                    solver__beta_numerator_,
                                                    PinholeFocal_num_);

  Point_alpha_denumerator_or_beta_nummerator(nodes__Point__r_k_,
                                             Point_num_,
                                             nodes__Point__z_,
                                             Point_num_,
                                             solver__beta_numerator_,
                                             Point_num_);

  Pose_alpha_denumerator_or_beta_nummerator(nodes__Pose__r_k_,
                                            Pose_num_,
                                            nodes__Pose__z_,
                                            Pose_num_,
                                            solver__beta_numerator_,
                                            Pose_num_);

  SimpleRadialExtraCalib_alpha_denumerator_or_beta_nummerator(
      nodes__SimpleRadialExtraCalib__r_k_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__z_,
      SimpleRadialExtraCalib_num_,
      solver__beta_numerator_,
      SimpleRadialExtraCalib_num_);

  SimpleRadialFocal_alpha_denumerator_or_beta_nummerator(
      nodes__SimpleRadialFocal__r_k_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__z_,
      SimpleRadialFocal_num_,
      solver__beta_numerator_,
      SimpleRadialFocal_num_);
  beta_from_num_denum(
      solver__beta_numerator_, solver__alpha_numerator_, solver__beta_);
}

void GraphSolver::do_update_p() {
  PinholeExtraCalib_update_p(nodes__PinholeExtraCalib__z_,
                             PinholeExtraCalib_num_,
                             nodes__PinholeExtraCalib__p_,
                             PinholeExtraCalib_num_,
                             solver__beta_,
                             nodes__PinholeExtraCalib__p_,
                             PinholeExtraCalib_num_,
                             PinholeExtraCalib_num_);
  PinholeFocal_update_p(nodes__PinholeFocal__z_,
                        PinholeFocal_num_,
                        nodes__PinholeFocal__p_,
                        PinholeFocal_num_,
                        solver__beta_,
                        nodes__PinholeFocal__p_,
                        PinholeFocal_num_,
                        PinholeFocal_num_);
  Point_update_p(nodes__Point__z_,
                 Point_num_,
                 nodes__Point__p_,
                 Point_num_,
                 solver__beta_,
                 nodes__Point__p_,
                 Point_num_,
                 Point_num_);
  Pose_update_p(nodes__Pose__z_,
                Pose_num_,
                nodes__Pose__p_,
                Pose_num_,
                solver__beta_,
                nodes__Pose__p_,
                Pose_num_,
                Pose_num_);
  SimpleRadialExtraCalib_update_p(nodes__SimpleRadialExtraCalib__z_,
                                  SimpleRadialExtraCalib_num_,
                                  nodes__SimpleRadialExtraCalib__p_,
                                  SimpleRadialExtraCalib_num_,
                                  solver__beta_,
                                  nodes__SimpleRadialExtraCalib__p_,
                                  SimpleRadialExtraCalib_num_,
                                  SimpleRadialExtraCalib_num_);
  SimpleRadialFocal_update_p(nodes__SimpleRadialFocal__z_,
                             SimpleRadialFocal_num_,
                             nodes__SimpleRadialFocal__p_,
                             SimpleRadialFocal_num_,
                             solver__beta_,
                             nodes__SimpleRadialFocal__p_,
                             SimpleRadialFocal_num_,
                             SimpleRadialFocal_num_);
}

float GraphSolver::get_pred_decrease() {
  zero(solver__pred_decrease_tot_, solver__pred_decrease_tot_ + 1);
  PinholeExtraCalib_pred_decrease_times_two(
      nodes__PinholeExtraCalib__step_,
      PinholeExtraCalib_num_,
      nodes__PinholeExtraCalib__precond_diag_,
      PinholeExtraCalib_num_,
      solver__current_diag_,
      nodes__PinholeExtraCalib__r_0_,
      PinholeExtraCalib_num_,
      solver__pred_decrease_tot_,
      PinholeExtraCalib_num_);
  PinholeFocal_pred_decrease_times_two(nodes__PinholeFocal__step_,
                                       PinholeFocal_num_,
                                       nodes__PinholeFocal__precond_diag_,
                                       PinholeFocal_num_,
                                       solver__current_diag_,
                                       nodes__PinholeFocal__r_0_,
                                       PinholeFocal_num_,
                                       solver__pred_decrease_tot_,
                                       PinholeFocal_num_);
  Point_pred_decrease_times_two(nodes__Point__step_,
                                Point_num_,
                                nodes__Point__precond_diag_,
                                Point_num_,
                                solver__current_diag_,
                                nodes__Point__r_0_,
                                Point_num_,
                                solver__pred_decrease_tot_,
                                Point_num_);
  Pose_pred_decrease_times_two(nodes__Pose__step_,
                               Pose_num_,
                               nodes__Pose__precond_diag_,
                               Pose_num_,
                               solver__current_diag_,
                               nodes__Pose__r_0_,
                               Pose_num_,
                               solver__pred_decrease_tot_,
                               Pose_num_);
  SimpleRadialExtraCalib_pred_decrease_times_two(
      nodes__SimpleRadialExtraCalib__step_,
      SimpleRadialExtraCalib_num_,
      nodes__SimpleRadialExtraCalib__precond_diag_,
      SimpleRadialExtraCalib_num_,
      solver__current_diag_,
      nodes__SimpleRadialExtraCalib__r_0_,
      SimpleRadialExtraCalib_num_,
      solver__pred_decrease_tot_,
      SimpleRadialExtraCalib_num_);
  SimpleRadialFocal_pred_decrease_times_two(
      nodes__SimpleRadialFocal__step_,
      SimpleRadialFocal_num_,
      nodes__SimpleRadialFocal__precond_diag_,
      SimpleRadialFocal_num_,
      solver__current_diag_,
      nodes__SimpleRadialFocal__r_0_,
      SimpleRadialFocal_num_,
      solver__pred_decrease_tot_,
      SimpleRadialFocal_num_);
  return 0.5 * read_cumem(solver__pred_decrease_tot_);
}

void GraphSolver::finish_indices() { indices_valid_ = true; }

void GraphSolver::set_PinholeExtraCalib_num(const size_t num) {
  if (num > PinholeExtraCalib_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > PinholeExtraCalib_num_max_");
  }
  PinholeExtraCalib_num_ = num;
}

void GraphSolver::set_PinholeExtraCalib_nodes_from_stacked_host(
    const float* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholeExtraCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeExtraCalib_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  PinholeExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      offset,
      num);
}

void GraphSolver::set_PinholeExtraCalib_nodes_from_stacked_device(
    const float* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholeExtraCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeExtraCalib_num_");
  }
  PinholeExtraCalib_stacked_to_caspar(
      data,
      nodes__PinholeExtraCalib__storage_current_,
      PinholeExtraCalib_num_max_,
      offset,
      num);
}

void GraphSolver::get_PinholeExtraCalib_nodes_to_stacked_host(
    float* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholeExtraCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeExtraCalib_num_");
  }
  PinholeExtraCalib_caspar_to_stacked(
      nodes__PinholeExtraCalib__storage_current_,
      marker__scratch_inout_,
      PinholeExtraCalib_num_max_,
      offset,
      num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             2 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::get_PinholeExtraCalib_nodes_to_stacked_device(
    float* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholeExtraCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeExtraCalib_num_");
  }
  PinholeExtraCalib_caspar_to_stacked(
      nodes__PinholeExtraCalib__storage_current_,
      data,
      PinholeExtraCalib_num_max_,
      offset,
      num);
}

void GraphSolver::set_PinholeFocal_num(const size_t num) {
  if (num > PinholeFocal_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > PinholeFocal_num_max_");
  }
  PinholeFocal_num_ = num;
}

void GraphSolver::set_PinholeFocal_nodes_from_stacked_host(
    const float* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholeFocal_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocal_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  PinholeFocal_stacked_to_caspar(marker__scratch_inout_,
                                 nodes__PinholeFocal__storage_current_,
                                 PinholeFocal_num_max_,
                                 offset,
                                 num);
}

void GraphSolver::set_PinholeFocal_nodes_from_stacked_device(
    const float* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholeFocal_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocal_num_");
  }
  PinholeFocal_stacked_to_caspar(data,
                                 nodes__PinholeFocal__storage_current_,
                                 PinholeFocal_num_max_,
                                 offset,
                                 num);
}

void GraphSolver::get_PinholeFocal_nodes_to_stacked_host(float* const data,
                                                         const size_t offset,
                                                         const size_t num) {
  if (offset + num > PinholeFocal_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocal_num_");
  }
  PinholeFocal_caspar_to_stacked(nodes__PinholeFocal__storage_current_,
                                 marker__scratch_inout_,
                                 PinholeFocal_num_max_,
                                 offset,
                                 num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             2 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::get_PinholeFocal_nodes_to_stacked_device(float* const data,
                                                           const size_t offset,
                                                           const size_t num) {
  if (offset + num > PinholeFocal_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocal_num_");
  }
  PinholeFocal_caspar_to_stacked(nodes__PinholeFocal__storage_current_,
                                 data,
                                 PinholeFocal_num_max_,
                                 offset,
                                 num);
}

void GraphSolver::set_Point_num(const size_t num) {
  if (num > Point_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > Point_num_max_");
  }
  Point_num_ = num;
}

void GraphSolver::set_Point_nodes_from_stacked_host(const float* const data,
                                                    const size_t offset,
                                                    const size_t num) {
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  Point_stacked_to_caspar(marker__scratch_inout_,
                          nodes__Point__storage_current_,
                          Point_num_max_,
                          offset,
                          num);
}

void GraphSolver::set_Point_nodes_from_stacked_device(const float* const data,
                                                      const size_t offset,
                                                      const size_t num) {
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  Point_stacked_to_caspar(
      data, nodes__Point__storage_current_, Point_num_max_, offset, num);
}

void GraphSolver::get_Point_nodes_to_stacked_host(float* const data,
                                                  const size_t offset,
                                                  const size_t num) {
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  Point_caspar_to_stacked(nodes__Point__storage_current_,
                          marker__scratch_inout_,
                          Point_num_max_,
                          offset,
                          num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             3 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::get_Point_nodes_to_stacked_device(float* const data,
                                                    const size_t offset,
                                                    const size_t num) {
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  Point_caspar_to_stacked(
      nodes__Point__storage_current_, data, Point_num_max_, offset, num);
}

void GraphSolver::set_Pose_num(const size_t num) {
  if (num > Pose_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > Pose_num_max_");
  }
  Pose_num_ = num;
}

void GraphSolver::set_Pose_nodes_from_stacked_host(const float* const data,
                                                   const size_t offset,
                                                   const size_t num) {
  if (offset + num > Pose_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Pose_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  Pose_stacked_to_caspar(marker__scratch_inout_,
                         nodes__Pose__storage_current_,
                         Pose_num_max_,
                         offset,
                         num);
}

void GraphSolver::set_Pose_nodes_from_stacked_device(const float* const data,
                                                     const size_t offset,
                                                     const size_t num) {
  if (offset + num > Pose_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Pose_num_");
  }
  Pose_stacked_to_caspar(
      data, nodes__Pose__storage_current_, Pose_num_max_, offset, num);
}

void GraphSolver::get_Pose_nodes_to_stacked_host(float* const data,
                                                 const size_t offset,
                                                 const size_t num) {
  if (offset + num > Pose_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Pose_num_");
  }
  Pose_caspar_to_stacked(nodes__Pose__storage_current_,
                         marker__scratch_inout_,
                         Pose_num_max_,
                         offset,
                         num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             7 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::get_Pose_nodes_to_stacked_device(float* const data,
                                                   const size_t offset,
                                                   const size_t num) {
  if (offset + num > Pose_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Pose_num_");
  }
  Pose_caspar_to_stacked(
      nodes__Pose__storage_current_, data, Pose_num_max_, offset, num);
}

void GraphSolver::set_SimpleRadialExtraCalib_num(const size_t num) {
  if (num > SimpleRadialExtraCalib_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > SimpleRadialExtraCalib_num_max_");
  }
  SimpleRadialExtraCalib_num_ = num;
}

void GraphSolver::set_SimpleRadialExtraCalib_nodes_from_stacked_host(
    const float* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialExtraCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialExtraCalib_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  SimpleRadialExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      offset,
      num);
}

void GraphSolver::set_SimpleRadialExtraCalib_nodes_from_stacked_device(
    const float* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialExtraCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialExtraCalib_num_");
  }
  SimpleRadialExtraCalib_stacked_to_caspar(
      data,
      nodes__SimpleRadialExtraCalib__storage_current_,
      SimpleRadialExtraCalib_num_max_,
      offset,
      num);
}

void GraphSolver::get_SimpleRadialExtraCalib_nodes_to_stacked_host(
    float* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialExtraCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialExtraCalib_num_");
  }
  SimpleRadialExtraCalib_caspar_to_stacked(
      nodes__SimpleRadialExtraCalib__storage_current_,
      marker__scratch_inout_,
      SimpleRadialExtraCalib_num_max_,
      offset,
      num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             3 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::get_SimpleRadialExtraCalib_nodes_to_stacked_device(
    float* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialExtraCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialExtraCalib_num_");
  }
  SimpleRadialExtraCalib_caspar_to_stacked(
      nodes__SimpleRadialExtraCalib__storage_current_,
      data,
      SimpleRadialExtraCalib_num_max_,
      offset,
      num);
}

void GraphSolver::set_SimpleRadialFocal_num(const size_t num) {
  if (num > SimpleRadialFocal_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > SimpleRadialFocal_num_max_");
  }
  SimpleRadialFocal_num_ = num;
}

void GraphSolver::set_SimpleRadialFocal_nodes_from_stacked_host(
    const float* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialFocal_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialFocal_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             1 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  SimpleRadialFocal_stacked_to_caspar(
      marker__scratch_inout_,
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      offset,
      num);
}

void GraphSolver::set_SimpleRadialFocal_nodes_from_stacked_device(
    const float* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialFocal_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialFocal_num_");
  }
  SimpleRadialFocal_stacked_to_caspar(
      data,
      nodes__SimpleRadialFocal__storage_current_,
      SimpleRadialFocal_num_max_,
      offset,
      num);
}

void GraphSolver::get_SimpleRadialFocal_nodes_to_stacked_host(
    float* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialFocal_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialFocal_num_");
  }
  SimpleRadialFocal_caspar_to_stacked(
      nodes__SimpleRadialFocal__storage_current_,
      marker__scratch_inout_,
      SimpleRadialFocal_num_max_,
      offset,
      num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             1 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::get_SimpleRadialFocal_nodes_to_stacked_device(
    float* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialFocal_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialFocal_num_");
  }
  SimpleRadialFocal_caspar_to_stacked(
      nodes__SimpleRadialFocal__storage_current_,
      data,
      SimpleRadialFocal_num_max_,
      offset,
      num);
}

void GraphSolver::set_simple_radial_num(const size_t num) {
  if (num > simple_radial_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > simple_radial_num_max_");
  }
  simple_radial_num_ = num;
}
void GraphSolver::set_simple_radial_pose_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != simple_radial_num_. Use "
                             "set_simple_radial_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_pose_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != simple_radial_num_. Use "
                             "set_simple_radial_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices, facs__simple_radial__args__pose__idx_shared_, num);
}
void GraphSolver::set_simple_radial_focal_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != simple_radial_num_. Use "
                             "set_simple_radial_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_focal_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != simple_radial_num_. Use "
                             "set_simple_radial_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices, facs__simple_radial__args__focal__idx_shared_, num);
}
void GraphSolver::set_simple_radial_extra_calib_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != simple_radial_num_. Use "
                             "set_simple_radial_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_extra_calib_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != simple_radial_num_. Use "
                             "set_simple_radial_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__simple_radial__args__extra_calib__idx_shared_, num);
}
void GraphSolver::set_simple_radial_point_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != simple_radial_num_. Use "
                             "set_simple_radial_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_point_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != simple_radial_num_. Use "
                             "set_simple_radial_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices, facs__simple_radial__args__point__idx_shared_, num);
}
void GraphSolver::set_simple_radial_pixel_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(marker__scratch_inout_,
                               facs__simple_radial__args__pixel__data_,
                               simple_radial_num_max_,
                               offset,
                               num);
}

void GraphSolver::set_simple_radial_pixel_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_num_max_");
  }
  ConstPixel_stacked_to_caspar(data,
                               facs__simple_radial__args__pixel__data_,
                               simple_radial_num_max_,
                               offset,
                               num);
}
void GraphSolver::set_simple_radial_fixed_pose_num(const size_t num) {
  if (num > simple_radial_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > simple_radial_fixed_pose_num_max_");
  }
  simple_radial_fixed_pose_num_ = num;
}
void GraphSolver::set_simple_radial_fixed_pose_focal_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_num_. Use "
        "set_simple_radial_fixed_pose_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_fixed_pose_focal_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_num_. Use "
        "set_simple_radial_fixed_pose_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__simple_radial_fixed_pose__args__focal__idx_shared_, num);
}
void GraphSolver::set_simple_radial_fixed_pose_extra_calib_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_num_. Use "
        "set_simple_radial_fixed_pose_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_fixed_pose_extra_calib_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_num_. Use "
        "set_simple_radial_fixed_pose_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices,
                 facs__simple_radial_fixed_pose__args__extra_calib__idx_shared_,
                 num);
}
void GraphSolver::set_simple_radial_fixed_pose_point_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_num_. Use "
        "set_simple_radial_fixed_pose_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_fixed_pose_point_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_num_. Use "
        "set_simple_radial_fixed_pose_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__simple_radial_fixed_pose__args__point__idx_shared_, num);
}
void GraphSolver::set_simple_radial_fixed_pose_pixel_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose__args__pixel__data_,
      simple_radial_fixed_pose_num_max_,
      offset,
      num);
}

void GraphSolver::set_simple_radial_fixed_pose_pixel_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_pose_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose__args__pixel__data_,
      simple_radial_fixed_pose_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_pose_pose_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(marker__scratch_inout_,
                              facs__simple_radial_fixed_pose__args__pose__data_,
                              simple_radial_fixed_pose_num_max_,
                              offset,
                              num);
}

void GraphSolver::set_simple_radial_fixed_pose_pose_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_pose_num_max_");
  }
  ConstPose_stacked_to_caspar(data,
                              facs__simple_radial_fixed_pose__args__pose__data_,
                              simple_radial_fixed_pose_num_max_,
                              offset,
                              num);
}
void GraphSolver::set_simple_radial_fixed_focal_num(const size_t num) {
  if (num > simple_radial_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > simple_radial_fixed_focal_num_max_");
  }
  simple_radial_fixed_focal_num_ = num;
}
void GraphSolver::set_simple_radial_fixed_focal_pose_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_num_. Use "
        "set_simple_radial_fixed_focal_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_fixed_focal_pose_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_num_. Use "
        "set_simple_radial_fixed_focal_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__simple_radial_fixed_focal__args__pose__idx_shared_, num);
}
void GraphSolver::set_simple_radial_fixed_focal_extra_calib_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_num_. Use "
        "set_simple_radial_fixed_focal_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_fixed_focal_extra_calib_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_num_. Use "
        "set_simple_radial_fixed_focal_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_focal__args__extra_calib__idx_shared_,
      num);
}
void GraphSolver::set_simple_radial_fixed_focal_point_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_num_. Use "
        "set_simple_radial_fixed_focal_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_fixed_focal_point_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_num_. Use "
        "set_simple_radial_fixed_focal_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__simple_radial_fixed_focal__args__point__idx_shared_, num);
}
void GraphSolver::set_simple_radial_fixed_focal_pixel_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal__args__pixel__data_,
      simple_radial_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::set_simple_radial_fixed_focal_pixel_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_focal_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal__args__pixel__data_,
      simple_radial_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_focal_focal_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             1 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocal_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal__args__focal__data_,
      simple_radial_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::set_simple_radial_fixed_focal_focal_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_focal_num_max_");
  }
  ConstSimpleRadialFocal_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal__args__focal__data_,
      simple_radial_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_extra_calib_num(const size_t num) {
  if (num > simple_radial_fixed_extra_calib_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > simple_radial_fixed_extra_calib_num_max_");
  }
  simple_radial_fixed_extra_calib_num_ = num;
}
void GraphSolver::set_simple_radial_fixed_extra_calib_pose_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_extra_calib_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_extra_calib_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_fixed_extra_calib_pose_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_extra_calib_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices,
                 facs__simple_radial_fixed_extra_calib__args__pose__idx_shared_,
                 num);
}
void GraphSolver::set_simple_radial_fixed_extra_calib_focal_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_extra_calib_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_extra_calib_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_fixed_extra_calib_focal_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_extra_calib_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_extra_calib__args__focal__idx_shared_,
      num);
}
void GraphSolver::set_simple_radial_fixed_extra_calib_point_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_extra_calib_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_extra_calib_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_fixed_extra_calib_point_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_extra_calib_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_extra_calib__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_extra_calib_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_extra_calib_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_extra_calib_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_extra_calib_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_extra_calib_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_extra_calib_extra_calib_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_extra_calib_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_extra_calib_extra_calib_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_extra_calib_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_extra_calib_num_max_");
  }
  ConstSimpleRadialExtraCalib_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_point_num(const size_t num) {
  if (num > simple_radial_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > simple_radial_fixed_point_num_max_");
  }
  simple_radial_fixed_point_num_ = num;
}
void GraphSolver::set_simple_radial_fixed_point_pose_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_point_num_. Use "
        "set_simple_radial_fixed_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_fixed_point_pose_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_point_num_. Use "
        "set_simple_radial_fixed_point_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__simple_radial_fixed_point__args__pose__idx_shared_, num);
}
void GraphSolver::set_simple_radial_fixed_point_focal_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_point_num_. Use "
        "set_simple_radial_fixed_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_point_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_fixed_point_focal_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_point_num_. Use "
        "set_simple_radial_fixed_point_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__simple_radial_fixed_point__args__focal__idx_shared_, num);
}
void GraphSolver::set_simple_radial_fixed_point_extra_calib_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_point_num_. Use "
        "set_simple_radial_fixed_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_point_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_fixed_point_extra_calib_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_point_num_. Use "
        "set_simple_radial_fixed_point_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_point__args__extra_calib__idx_shared_,
      num);
}
void GraphSolver::set_simple_radial_fixed_point_pixel_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_point__args__pixel__data_,
      simple_radial_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::set_simple_radial_fixed_point_pixel_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_point__args__pixel__data_,
      simple_radial_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_point_point_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_point__args__point__data_,
      simple_radial_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::set_simple_radial_fixed_point_point_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_point__args__point__data_,
      simple_radial_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_pose_fixed_focal_num(
    const size_t num) {
  if (num > simple_radial_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_pose_fixed_focal_num_max_");
  }
  simple_radial_fixed_pose_fixed_focal_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_extra_calib_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_num_. Use "
        "set_simple_radial_fixed_pose_fixed_focal_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_focal_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_extra_calib_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_num_. Use "
        "set_simple_radial_fixed_pose_fixed_focal_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_pose_fixed_focal__args__extra_calib__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_num_. Use "
        "set_simple_radial_fixed_pose_fixed_focal_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_focal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_num_. Use "
        "set_simple_radial_fixed_pose_fixed_focal_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_pose_fixed_focal__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_pose_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_pose_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_focal_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             1 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocal_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_focal_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_num_max_");
  }
  ConstSimpleRadialFocal_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_pose_fixed_extra_calib_num(
    const size_t num) {
  if (num > simple_radial_fixed_pose_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_num_max_");
  }
  simple_radial_fixed_pose_fixed_extra_calib_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_focal_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_pose_fixed_extra_calib_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_extra_calib_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_focal_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_pose_fixed_extra_calib_num before setting "
        "indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__focal__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_pose_fixed_extra_calib_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_extra_calib_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_pose_fixed_extra_calib_num before setting "
        "indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_pose_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__pose__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_pose_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__pose__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_extra_calib_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_extra_calib_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_num_max_");
  }
  ConstSimpleRadialExtraCalib_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_pose_fixed_point_num(
    const size_t num) {
  if (num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  simple_radial_fixed_pose_fixed_point_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_point_focal_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_point_num_. Use "
        "set_simple_radial_fixed_pose_fixed_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_point_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_point_focal_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_point_num_. Use "
        "set_simple_radial_fixed_pose_fixed_point_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_pose_fixed_point__args__focal__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_point_extra_calib_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_point_num_. Use "
        "set_simple_radial_fixed_pose_fixed_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_point_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_point_extra_calib_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_point_num_. Use "
        "set_simple_radial_fixed_pose_fixed_point_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_pose_fixed_point__args__extra_calib__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_point_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_point_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_point_pose_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_point_pose_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_point_point_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_point_point_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_focal_fixed_extra_calib_num(
    const size_t num) {
  if (num > simple_radial_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_num_max_");
  }
  simple_radial_fixed_focal_fixed_extra_calib_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_focal_fixed_extra_calib_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_fixed_extra_calib_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_focal_fixed_extra_calib_num before setting "
        "indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_focal_fixed_extra_calib_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_fixed_extra_calib_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_focal_fixed_extra_calib_num before setting "
        "indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_focal_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             1 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocal_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__focal__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_focal_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_num_max_");
  }
  ConstSimpleRadialFocal_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__focal__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_num_max_");
  }
  ConstSimpleRadialExtraCalib_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_focal_fixed_point_num(
    const size_t num) {
  if (num > simple_radial_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_focal_fixed_point_num_max_");
  }
  simple_radial_fixed_focal_fixed_point_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_point_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_fixed_point_num_. Use "
        "set_simple_radial_fixed_focal_fixed_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_fixed_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_point_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_fixed_point_num_. Use "
        "set_simple_radial_fixed_focal_fixed_point_num before setting "
        "indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_focal_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_point_extra_calib_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_fixed_point_num_. Use "
        "set_simple_radial_fixed_focal_fixed_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_fixed_point_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_point_extra_calib_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_fixed_point_num_. Use "
        "set_simple_radial_fixed_focal_fixed_point_num before setting "
        "indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_point_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_point_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_point_focal_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             1 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocal_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_fixed_point__args__focal__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_point_focal_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_point_num_max_");
  }
  ConstSimpleRadialFocal_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_fixed_point__args__focal__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_point_point_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_fixed_point__args__point__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_point_point_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_fixed_point__args__point__data_,
      simple_radial_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_extra_calib_fixed_point_num(
    const size_t num) {
  if (num > simple_radial_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_extra_calib_fixed_point_num_max_");
  }
  simple_radial_fixed_extra_calib_fixed_point_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_extra_calib_fixed_point_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_extra_calib_fixed_point_num_. Use "
        "set_simple_radial_fixed_extra_calib_fixed_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_extra_calib_fixed_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_extra_calib_fixed_point_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_extra_calib_fixed_point_num_. Use "
        "set_simple_radial_fixed_extra_calib_fixed_point_num before setting "
        "indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_extra_calib_fixed_point_focal_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_extra_calib_fixed_point_num_. Use "
        "set_simple_radial_fixed_extra_calib_fixed_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_extra_calib_fixed_point_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_extra_calib_fixed_point_focal_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_extra_calib_fixed_point_num_. Use "
        "set_simple_radial_fixed_extra_calib_fixed_point_num before setting "
        "indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstSimpleRadialExtraCalib_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num(
        const size_t num) {
  if (num > simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_. Use "
        "set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_pose_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_pose_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_focal_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             1 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocal_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_focal_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  ConstSimpleRadialFocal_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  ConstSimpleRadialExtraCalib_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_pose_fixed_focal_fixed_point_num(
    const size_t num) {
  if (num > simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  simple_radial_fixed_pose_fixed_focal_fixed_point_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_fixed_point_num_. Use "
        "set_simple_radial_fixed_pose_fixed_focal_fixed_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_fixed_point_num_. Use "
        "set_simple_radial_fixed_pose_fixed_focal_fixed_point_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             1 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocal_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  ConstSimpleRadialFocal_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num(
        const size_t num) {
  if (num > simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_. Use "
        "set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_. Use "
        "set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_pose_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_pose_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstSimpleRadialExtraCalib_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num(
        const size_t num) {
  if (num > simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_. Use "
        "set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_. Use "
        "set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num "
        "before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_focal_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             1 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocal_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__focal__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_focal_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstSimpleRadialFocal_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__focal__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstSimpleRadialExtraCalib_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__point__data_,
      simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_num(const size_t num) {
  if (num > pinhole_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > pinhole_num_max_");
  }
  pinhole_num_ = num;
}
void GraphSolver::set_pinhole_pose_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_num_. Use set_pinhole_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_pose_indices_from_device((unsigned int*)marker__scratch_inout_,
                                       num);
}

void GraphSolver::set_pinhole_pose_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_num_. Use set_pinhole_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices, facs__pinhole__args__pose__idx_shared_, num);
}
void GraphSolver::set_pinhole_focal_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_num_. Use set_pinhole_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_focal_indices_from_device((unsigned int*)marker__scratch_inout_,
                                        num);
}

void GraphSolver::set_pinhole_focal_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_num_. Use set_pinhole_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices, facs__pinhole__args__focal__idx_shared_, num);
}
void GraphSolver::set_pinhole_extra_calib_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_num_. Use set_pinhole_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_extra_calib_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_num_. Use set_pinhole_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices, facs__pinhole__args__extra_calib__idx_shared_, num);
}
void GraphSolver::set_pinhole_point_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_num_. Use set_pinhole_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_point_indices_from_device((unsigned int*)marker__scratch_inout_,
                                        num);
}

void GraphSolver::set_pinhole_point_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_num_. Use set_pinhole_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices, facs__pinhole__args__point__idx_shared_, num);
}
void GraphSolver::set_pinhole_pixel_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(marker__scratch_inout_,
                               facs__pinhole__args__pixel__data_,
                               pinhole_num_max_,
                               offset,
                               num);
}

void GraphSolver::set_pinhole_pixel_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data, facs__pinhole__args__pixel__data_, pinhole_num_max_, offset, num);
}
void GraphSolver::set_pinhole_fixed_pose_num(const size_t num) {
  if (num > pinhole_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_fixed_pose_num_max_");
  }
  pinhole_fixed_pose_num_ = num;
}
void GraphSolver::set_pinhole_fixed_pose_focal_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_num_. Use set_pinhole_fixed_pose_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_pose_focal_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_num_. Use set_pinhole_fixed_pose_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__pinhole_fixed_pose__args__focal__idx_shared_, num);
}
void GraphSolver::set_pinhole_fixed_pose_extra_calib_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_num_. Use set_pinhole_fixed_pose_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_pose_extra_calib_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_num_. Use set_pinhole_fixed_pose_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__pinhole_fixed_pose__args__extra_calib__idx_shared_, num);
}
void GraphSolver::set_pinhole_fixed_pose_point_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_num_. Use set_pinhole_fixed_pose_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_pose_point_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_num_. Use set_pinhole_fixed_pose_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__pinhole_fixed_pose__args__point__idx_shared_, num);
}
void GraphSolver::set_pinhole_fixed_pose_pixel_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(marker__scratch_inout_,
                               facs__pinhole_fixed_pose__args__pixel__data_,
                               pinhole_fixed_pose_num_max_,
                               offset,
                               num);
}

void GraphSolver::set_pinhole_fixed_pose_pixel_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_num_max_");
  }
  ConstPixel_stacked_to_caspar(data,
                               facs__pinhole_fixed_pose__args__pixel__data_,
                               pinhole_fixed_pose_num_max_,
                               offset,
                               num);
}
void GraphSolver::set_pinhole_fixed_pose_pose_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(marker__scratch_inout_,
                              facs__pinhole_fixed_pose__args__pose__data_,
                              pinhole_fixed_pose_num_max_,
                              offset,
                              num);
}

void GraphSolver::set_pinhole_fixed_pose_pose_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_num_max_");
  }
  ConstPose_stacked_to_caspar(data,
                              facs__pinhole_fixed_pose__args__pose__data_,
                              pinhole_fixed_pose_num_max_,
                              offset,
                              num);
}
void GraphSolver::set_pinhole_fixed_focal_num(const size_t num) {
  if (num > pinhole_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_fixed_focal_num_max_");
  }
  pinhole_fixed_focal_num_ = num;
}
void GraphSolver::set_pinhole_fixed_focal_pose_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_num_. Use set_pinhole_fixed_focal_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_focal_pose_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_num_. Use set_pinhole_fixed_focal_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__pinhole_fixed_focal__args__pose__idx_shared_, num);
}
void GraphSolver::set_pinhole_fixed_focal_extra_calib_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_num_. Use set_pinhole_fixed_focal_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_focal_extra_calib_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_num_. Use set_pinhole_fixed_focal_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__pinhole_fixed_focal__args__extra_calib__idx_shared_, num);
}
void GraphSolver::set_pinhole_fixed_focal_point_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_num_. Use set_pinhole_fixed_focal_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_focal_point_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_num_. Use set_pinhole_fixed_focal_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__pinhole_fixed_focal__args__point__idx_shared_, num);
}
void GraphSolver::set_pinhole_fixed_focal_pixel_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(marker__scratch_inout_,
                               facs__pinhole_fixed_focal__args__pixel__data_,
                               pinhole_fixed_focal_num_max_,
                               offset,
                               num);
}

void GraphSolver::set_pinhole_fixed_focal_pixel_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_num_max_");
  }
  ConstPixel_stacked_to_caspar(data,
                               facs__pinhole_fixed_focal__args__pixel__data_,
                               pinhole_fixed_focal_num_max_,
                               offset,
                               num);
}
void GraphSolver::set_pinhole_fixed_focal_focal_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeFocal_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal__args__focal__data_,
      pinhole_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::set_pinhole_fixed_focal_focal_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_num_max_");
  }
  ConstPinholeFocal_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal__args__focal__data_,
      pinhole_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_extra_calib_num(const size_t num) {
  if (num > pinhole_fixed_extra_calib_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_fixed_extra_calib_num_max_");
  }
  pinhole_fixed_extra_calib_num_ = num;
}
void GraphSolver::set_pinhole_fixed_extra_calib_pose_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_extra_calib_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_extra_calib_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_extra_calib_pose_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_extra_calib_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__pinhole_fixed_extra_calib__args__pose__idx_shared_, num);
}
void GraphSolver::set_pinhole_fixed_extra_calib_focal_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_extra_calib_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_extra_calib_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_extra_calib_focal_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_extra_calib_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__pinhole_fixed_extra_calib__args__focal__idx_shared_, num);
}
void GraphSolver::set_pinhole_fixed_extra_calib_point_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_extra_calib_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_extra_calib_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_extra_calib_point_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_extra_calib_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__pinhole_fixed_extra_calib__args__point__idx_shared_, num);
}
void GraphSolver::set_pinhole_fixed_extra_calib_pixel_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_extra_calib_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::set_pinhole_fixed_extra_calib_pixel_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_extra_calib_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_extra_calib_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_extra_calib_extra_calib_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_extra_calib_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_extra_calib_extra_calib_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_extra_calib_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_extra_calib_num_max_");
  }
  ConstPinholeExtraCalib_stacked_to_caspar(
      data,
      facs__pinhole_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_point_num(const size_t num) {
  if (num > pinhole_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_fixed_point_num_max_");
  }
  pinhole_fixed_point_num_ = num;
}
void GraphSolver::set_pinhole_fixed_point_pose_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_point_num_. Use set_pinhole_fixed_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_point_pose_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_point_num_. Use set_pinhole_fixed_point_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__pinhole_fixed_point__args__pose__idx_shared_, num);
}
void GraphSolver::set_pinhole_fixed_point_focal_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_point_num_. Use set_pinhole_fixed_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_point_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_point_focal_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_point_num_. Use set_pinhole_fixed_point_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__pinhole_fixed_point__args__focal__idx_shared_, num);
}
void GraphSolver::set_pinhole_fixed_point_extra_calib_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_point_num_. Use set_pinhole_fixed_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_point_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_point_extra_calib_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_point_num_. Use set_pinhole_fixed_point_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices, facs__pinhole_fixed_point__args__extra_calib__idx_shared_, num);
}
void GraphSolver::set_pinhole_fixed_point_pixel_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(marker__scratch_inout_,
                               facs__pinhole_fixed_point__args__pixel__data_,
                               pinhole_fixed_point_num_max_,
                               offset,
                               num);
}

void GraphSolver::set_pinhole_fixed_point_pixel_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(data,
                               facs__pinhole_fixed_point__args__pixel__data_,
                               pinhole_fixed_point_num_max_,
                               offset,
                               num);
}
void GraphSolver::set_pinhole_fixed_point_point_data_from_stacked_host(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(marker__scratch_inout_,
                               facs__pinhole_fixed_point__args__point__data_,
                               pinhole_fixed_point_num_max_,
                               offset,
                               num);
}

void GraphSolver::set_pinhole_fixed_point_point_data_from_stacked_device(
    const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(data,
                               facs__pinhole_fixed_point__args__point__data_,
                               pinhole_fixed_point_num_max_,
                               offset,
                               num);
}
void GraphSolver::set_pinhole_fixed_pose_fixed_focal_num(const size_t num) {
  if (num > pinhole_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_fixed_pose_fixed_focal_num_max_");
  }
  pinhole_fixed_pose_fixed_focal_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_extra_calib_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_num_. Use "
        "set_pinhole_fixed_pose_fixed_focal_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_focal_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_extra_calib_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_num_. Use "
        "set_pinhole_fixed_pose_fixed_focal_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__pinhole_fixed_pose_fixed_focal__args__extra_calib__idx_shared_,
      num);
}
void GraphSolver::set_pinhole_fixed_pose_fixed_focal_point_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_num_. Use "
        "set_pinhole_fixed_pose_fixed_focal_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_focal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_pose_fixed_focal_point_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_num_. Use "
        "set_pinhole_fixed_pose_fixed_focal_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices,
                 facs__pinhole_fixed_pose_fixed_focal__args__point__idx_shared_,
                 num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_focal_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_pose_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_pose_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_focal_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_focal_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeFocal_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_focal_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_focal_num_max_");
  }
  ConstPinholeFocal_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_pose_fixed_extra_calib_num(
    const size_t num) {
  if (num > pinhole_fixed_pose_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_pose_fixed_extra_calib_num_max_");
  }
  pinhole_fixed_pose_fixed_extra_calib_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_focal_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_pose_fixed_extra_calib_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_extra_calib_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_focal_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_pose_fixed_extra_calib_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__focal__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_pose_fixed_extra_calib_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_extra_calib_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_pose_fixed_extra_calib_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_extra_calib_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_pose_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__pose__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_pose_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_extra_calib_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__pose__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_extra_calib_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_extra_calib_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_extra_calib_num_max_");
  }
  ConstPinholeExtraCalib_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_pose_fixed_point_num(const size_t num) {
  if (num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  pinhole_fixed_pose_fixed_point_num_ = num;
}
void GraphSolver::set_pinhole_fixed_pose_fixed_point_focal_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_point_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_pose_fixed_point_focal_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_point_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices,
                 facs__pinhole_fixed_pose_fixed_point__args__focal__idx_shared_,
                 num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_point_extra_calib_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_point_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_point_extra_calib_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_point_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__pinhole_fixed_pose_fixed_point__args__extra_calib__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_point_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_point_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_point_pose_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_point_pose_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_point_point_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_point_point_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_focal_fixed_extra_calib_num(
    const size_t num) {
  if (num > pinhole_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_focal_fixed_extra_calib_num_max_");
  }
  pinhole_fixed_focal_fixed_extra_calib_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_focal_fixed_extra_calib_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_fixed_extra_calib_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_focal_fixed_extra_calib_num before setting "
        "indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_focal_fixed_extra_calib_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_fixed_extra_calib_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_focal_fixed_extra_calib_num before setting "
        "indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_fixed_extra_calib_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_focal_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeFocal_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__focal__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_focal_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_fixed_extra_calib_num_max_");
  }
  ConstPinholeFocal_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__focal__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_fixed_extra_calib_num_max_");
  }
  ConstPinholeExtraCalib_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_focal_fixed_point_num(const size_t num) {
  if (num > pinhole_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_fixed_focal_fixed_point_num_max_");
  }
  pinhole_fixed_focal_fixed_point_num_ = num;
}
void GraphSolver::set_pinhole_fixed_focal_fixed_point_pose_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_fixed_point_num_. Use "
        "set_pinhole_fixed_focal_fixed_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_fixed_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_focal_fixed_point_pose_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_fixed_point_num_. Use "
        "set_pinhole_fixed_focal_fixed_point_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices,
                 facs__pinhole_fixed_focal_fixed_point__args__pose__idx_shared_,
                 num);
}
void GraphSolver::
    set_pinhole_fixed_focal_fixed_point_extra_calib_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_fixed_point_num_. Use "
        "set_pinhole_fixed_focal_fixed_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_fixed_point_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_focal_fixed_point_extra_calib_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_fixed_point_num_. Use "
        "set_pinhole_fixed_focal_fixed_point_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__pinhole_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_fixed_point_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_fixed_point_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_fixed_point_focal_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeFocal_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_fixed_point__args__focal__data_,
      pinhole_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_fixed_point_focal_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_fixed_point_num_max_");
  }
  ConstPinholeFocal_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_fixed_point__args__focal__data_,
      pinhole_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_fixed_point_point_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_fixed_point__args__point__data_,
      pinhole_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_fixed_point_point_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_fixed_point__args__point__data_,
      pinhole_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_extra_calib_fixed_point_num(
    const size_t num) {
  if (num > pinhole_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_extra_calib_fixed_point_num_max_");
  }
  pinhole_fixed_extra_calib_fixed_point_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_extra_calib_fixed_point_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_extra_calib_fixed_point_num_. Use "
        "set_pinhole_fixed_extra_calib_fixed_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_extra_calib_fixed_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_extra_calib_fixed_point_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_extra_calib_fixed_point_num_. Use "
        "set_pinhole_fixed_extra_calib_fixed_point_num before setting "
        "indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__pinhole_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_extra_calib_fixed_point_focal_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_extra_calib_fixed_point_num_. Use "
        "set_pinhole_fixed_extra_calib_fixed_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_extra_calib_fixed_point_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_extra_calib_fixed_point_focal_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_extra_calib_fixed_point_num_. Use "
        "set_pinhole_fixed_extra_calib_fixed_point_num before setting "
        "indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__pinhole_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPinholeExtraCalib_stacked_to_caspar(
      data,
      facs__pinhole_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num(
    const size_t num) {
  if (num > pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_. Use "
        "set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_pose_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_pose_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_focal_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeFocal_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_focal_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  ConstPinholeFocal_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_");
  }
  ConstPinholeExtraCalib_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_pose_fixed_focal_fixed_point_num(
    const size_t num) {
  if (num > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  pinhole_fixed_pose_fixed_focal_fixed_point_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_fixed_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_focal_fixed_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_fixed_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_focal_fixed_point_num before setting "
        "indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__extra_calib__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeFocal_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  ConstPinholeFocal_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_num(
    const size_t num) {
  if (num > pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__focal__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_pose_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_pose_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPinholeExtraCalib_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_num(
    const size_t num) {
  if (num > pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_. Use "
        "set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_. Use "
        "set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_num before "
        "setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(
      indices,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_focal_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeFocal_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__focal__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_focal_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPinholeFocal_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__focal__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeExtraCalib_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPinholeExtraCalib_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__extra_calib__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
        const float* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__point__data_,
      pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_,
      offset,
      num);
}

size_t GraphSolver::get_nbytes() {
  size_t offset = 0;
  size_t at_least = 0;
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholeExtraCalib_num_, 4);
  increment_offset<float>(offset, 2 * PinholeExtraCalib_num_, 4);
  increment_offset<float>(offset, 2 * PinholeExtraCalib_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 8 * Pose_num_, 4);
  increment_offset<float>(offset, 8 * Pose_num_, 4);
  increment_offset<float>(offset, 8 * Pose_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialExtraCalib_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialExtraCalib_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialExtraCalib_num_, 4);
  increment_offset<float>(offset, 1 * SimpleRadialFocal_num_, 4);
  increment_offset<float>(offset, 1 * SimpleRadialFocal_num_, 4);
  increment_offset<float>(offset, 1 * SimpleRadialFocal_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_pose_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_pose_num_, 4);
  increment_offset<float>(offset, 8 * simple_radial_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_focal_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_focal_num_, 4);
  increment_offset<float>(offset, 1 * simple_radial_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 4 * simple_radial_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_point_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_point_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(offset, 4 * simple_radial_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_pose_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 8 * simple_radial_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 1 * simple_radial_fixed_pose_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 8 * simple_radial_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 4 * simple_radial_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 8 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 4 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 1 * simple_radial_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 4 * simple_radial_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_focal_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 1 * simple_radial_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 4 * simple_radial_fixed_focal_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 4 * simple_radial_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 4 * simple_radial_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
      4);
  increment_offset<float>(
      offset,
      8 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
      4);
  increment_offset<float>(
      offset,
      1 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
      4);
  increment_offset<float>(
      offset,
      4 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
      4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 8 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 1 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 4 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      8 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      4 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      4 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      1 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      4 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      4 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_pose_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_num_, 4);
  increment_offset<float>(offset, 8 * pinhole_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_focal_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_focal_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_point_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_point_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 4 * pinhole_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(offset, 8 * pinhole_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 8 * pinhole_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(offset, 8 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(offset, 4 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_focal_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(offset, 4 * pinhole_fixed_focal_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 4 * pinhole_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 8 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 8 * pinhole_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 4 * pinhole_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 8 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 4 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 4 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_, 4);
  at_least = std::max(at_least,
                      offset + std::max({2 * PinholeExtraCalib_num_max_,
                                         2 * PinholeFocal_num_max_,
                                         3 * Point_num_max_,
                                         7 * Pose_num_max_,
                                         3 * SimpleRadialExtraCalib_num_max_,
                                         1 * SimpleRadialFocal_num_max_}) *
                                   sizeof(float));
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * simple_radial_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_pose_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_focal_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset,
      2 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
      4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset,
      2 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<float>(offset, 2 * pinhole_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_focal_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(offset, 12 * simple_radial_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_num_, 4);
  increment_offset<float>(offset, 6 * simple_radial_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_pose_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_pose_num_, 4);
  increment_offset<float>(offset, 6 * simple_radial_fixed_pose_num_, 4);
  increment_offset<float>(offset, 12 * simple_radial_fixed_focal_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_focal_num_, 4);
  increment_offset<float>(offset, 6 * simple_radial_fixed_focal_num_, 4);
  increment_offset<float>(offset, 12 * simple_radial_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 6 * simple_radial_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 12 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 6 * simple_radial_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 6 * simple_radial_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 12 * simple_radial_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 6 * simple_radial_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 12 * simple_radial_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 12 * simple_radial_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset,
      6 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
      4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset,
      2 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      12 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<float>(offset, 10 * pinhole_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_num_, 4);
  increment_offset<float>(offset, 0 * pinhole_num_, 4);
  increment_offset<float>(offset, 6 * pinhole_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_num_, 4);
  increment_offset<float>(offset, 0 * pinhole_fixed_pose_num_, 4);
  increment_offset<float>(offset, 6 * pinhole_fixed_pose_num_, 4);
  increment_offset<float>(offset, 10 * pinhole_fixed_focal_num_, 4);
  increment_offset<float>(offset, 0 * pinhole_fixed_focal_num_, 4);
  increment_offset<float>(offset, 6 * pinhole_fixed_focal_num_, 4);
  increment_offset<float>(offset, 10 * pinhole_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 6 * pinhole_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 10 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 0 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 0 * pinhole_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(offset, 6 * pinhole_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 6 * pinhole_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(offset, 0 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 10 * pinhole_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 6 * pinhole_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 10 * pinhole_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(offset, 0 * pinhole_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 10 * pinhole_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 6 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 0 * pinhole_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 10 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * PinholeExtraCalib_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 6 * Pose_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * SimpleRadialExtraCalib_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 1 * SimpleRadialFocal_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholeExtraCalib_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 6 * Pose_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * SimpleRadialExtraCalib_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 1 * SimpleRadialFocal_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholeExtraCalib_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 6 * Pose_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * SimpleRadialExtraCalib_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 1 * SimpleRadialFocal_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholeExtraCalib_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 6 * Pose_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialExtraCalib_num_, 4);
  increment_offset<float>(offset, 1 * SimpleRadialFocal_num_, 4);
  increment_offset<float>(offset, 0 * 0, 1);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholeExtraCalib_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 6 * Pose_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialExtraCalib_num_, 4);
  increment_offset<float>(offset, 1 * SimpleRadialFocal_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholeExtraCalib_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 6 * Pose_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialExtraCalib_num_, 4);
  increment_offset<float>(offset, 1 * SimpleRadialFocal_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholeExtraCalib_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 6 * Pose_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialExtraCalib_num_, 4);
  increment_offset<float>(offset, 1 * SimpleRadialFocal_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholeExtraCalib_num_, 4);
  increment_offset<float>(offset, 1 * PinholeExtraCalib_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 1 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 6 * Pose_num_, 4);
  increment_offset<float>(offset, 16 * Pose_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialExtraCalib_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialExtraCalib_num_, 4);
  increment_offset<float>(offset, 1 * SimpleRadialFocal_num_, 4);
  increment_offset<float>(offset, 0 * SimpleRadialFocal_num_, 4);
  increment_offset<float>(offset, 0 * 0, 1);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * simple_radial_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_pose_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_focal_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset,
      2 * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_,
      4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset,
      2 * simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_,
      4);
  increment_offset<float>(offset, 2 * pinhole_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_focal_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);
  increment_offset<float>(offset, 1 * 1, 1);

  return std::max(offset, at_least);
}

}  // namespace caspar