#include "solver.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "caspar_mappings.h"
#include "kernel_PinholeCalib_alpha_denominator_or_beta_numerator.h"
#include "kernel_PinholeCalib_alpha_numerator_denominator.h"
#include "kernel_PinholeCalib_normalize.h"
#include "kernel_PinholeCalib_pred_decrease_times_two.h"
#include "kernel_PinholeCalib_retract.h"
#include "kernel_PinholeCalib_start_w.h"
#include "kernel_PinholeCalib_start_w_contribute.h"
#include "kernel_PinholeCalib_update_Mp.h"
#include "kernel_PinholeCalib_update_p.h"
#include "kernel_PinholeCalib_update_r.h"
#include "kernel_PinholeCalib_update_r_first.h"
#include "kernel_PinholeCalib_update_step.h"
#include "kernel_PinholeCalib_update_step_first.h"
#include "kernel_PinholeFocalAndExtra_alpha_denominator_or_beta_numerator.h"
#include "kernel_PinholeFocalAndExtra_alpha_numerator_denominator.h"
#include "kernel_PinholeFocalAndExtra_normalize.h"
#include "kernel_PinholeFocalAndExtra_pred_decrease_times_two.h"
#include "kernel_PinholeFocalAndExtra_retract.h"
#include "kernel_PinholeFocalAndExtra_start_w.h"
#include "kernel_PinholeFocalAndExtra_start_w_contribute.h"
#include "kernel_PinholeFocalAndExtra_update_Mp.h"
#include "kernel_PinholeFocalAndExtra_update_p.h"
#include "kernel_PinholeFocalAndExtra_update_r.h"
#include "kernel_PinholeFocalAndExtra_update_r_first.h"
#include "kernel_PinholeFocalAndExtra_update_step.h"
#include "kernel_PinholeFocalAndExtra_update_step_first.h"
#include "kernel_PinholePose_alpha_denominator_or_beta_numerator.h"
#include "kernel_PinholePose_alpha_numerator_denominator.h"
#include "kernel_PinholePose_normalize.h"
#include "kernel_PinholePose_pred_decrease_times_two.h"
#include "kernel_PinholePose_retract.h"
#include "kernel_PinholePose_start_w.h"
#include "kernel_PinholePose_start_w_contribute.h"
#include "kernel_PinholePose_update_Mp.h"
#include "kernel_PinholePose_update_p.h"
#include "kernel_PinholePose_update_r.h"
#include "kernel_PinholePose_update_r_first.h"
#include "kernel_PinholePose_update_step.h"
#include "kernel_PinholePose_update_step_first.h"
#include "kernel_PinholePrincipalPoint_alpha_denominator_or_beta_numerator.h"
#include "kernel_PinholePrincipalPoint_alpha_numerator_denominator.h"
#include "kernel_PinholePrincipalPoint_normalize.h"
#include "kernel_PinholePrincipalPoint_pred_decrease_times_two.h"
#include "kernel_PinholePrincipalPoint_retract.h"
#include "kernel_PinholePrincipalPoint_start_w.h"
#include "kernel_PinholePrincipalPoint_start_w_contribute.h"
#include "kernel_PinholePrincipalPoint_update_Mp.h"
#include "kernel_PinholePrincipalPoint_update_p.h"
#include "kernel_PinholePrincipalPoint_update_r.h"
#include "kernel_PinholePrincipalPoint_update_r_first.h"
#include "kernel_PinholePrincipalPoint_update_step.h"
#include "kernel_PinholePrincipalPoint_update_step_first.h"
#include "kernel_Point_alpha_denominator_or_beta_numerator.h"
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
#include "kernel_SimpleRadialCalib_alpha_denominator_or_beta_numerator.h"
#include "kernel_SimpleRadialCalib_alpha_numerator_denominator.h"
#include "kernel_SimpleRadialCalib_normalize.h"
#include "kernel_SimpleRadialCalib_pred_decrease_times_two.h"
#include "kernel_SimpleRadialCalib_retract.h"
#include "kernel_SimpleRadialCalib_start_w.h"
#include "kernel_SimpleRadialCalib_start_w_contribute.h"
#include "kernel_SimpleRadialCalib_update_Mp.h"
#include "kernel_SimpleRadialCalib_update_p.h"
#include "kernel_SimpleRadialCalib_update_r.h"
#include "kernel_SimpleRadialCalib_update_r_first.h"
#include "kernel_SimpleRadialCalib_update_step.h"
#include "kernel_SimpleRadialCalib_update_step_first.h"
#include "kernel_SimpleRadialFocalAndExtra_alpha_denominator_or_beta_numerator.h"
#include "kernel_SimpleRadialFocalAndExtra_alpha_numerator_denominator.h"
#include "kernel_SimpleRadialFocalAndExtra_normalize.h"
#include "kernel_SimpleRadialFocalAndExtra_pred_decrease_times_two.h"
#include "kernel_SimpleRadialFocalAndExtra_retract.h"
#include "kernel_SimpleRadialFocalAndExtra_start_w.h"
#include "kernel_SimpleRadialFocalAndExtra_start_w_contribute.h"
#include "kernel_SimpleRadialFocalAndExtra_update_Mp.h"
#include "kernel_SimpleRadialFocalAndExtra_update_p.h"
#include "kernel_SimpleRadialFocalAndExtra_update_r.h"
#include "kernel_SimpleRadialFocalAndExtra_update_r_first.h"
#include "kernel_SimpleRadialFocalAndExtra_update_step.h"
#include "kernel_SimpleRadialFocalAndExtra_update_step_first.h"
#include "kernel_SimpleRadialPose_alpha_denominator_or_beta_numerator.h"
#include "kernel_SimpleRadialPose_alpha_numerator_denominator.h"
#include "kernel_SimpleRadialPose_normalize.h"
#include "kernel_SimpleRadialPose_pred_decrease_times_two.h"
#include "kernel_SimpleRadialPose_retract.h"
#include "kernel_SimpleRadialPose_start_w.h"
#include "kernel_SimpleRadialPose_start_w_contribute.h"
#include "kernel_SimpleRadialPose_update_Mp.h"
#include "kernel_SimpleRadialPose_update_p.h"
#include "kernel_SimpleRadialPose_update_r.h"
#include "kernel_SimpleRadialPose_update_r_first.h"
#include "kernel_SimpleRadialPose_update_step.h"
#include "kernel_SimpleRadialPose_update_step_first.h"
#include "kernel_SimpleRadialPrincipalPoint_alpha_denominator_or_beta_numerator.h"
#include "kernel_SimpleRadialPrincipalPoint_alpha_numerator_denominator.h"
#include "kernel_SimpleRadialPrincipalPoint_normalize.h"
#include "kernel_SimpleRadialPrincipalPoint_pred_decrease_times_two.h"
#include "kernel_SimpleRadialPrincipalPoint_retract.h"
#include "kernel_SimpleRadialPrincipalPoint_start_w.h"
#include "kernel_SimpleRadialPrincipalPoint_start_w_contribute.h"
#include "kernel_SimpleRadialPrincipalPoint_update_Mp.h"
#include "kernel_SimpleRadialPrincipalPoint_update_p.h"
#include "kernel_SimpleRadialPrincipalPoint_update_r.h"
#include "kernel_SimpleRadialPrincipalPoint_update_r_first.h"
#include "kernel_SimpleRadialPrincipalPoint_update_step.h"
#include "kernel_SimpleRadialPrincipalPoint_update_step_first.h"
#include "kernel_pinhole_fixed_focal_and_extra_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_focal_and_extra_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_focal_and_extra_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_focal_and_extra_fixed_point_score.h"
#include "kernel_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_score.h"
#include "kernel_pinhole_fixed_focal_and_extra_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_focal_and_extra_fixed_principal_point_res_jac.h"
#include "kernel_pinhole_fixed_focal_and_extra_fixed_principal_point_res_jac_first.h"
#include "kernel_pinhole_fixed_focal_and_extra_fixed_principal_point_score.h"
#include "kernel_pinhole_fixed_focal_and_extra_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_focal_and_extra_res_jac.h"
#include "kernel_pinhole_fixed_focal_and_extra_res_jac_first.h"
#include "kernel_pinhole_fixed_focal_and_extra_score.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_score.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_score.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_and_extra_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_and_extra_res_jac.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_and_extra_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_fixed_focal_and_extra_score.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_fixed_point_score.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_res_jac.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_score.h"
#include "kernel_pinhole_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_principal_point_fixed_point_score.h"
#include "kernel_pinhole_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_principal_point_res_jac.h"
#include "kernel_pinhole_fixed_principal_point_res_jac_first.h"
#include "kernel_pinhole_fixed_principal_point_score.h"
#include "kernel_pinhole_merged_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_merged_fixed_point_res_jac.h"
#include "kernel_pinhole_merged_fixed_point_res_jac_first.h"
#include "kernel_pinhole_merged_fixed_point_score.h"
#include "kernel_pinhole_merged_fixed_pose_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_merged_fixed_pose_fixed_point_res_jac.h"
#include "kernel_pinhole_merged_fixed_pose_fixed_point_res_jac_first.h"
#include "kernel_pinhole_merged_fixed_pose_fixed_point_score.h"
#include "kernel_pinhole_merged_fixed_pose_jtjnjtr_direct.h"
#include "kernel_pinhole_merged_fixed_pose_res_jac.h"
#include "kernel_pinhole_merged_fixed_pose_res_jac_first.h"
#include "kernel_pinhole_merged_fixed_pose_score.h"
#include "kernel_pinhole_merged_jtjnjtr_direct.h"
#include "kernel_pinhole_merged_res_jac.h"
#include "kernel_pinhole_merged_res_jac_first.h"
#include "kernel_pinhole_merged_score.h"
#include "kernel_simple_radial_fixed_focal_and_extra_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_focal_and_extra_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_focal_and_extra_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_focal_and_extra_fixed_point_score.h"
#include "kernel_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_score.h"
#include "kernel_simple_radial_fixed_focal_and_extra_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_focal_and_extra_fixed_principal_point_res_jac.h"
#include "kernel_simple_radial_fixed_focal_and_extra_fixed_principal_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_focal_and_extra_fixed_principal_point_score.h"
#include "kernel_simple_radial_fixed_focal_and_extra_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_focal_and_extra_res_jac.h"
#include "kernel_simple_radial_fixed_focal_and_extra_res_jac_first.h"
#include "kernel_simple_radial_fixed_focal_and_extra_score.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_score.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_score.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_and_extra_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_and_extra_res_jac.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_and_extra_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_fixed_focal_and_extra_score.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_fixed_point_score.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_res_jac.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_score.h"
#include "kernel_simple_radial_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_principal_point_fixed_point_score.h"
#include "kernel_simple_radial_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_principal_point_res_jac.h"
#include "kernel_simple_radial_fixed_principal_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_principal_point_score.h"
#include "kernel_simple_radial_merged_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_merged_fixed_point_res_jac.h"
#include "kernel_simple_radial_merged_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_merged_fixed_point_score.h"
#include "kernel_simple_radial_merged_fixed_pose_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_merged_fixed_pose_fixed_point_res_jac.h"
#include "kernel_simple_radial_merged_fixed_pose_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_merged_fixed_pose_fixed_point_score.h"
#include "kernel_simple_radial_merged_fixed_pose_jtjnjtr_direct.h"
#include "kernel_simple_radial_merged_fixed_pose_res_jac.h"
#include "kernel_simple_radial_merged_fixed_pose_res_jac_first.h"
#include "kernel_simple_radial_merged_fixed_pose_score.h"
#include "kernel_simple_radial_merged_jtjnjtr_direct.h"
#include "kernel_simple_radial_merged_res_jac.h"
#include "kernel_simple_radial_merged_res_jac_first.h"
#include "kernel_simple_radial_merged_score.h"
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
    size_t PinholeCalib_num_max,
    size_t PinholeFocalAndExtra_num_max,
    size_t PinholePose_num_max,
    size_t PinholePrincipalPoint_num_max,
    size_t Point_num_max,
    size_t SimpleRadialCalib_num_max,
    size_t SimpleRadialFocalAndExtra_num_max,
    size_t SimpleRadialPose_num_max,
    size_t SimpleRadialPrincipalPoint_num_max,
    size_t simple_radial_merged_num_max,
    size_t simple_radial_merged_fixed_pose_num_max,
    size_t simple_radial_merged_fixed_point_num_max,
    size_t simple_radial_merged_fixed_pose_fixed_point_num_max,
    size_t pinhole_merged_num_max,
    size_t pinhole_merged_fixed_pose_num_max,
    size_t pinhole_merged_fixed_point_num_max,
    size_t pinhole_merged_fixed_pose_fixed_point_num_max,
    size_t simple_radial_fixed_focal_and_extra_num_max,
    size_t simple_radial_fixed_principal_point_num_max,
    size_t simple_radial_fixed_pose_fixed_focal_and_extra_num_max,
    size_t simple_radial_fixed_pose_fixed_principal_point_num_max,
    size_t simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max,
    size_t simple_radial_fixed_focal_and_extra_fixed_point_num_max,
    size_t simple_radial_fixed_principal_point_fixed_point_num_max,
    size_t
        simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max,
    size_t simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max,
    size_t simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max,
    size_t
        simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max,
    size_t pinhole_fixed_focal_and_extra_num_max,
    size_t pinhole_fixed_principal_point_num_max,
    size_t pinhole_fixed_pose_fixed_focal_and_extra_num_max,
    size_t pinhole_fixed_pose_fixed_principal_point_num_max,
    size_t pinhole_fixed_focal_and_extra_fixed_principal_point_num_max,
    size_t pinhole_fixed_focal_and_extra_fixed_point_num_max,
    size_t pinhole_fixed_principal_point_fixed_point_num_max,
    size_t
        pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max,
    size_t pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max,
    size_t pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max,
    size_t
        pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max)
    : params_(params),
      PinholeCalib_num_(PinholeCalib_num_max),
      PinholeCalib_num_max_(PinholeCalib_num_max),
      PinholeFocalAndExtra_num_(PinholeFocalAndExtra_num_max),
      PinholeFocalAndExtra_num_max_(PinholeFocalAndExtra_num_max),
      PinholePose_num_(PinholePose_num_max),
      PinholePose_num_max_(PinholePose_num_max),
      PinholePrincipalPoint_num_(PinholePrincipalPoint_num_max),
      PinholePrincipalPoint_num_max_(PinholePrincipalPoint_num_max),
      Point_num_(Point_num_max),
      Point_num_max_(Point_num_max),
      SimpleRadialCalib_num_(SimpleRadialCalib_num_max),
      SimpleRadialCalib_num_max_(SimpleRadialCalib_num_max),
      SimpleRadialFocalAndExtra_num_(SimpleRadialFocalAndExtra_num_max),
      SimpleRadialFocalAndExtra_num_max_(SimpleRadialFocalAndExtra_num_max),
      SimpleRadialPose_num_(SimpleRadialPose_num_max),
      SimpleRadialPose_num_max_(SimpleRadialPose_num_max),
      SimpleRadialPrincipalPoint_num_(SimpleRadialPrincipalPoint_num_max),
      SimpleRadialPrincipalPoint_num_max_(SimpleRadialPrincipalPoint_num_max),
      simple_radial_merged_num_(simple_radial_merged_num_max),
      simple_radial_merged_num_max_(simple_radial_merged_num_max),
      simple_radial_merged_fixed_pose_num_(
          simple_radial_merged_fixed_pose_num_max),
      simple_radial_merged_fixed_pose_num_max_(
          simple_radial_merged_fixed_pose_num_max),
      simple_radial_merged_fixed_point_num_(
          simple_radial_merged_fixed_point_num_max),
      simple_radial_merged_fixed_point_num_max_(
          simple_radial_merged_fixed_point_num_max),
      simple_radial_merged_fixed_pose_fixed_point_num_(
          simple_radial_merged_fixed_pose_fixed_point_num_max),
      simple_radial_merged_fixed_pose_fixed_point_num_max_(
          simple_radial_merged_fixed_pose_fixed_point_num_max),
      pinhole_merged_num_(pinhole_merged_num_max),
      pinhole_merged_num_max_(pinhole_merged_num_max),
      pinhole_merged_fixed_pose_num_(pinhole_merged_fixed_pose_num_max),
      pinhole_merged_fixed_pose_num_max_(pinhole_merged_fixed_pose_num_max),
      pinhole_merged_fixed_point_num_(pinhole_merged_fixed_point_num_max),
      pinhole_merged_fixed_point_num_max_(pinhole_merged_fixed_point_num_max),
      pinhole_merged_fixed_pose_fixed_point_num_(
          pinhole_merged_fixed_pose_fixed_point_num_max),
      pinhole_merged_fixed_pose_fixed_point_num_max_(
          pinhole_merged_fixed_pose_fixed_point_num_max),
      simple_radial_fixed_focal_and_extra_num_(
          simple_radial_fixed_focal_and_extra_num_max),
      simple_radial_fixed_focal_and_extra_num_max_(
          simple_radial_fixed_focal_and_extra_num_max),
      simple_radial_fixed_principal_point_num_(
          simple_radial_fixed_principal_point_num_max),
      simple_radial_fixed_principal_point_num_max_(
          simple_radial_fixed_principal_point_num_max),
      simple_radial_fixed_pose_fixed_focal_and_extra_num_(
          simple_radial_fixed_pose_fixed_focal_and_extra_num_max),
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_(
          simple_radial_fixed_pose_fixed_focal_and_extra_num_max),
      simple_radial_fixed_pose_fixed_principal_point_num_(
          simple_radial_fixed_pose_fixed_principal_point_num_max),
      simple_radial_fixed_pose_fixed_principal_point_num_max_(
          simple_radial_fixed_pose_fixed_principal_point_num_max),
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_(
          simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max),
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_(
          simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max),
      simple_radial_fixed_focal_and_extra_fixed_point_num_(
          simple_radial_fixed_focal_and_extra_fixed_point_num_max),
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_(
          simple_radial_fixed_focal_and_extra_fixed_point_num_max),
      simple_radial_fixed_principal_point_fixed_point_num_(
          simple_radial_fixed_principal_point_fixed_point_num_max),
      simple_radial_fixed_principal_point_fixed_point_num_max_(
          simple_radial_fixed_principal_point_fixed_point_num_max),
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_(
          simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max),
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_(
          simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max),
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_(
          simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max),
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_(
          simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max),
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_(
          simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max),
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_(
          simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max),
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_(
          simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max),
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_(
          simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max),
      pinhole_fixed_focal_and_extra_num_(pinhole_fixed_focal_and_extra_num_max),
      pinhole_fixed_focal_and_extra_num_max_(
          pinhole_fixed_focal_and_extra_num_max),
      pinhole_fixed_principal_point_num_(pinhole_fixed_principal_point_num_max),
      pinhole_fixed_principal_point_num_max_(
          pinhole_fixed_principal_point_num_max),
      pinhole_fixed_pose_fixed_focal_and_extra_num_(
          pinhole_fixed_pose_fixed_focal_and_extra_num_max),
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_(
          pinhole_fixed_pose_fixed_focal_and_extra_num_max),
      pinhole_fixed_pose_fixed_principal_point_num_(
          pinhole_fixed_pose_fixed_principal_point_num_max),
      pinhole_fixed_pose_fixed_principal_point_num_max_(
          pinhole_fixed_pose_fixed_principal_point_num_max),
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_(
          pinhole_fixed_focal_and_extra_fixed_principal_point_num_max),
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_(
          pinhole_fixed_focal_and_extra_fixed_principal_point_num_max),
      pinhole_fixed_focal_and_extra_fixed_point_num_(
          pinhole_fixed_focal_and_extra_fixed_point_num_max),
      pinhole_fixed_focal_and_extra_fixed_point_num_max_(
          pinhole_fixed_focal_and_extra_fixed_point_num_max),
      pinhole_fixed_principal_point_fixed_point_num_(
          pinhole_fixed_principal_point_fixed_point_num_max),
      pinhole_fixed_principal_point_fixed_point_num_max_(
          pinhole_fixed_principal_point_fixed_point_num_max),
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_(
          pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max),
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_(
          pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max),
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_(
          pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max),
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_(
          pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max),
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_(
          pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max),
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_(
          pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max),
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_(
          pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max),
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_(
          pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max) {
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
  marker__start_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__storage_current_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeCalib__storage_check_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeCalib__storage_new_best_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeFocalAndExtra__storage_current_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholeFocalAndExtra__storage_check_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholeFocalAndExtra__storage_new_best_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholePose__storage_current_ = assign_and_increment<double>(
      origin_ptr_, offset, 8 * PinholePose_num_, 4);
  nodes__PinholePose__storage_check_ = assign_and_increment<double>(
      origin_ptr_, offset, 8 * PinholePose_num_, 4);
  nodes__PinholePose__storage_new_best_ = assign_and_increment<double>(
      origin_ptr_, offset, 8 * PinholePose_num_, 4);
  nodes__PinholePrincipalPoint__storage_current_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__storage_check_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__storage_new_best_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__Point__storage_current_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__storage_check_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__storage_new_best_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__SimpleRadialCalib__storage_current_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialCalib__storage_check_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialCalib__storage_new_best_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialFocalAndExtra__storage_current_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__storage_check_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__storage_new_best_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPose__storage_current_ = assign_and_increment<double>(
      origin_ptr_, offset, 8 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPose__storage_check_ = assign_and_increment<double>(
      origin_ptr_, offset, 8 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPose__storage_new_best_ = assign_and_increment<double>(
      origin_ptr_, offset, 8 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPrincipalPoint__storage_current_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__storage_check_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__storage_new_best_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  facs__simple_radial_merged__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_merged_num_, 4);
  facs__simple_radial_merged__args__calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_merged_num_, 4);
  facs__simple_radial_merged__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_merged_num_, 4);
  facs__simple_radial_merged__args__pixel__data_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_merged_num_, 4);
  facs__simple_radial_merged_fixed_pose__args__calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_merged_fixed_pose_num_, 4);
  facs__simple_radial_merged_fixed_pose__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_merged_fixed_pose_num_, 4);
  facs__simple_radial_merged_fixed_pose__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * simple_radial_merged_fixed_pose_num_, 4);
  facs__simple_radial_merged_fixed_pose__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 8 * simple_radial_merged_fixed_pose_num_, 4);
  facs__simple_radial_merged_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_merged_fixed_point_num_, 4);
  facs__simple_radial_merged_fixed_point__args__calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_merged_fixed_point_num_, 4);
  facs__simple_radial_merged_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * simple_radial_merged_fixed_point_num_, 4);
  facs__simple_radial_merged_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 4 * simple_radial_merged_fixed_point_num_, 4);
  facs__simple_radial_merged_fixed_pose_fixed_point__args__calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_merged_fixed_pose_fixed_point_num_,
          4);
  facs__simple_radial_merged_fixed_pose_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_merged_fixed_pose_fixed_point_num_,
          4);
  facs__simple_radial_merged_fixed_pose_fixed_point__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          8 * simple_radial_merged_fixed_pose_fixed_point_num_,
          4);
  facs__simple_radial_merged_fixed_pose_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * simple_radial_merged_fixed_pose_fixed_point_num_,
          4);
  facs__pinhole_merged__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_merged_num_, 4);
  facs__pinhole_merged__args__calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_merged_num_, 4);
  facs__pinhole_merged__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_merged_num_, 4);
  facs__pinhole_merged__args__pixel__data_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_merged_num_, 4);
  facs__pinhole_merged_fixed_pose__args__calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_merged_fixed_pose_num_, 4);
  facs__pinhole_merged_fixed_pose__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_merged_fixed_pose_num_, 4);
  facs__pinhole_merged_fixed_pose__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * pinhole_merged_fixed_pose_num_, 4);
  facs__pinhole_merged_fixed_pose__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 8 * pinhole_merged_fixed_pose_num_, 4);
  facs__pinhole_merged_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_merged_fixed_point_num_, 4);
  facs__pinhole_merged_fixed_point__args__calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_merged_fixed_point_num_, 4);
  facs__pinhole_merged_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * pinhole_merged_fixed_point_num_, 4);
  facs__pinhole_merged_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 4 * pinhole_merged_fixed_point_num_, 4);
  facs__pinhole_merged_fixed_pose_fixed_point__args__calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_merged_fixed_pose_fixed_point_num_,
          4);
  facs__pinhole_merged_fixed_pose_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_merged_fixed_pose_fixed_point_num_,
          4);
  facs__pinhole_merged_fixed_pose_fixed_point__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          8 * pinhole_merged_fixed_pose_fixed_point_num_,
          4);
  facs__pinhole_merged_fixed_pose_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * pinhole_merged_fixed_pose_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_focal_and_extra_num_, 4);
  facs__simple_radial_fixed_focal_and_extra__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_focal_and_extra_num_, 4);
  facs__simple_radial_fixed_focal_and_extra__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_focal_and_extra_num_, 4);
  facs__simple_radial_fixed_focal_and_extra__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * simple_radial_fixed_focal_and_extra_num_, 4);
  facs__simple_radial_fixed_focal_and_extra__args__focal_and_extra__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * simple_radial_fixed_focal_and_extra_num_, 4);
  facs__simple_radial_fixed_principal_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_principal_point_num_, 4);
  facs__simple_radial_fixed_principal_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_principal_point_num_, 4);
  facs__simple_radial_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_principal_point_num_, 4);
  facs__simple_radial_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * simple_radial_fixed_principal_point_num_, 4);
  facs__simple_radial_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * simple_radial_fixed_principal_point_num_, 4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          8 * simple_radial_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          8 * simple_radial_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_principal_point_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_principal_point_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_principal_point_fixed_point__args__principal_point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_principal_point_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          8 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          8 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          8 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_focal_and_extra_num_, 4);
  facs__pinhole_fixed_focal_and_extra__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_focal_and_extra_num_, 4);
  facs__pinhole_fixed_focal_and_extra__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_focal_and_extra_num_, 4);
  facs__pinhole_fixed_focal_and_extra__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * pinhole_fixed_focal_and_extra_num_, 4);
  facs__pinhole_fixed_focal_and_extra__args__focal_and_extra__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * pinhole_fixed_focal_and_extra_num_, 4);
  facs__pinhole_fixed_principal_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_principal_point_num_, 4);
  facs__pinhole_fixed_principal_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_principal_point_num_, 4);
  facs__pinhole_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_principal_point_num_, 4);
  facs__pinhole_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * pinhole_fixed_principal_point_num_, 4);
  facs__pinhole_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * pinhole_fixed_principal_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          8 * pinhole_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          8 * pinhole_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * pinhole_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_principal_point_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_principal_point_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_principal_point_fixed_point__args__principal_point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_principal_point_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * pinhole_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          8 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          8 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          8 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  marker__scratch_inout_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  facs__simple_radial_merged__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_merged_num_, 4);
  facs__simple_radial_merged_fixed_pose__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_merged_fixed_pose_num_, 4);
  facs__simple_radial_merged_fixed_point__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_merged_fixed_point_num_, 4);
  facs__simple_radial_merged_fixed_pose_fixed_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_merged_fixed_pose_fixed_point_num_,
          4);
  facs__pinhole_merged__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_merged_num_, 4);
  facs__pinhole_merged_fixed_pose__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_merged_fixed_pose_num_, 4);
  facs__pinhole_merged_fixed_point__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_merged_fixed_point_num_, 4);
  facs__pinhole_merged_fixed_pose_fixed_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_merged_fixed_pose_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra__res_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * simple_radial_fixed_focal_and_extra_num_, 4);
  facs__simple_radial_fixed_principal_point__res_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * simple_radial_fixed_principal_point_num_, 4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_principal_point_fixed_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_focal_and_extra_num_, 4);
  facs__pinhole_fixed_principal_point__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_principal_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_principal_point_fixed_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_merged__args__pose__jac_ = assign_and_increment<double>(
      origin_ptr_, offset, 12 * simple_radial_merged_num_, 4);
  facs__simple_radial_merged__args__calib__jac_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * simple_radial_merged_num_, 4);
  facs__simple_radial_merged__args__point__jac_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * simple_radial_merged_num_, 4);
  facs__simple_radial_merged_fixed_pose__args__calib__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 4 * simple_radial_merged_fixed_pose_num_, 4);
  facs__simple_radial_merged_fixed_pose__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 6 * simple_radial_merged_fixed_pose_num_, 4);
  facs__simple_radial_merged_fixed_point__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 12 * simple_radial_merged_fixed_point_num_, 4);
  facs__simple_radial_merged_fixed_point__args__calib__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 4 * simple_radial_merged_fixed_point_num_, 4);
  facs__simple_radial_merged_fixed_pose_fixed_point__args__calib__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * simple_radial_merged_fixed_pose_fixed_point_num_,
          4);
  facs__pinhole_merged__args__pose__jac_ = assign_and_increment<double>(
      origin_ptr_, offset, 10 * pinhole_merged_num_, 4);
  facs__pinhole_merged__args__calib__jac_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_merged_num_, 4);
  facs__pinhole_merged__args__point__jac_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * pinhole_merged_num_, 4);
  facs__pinhole_merged_fixed_pose__args__calib__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * pinhole_merged_fixed_pose_num_, 4);
  facs__pinhole_merged_fixed_pose__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 6 * pinhole_merged_fixed_pose_num_, 4);
  facs__pinhole_merged_fixed_point__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 10 * pinhole_merged_fixed_point_num_, 4);
  facs__pinhole_merged_fixed_point__args__calib__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * pinhole_merged_fixed_point_num_, 4);
  facs__pinhole_merged_fixed_pose_fixed_point__args__calib__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_merged_fixed_pose_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          12 * simple_radial_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 0 * simple_radial_fixed_focal_and_extra_num_, 4);
  facs__simple_radial_fixed_focal_and_extra__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 6 * simple_radial_fixed_focal_and_extra_num_, 4);
  facs__simple_radial_fixed_principal_point__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          12 * simple_radial_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_principal_point__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 4 * simple_radial_fixed_principal_point_num_, 4);
  facs__simple_radial_fixed_principal_point__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 6 * simple_radial_fixed_principal_point_num_, 4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          0 * simple_radial_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          6 * simple_radial_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          6 * simple_radial_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          12 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          6 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          12 * simple_radial_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          0 * simple_radial_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_principal_point_fixed_point__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          12 * simple_radial_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          6 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          0 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          12 *
              simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 10 * pinhole_fixed_focal_and_extra_num_, 4);
  facs__pinhole_fixed_focal_and_extra__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 0 * pinhole_fixed_focal_and_extra_num_, 4);
  facs__pinhole_fixed_focal_and_extra__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 6 * pinhole_fixed_focal_and_extra_num_, 4);
  facs__pinhole_fixed_principal_point__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 10 * pinhole_fixed_principal_point_num_, 4);
  facs__pinhole_fixed_principal_point__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * pinhole_fixed_principal_point_num_, 4);
  facs__pinhole_fixed_principal_point__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 6 * pinhole_fixed_principal_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          0 * pinhole_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          6 * pinhole_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          6 * pinhole_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          10 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          6 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          10 * pinhole_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          0 * pinhole_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_principal_point_fixed_point__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          10 * pinhole_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          6 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          0 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          10 *
              pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  nodes__PinholeCalib__z_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeCalib__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocalAndExtra__z_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholeFocalAndExtra__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePose__z_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePose__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePrincipalPoint__z_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__z_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialCalib__z_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialCalib__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialFocalAndExtra__z_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPose__z_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPose__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPrincipalPoint__z_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__p_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeCalib__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocalAndExtra__p_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholeFocalAndExtra__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePose__p_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePose__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePrincipalPoint__p_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__p_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialCalib__p_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialCalib__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialFocalAndExtra__p_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPose__p_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPose__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPrincipalPoint__p_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__step_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeCalib__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocalAndExtra__step_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholeFocalAndExtra__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePose__step_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePose__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePrincipalPoint__step_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__step_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialCalib__step_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialCalib__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialFocalAndExtra__step_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPose__step_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPose__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPrincipalPoint__step_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__w_start_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__w_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeFocalAndExtra__w_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholePose__w_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePrincipalPoint__w_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__Point__w_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__SimpleRadialCalib__w_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialFocalAndExtra__w_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPose__w_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPrincipalPoint__w_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  marker__w_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 1);
  marker__r_0_start_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__r_0_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeFocalAndExtra__r_0_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholePose__r_0_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePrincipalPoint__r_0_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__Point__r_0_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__SimpleRadialCalib__r_0_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialFocalAndExtra__r_0_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPose__r_0_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPrincipalPoint__r_0_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  marker__r_0_end_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__r_k_start_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__r_k_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeFocalAndExtra__r_k_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholePose__r_k_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePrincipalPoint__r_k_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__Point__r_k_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__SimpleRadialCalib__r_k_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialFocalAndExtra__r_k_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPose__r_k_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPrincipalPoint__r_k_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  marker__r_k_end_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__Mp_start_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__Mp_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeFocalAndExtra__Mp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholePose__Mp_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePrincipalPoint__Mp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__Point__Mp_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__SimpleRadialCalib__Mp_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialFocalAndExtra__Mp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPose__Mp_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPrincipalPoint__Mp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  marker__Mp_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__precond_start_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__precond_diag_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeCalib__precond_tril_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * PinholeCalib_num_, 4);
  nodes__PinholeFocalAndExtra__precond_diag_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholeFocalAndExtra__precond_tril_ = assign_and_increment<double>(
      origin_ptr_, offset, 1 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholePose__precond_diag_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePose__precond_tril_ = assign_and_increment<double>(
      origin_ptr_, offset, 16 * PinholePose_num_, 4);
  nodes__PinholePrincipalPoint__precond_diag_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__precond_tril_ = assign_and_increment<double>(
      origin_ptr_, offset, 1 * PinholePrincipalPoint_num_, 4);
  nodes__Point__precond_diag_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__precond_tril_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__SimpleRadialCalib__precond_diag_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialCalib__precond_tril_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialFocalAndExtra__precond_diag_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__precond_tril_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 1 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPose__precond_diag_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPose__precond_tril_ = assign_and_increment<double>(
      origin_ptr_, offset, 16 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPrincipalPoint__precond_diag_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__precond_tril_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 1 * SimpleRadialPrincipalPoint_num_, 4);
  marker__precond_end_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 1);
  marker__jp_start_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  facs__simple_radial_merged__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_merged_num_, 4);
  facs__simple_radial_merged_fixed_pose__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_merged_fixed_pose_num_, 4);
  facs__simple_radial_merged_fixed_point__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_merged_fixed_point_num_, 4);
  facs__simple_radial_merged_fixed_pose_fixed_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_merged_fixed_pose_fixed_point_num_,
          4);
  facs__pinhole_merged__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_merged_num_, 4);
  facs__pinhole_merged_fixed_pose__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_merged_fixed_pose_num_, 4);
  facs__pinhole_merged_fixed_point__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_merged_fixed_point_num_, 4);
  facs__pinhole_merged_fixed_pose_fixed_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_merged_fixed_pose_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_fixed_focal_and_extra_num_, 4);
  facs__simple_radial_fixed_principal_point__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_fixed_principal_point_num_, 4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_principal_point_fixed_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_focal_and_extra_num_, 4);
  facs__pinhole_fixed_principal_point__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_principal_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_principal_point_fixed_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  marker__jp_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 1);
  solver__current_diag_ =
      assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__alpha_numerator_ =
      assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__alpha_denominator_ =
      assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__alpha_ = assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__neg_alpha_ =
      assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__beta_numerator_ =
      assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__beta_ = assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__r_0_norm2_tot_ =
      assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__r_kp1_norm2_tot_ =
      assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__pred_decrease_tot_ =
      assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);
  solver__res_tot_ =
      assign_and_increment<double>(origin_ptr_, offset, 1 * 1, 1);

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
  double score_best;
  double score_best_pcg;
  double diag = params_.diag_init;
  cudaMemcpy(
      solver__current_diag_, &diag, sizeof(double), cudaMemcpyHostToDevice);

  double up_scale = params_.diag_scaling_up;
  double quality;

  std::chrono::time_point<std::chrono::steady_clock> t0 =
      std::chrono::steady_clock::now();
  std::chrono::time_point<std::chrono::steady_clock> t_prev = t0;
  score_best = DoResJacFirst();
  if (print_progress) {
    printf("                                 score_init: % .6e\n", score_best);
  }

  for (solver_iter_ = 0; solver_iter_ < params_.solver_iter_max;
       solver_iter_++) {
    if (solver_iter_ != 0 && solver_iter_ < params_.solver_iter_max - 1) {
      DoResJac();
    }
    score_best_pcg = score_best;
    for (pcg_iter_ = 0; pcg_iter_ < params_.pcg_iter_max; pcg_iter_++) {
      DoNormalize();

      if (pcg_iter_ == 0) {
        Copy(marker__r_k_start_, marker__r_k_end_, marker__w_start_);
        DoJtjpDirect();
        DoAlphaFirst();
        DoUpdateStepFirst();
        DoUpdateRFirst();
      } else {
        DoBeta();
        DoUpdateP();
        DoUpdateMp();
        DoJtjpDirect();
        DoAlpha();
        DoUpdateStep();
        DoUpdateR();
      }
      if (params_.pcg_rel_decrease_min != -1.0f ||
          params_.pcg_rel_score_exit != -1.0f) {
        double score_new_pcg = DoRetractScore();
        if (!(score_new_pcg <= score_best_pcg * params_.pcg_rel_decrease_min)) {
          break;
        }
        std::swap(nodes__PinholeCalib__storage_check_,
                  nodes__PinholeCalib__storage_new_best_);
        std::swap(nodes__PinholeFocalAndExtra__storage_check_,
                  nodes__PinholeFocalAndExtra__storage_new_best_);
        std::swap(nodes__PinholePose__storage_check_,
                  nodes__PinholePose__storage_new_best_);
        std::swap(nodes__PinholePrincipalPoint__storage_check_,
                  nodes__PinholePrincipalPoint__storage_new_best_);
        std::swap(nodes__Point__storage_check_,
                  nodes__Point__storage_new_best_);
        std::swap(nodes__SimpleRadialCalib__storage_check_,
                  nodes__SimpleRadialCalib__storage_new_best_);
        std::swap(nodes__SimpleRadialFocalAndExtra__storage_check_,
                  nodes__SimpleRadialFocalAndExtra__storage_new_best_);
        std::swap(nodes__SimpleRadialPose__storage_check_,
                  nodes__SimpleRadialPose__storage_new_best_);
        std::swap(nodes__SimpleRadialPrincipalPoint__storage_check_,
                  nodes__SimpleRadialPrincipalPoint__storage_new_best_);
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
      score_best_pcg = DoRetractScore();
      std::swap(nodes__PinholeCalib__storage_check_,
                nodes__PinholeCalib__storage_new_best_);
      std::swap(nodes__PinholeFocalAndExtra__storage_check_,
                nodes__PinholeFocalAndExtra__storage_new_best_);
      std::swap(nodes__PinholePose__storage_check_,
                nodes__PinholePose__storage_new_best_);
      std::swap(nodes__PinholePrincipalPoint__storage_check_,
                nodes__PinholePrincipalPoint__storage_new_best_);
      std::swap(nodes__Point__storage_check_, nodes__Point__storage_new_best_);
      std::swap(nodes__SimpleRadialCalib__storage_check_,
                nodes__SimpleRadialCalib__storage_new_best_);
      std::swap(nodes__SimpleRadialFocalAndExtra__storage_check_,
                nodes__SimpleRadialFocalAndExtra__storage_new_best_);
      std::swap(nodes__SimpleRadialPose__storage_check_,
                nodes__SimpleRadialPose__storage_new_best_);
      std::swap(nodes__SimpleRadialPrincipalPoint__storage_check_,
                nodes__SimpleRadialPrincipalPoint__storage_new_best_);
    }

    const double diag_current = diag;
    bool step_accepted = false;
    if (score_best_pcg < score_best * params_.solver_rel_decrease_min) {
      quality = (score_best - score_best_pcg) / GetPredDecrease();
      const double quality_tmp = 2 * quality - 1;
      double scale = std::max(params_.diag_scaling_down,
                              1.0f - quality_tmp * quality_tmp * quality_tmp);
      diag = std::max(params_.diag_min, diag * scale);
      cudaMemcpy(
          solver__current_diag_, &diag, sizeof(double), cudaMemcpyHostToDevice);
      up_scale = params_.diag_scaling_up;
      score_best = score_best_pcg;
      std::swap(nodes__PinholeCalib__storage_current_,
                nodes__PinholeCalib__storage_new_best_);
      std::swap(nodes__PinholeFocalAndExtra__storage_current_,
                nodes__PinholeFocalAndExtra__storage_new_best_);
      std::swap(nodes__PinholePose__storage_current_,
                nodes__PinholePose__storage_new_best_);
      std::swap(nodes__PinholePrincipalPoint__storage_current_,
                nodes__PinholePrincipalPoint__storage_new_best_);
      std::swap(nodes__Point__storage_current_,
                nodes__Point__storage_new_best_);
      std::swap(nodes__SimpleRadialCalib__storage_current_,
                nodes__SimpleRadialCalib__storage_new_best_);
      std::swap(nodes__SimpleRadialFocalAndExtra__storage_current_,
                nodes__SimpleRadialFocalAndExtra__storage_new_best_);
      std::swap(nodes__SimpleRadialPose__storage_current_,
                nodes__SimpleRadialPose__storage_new_best_);
      std::swap(nodes__SimpleRadialPrincipalPoint__storage_current_,
                nodes__SimpleRadialPrincipalPoint__storage_new_best_);

    } else {
      quality = 0.0f;
      diag = diag * up_scale;
      if (diag > params_.diag_exit_value) {
        result.exit_reason = ExitReason::CONVERGED_DIAG_EXIT;
        break;
      }
      cudaMemcpy(
          solver__current_diag_, &diag, sizeof(double), cudaMemcpyHostToDevice);
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

double GraphSolver::DoResJacFirst() {
  Zero(solver__res_tot_, solver__res_tot_ + 1);
  Zero(marker__r_0_start_, marker__precond_end_);

  SimpleRadialMergedResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_merged__args__pose__idx_shared_,
      nodes__SimpleRadialCalib__storage_current_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_merged__args__calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_merged__args__point__idx_shared_,
      facs__simple_radial_merged__args__pixel__data_,
      simple_radial_merged_num_max_,

      facs__simple_radial_merged__res_,
      simple_radial_merged_num_,
      solver__res_tot_,
      facs__simple_radial_merged__args__pose__jac_,
      simple_radial_merged_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_merged__args__calib__jac_,
      simple_radial_merged_num_,
      nodes__SimpleRadialCalib__r_k_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_diag_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_tril_,
      SimpleRadialCalib_num_,
      facs__simple_radial_merged__args__point__jac_,
      simple_radial_merged_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_merged_num_);

  SimpleRadialMergedFixedPoseResJacFirst(
      nodes__SimpleRadialCalib__storage_current_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_merged_fixed_pose__args__calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_merged_fixed_pose__args__point__idx_shared_,
      facs__simple_radial_merged_fixed_pose__args__pixel__data_,
      simple_radial_merged_fixed_pose_num_max_,
      facs__simple_radial_merged_fixed_pose__args__pose__data_,
      simple_radial_merged_fixed_pose_num_max_,

      facs__simple_radial_merged_fixed_pose__res_,
      simple_radial_merged_fixed_pose_num_,
      solver__res_tot_,
      facs__simple_radial_merged_fixed_pose__args__calib__jac_,
      simple_radial_merged_fixed_pose_num_,
      nodes__SimpleRadialCalib__r_k_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_diag_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_tril_,
      SimpleRadialCalib_num_,
      facs__simple_radial_merged_fixed_pose__args__point__jac_,
      simple_radial_merged_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_merged_fixed_pose_num_);

  SimpleRadialMergedFixedPointResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_merged_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialCalib__storage_current_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_merged_fixed_point__args__calib__idx_shared_,
      facs__simple_radial_merged_fixed_point__args__pixel__data_,
      simple_radial_merged_fixed_point_num_max_,
      facs__simple_radial_merged_fixed_point__args__point__data_,
      simple_radial_merged_fixed_point_num_max_,

      facs__simple_radial_merged_fixed_point__res_,
      simple_radial_merged_fixed_point_num_,
      solver__res_tot_,
      facs__simple_radial_merged_fixed_point__args__pose__jac_,
      simple_radial_merged_fixed_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_merged_fixed_point__args__calib__jac_,
      simple_radial_merged_fixed_point_num_,
      nodes__SimpleRadialCalib__r_k_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_diag_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_tril_,
      SimpleRadialCalib_num_,
      simple_radial_merged_fixed_point_num_);

  SimpleRadialMergedFixedPoseFixedPointResJacFirst(
      nodes__SimpleRadialCalib__storage_current_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__calib__idx_shared_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__point__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,

      facs__simple_radial_merged_fixed_pose_fixed_point__res_,
      simple_radial_merged_fixed_pose_fixed_point_num_,
      solver__res_tot_,
      nodes__SimpleRadialCalib__r_k_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_diag_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_tril_,
      SimpleRadialCalib_num_,
      simple_radial_merged_fixed_pose_fixed_point_num_);

  PinholeMergedResJacFirst(nodes__PinholePose__storage_current_,
                           PinholePose_num_max_,
                           facs__pinhole_merged__args__pose__idx_shared_,
                           nodes__PinholeCalib__storage_current_,
                           PinholeCalib_num_max_,
                           facs__pinhole_merged__args__calib__idx_shared_,
                           nodes__Point__storage_current_,
                           Point_num_max_,
                           facs__pinhole_merged__args__point__idx_shared_,
                           facs__pinhole_merged__args__pixel__data_,
                           pinhole_merged_num_max_,

                           facs__pinhole_merged__res_,
                           pinhole_merged_num_,
                           solver__res_tot_,
                           facs__pinhole_merged__args__pose__jac_,
                           pinhole_merged_num_,
                           nodes__PinholePose__r_k_,
                           PinholePose_num_,
                           nodes__PinholePose__precond_diag_,
                           PinholePose_num_,
                           nodes__PinholePose__precond_tril_,
                           PinholePose_num_,
                           facs__pinhole_merged__args__calib__jac_,
                           pinhole_merged_num_,
                           nodes__PinholeCalib__r_k_,
                           PinholeCalib_num_,
                           nodes__PinholeCalib__precond_diag_,
                           PinholeCalib_num_,
                           nodes__PinholeCalib__precond_tril_,
                           PinholeCalib_num_,
                           facs__pinhole_merged__args__point__jac_,
                           pinhole_merged_num_,
                           nodes__Point__r_k_,
                           Point_num_,
                           nodes__Point__precond_diag_,
                           Point_num_,
                           nodes__Point__precond_tril_,
                           Point_num_,
                           pinhole_merged_num_);

  PinholeMergedFixedPoseResJacFirst(
      nodes__PinholeCalib__storage_current_,
      PinholeCalib_num_max_,
      facs__pinhole_merged_fixed_pose__args__calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_merged_fixed_pose__args__point__idx_shared_,
      facs__pinhole_merged_fixed_pose__args__pixel__data_,
      pinhole_merged_fixed_pose_num_max_,
      facs__pinhole_merged_fixed_pose__args__pose__data_,
      pinhole_merged_fixed_pose_num_max_,

      facs__pinhole_merged_fixed_pose__res_,
      pinhole_merged_fixed_pose_num_,
      solver__res_tot_,
      facs__pinhole_merged_fixed_pose__args__calib__jac_,
      pinhole_merged_fixed_pose_num_,
      nodes__PinholeCalib__r_k_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_diag_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_tril_,
      PinholeCalib_num_,
      facs__pinhole_merged_fixed_pose__args__point__jac_,
      pinhole_merged_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_merged_fixed_pose_num_);

  PinholeMergedFixedPointResJacFirst(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_merged_fixed_point__args__pose__idx_shared_,
      nodes__PinholeCalib__storage_current_,
      PinholeCalib_num_max_,
      facs__pinhole_merged_fixed_point__args__calib__idx_shared_,
      facs__pinhole_merged_fixed_point__args__pixel__data_,
      pinhole_merged_fixed_point_num_max_,
      facs__pinhole_merged_fixed_point__args__point__data_,
      pinhole_merged_fixed_point_num_max_,

      facs__pinhole_merged_fixed_point__res_,
      pinhole_merged_fixed_point_num_,
      solver__res_tot_,
      facs__pinhole_merged_fixed_point__args__pose__jac_,
      pinhole_merged_fixed_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_merged_fixed_point__args__calib__jac_,
      pinhole_merged_fixed_point_num_,
      nodes__PinholeCalib__r_k_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_diag_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_tril_,
      PinholeCalib_num_,
      pinhole_merged_fixed_point_num_);

  PinholeMergedFixedPoseFixedPointResJacFirst(
      nodes__PinholeCalib__storage_current_,
      PinholeCalib_num_max_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__calib__idx_shared_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__pose__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__point__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,

      facs__pinhole_merged_fixed_pose_fixed_point__res_,
      pinhole_merged_fixed_pose_fixed_point_num_,
      solver__res_tot_,
      nodes__PinholeCalib__r_k_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_diag_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_tril_,
      PinholeCalib_num_,
      pinhole_merged_fixed_pose_fixed_point_num_);

  SimpleRadialFixedFocalAndExtraResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_focal_and_extra__args__pose__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_num_max_,
      facs__simple_radial_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_num_max_,

      facs__simple_radial_fixed_focal_and_extra__res_,
      simple_radial_fixed_focal_and_extra_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_focal_and_extra__args__pose__jac_,
      simple_radial_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_focal_and_extra__args__principal_point__jac_,
      simple_radial_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_fixed_focal_and_extra__args__point__jac_,
      simple_radial_fixed_focal_and_extra_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_focal_and_extra_num_);

  SimpleRadialFixedPrincipalPointResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_principal_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_principal_point_num_max_,

      facs__simple_radial_fixed_principal_point__res_,
      simple_radial_fixed_principal_point_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_principal_point__args__pose__jac_,
      simple_radial_fixed_principal_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_principal_point__args__focal_and_extra__jac_,
      simple_radial_fixed_principal_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_principal_point__args__point__jac_,
      simple_radial_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_principal_point_num_);

  SimpleRadialFixedPoseFixedFocalAndExtraResJacFirst(
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,

      facs__simple_radial_fixed_pose_fixed_focal_and_extra__res_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__point__jac_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_);

  SimpleRadialFixedPoseFixedPrincipalPointResJacFirst(
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_principal_point__res_,
      simple_radial_fixed_pose_fixed_principal_point_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_,
      simple_radial_fixed_pose_fixed_principal_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__point__jac_,
      simple_radial_fixed_pose_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_fixed_principal_point_num_);

  SimpleRadialFixedFocalAndExtraFixedPrincipalPointResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,

      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__res_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_);

  SimpleRadialFixedFocalAndExtraFixedPointResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,

      facs__simple_radial_fixed_focal_and_extra_fixed_point__res_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__jac_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_);

  SimpleRadialFixedPrincipalPointFixedPointResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,

      facs__simple_radial_fixed_principal_point_fixed_point__res_,
      simple_radial_fixed_principal_point_fixed_point_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pose__jac_,
      simple_radial_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      simple_radial_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      simple_radial_fixed_principal_point_fixed_point_num_);

  SimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointResJacFirst(
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__res_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      solver__res_tot_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_);

  SimpleRadialFixedPoseFixedFocalAndExtraFixedPointResJacFirst(
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__res_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      solver__res_tot_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_);

  SimpleRadialFixedPoseFixedPrincipalPointFixedPointResJacFirst(
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__res_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
      solver__res_tot_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_);

  SimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,

      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__res_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      solver__res_tot_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_);

  PinholeFixedFocalAndExtraResJacFirst(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_fixed_focal_and_extra__args__pose__idx_shared_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_focal_and_extra__args__point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra__args__pixel__data_,
      pinhole_fixed_focal_and_extra_num_max_,
      facs__pinhole_fixed_focal_and_extra__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_num_max_,

      facs__pinhole_fixed_focal_and_extra__res_,
      pinhole_fixed_focal_and_extra_num_,
      solver__res_tot_,
      facs__pinhole_fixed_focal_and_extra__args__pose__jac_,
      pinhole_fixed_focal_and_extra_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_fixed_focal_and_extra__args__principal_point__jac_,
      pinhole_fixed_focal_and_extra_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      facs__pinhole_fixed_focal_and_extra__args__point__jac_,
      pinhole_fixed_focal_and_extra_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_focal_and_extra_num_);

  PinholeFixedPrincipalPointResJacFirst(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_fixed_principal_point__args__pose__idx_shared_,
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_principal_point_num_max_,
      facs__pinhole_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_principal_point_num_max_,

      facs__pinhole_fixed_principal_point__res_,
      pinhole_fixed_principal_point_num_,
      solver__res_tot_,
      facs__pinhole_fixed_principal_point__args__pose__jac_,
      pinhole_fixed_principal_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_fixed_principal_point__args__focal_and_extra__jac_,
      pinhole_fixed_principal_point_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_principal_point__args__point__jac_,
      pinhole_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_principal_point_num_);

  PinholeFixedPoseFixedFocalAndExtraResJacFirst(
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,

      facs__pinhole_fixed_pose_fixed_focal_and_extra__res_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_,
      solver__res_tot_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__point__jac_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_);

  PinholeFixedPoseFixedPrincipalPointResJacFirst(
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__pose__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,

      facs__pinhole_fixed_pose_fixed_principal_point__res_,
      pinhole_fixed_pose_fixed_principal_point_num_,
      solver__res_tot_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_,
      pinhole_fixed_pose_fixed_principal_point_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__point__jac_,
      pinhole_fixed_pose_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_fixed_principal_point_num_);

  PinholeFixedFocalAndExtraFixedPrincipalPointResJacFirst(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,

      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__res_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
      solver__res_tot_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_);

  PinholeFixedFocalAndExtraFixedPointResJacFirst(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__point__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,

      facs__pinhole_fixed_focal_and_extra_fixed_point__res_,
      pinhole_fixed_focal_and_extra_fixed_point_num_,
      solver__res_tot_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__jac_,
      pinhole_fixed_focal_and_extra_fixed_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      pinhole_fixed_focal_and_extra_fixed_point_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      pinhole_fixed_focal_and_extra_fixed_point_num_);

  PinholeFixedPrincipalPointFixedPointResJacFirst(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__pinhole_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,

      facs__pinhole_fixed_principal_point_fixed_point__res_,
      pinhole_fixed_principal_point_fixed_point_num_,
      solver__res_tot_,
      facs__pinhole_fixed_principal_point_fixed_point__args__pose__jac_,
      pinhole_fixed_principal_point_fixed_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      pinhole_fixed_principal_point_fixed_point_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      pinhole_fixed_principal_point_fixed_point_num_);

  PinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointResJacFirst(
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,

      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__res_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      solver__res_tot_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_);

  PinholeFixedPoseFixedFocalAndExtraFixedPointResJacFirst(
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,

      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__res_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      solver__res_tot_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_);

  PinholeFixedPoseFixedPrincipalPointFixedPointResJacFirst(
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,

      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__res_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_,
      solver__res_tot_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_);

  PinholeFixedFocalAndExtraFixedPrincipalPointFixedPointResJacFirst(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,

      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__res_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      solver__res_tot_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__r_0_start_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
  return 0.5 * ReadCuMem(solver__res_tot_);
}
void GraphSolver::DoResJac() {
  Zero(solver__res_tot_, solver__res_tot_ + 1);
  Zero(marker__r_0_start_, marker__precond_end_);

  SimpleRadialMergedResJac(nodes__SimpleRadialPose__storage_current_,
                           SimpleRadialPose_num_max_,
                           facs__simple_radial_merged__args__pose__idx_shared_,
                           nodes__SimpleRadialCalib__storage_current_,
                           SimpleRadialCalib_num_max_,
                           facs__simple_radial_merged__args__calib__idx_shared_,
                           nodes__Point__storage_current_,
                           Point_num_max_,
                           facs__simple_radial_merged__args__point__idx_shared_,
                           facs__simple_radial_merged__args__pixel__data_,
                           simple_radial_merged_num_max_,

                           facs__simple_radial_merged__res_,
                           simple_radial_merged_num_,

                           facs__simple_radial_merged__args__pose__jac_,
                           simple_radial_merged_num_,
                           nodes__SimpleRadialPose__r_k_,
                           SimpleRadialPose_num_,
                           nodes__SimpleRadialPose__precond_diag_,
                           SimpleRadialPose_num_,
                           nodes__SimpleRadialPose__precond_tril_,
                           SimpleRadialPose_num_,
                           facs__simple_radial_merged__args__calib__jac_,
                           simple_radial_merged_num_,
                           nodes__SimpleRadialCalib__r_k_,
                           SimpleRadialCalib_num_,
                           nodes__SimpleRadialCalib__precond_diag_,
                           SimpleRadialCalib_num_,
                           nodes__SimpleRadialCalib__precond_tril_,
                           SimpleRadialCalib_num_,
                           facs__simple_radial_merged__args__point__jac_,
                           simple_radial_merged_num_,
                           nodes__Point__r_k_,
                           Point_num_,
                           nodes__Point__precond_diag_,
                           Point_num_,
                           nodes__Point__precond_tril_,
                           Point_num_,
                           simple_radial_merged_num_);

  SimpleRadialMergedFixedPoseResJac(
      nodes__SimpleRadialCalib__storage_current_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_merged_fixed_pose__args__calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_merged_fixed_pose__args__point__idx_shared_,
      facs__simple_radial_merged_fixed_pose__args__pixel__data_,
      simple_radial_merged_fixed_pose_num_max_,
      facs__simple_radial_merged_fixed_pose__args__pose__data_,
      simple_radial_merged_fixed_pose_num_max_,

      facs__simple_radial_merged_fixed_pose__res_,
      simple_radial_merged_fixed_pose_num_,

      facs__simple_radial_merged_fixed_pose__args__calib__jac_,
      simple_radial_merged_fixed_pose_num_,
      nodes__SimpleRadialCalib__r_k_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_diag_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_tril_,
      SimpleRadialCalib_num_,
      facs__simple_radial_merged_fixed_pose__args__point__jac_,
      simple_radial_merged_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_merged_fixed_pose_num_);

  SimpleRadialMergedFixedPointResJac(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_merged_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialCalib__storage_current_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_merged_fixed_point__args__calib__idx_shared_,
      facs__simple_radial_merged_fixed_point__args__pixel__data_,
      simple_radial_merged_fixed_point_num_max_,
      facs__simple_radial_merged_fixed_point__args__point__data_,
      simple_radial_merged_fixed_point_num_max_,

      facs__simple_radial_merged_fixed_point__res_,
      simple_radial_merged_fixed_point_num_,

      facs__simple_radial_merged_fixed_point__args__pose__jac_,
      simple_radial_merged_fixed_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_merged_fixed_point__args__calib__jac_,
      simple_radial_merged_fixed_point_num_,
      nodes__SimpleRadialCalib__r_k_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_diag_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_tril_,
      SimpleRadialCalib_num_,
      simple_radial_merged_fixed_point_num_);

  SimpleRadialMergedFixedPoseFixedPointResJac(
      nodes__SimpleRadialCalib__storage_current_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__calib__idx_shared_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__point__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,

      facs__simple_radial_merged_fixed_pose_fixed_point__res_,
      simple_radial_merged_fixed_pose_fixed_point_num_,

      nodes__SimpleRadialCalib__r_k_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_diag_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_tril_,
      SimpleRadialCalib_num_,
      simple_radial_merged_fixed_pose_fixed_point_num_);

  PinholeMergedResJac(nodes__PinholePose__storage_current_,
                      PinholePose_num_max_,
                      facs__pinhole_merged__args__pose__idx_shared_,
                      nodes__PinholeCalib__storage_current_,
                      PinholeCalib_num_max_,
                      facs__pinhole_merged__args__calib__idx_shared_,
                      nodes__Point__storage_current_,
                      Point_num_max_,
                      facs__pinhole_merged__args__point__idx_shared_,
                      facs__pinhole_merged__args__pixel__data_,
                      pinhole_merged_num_max_,

                      facs__pinhole_merged__res_,
                      pinhole_merged_num_,

                      facs__pinhole_merged__args__pose__jac_,
                      pinhole_merged_num_,
                      nodes__PinholePose__r_k_,
                      PinholePose_num_,
                      nodes__PinholePose__precond_diag_,
                      PinholePose_num_,
                      nodes__PinholePose__precond_tril_,
                      PinholePose_num_,
                      facs__pinhole_merged__args__calib__jac_,
                      pinhole_merged_num_,
                      nodes__PinholeCalib__r_k_,
                      PinholeCalib_num_,
                      nodes__PinholeCalib__precond_diag_,
                      PinholeCalib_num_,
                      nodes__PinholeCalib__precond_tril_,
                      PinholeCalib_num_,
                      facs__pinhole_merged__args__point__jac_,
                      pinhole_merged_num_,
                      nodes__Point__r_k_,
                      Point_num_,
                      nodes__Point__precond_diag_,
                      Point_num_,
                      nodes__Point__precond_tril_,
                      Point_num_,
                      pinhole_merged_num_);

  PinholeMergedFixedPoseResJac(
      nodes__PinholeCalib__storage_current_,
      PinholeCalib_num_max_,
      facs__pinhole_merged_fixed_pose__args__calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_merged_fixed_pose__args__point__idx_shared_,
      facs__pinhole_merged_fixed_pose__args__pixel__data_,
      pinhole_merged_fixed_pose_num_max_,
      facs__pinhole_merged_fixed_pose__args__pose__data_,
      pinhole_merged_fixed_pose_num_max_,

      facs__pinhole_merged_fixed_pose__res_,
      pinhole_merged_fixed_pose_num_,

      facs__pinhole_merged_fixed_pose__args__calib__jac_,
      pinhole_merged_fixed_pose_num_,
      nodes__PinholeCalib__r_k_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_diag_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_tril_,
      PinholeCalib_num_,
      facs__pinhole_merged_fixed_pose__args__point__jac_,
      pinhole_merged_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_merged_fixed_pose_num_);

  PinholeMergedFixedPointResJac(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_merged_fixed_point__args__pose__idx_shared_,
      nodes__PinholeCalib__storage_current_,
      PinholeCalib_num_max_,
      facs__pinhole_merged_fixed_point__args__calib__idx_shared_,
      facs__pinhole_merged_fixed_point__args__pixel__data_,
      pinhole_merged_fixed_point_num_max_,
      facs__pinhole_merged_fixed_point__args__point__data_,
      pinhole_merged_fixed_point_num_max_,

      facs__pinhole_merged_fixed_point__res_,
      pinhole_merged_fixed_point_num_,

      facs__pinhole_merged_fixed_point__args__pose__jac_,
      pinhole_merged_fixed_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_merged_fixed_point__args__calib__jac_,
      pinhole_merged_fixed_point_num_,
      nodes__PinholeCalib__r_k_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_diag_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_tril_,
      PinholeCalib_num_,
      pinhole_merged_fixed_point_num_);

  PinholeMergedFixedPoseFixedPointResJac(
      nodes__PinholeCalib__storage_current_,
      PinholeCalib_num_max_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__calib__idx_shared_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__pose__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__point__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,

      facs__pinhole_merged_fixed_pose_fixed_point__res_,
      pinhole_merged_fixed_pose_fixed_point_num_,

      nodes__PinholeCalib__r_k_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_diag_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_tril_,
      PinholeCalib_num_,
      pinhole_merged_fixed_pose_fixed_point_num_);

  SimpleRadialFixedFocalAndExtraResJac(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_focal_and_extra__args__pose__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_num_max_,
      facs__simple_radial_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_num_max_,

      facs__simple_radial_fixed_focal_and_extra__res_,
      simple_radial_fixed_focal_and_extra_num_,

      facs__simple_radial_fixed_focal_and_extra__args__pose__jac_,
      simple_radial_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_focal_and_extra__args__principal_point__jac_,
      simple_radial_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_fixed_focal_and_extra__args__point__jac_,
      simple_radial_fixed_focal_and_extra_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_focal_and_extra_num_);

  SimpleRadialFixedPrincipalPointResJac(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_principal_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_principal_point_num_max_,

      facs__simple_radial_fixed_principal_point__res_,
      simple_radial_fixed_principal_point_num_,

      facs__simple_radial_fixed_principal_point__args__pose__jac_,
      simple_radial_fixed_principal_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_principal_point__args__focal_and_extra__jac_,
      simple_radial_fixed_principal_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_principal_point__args__point__jac_,
      simple_radial_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_principal_point_num_);

  SimpleRadialFixedPoseFixedFocalAndExtraResJac(
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,

      facs__simple_radial_fixed_pose_fixed_focal_and_extra__res_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_,

      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__point__jac_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_);

  SimpleRadialFixedPoseFixedPrincipalPointResJac(
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_principal_point__res_,
      simple_radial_fixed_pose_fixed_principal_point_num_,

      facs__simple_radial_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_,
      simple_radial_fixed_pose_fixed_principal_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__point__jac_,
      simple_radial_fixed_pose_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_fixed_principal_point_num_);

  SimpleRadialFixedFocalAndExtraFixedPrincipalPointResJac(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,

      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__res_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,

      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_);

  SimpleRadialFixedFocalAndExtraFixedPointResJac(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,

      facs__simple_radial_fixed_focal_and_extra_fixed_point__res_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_,

      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__jac_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_);

  SimpleRadialFixedPrincipalPointFixedPointResJac(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,

      facs__simple_radial_fixed_principal_point_fixed_point__res_,
      simple_radial_fixed_principal_point_fixed_point_num_,

      facs__simple_radial_fixed_principal_point_fixed_point__args__pose__jac_,
      simple_radial_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      simple_radial_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      simple_radial_fixed_principal_point_fixed_point_num_);

  SimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointResJac(
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__res_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,

      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_);

  SimpleRadialFixedPoseFixedFocalAndExtraFixedPointResJac(
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__res_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,

      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_);

  SimpleRadialFixedPoseFixedPrincipalPointFixedPointResJac(
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__res_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,

      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_);

  SimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointResJac(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,

      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__res_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,

      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_);

  PinholeFixedFocalAndExtraResJac(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_fixed_focal_and_extra__args__pose__idx_shared_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_focal_and_extra__args__point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra__args__pixel__data_,
      pinhole_fixed_focal_and_extra_num_max_,
      facs__pinhole_fixed_focal_and_extra__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_num_max_,

      facs__pinhole_fixed_focal_and_extra__res_,
      pinhole_fixed_focal_and_extra_num_,

      facs__pinhole_fixed_focal_and_extra__args__pose__jac_,
      pinhole_fixed_focal_and_extra_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_fixed_focal_and_extra__args__principal_point__jac_,
      pinhole_fixed_focal_and_extra_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      facs__pinhole_fixed_focal_and_extra__args__point__jac_,
      pinhole_fixed_focal_and_extra_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_focal_and_extra_num_);

  PinholeFixedPrincipalPointResJac(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_fixed_principal_point__args__pose__idx_shared_,
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_principal_point_num_max_,
      facs__pinhole_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_principal_point_num_max_,

      facs__pinhole_fixed_principal_point__res_,
      pinhole_fixed_principal_point_num_,

      facs__pinhole_fixed_principal_point__args__pose__jac_,
      pinhole_fixed_principal_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_fixed_principal_point__args__focal_and_extra__jac_,
      pinhole_fixed_principal_point_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_principal_point__args__point__jac_,
      pinhole_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_principal_point_num_);

  PinholeFixedPoseFixedFocalAndExtraResJac(
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,

      facs__pinhole_fixed_pose_fixed_focal_and_extra__res_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_,

      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__point__jac_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_);

  PinholeFixedPoseFixedPrincipalPointResJac(
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__pose__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,

      facs__pinhole_fixed_pose_fixed_principal_point__res_,
      pinhole_fixed_pose_fixed_principal_point_num_,

      facs__pinhole_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_,
      pinhole_fixed_pose_fixed_principal_point_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__point__jac_,
      pinhole_fixed_pose_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_fixed_principal_point_num_);

  PinholeFixedFocalAndExtraFixedPrincipalPointResJac(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,

      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__res_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_,

      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_);

  PinholeFixedFocalAndExtraFixedPointResJac(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__point__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,

      facs__pinhole_fixed_focal_and_extra_fixed_point__res_,
      pinhole_fixed_focal_and_extra_fixed_point_num_,

      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__jac_,
      pinhole_fixed_focal_and_extra_fixed_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      pinhole_fixed_focal_and_extra_fixed_point_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      pinhole_fixed_focal_and_extra_fixed_point_num_);

  PinholeFixedPrincipalPointFixedPointResJac(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__pinhole_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,

      facs__pinhole_fixed_principal_point_fixed_point__res_,
      pinhole_fixed_principal_point_fixed_point_num_,

      facs__pinhole_fixed_principal_point_fixed_point__args__pose__jac_,
      pinhole_fixed_principal_point_fixed_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      pinhole_fixed_principal_point_fixed_point_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      pinhole_fixed_principal_point_fixed_point_num_);

  PinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointResJac(
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,

      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__res_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,

      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_);

  PinholeFixedPoseFixedFocalAndExtraFixedPointResJac(
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,

      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__res_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_,

      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_);

  PinholeFixedPoseFixedPrincipalPointFixedPointResJac(
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,

      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__res_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_,

      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_);

  PinholeFixedFocalAndExtraFixedPrincipalPointFixedPointResJac(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,

      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__res_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,

      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__r_0_start_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
}

void GraphSolver::DoNormalize() {
  double* r_k;
  double* z;
  z = pcg_iter_ == 0 ? nodes__PinholeCalib__p_ : nodes__PinholeCalib__z_;
  PinholeCalibNormalize(nodes__PinholeCalib__precond_diag_,
                        PinholeCalib_num_,
                        nodes__PinholeCalib__precond_tril_,
                        PinholeCalib_num_,
                        nodes__PinholeCalib__r_k_,
                        PinholeCalib_num_,
                        solver__current_diag_,
                        z,
                        PinholeCalib_num_,
                        PinholeCalib_num_);
  z = pcg_iter_ == 0 ? nodes__PinholeFocalAndExtra__p_
                     : nodes__PinholeFocalAndExtra__z_;
  PinholeFocalAndExtraNormalize(nodes__PinholeFocalAndExtra__precond_diag_,
                                PinholeFocalAndExtra_num_,
                                nodes__PinholeFocalAndExtra__precond_tril_,
                                PinholeFocalAndExtra_num_,
                                nodes__PinholeFocalAndExtra__r_k_,
                                PinholeFocalAndExtra_num_,
                                solver__current_diag_,
                                z,
                                PinholeFocalAndExtra_num_,
                                PinholeFocalAndExtra_num_);
  z = pcg_iter_ == 0 ? nodes__PinholePose__p_ : nodes__PinholePose__z_;
  PinholePoseNormalize(nodes__PinholePose__precond_diag_,
                       PinholePose_num_,
                       nodes__PinholePose__precond_tril_,
                       PinholePose_num_,
                       nodes__PinholePose__r_k_,
                       PinholePose_num_,
                       solver__current_diag_,
                       z,
                       PinholePose_num_,
                       PinholePose_num_);
  z = pcg_iter_ == 0 ? nodes__PinholePrincipalPoint__p_
                     : nodes__PinholePrincipalPoint__z_;
  PinholePrincipalPointNormalize(nodes__PinholePrincipalPoint__precond_diag_,
                                 PinholePrincipalPoint_num_,
                                 nodes__PinholePrincipalPoint__precond_tril_,
                                 PinholePrincipalPoint_num_,
                                 nodes__PinholePrincipalPoint__r_k_,
                                 PinholePrincipalPoint_num_,
                                 solver__current_diag_,
                                 z,
                                 PinholePrincipalPoint_num_,
                                 PinholePrincipalPoint_num_);
  z = pcg_iter_ == 0 ? nodes__Point__p_ : nodes__Point__z_;
  PointNormalize(nodes__Point__precond_diag_,
                 Point_num_,
                 nodes__Point__precond_tril_,
                 Point_num_,
                 nodes__Point__r_k_,
                 Point_num_,
                 solver__current_diag_,
                 z,
                 Point_num_,
                 Point_num_);
  z = pcg_iter_ == 0 ? nodes__SimpleRadialCalib__p_
                     : nodes__SimpleRadialCalib__z_;
  SimpleRadialCalibNormalize(nodes__SimpleRadialCalib__precond_diag_,
                             SimpleRadialCalib_num_,
                             nodes__SimpleRadialCalib__precond_tril_,
                             SimpleRadialCalib_num_,
                             nodes__SimpleRadialCalib__r_k_,
                             SimpleRadialCalib_num_,
                             solver__current_diag_,
                             z,
                             SimpleRadialCalib_num_,
                             SimpleRadialCalib_num_);
  z = pcg_iter_ == 0 ? nodes__SimpleRadialFocalAndExtra__p_
                     : nodes__SimpleRadialFocalAndExtra__z_;
  SimpleRadialFocalAndExtraNormalize(
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      solver__current_diag_,
      z,
      SimpleRadialFocalAndExtra_num_,
      SimpleRadialFocalAndExtra_num_);
  z = pcg_iter_ == 0 ? nodes__SimpleRadialPose__p_
                     : nodes__SimpleRadialPose__z_;
  SimpleRadialPoseNormalize(nodes__SimpleRadialPose__precond_diag_,
                            SimpleRadialPose_num_,
                            nodes__SimpleRadialPose__precond_tril_,
                            SimpleRadialPose_num_,
                            nodes__SimpleRadialPose__r_k_,
                            SimpleRadialPose_num_,
                            solver__current_diag_,
                            z,
                            SimpleRadialPose_num_,
                            SimpleRadialPose_num_);
  z = pcg_iter_ == 0 ? nodes__SimpleRadialPrincipalPoint__p_
                     : nodes__SimpleRadialPrincipalPoint__z_;
  SimpleRadialPrincipalPointNormalize(
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      solver__current_diag_,
      z,
      SimpleRadialPrincipalPoint_num_,
      SimpleRadialPrincipalPoint_num_);
}

void GraphSolver::DoUpdateMp() {
  PinholeCalibUpdateMp(nodes__PinholeCalib__r_k_,
                       PinholeCalib_num_,
                       nodes__PinholeCalib__Mp_,
                       PinholeCalib_num_,
                       solver__beta_,
                       nodes__PinholeCalib__Mp_,
                       PinholeCalib_num_,
                       nodes__PinholeCalib__w_,
                       PinholeCalib_num_,
                       PinholeCalib_num_);
  PinholeFocalAndExtraUpdateMp(nodes__PinholeFocalAndExtra__r_k_,
                               PinholeFocalAndExtra_num_,
                               nodes__PinholeFocalAndExtra__Mp_,
                               PinholeFocalAndExtra_num_,
                               solver__beta_,
                               nodes__PinholeFocalAndExtra__Mp_,
                               PinholeFocalAndExtra_num_,
                               nodes__PinholeFocalAndExtra__w_,
                               PinholeFocalAndExtra_num_,
                               PinholeFocalAndExtra_num_);
  PinholePoseUpdateMp(nodes__PinholePose__r_k_,
                      PinholePose_num_,
                      nodes__PinholePose__Mp_,
                      PinholePose_num_,
                      solver__beta_,
                      nodes__PinholePose__Mp_,
                      PinholePose_num_,
                      nodes__PinholePose__w_,
                      PinholePose_num_,
                      PinholePose_num_);
  PinholePrincipalPointUpdateMp(nodes__PinholePrincipalPoint__r_k_,
                                PinholePrincipalPoint_num_,
                                nodes__PinholePrincipalPoint__Mp_,
                                PinholePrincipalPoint_num_,
                                solver__beta_,
                                nodes__PinholePrincipalPoint__Mp_,
                                PinholePrincipalPoint_num_,
                                nodes__PinholePrincipalPoint__w_,
                                PinholePrincipalPoint_num_,
                                PinholePrincipalPoint_num_);
  PointUpdateMp(nodes__Point__r_k_,
                Point_num_,
                nodes__Point__Mp_,
                Point_num_,
                solver__beta_,
                nodes__Point__Mp_,
                Point_num_,
                nodes__Point__w_,
                Point_num_,
                Point_num_);
  SimpleRadialCalibUpdateMp(nodes__SimpleRadialCalib__r_k_,
                            SimpleRadialCalib_num_,
                            nodes__SimpleRadialCalib__Mp_,
                            SimpleRadialCalib_num_,
                            solver__beta_,
                            nodes__SimpleRadialCalib__Mp_,
                            SimpleRadialCalib_num_,
                            nodes__SimpleRadialCalib__w_,
                            SimpleRadialCalib_num_,
                            SimpleRadialCalib_num_);
  SimpleRadialFocalAndExtraUpdateMp(nodes__SimpleRadialFocalAndExtra__r_k_,
                                    SimpleRadialFocalAndExtra_num_,
                                    nodes__SimpleRadialFocalAndExtra__Mp_,
                                    SimpleRadialFocalAndExtra_num_,
                                    solver__beta_,
                                    nodes__SimpleRadialFocalAndExtra__Mp_,
                                    SimpleRadialFocalAndExtra_num_,
                                    nodes__SimpleRadialFocalAndExtra__w_,
                                    SimpleRadialFocalAndExtra_num_,
                                    SimpleRadialFocalAndExtra_num_);
  SimpleRadialPoseUpdateMp(nodes__SimpleRadialPose__r_k_,
                           SimpleRadialPose_num_,
                           nodes__SimpleRadialPose__Mp_,
                           SimpleRadialPose_num_,
                           solver__beta_,
                           nodes__SimpleRadialPose__Mp_,
                           SimpleRadialPose_num_,
                           nodes__SimpleRadialPose__w_,
                           SimpleRadialPose_num_,
                           SimpleRadialPose_num_);
  SimpleRadialPrincipalPointUpdateMp(nodes__SimpleRadialPrincipalPoint__r_k_,
                                     SimpleRadialPrincipalPoint_num_,
                                     nodes__SimpleRadialPrincipalPoint__Mp_,
                                     SimpleRadialPrincipalPoint_num_,
                                     solver__beta_,
                                     nodes__SimpleRadialPrincipalPoint__Mp_,
                                     SimpleRadialPrincipalPoint_num_,
                                     nodes__SimpleRadialPrincipalPoint__w_,
                                     SimpleRadialPrincipalPoint_num_,
                                     SimpleRadialPrincipalPoint_num_);
}

void GraphSolver::DoJtjpDirect() {
  SimpleRadialMergedJtjnjtrDirect(
      nodes__SimpleRadialPose__p_,
      SimpleRadialPose_num_,
      facs__simple_radial_merged__args__pose__idx_shared_,
      facs__simple_radial_merged__args__pose__jac_,
      simple_radial_merged_num_,
      nodes__SimpleRadialCalib__p_,
      SimpleRadialCalib_num_,
      facs__simple_radial_merged__args__calib__idx_shared_,
      facs__simple_radial_merged__args__calib__jac_,
      simple_radial_merged_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_merged__args__point__idx_shared_,
      facs__simple_radial_merged__args__point__jac_,
      simple_radial_merged_num_,
      nodes__SimpleRadialPose__w_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialCalib__w_,
      SimpleRadialCalib_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_merged_num_);
  SimpleRadialMergedFixedPoseJtjnjtrDirect(
      nodes__SimpleRadialCalib__p_,
      SimpleRadialCalib_num_,
      facs__simple_radial_merged_fixed_pose__args__calib__idx_shared_,
      facs__simple_radial_merged_fixed_pose__args__calib__jac_,
      simple_radial_merged_fixed_pose_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_merged_fixed_pose__args__point__idx_shared_,
      facs__simple_radial_merged_fixed_pose__args__point__jac_,
      simple_radial_merged_fixed_pose_num_,
      nodes__SimpleRadialCalib__w_,
      SimpleRadialCalib_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_merged_fixed_pose_num_);
  SimpleRadialMergedFixedPointJtjnjtrDirect(
      nodes__SimpleRadialPose__p_,
      SimpleRadialPose_num_,
      facs__simple_radial_merged_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_merged_fixed_point__args__pose__jac_,
      simple_radial_merged_fixed_point_num_,
      nodes__SimpleRadialCalib__p_,
      SimpleRadialCalib_num_,
      facs__simple_radial_merged_fixed_point__args__calib__idx_shared_,
      facs__simple_radial_merged_fixed_point__args__calib__jac_,
      simple_radial_merged_fixed_point_num_,
      nodes__SimpleRadialPose__w_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialCalib__w_,
      SimpleRadialCalib_num_,
      simple_radial_merged_fixed_point_num_);
  PinholeMergedJtjnjtrDirect(nodes__PinholePose__p_,
                             PinholePose_num_,
                             facs__pinhole_merged__args__pose__idx_shared_,
                             facs__pinhole_merged__args__pose__jac_,
                             pinhole_merged_num_,
                             nodes__PinholeCalib__p_,
                             PinholeCalib_num_,
                             facs__pinhole_merged__args__calib__idx_shared_,
                             facs__pinhole_merged__args__calib__jac_,
                             pinhole_merged_num_,
                             nodes__Point__p_,
                             Point_num_,
                             facs__pinhole_merged__args__point__idx_shared_,
                             facs__pinhole_merged__args__point__jac_,
                             pinhole_merged_num_,
                             nodes__PinholePose__w_,
                             PinholePose_num_,
                             nodes__PinholeCalib__w_,
                             PinholeCalib_num_,
                             nodes__Point__w_,
                             Point_num_,
                             pinhole_merged_num_);
  PinholeMergedFixedPoseJtjnjtrDirect(
      nodes__PinholeCalib__p_,
      PinholeCalib_num_,
      facs__pinhole_merged_fixed_pose__args__calib__idx_shared_,
      facs__pinhole_merged_fixed_pose__args__calib__jac_,
      pinhole_merged_fixed_pose_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_merged_fixed_pose__args__point__idx_shared_,
      facs__pinhole_merged_fixed_pose__args__point__jac_,
      pinhole_merged_fixed_pose_num_,
      nodes__PinholeCalib__w_,
      PinholeCalib_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_merged_fixed_pose_num_);
  PinholeMergedFixedPointJtjnjtrDirect(
      nodes__PinholePose__p_,
      PinholePose_num_,
      facs__pinhole_merged_fixed_point__args__pose__idx_shared_,
      facs__pinhole_merged_fixed_point__args__pose__jac_,
      pinhole_merged_fixed_point_num_,
      nodes__PinholeCalib__p_,
      PinholeCalib_num_,
      facs__pinhole_merged_fixed_point__args__calib__idx_shared_,
      facs__pinhole_merged_fixed_point__args__calib__jac_,
      pinhole_merged_fixed_point_num_,
      nodes__PinholePose__w_,
      PinholePose_num_,
      nodes__PinholeCalib__w_,
      PinholeCalib_num_,
      pinhole_merged_fixed_point_num_);
  SimpleRadialFixedFocalAndExtraJtjnjtrDirect(
      nodes__SimpleRadialPose__p_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_focal_and_extra__args__pose__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra__args__pose__jac_,
      simple_radial_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_fixed_focal_and_extra__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra__args__principal_point__jac_,
      simple_radial_fixed_focal_and_extra_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra__args__point__jac_,
      simple_radial_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPose__w_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_focal_and_extra_num_);
  SimpleRadialFixedPrincipalPointJtjnjtrDirect(
      nodes__SimpleRadialPose__p_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_principal_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_principal_point__args__pose__jac_,
      simple_radial_fixed_principal_point_num_,
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_principal_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_fixed_principal_point__args__focal_and_extra__jac_,
      simple_radial_fixed_principal_point_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_principal_point__args__point__jac_,
      simple_radial_fixed_principal_point_num_,
      nodes__SimpleRadialPose__w_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_principal_point_num_);
  SimpleRadialFixedPoseFixedFocalAndExtraJtjnjtrDirect(
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__point__jac_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_);
  SimpleRadialFixedPoseFixedPrincipalPointJtjnjtrDirect(
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_,
      simple_radial_fixed_pose_fixed_principal_point_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__point__jac_,
      simple_radial_fixed_pose_fixed_principal_point_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_pose_fixed_principal_point_num_);
  SimpleRadialFixedFocalAndExtraFixedPrincipalPointJtjnjtrDirect(
      nodes__SimpleRadialPose__p_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__SimpleRadialPose__w_,
      SimpleRadialPose_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_);
  SimpleRadialFixedFocalAndExtraFixedPointJtjnjtrDirect(
      nodes__SimpleRadialPose__p_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__jac_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPose__w_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_);
  SimpleRadialFixedPrincipalPointFixedPointJtjnjtrDirect(
      nodes__SimpleRadialPose__p_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pose__jac_,
      simple_radial_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      simple_radial_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialPose__w_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      simple_radial_fixed_principal_point_fixed_point_num_);
  PinholeFixedFocalAndExtraJtjnjtrDirect(
      nodes__PinholePose__p_,
      PinholePose_num_,
      facs__pinhole_fixed_focal_and_extra__args__pose__idx_shared_,
      facs__pinhole_fixed_focal_and_extra__args__pose__jac_,
      pinhole_fixed_focal_and_extra_num_,
      nodes__PinholePrincipalPoint__p_,
      PinholePrincipalPoint_num_,
      facs__pinhole_fixed_focal_and_extra__args__principal_point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra__args__principal_point__jac_,
      pinhole_fixed_focal_and_extra_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_fixed_focal_and_extra__args__point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra__args__point__jac_,
      pinhole_fixed_focal_and_extra_num_,
      nodes__PinholePose__w_,
      PinholePose_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_focal_and_extra_num_);
  PinholeFixedPrincipalPointJtjnjtrDirect(
      nodes__PinholePose__p_,
      PinholePose_num_,
      facs__pinhole_fixed_principal_point__args__pose__idx_shared_,
      facs__pinhole_fixed_principal_point__args__pose__jac_,
      pinhole_fixed_principal_point_num_,
      nodes__PinholeFocalAndExtra__p_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_principal_point__args__focal_and_extra__idx_shared_,
      facs__pinhole_fixed_principal_point__args__focal_and_extra__jac_,
      pinhole_fixed_principal_point_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_principal_point__args__point__jac_,
      pinhole_fixed_principal_point_num_,
      nodes__PinholePose__w_,
      PinholePose_num_,
      nodes__PinholeFocalAndExtra__w_,
      PinholeFocalAndExtra_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_principal_point_num_);
  PinholeFixedPoseFixedFocalAndExtraJtjnjtrDirect(
      nodes__PinholePrincipalPoint__p_,
      PinholePrincipalPoint_num_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__point__jac_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_);
  PinholeFixedPoseFixedPrincipalPointJtjnjtrDirect(
      nodes__PinholeFocalAndExtra__p_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_,
      pinhole_fixed_pose_fixed_principal_point_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__point__jac_,
      pinhole_fixed_pose_fixed_principal_point_num_,
      nodes__PinholeFocalAndExtra__w_,
      PinholeFocalAndExtra_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_pose_fixed_principal_point_num_);
  PinholeFixedFocalAndExtraFixedPrincipalPointJtjnjtrDirect(
      nodes__PinholePose__p_,
      PinholePose_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__PinholePose__w_,
      PinholePose_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_);
  PinholeFixedFocalAndExtraFixedPointJtjnjtrDirect(
      nodes__PinholePose__p_,
      PinholePose_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__jac_,
      pinhole_fixed_focal_and_extra_fixed_point_num_,
      nodes__PinholePrincipalPoint__p_,
      PinholePrincipalPoint_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      pinhole_fixed_focal_and_extra_fixed_point_num_,
      nodes__PinholePose__w_,
      PinholePose_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      pinhole_fixed_focal_and_extra_fixed_point_num_);
  PinholeFixedPrincipalPointFixedPointJtjnjtrDirect(
      nodes__PinholePose__p_,
      PinholePose_num_,
      facs__pinhole_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_principal_point_fixed_point__args__pose__jac_,
      pinhole_fixed_principal_point_fixed_point_num_,
      nodes__PinholeFocalAndExtra__p_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      pinhole_fixed_principal_point_fixed_point_num_,
      nodes__PinholePose__w_,
      PinholePose_num_,
      nodes__PinholeFocalAndExtra__w_,
      PinholeFocalAndExtra_num_,
      pinhole_fixed_principal_point_fixed_point_num_);
}

void GraphSolver::DoAlphaFirst() {
  Zero(solver__alpha_numerator_, solver__alpha_denominator_ + 1);
  double* p_kp1;
  double* r_k;
  PinholeCalibAlphaNumeratorDenominator(nodes__PinholeCalib__p_,
                                        PinholeCalib_num_,
                                        nodes__PinholeCalib__r_k_,
                                        PinholeCalib_num_,
                                        nodes__PinholeCalib__w_,
                                        PinholeCalib_num_,
                                        solver__alpha_numerator_,
                                        solver__alpha_denominator_,
                                        PinholeCalib_num_);
  PinholeFocalAndExtraAlphaNumeratorDenominator(
      nodes__PinholeFocalAndExtra__p_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__w_,
      PinholeFocalAndExtra_num_,
      solver__alpha_numerator_,
      solver__alpha_denominator_,
      PinholeFocalAndExtra_num_);
  PinholePoseAlphaNumeratorDenominator(nodes__PinholePose__p_,
                                       PinholePose_num_,
                                       nodes__PinholePose__r_k_,
                                       PinholePose_num_,
                                       nodes__PinholePose__w_,
                                       PinholePose_num_,
                                       solver__alpha_numerator_,
                                       solver__alpha_denominator_,
                                       PinholePose_num_);
  PinholePrincipalPointAlphaNumeratorDenominator(
      nodes__PinholePrincipalPoint__p_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      solver__alpha_numerator_,
      solver__alpha_denominator_,
      PinholePrincipalPoint_num_);
  PointAlphaNumeratorDenominator(nodes__Point__p_,
                                 Point_num_,
                                 nodes__Point__r_k_,
                                 Point_num_,
                                 nodes__Point__w_,
                                 Point_num_,
                                 solver__alpha_numerator_,
                                 solver__alpha_denominator_,
                                 Point_num_);
  SimpleRadialCalibAlphaNumeratorDenominator(nodes__SimpleRadialCalib__p_,
                                             SimpleRadialCalib_num_,
                                             nodes__SimpleRadialCalib__r_k_,
                                             SimpleRadialCalib_num_,
                                             nodes__SimpleRadialCalib__w_,
                                             SimpleRadialCalib_num_,
                                             solver__alpha_numerator_,
                                             solver__alpha_denominator_,
                                             SimpleRadialCalib_num_);
  SimpleRadialFocalAndExtraAlphaNumeratorDenominator(
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      solver__alpha_numerator_,
      solver__alpha_denominator_,
      SimpleRadialFocalAndExtra_num_);
  SimpleRadialPoseAlphaNumeratorDenominator(nodes__SimpleRadialPose__p_,
                                            SimpleRadialPose_num_,
                                            nodes__SimpleRadialPose__r_k_,
                                            SimpleRadialPose_num_,
                                            nodes__SimpleRadialPose__w_,
                                            SimpleRadialPose_num_,
                                            solver__alpha_numerator_,
                                            solver__alpha_denominator_,
                                            SimpleRadialPose_num_);
  SimpleRadialPrincipalPointAlphaNumeratorDenominator(
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      solver__alpha_numerator_,
      solver__alpha_denominator_,
      SimpleRadialPrincipalPoint_num_);

  AlphaFromNumDenom(solver__alpha_numerator_,
                    solver__alpha_denominator_,
                    solver__alpha_,
                    solver__neg_alpha_);
}

void GraphSolver::DoAlpha() {
  Zero(solver__alpha_denominator_, solver__alpha_denominator_ + 1);
  PinholeCalibAlphaDenominatorOrBetaNumerator(nodes__PinholeCalib__p_,
                                              PinholeCalib_num_,
                                              nodes__PinholeCalib__w_,
                                              PinholeCalib_num_,
                                              solver__alpha_denominator_,
                                              PinholeCalib_num_);
  PinholeFocalAndExtraAlphaDenominatorOrBetaNumerator(
      nodes__PinholeFocalAndExtra__p_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__w_,
      PinholeFocalAndExtra_num_,
      solver__alpha_denominator_,
      PinholeFocalAndExtra_num_);
  PinholePoseAlphaDenominatorOrBetaNumerator(nodes__PinholePose__p_,
                                             PinholePose_num_,
                                             nodes__PinholePose__w_,
                                             PinholePose_num_,
                                             solver__alpha_denominator_,
                                             PinholePose_num_);
  PinholePrincipalPointAlphaDenominatorOrBetaNumerator(
      nodes__PinholePrincipalPoint__p_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      solver__alpha_denominator_,
      PinholePrincipalPoint_num_);
  PointAlphaDenominatorOrBetaNumerator(nodes__Point__p_,
                                       Point_num_,
                                       nodes__Point__w_,
                                       Point_num_,
                                       solver__alpha_denominator_,
                                       Point_num_);
  SimpleRadialCalibAlphaDenominatorOrBetaNumerator(nodes__SimpleRadialCalib__p_,
                                                   SimpleRadialCalib_num_,
                                                   nodes__SimpleRadialCalib__w_,
                                                   SimpleRadialCalib_num_,
                                                   solver__alpha_denominator_,
                                                   SimpleRadialCalib_num_);
  SimpleRadialFocalAndExtraAlphaDenominatorOrBetaNumerator(
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      solver__alpha_denominator_,
      SimpleRadialFocalAndExtra_num_);
  SimpleRadialPoseAlphaDenominatorOrBetaNumerator(nodes__SimpleRadialPose__p_,
                                                  SimpleRadialPose_num_,
                                                  nodes__SimpleRadialPose__w_,
                                                  SimpleRadialPose_num_,
                                                  solver__alpha_denominator_,
                                                  SimpleRadialPose_num_);
  SimpleRadialPrincipalPointAlphaDenominatorOrBetaNumerator(
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      solver__alpha_denominator_,
      SimpleRadialPrincipalPoint_num_);

  AlphaFromNumDenom(solver__beta_numerator_,
                    solver__alpha_denominator_,
                    solver__alpha_,
                    solver__neg_alpha_);
}

void GraphSolver::DoUpdateStepFirst() {
  PinholeCalibUpdateStepFirst(nodes__PinholeCalib__p_,
                              PinholeCalib_num_,
                              solver__alpha_,
                              nodes__PinholeCalib__step_,
                              PinholeCalib_num_,
                              PinholeCalib_num_);
  PinholeFocalAndExtraUpdateStepFirst(nodes__PinholeFocalAndExtra__p_,
                                      PinholeFocalAndExtra_num_,
                                      solver__alpha_,
                                      nodes__PinholeFocalAndExtra__step_,
                                      PinholeFocalAndExtra_num_,
                                      PinholeFocalAndExtra_num_);
  PinholePoseUpdateStepFirst(nodes__PinholePose__p_,
                             PinholePose_num_,
                             solver__alpha_,
                             nodes__PinholePose__step_,
                             PinholePose_num_,
                             PinholePose_num_);
  PinholePrincipalPointUpdateStepFirst(nodes__PinholePrincipalPoint__p_,
                                       PinholePrincipalPoint_num_,
                                       solver__alpha_,
                                       nodes__PinholePrincipalPoint__step_,
                                       PinholePrincipalPoint_num_,
                                       PinholePrincipalPoint_num_);
  PointUpdateStepFirst(nodes__Point__p_,
                       Point_num_,
                       solver__alpha_,
                       nodes__Point__step_,
                       Point_num_,
                       Point_num_);
  SimpleRadialCalibUpdateStepFirst(nodes__SimpleRadialCalib__p_,
                                   SimpleRadialCalib_num_,
                                   solver__alpha_,
                                   nodes__SimpleRadialCalib__step_,
                                   SimpleRadialCalib_num_,
                                   SimpleRadialCalib_num_);
  SimpleRadialFocalAndExtraUpdateStepFirst(
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      solver__alpha_,
      nodes__SimpleRadialFocalAndExtra__step_,
      SimpleRadialFocalAndExtra_num_,
      SimpleRadialFocalAndExtra_num_);
  SimpleRadialPoseUpdateStepFirst(nodes__SimpleRadialPose__p_,
                                  SimpleRadialPose_num_,
                                  solver__alpha_,
                                  nodes__SimpleRadialPose__step_,
                                  SimpleRadialPose_num_,
                                  SimpleRadialPose_num_);
  SimpleRadialPrincipalPointUpdateStepFirst(
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      solver__alpha_,
      nodes__SimpleRadialPrincipalPoint__step_,
      SimpleRadialPrincipalPoint_num_,
      SimpleRadialPrincipalPoint_num_);
}

void GraphSolver::DoUpdateStep() {
  PinholeCalibUpdateStep(nodes__PinholeCalib__step_,
                         PinholeCalib_num_,
                         nodes__PinholeCalib__p_,
                         PinholeCalib_num_,
                         solver__alpha_,
                         nodes__PinholeCalib__step_,
                         PinholeCalib_num_,
                         PinholeCalib_num_);
  PinholeFocalAndExtraUpdateStep(nodes__PinholeFocalAndExtra__step_,
                                 PinholeFocalAndExtra_num_,
                                 nodes__PinholeFocalAndExtra__p_,
                                 PinholeFocalAndExtra_num_,
                                 solver__alpha_,
                                 nodes__PinholeFocalAndExtra__step_,
                                 PinholeFocalAndExtra_num_,
                                 PinholeFocalAndExtra_num_);
  PinholePoseUpdateStep(nodes__PinholePose__step_,
                        PinholePose_num_,
                        nodes__PinholePose__p_,
                        PinholePose_num_,
                        solver__alpha_,
                        nodes__PinholePose__step_,
                        PinholePose_num_,
                        PinholePose_num_);
  PinholePrincipalPointUpdateStep(nodes__PinholePrincipalPoint__step_,
                                  PinholePrincipalPoint_num_,
                                  nodes__PinholePrincipalPoint__p_,
                                  PinholePrincipalPoint_num_,
                                  solver__alpha_,
                                  nodes__PinholePrincipalPoint__step_,
                                  PinholePrincipalPoint_num_,
                                  PinholePrincipalPoint_num_);
  PointUpdateStep(nodes__Point__step_,
                  Point_num_,
                  nodes__Point__p_,
                  Point_num_,
                  solver__alpha_,
                  nodes__Point__step_,
                  Point_num_,
                  Point_num_);
  SimpleRadialCalibUpdateStep(nodes__SimpleRadialCalib__step_,
                              SimpleRadialCalib_num_,
                              nodes__SimpleRadialCalib__p_,
                              SimpleRadialCalib_num_,
                              solver__alpha_,
                              nodes__SimpleRadialCalib__step_,
                              SimpleRadialCalib_num_,
                              SimpleRadialCalib_num_);
  SimpleRadialFocalAndExtraUpdateStep(nodes__SimpleRadialFocalAndExtra__step_,
                                      SimpleRadialFocalAndExtra_num_,
                                      nodes__SimpleRadialFocalAndExtra__p_,
                                      SimpleRadialFocalAndExtra_num_,
                                      solver__alpha_,
                                      nodes__SimpleRadialFocalAndExtra__step_,
                                      SimpleRadialFocalAndExtra_num_,
                                      SimpleRadialFocalAndExtra_num_);
  SimpleRadialPoseUpdateStep(nodes__SimpleRadialPose__step_,
                             SimpleRadialPose_num_,
                             nodes__SimpleRadialPose__p_,
                             SimpleRadialPose_num_,
                             solver__alpha_,
                             nodes__SimpleRadialPose__step_,
                             SimpleRadialPose_num_,
                             SimpleRadialPose_num_);
  SimpleRadialPrincipalPointUpdateStep(nodes__SimpleRadialPrincipalPoint__step_,
                                       SimpleRadialPrincipalPoint_num_,
                                       nodes__SimpleRadialPrincipalPoint__p_,
                                       SimpleRadialPrincipalPoint_num_,
                                       solver__alpha_,
                                       nodes__SimpleRadialPrincipalPoint__step_,
                                       SimpleRadialPrincipalPoint_num_,
                                       SimpleRadialPrincipalPoint_num_);
}

void GraphSolver::DoUpdateRFirst() {
  Zero(solver__r_0_norm2_tot_, solver__r_0_norm2_tot_ + 1);

  PinholeCalibUpdateRFirst(nodes__PinholeCalib__r_k_,
                           PinholeCalib_num_,
                           nodes__PinholeCalib__w_,
                           PinholeCalib_num_,
                           solver__neg_alpha_,
                           nodes__PinholeCalib__r_k_,
                           PinholeCalib_num_,
                           solver__r_0_norm2_tot_,
                           solver__r_kp1_norm2_tot_,
                           PinholeCalib_num_);

  PinholeFocalAndExtraUpdateRFirst(nodes__PinholeFocalAndExtra__r_k_,
                                   PinholeFocalAndExtra_num_,
                                   nodes__PinholeFocalAndExtra__w_,
                                   PinholeFocalAndExtra_num_,
                                   solver__neg_alpha_,
                                   nodes__PinholeFocalAndExtra__r_k_,
                                   PinholeFocalAndExtra_num_,
                                   solver__r_0_norm2_tot_,
                                   solver__r_kp1_norm2_tot_,
                                   PinholeFocalAndExtra_num_);

  PinholePoseUpdateRFirst(nodes__PinholePose__r_k_,
                          PinholePose_num_,
                          nodes__PinholePose__w_,
                          PinholePose_num_,
                          solver__neg_alpha_,
                          nodes__PinholePose__r_k_,
                          PinholePose_num_,
                          solver__r_0_norm2_tot_,
                          solver__r_kp1_norm2_tot_,
                          PinholePose_num_);

  PinholePrincipalPointUpdateRFirst(nodes__PinholePrincipalPoint__r_k_,
                                    PinholePrincipalPoint_num_,
                                    nodes__PinholePrincipalPoint__w_,
                                    PinholePrincipalPoint_num_,
                                    solver__neg_alpha_,
                                    nodes__PinholePrincipalPoint__r_k_,
                                    PinholePrincipalPoint_num_,
                                    solver__r_0_norm2_tot_,
                                    solver__r_kp1_norm2_tot_,
                                    PinholePrincipalPoint_num_);

  PointUpdateRFirst(nodes__Point__r_k_,
                    Point_num_,
                    nodes__Point__w_,
                    Point_num_,
                    solver__neg_alpha_,
                    nodes__Point__r_k_,
                    Point_num_,
                    solver__r_0_norm2_tot_,
                    solver__r_kp1_norm2_tot_,
                    Point_num_);

  SimpleRadialCalibUpdateRFirst(nodes__SimpleRadialCalib__r_k_,
                                SimpleRadialCalib_num_,
                                nodes__SimpleRadialCalib__w_,
                                SimpleRadialCalib_num_,
                                solver__neg_alpha_,
                                nodes__SimpleRadialCalib__r_k_,
                                SimpleRadialCalib_num_,
                                solver__r_0_norm2_tot_,
                                solver__r_kp1_norm2_tot_,
                                SimpleRadialCalib_num_);

  SimpleRadialFocalAndExtraUpdateRFirst(nodes__SimpleRadialFocalAndExtra__r_k_,
                                        SimpleRadialFocalAndExtra_num_,
                                        nodes__SimpleRadialFocalAndExtra__w_,
                                        SimpleRadialFocalAndExtra_num_,
                                        solver__neg_alpha_,
                                        nodes__SimpleRadialFocalAndExtra__r_k_,
                                        SimpleRadialFocalAndExtra_num_,
                                        solver__r_0_norm2_tot_,
                                        solver__r_kp1_norm2_tot_,
                                        SimpleRadialFocalAndExtra_num_);

  SimpleRadialPoseUpdateRFirst(nodes__SimpleRadialPose__r_k_,
                               SimpleRadialPose_num_,
                               nodes__SimpleRadialPose__w_,
                               SimpleRadialPose_num_,
                               solver__neg_alpha_,
                               nodes__SimpleRadialPose__r_k_,
                               SimpleRadialPose_num_,
                               solver__r_0_norm2_tot_,
                               solver__r_kp1_norm2_tot_,
                               SimpleRadialPose_num_);

  SimpleRadialPrincipalPointUpdateRFirst(
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      solver__neg_alpha_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      solver__r_0_norm2_tot_,
      solver__r_kp1_norm2_tot_,
      SimpleRadialPrincipalPoint_num_);

  pcg_r_0_norm2_ = ReadCuMem(solver__r_0_norm2_tot_);
  pcg_r_kp1_norm2_ = ReadCuMem(solver__r_kp1_norm2_tot_);
}

void GraphSolver::DoUpdateR() {
  Zero(solver__r_kp1_norm2_tot_, solver__r_kp1_norm2_tot_ + 1);

  PinholeCalibUpdateR(nodes__PinholeCalib__r_k_,
                      PinholeCalib_num_,
                      nodes__PinholeCalib__w_,
                      PinholeCalib_num_,
                      solver__neg_alpha_,
                      nodes__PinholeCalib__r_k_,
                      PinholeCalib_num_,
                      solver__r_kp1_norm2_tot_,
                      PinholeCalib_num_);
  PinholeFocalAndExtraUpdateR(nodes__PinholeFocalAndExtra__r_k_,
                              PinholeFocalAndExtra_num_,
                              nodes__PinholeFocalAndExtra__w_,
                              PinholeFocalAndExtra_num_,
                              solver__neg_alpha_,
                              nodes__PinholeFocalAndExtra__r_k_,
                              PinholeFocalAndExtra_num_,
                              solver__r_kp1_norm2_tot_,
                              PinholeFocalAndExtra_num_);
  PinholePoseUpdateR(nodes__PinholePose__r_k_,
                     PinholePose_num_,
                     nodes__PinholePose__w_,
                     PinholePose_num_,
                     solver__neg_alpha_,
                     nodes__PinholePose__r_k_,
                     PinholePose_num_,
                     solver__r_kp1_norm2_tot_,
                     PinholePose_num_);
  PinholePrincipalPointUpdateR(nodes__PinholePrincipalPoint__r_k_,
                               PinholePrincipalPoint_num_,
                               nodes__PinholePrincipalPoint__w_,
                               PinholePrincipalPoint_num_,
                               solver__neg_alpha_,
                               nodes__PinholePrincipalPoint__r_k_,
                               PinholePrincipalPoint_num_,
                               solver__r_kp1_norm2_tot_,
                               PinholePrincipalPoint_num_);
  PointUpdateR(nodes__Point__r_k_,
               Point_num_,
               nodes__Point__w_,
               Point_num_,
               solver__neg_alpha_,
               nodes__Point__r_k_,
               Point_num_,
               solver__r_kp1_norm2_tot_,
               Point_num_);
  SimpleRadialCalibUpdateR(nodes__SimpleRadialCalib__r_k_,
                           SimpleRadialCalib_num_,
                           nodes__SimpleRadialCalib__w_,
                           SimpleRadialCalib_num_,
                           solver__neg_alpha_,
                           nodes__SimpleRadialCalib__r_k_,
                           SimpleRadialCalib_num_,
                           solver__r_kp1_norm2_tot_,
                           SimpleRadialCalib_num_);
  SimpleRadialFocalAndExtraUpdateR(nodes__SimpleRadialFocalAndExtra__r_k_,
                                   SimpleRadialFocalAndExtra_num_,
                                   nodes__SimpleRadialFocalAndExtra__w_,
                                   SimpleRadialFocalAndExtra_num_,
                                   solver__neg_alpha_,
                                   nodes__SimpleRadialFocalAndExtra__r_k_,
                                   SimpleRadialFocalAndExtra_num_,
                                   solver__r_kp1_norm2_tot_,
                                   SimpleRadialFocalAndExtra_num_);
  SimpleRadialPoseUpdateR(nodes__SimpleRadialPose__r_k_,
                          SimpleRadialPose_num_,
                          nodes__SimpleRadialPose__w_,
                          SimpleRadialPose_num_,
                          solver__neg_alpha_,
                          nodes__SimpleRadialPose__r_k_,
                          SimpleRadialPose_num_,
                          solver__r_kp1_norm2_tot_,
                          SimpleRadialPose_num_);
  SimpleRadialPrincipalPointUpdateR(nodes__SimpleRadialPrincipalPoint__r_k_,
                                    SimpleRadialPrincipalPoint_num_,
                                    nodes__SimpleRadialPrincipalPoint__w_,
                                    SimpleRadialPrincipalPoint_num_,
                                    solver__neg_alpha_,
                                    nodes__SimpleRadialPrincipalPoint__r_k_,
                                    SimpleRadialPrincipalPoint_num_,
                                    solver__r_kp1_norm2_tot_,
                                    SimpleRadialPrincipalPoint_num_);
  pcg_r_kp1_norm2_ = ReadCuMem(solver__r_kp1_norm2_tot_);
}

double GraphSolver::DoRetractScore() {
  PinholeCalibRetract(nodes__PinholeCalib__storage_current_,
                      PinholeCalib_num_max_,
                      nodes__PinholeCalib__step_,
                      PinholeCalib_num_,
                      nodes__PinholeCalib__storage_check_,
                      PinholeCalib_num_max_,
                      PinholeCalib_num_);
  PinholeFocalAndExtraRetract(nodes__PinholeFocalAndExtra__storage_current_,
                              PinholeFocalAndExtra_num_max_,
                              nodes__PinholeFocalAndExtra__step_,
                              PinholeFocalAndExtra_num_,
                              nodes__PinholeFocalAndExtra__storage_check_,
                              PinholeFocalAndExtra_num_max_,
                              PinholeFocalAndExtra_num_);
  PinholePoseRetract(nodes__PinholePose__storage_current_,
                     PinholePose_num_max_,
                     nodes__PinholePose__step_,
                     PinholePose_num_,
                     nodes__PinholePose__storage_check_,
                     PinholePose_num_max_,
                     PinholePose_num_);
  PinholePrincipalPointRetract(nodes__PinholePrincipalPoint__storage_current_,
                               PinholePrincipalPoint_num_max_,
                               nodes__PinholePrincipalPoint__step_,
                               PinholePrincipalPoint_num_,
                               nodes__PinholePrincipalPoint__storage_check_,
                               PinholePrincipalPoint_num_max_,
                               PinholePrincipalPoint_num_);
  PointRetract(nodes__Point__storage_current_,
               Point_num_max_,
               nodes__Point__step_,
               Point_num_,
               nodes__Point__storage_check_,
               Point_num_max_,
               Point_num_);
  SimpleRadialCalibRetract(nodes__SimpleRadialCalib__storage_current_,
                           SimpleRadialCalib_num_max_,
                           nodes__SimpleRadialCalib__step_,
                           SimpleRadialCalib_num_,
                           nodes__SimpleRadialCalib__storage_check_,
                           SimpleRadialCalib_num_max_,
                           SimpleRadialCalib_num_);
  SimpleRadialFocalAndExtraRetract(
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      nodes__SimpleRadialFocalAndExtra__step_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__storage_check_,
      SimpleRadialFocalAndExtra_num_max_,
      SimpleRadialFocalAndExtra_num_);
  SimpleRadialPoseRetract(nodes__SimpleRadialPose__storage_current_,
                          SimpleRadialPose_num_max_,
                          nodes__SimpleRadialPose__step_,
                          SimpleRadialPose_num_,
                          nodes__SimpleRadialPose__storage_check_,
                          SimpleRadialPose_num_max_,
                          SimpleRadialPose_num_);
  SimpleRadialPrincipalPointRetract(
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      nodes__SimpleRadialPrincipalPoint__step_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__storage_check_,
      SimpleRadialPrincipalPoint_num_max_,
      SimpleRadialPrincipalPoint_num_);
  Zero(solver__res_tot_, solver__res_tot_ + 1);
  SimpleRadialMergedScore(nodes__SimpleRadialPose__storage_check_,
                          SimpleRadialPose_num_max_,
                          facs__simple_radial_merged__args__pose__idx_shared_,
                          nodes__SimpleRadialCalib__storage_check_,
                          SimpleRadialCalib_num_max_,
                          facs__simple_radial_merged__args__calib__idx_shared_,
                          nodes__Point__storage_check_,
                          Point_num_max_,
                          facs__simple_radial_merged__args__point__idx_shared_,
                          facs__simple_radial_merged__args__pixel__data_,
                          simple_radial_merged_num_max_,
                          solver__res_tot_,
                          simple_radial_merged_num_);
  SimpleRadialMergedFixedPoseScore(
      nodes__SimpleRadialCalib__storage_check_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_merged_fixed_pose__args__calib__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_merged_fixed_pose__args__point__idx_shared_,
      facs__simple_radial_merged_fixed_pose__args__pixel__data_,
      simple_radial_merged_fixed_pose_num_max_,
      facs__simple_radial_merged_fixed_pose__args__pose__data_,
      simple_radial_merged_fixed_pose_num_max_,
      solver__res_tot_,
      simple_radial_merged_fixed_pose_num_);
  SimpleRadialMergedFixedPointScore(
      nodes__SimpleRadialPose__storage_check_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_merged_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialCalib__storage_check_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_merged_fixed_point__args__calib__idx_shared_,
      facs__simple_radial_merged_fixed_point__args__pixel__data_,
      simple_radial_merged_fixed_point_num_max_,
      facs__simple_radial_merged_fixed_point__args__point__data_,
      simple_radial_merged_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_merged_fixed_point_num_);
  SimpleRadialMergedFixedPoseFixedPointScore(
      nodes__SimpleRadialCalib__storage_check_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__calib__idx_shared_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__point__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_merged_fixed_pose_fixed_point_num_);
  PinholeMergedScore(nodes__PinholePose__storage_check_,
                     PinholePose_num_max_,
                     facs__pinhole_merged__args__pose__idx_shared_,
                     nodes__PinholeCalib__storage_check_,
                     PinholeCalib_num_max_,
                     facs__pinhole_merged__args__calib__idx_shared_,
                     nodes__Point__storage_check_,
                     Point_num_max_,
                     facs__pinhole_merged__args__point__idx_shared_,
                     facs__pinhole_merged__args__pixel__data_,
                     pinhole_merged_num_max_,
                     solver__res_tot_,
                     pinhole_merged_num_);
  PinholeMergedFixedPoseScore(
      nodes__PinholeCalib__storage_check_,
      PinholeCalib_num_max_,
      facs__pinhole_merged_fixed_pose__args__calib__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_merged_fixed_pose__args__point__idx_shared_,
      facs__pinhole_merged_fixed_pose__args__pixel__data_,
      pinhole_merged_fixed_pose_num_max_,
      facs__pinhole_merged_fixed_pose__args__pose__data_,
      pinhole_merged_fixed_pose_num_max_,
      solver__res_tot_,
      pinhole_merged_fixed_pose_num_);
  PinholeMergedFixedPointScore(
      nodes__PinholePose__storage_check_,
      PinholePose_num_max_,
      facs__pinhole_merged_fixed_point__args__pose__idx_shared_,
      nodes__PinholeCalib__storage_check_,
      PinholeCalib_num_max_,
      facs__pinhole_merged_fixed_point__args__calib__idx_shared_,
      facs__pinhole_merged_fixed_point__args__pixel__data_,
      pinhole_merged_fixed_point_num_max_,
      facs__pinhole_merged_fixed_point__args__point__data_,
      pinhole_merged_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_merged_fixed_point_num_);
  PinholeMergedFixedPoseFixedPointScore(
      nodes__PinholeCalib__storage_check_,
      PinholeCalib_num_max_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__calib__idx_shared_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__pose__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__point__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_merged_fixed_pose_fixed_point_num_);
  SimpleRadialFixedFocalAndExtraScore(
      nodes__SimpleRadialPose__storage_check_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_focal_and_extra__args__pose__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_check_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_num_max_,
      facs__simple_radial_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_num_max_,
      solver__res_tot_,
      simple_radial_fixed_focal_and_extra_num_);
  SimpleRadialFixedPrincipalPointScore(
      nodes__SimpleRadialPose__storage_check_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_principal_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocalAndExtra__storage_check_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_principal_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_principal_point_num_);
  SimpleRadialFixedPoseFixedFocalAndExtraScore(
      nodes__SimpleRadialPrincipalPoint__storage_check_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_);
  SimpleRadialFixedPoseFixedPrincipalPointScore(
      nodes__SimpleRadialFocalAndExtra__storage_check_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_fixed_principal_point_num_);
  SimpleRadialFixedFocalAndExtraFixedPrincipalPointScore(
      nodes__SimpleRadialPose__storage_check_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_);
  SimpleRadialFixedFocalAndExtraFixedPointScore(
      nodes__SimpleRadialPose__storage_check_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_check_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_);
  SimpleRadialFixedPrincipalPointFixedPointScore(
      nodes__SimpleRadialPose__storage_check_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocalAndExtra__storage_check_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_principal_point_fixed_point_num_);
  SimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointScore(
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_);
  SimpleRadialFixedPoseFixedFocalAndExtraFixedPointScore(
      nodes__SimpleRadialPrincipalPoint__storage_check_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_);
  SimpleRadialFixedPoseFixedPrincipalPointFixedPointScore(
      nodes__SimpleRadialFocalAndExtra__storage_check_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_);
  SimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointScore(
      nodes__SimpleRadialPose__storage_check_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_);
  PinholeFixedFocalAndExtraScore(
      nodes__PinholePose__storage_check_,
      PinholePose_num_max_,
      facs__pinhole_fixed_focal_and_extra__args__pose__idx_shared_,
      nodes__PinholePrincipalPoint__storage_check_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_fixed_focal_and_extra__args__point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra__args__pixel__data_,
      pinhole_fixed_focal_and_extra_num_max_,
      facs__pinhole_fixed_focal_and_extra__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_num_max_,
      solver__res_tot_,
      pinhole_fixed_focal_and_extra_num_);
  PinholeFixedPrincipalPointScore(
      nodes__PinholePose__storage_check_,
      PinholePose_num_max_,
      facs__pinhole_fixed_principal_point__args__pose__idx_shared_,
      nodes__PinholeFocalAndExtra__storage_check_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_principal_point_num_max_,
      facs__pinhole_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_principal_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_principal_point_num_);
  PinholeFixedPoseFixedFocalAndExtraScore(
      nodes__PinholePrincipalPoint__storage_check_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_);
  PinholeFixedPoseFixedPrincipalPointScore(
      nodes__PinholeFocalAndExtra__storage_check_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__pose__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_fixed_principal_point_num_);
  PinholeFixedFocalAndExtraFixedPrincipalPointScore(
      nodes__PinholePose__storage_check_,
      PinholePose_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_);
  PinholeFixedFocalAndExtraFixedPointScore(
      nodes__PinholePose__storage_check_,
      PinholePose_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      nodes__PinholePrincipalPoint__storage_check_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__point__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_focal_and_extra_fixed_point_num_);
  PinholeFixedPrincipalPointFixedPointScore(
      nodes__PinholePose__storage_check_,
      PinholePose_num_max_,
      facs__pinhole_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      nodes__PinholeFocalAndExtra__storage_check_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__pinhole_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_principal_point_fixed_point_num_);
  PinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointScore(
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_);
  PinholeFixedPoseFixedFocalAndExtraFixedPointScore(
      nodes__PinholePrincipalPoint__storage_check_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_);
  PinholeFixedPoseFixedPrincipalPointFixedPointScore(
      nodes__PinholeFocalAndExtra__storage_check_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_);
  PinholeFixedFocalAndExtraFixedPrincipalPointFixedPointScore(
      nodes__PinholePose__storage_check_,
      PinholePose_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_);
  return 0.5 * ReadCuMem(solver__res_tot_);
}

void GraphSolver::DoBeta() {
  Zero(solver__beta_numerator_, solver__beta_numerator_ + 1);

  PinholeCalibAlphaDenominatorOrBetaNumerator(nodes__PinholeCalib__r_k_,
                                              PinholeCalib_num_,
                                              nodes__PinholeCalib__z_,
                                              PinholeCalib_num_,
                                              solver__beta_numerator_,
                                              PinholeCalib_num_);

  PinholeFocalAndExtraAlphaDenominatorOrBetaNumerator(
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__z_,
      PinholeFocalAndExtra_num_,
      solver__beta_numerator_,
      PinholeFocalAndExtra_num_);

  PinholePoseAlphaDenominatorOrBetaNumerator(nodes__PinholePose__r_k_,
                                             PinholePose_num_,
                                             nodes__PinholePose__z_,
                                             PinholePose_num_,
                                             solver__beta_numerator_,
                                             PinholePose_num_);

  PinholePrincipalPointAlphaDenominatorOrBetaNumerator(
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__z_,
      PinholePrincipalPoint_num_,
      solver__beta_numerator_,
      PinholePrincipalPoint_num_);

  PointAlphaDenominatorOrBetaNumerator(nodes__Point__r_k_,
                                       Point_num_,
                                       nodes__Point__z_,
                                       Point_num_,
                                       solver__beta_numerator_,
                                       Point_num_);

  SimpleRadialCalibAlphaDenominatorOrBetaNumerator(
      nodes__SimpleRadialCalib__r_k_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__z_,
      SimpleRadialCalib_num_,
      solver__beta_numerator_,
      SimpleRadialCalib_num_);

  SimpleRadialFocalAndExtraAlphaDenominatorOrBetaNumerator(
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__z_,
      SimpleRadialFocalAndExtra_num_,
      solver__beta_numerator_,
      SimpleRadialFocalAndExtra_num_);

  SimpleRadialPoseAlphaDenominatorOrBetaNumerator(nodes__SimpleRadialPose__r_k_,
                                                  SimpleRadialPose_num_,
                                                  nodes__SimpleRadialPose__z_,
                                                  SimpleRadialPose_num_,
                                                  solver__beta_numerator_,
                                                  SimpleRadialPose_num_);

  SimpleRadialPrincipalPointAlphaDenominatorOrBetaNumerator(
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__z_,
      SimpleRadialPrincipalPoint_num_,
      solver__beta_numerator_,
      SimpleRadialPrincipalPoint_num_);
  BetaFromNumDenom(
      solver__beta_numerator_, solver__alpha_numerator_, solver__beta_);
}

void GraphSolver::DoUpdateP() {
  PinholeCalibUpdateP(nodes__PinholeCalib__z_,
                      PinholeCalib_num_,
                      nodes__PinholeCalib__p_,
                      PinholeCalib_num_,
                      solver__beta_,
                      nodes__PinholeCalib__p_,
                      PinholeCalib_num_,
                      PinholeCalib_num_);
  PinholeFocalAndExtraUpdateP(nodes__PinholeFocalAndExtra__z_,
                              PinholeFocalAndExtra_num_,
                              nodes__PinholeFocalAndExtra__p_,
                              PinholeFocalAndExtra_num_,
                              solver__beta_,
                              nodes__PinholeFocalAndExtra__p_,
                              PinholeFocalAndExtra_num_,
                              PinholeFocalAndExtra_num_);
  PinholePoseUpdateP(nodes__PinholePose__z_,
                     PinholePose_num_,
                     nodes__PinholePose__p_,
                     PinholePose_num_,
                     solver__beta_,
                     nodes__PinholePose__p_,
                     PinholePose_num_,
                     PinholePose_num_);
  PinholePrincipalPointUpdateP(nodes__PinholePrincipalPoint__z_,
                               PinholePrincipalPoint_num_,
                               nodes__PinholePrincipalPoint__p_,
                               PinholePrincipalPoint_num_,
                               solver__beta_,
                               nodes__PinholePrincipalPoint__p_,
                               PinholePrincipalPoint_num_,
                               PinholePrincipalPoint_num_);
  PointUpdateP(nodes__Point__z_,
               Point_num_,
               nodes__Point__p_,
               Point_num_,
               solver__beta_,
               nodes__Point__p_,
               Point_num_,
               Point_num_);
  SimpleRadialCalibUpdateP(nodes__SimpleRadialCalib__z_,
                           SimpleRadialCalib_num_,
                           nodes__SimpleRadialCalib__p_,
                           SimpleRadialCalib_num_,
                           solver__beta_,
                           nodes__SimpleRadialCalib__p_,
                           SimpleRadialCalib_num_,
                           SimpleRadialCalib_num_);
  SimpleRadialFocalAndExtraUpdateP(nodes__SimpleRadialFocalAndExtra__z_,
                                   SimpleRadialFocalAndExtra_num_,
                                   nodes__SimpleRadialFocalAndExtra__p_,
                                   SimpleRadialFocalAndExtra_num_,
                                   solver__beta_,
                                   nodes__SimpleRadialFocalAndExtra__p_,
                                   SimpleRadialFocalAndExtra_num_,
                                   SimpleRadialFocalAndExtra_num_);
  SimpleRadialPoseUpdateP(nodes__SimpleRadialPose__z_,
                          SimpleRadialPose_num_,
                          nodes__SimpleRadialPose__p_,
                          SimpleRadialPose_num_,
                          solver__beta_,
                          nodes__SimpleRadialPose__p_,
                          SimpleRadialPose_num_,
                          SimpleRadialPose_num_);
  SimpleRadialPrincipalPointUpdateP(nodes__SimpleRadialPrincipalPoint__z_,
                                    SimpleRadialPrincipalPoint_num_,
                                    nodes__SimpleRadialPrincipalPoint__p_,
                                    SimpleRadialPrincipalPoint_num_,
                                    solver__beta_,
                                    nodes__SimpleRadialPrincipalPoint__p_,
                                    SimpleRadialPrincipalPoint_num_,
                                    SimpleRadialPrincipalPoint_num_);
}

double GraphSolver::GetPredDecrease() {
  Zero(solver__pred_decrease_tot_, solver__pred_decrease_tot_ + 1);
  PinholeCalibPredDecreaseTimesTwo(nodes__PinholeCalib__step_,
                                   PinholeCalib_num_,
                                   nodes__PinholeCalib__precond_diag_,
                                   PinholeCalib_num_,
                                   solver__current_diag_,
                                   nodes__PinholeCalib__r_0_,
                                   PinholeCalib_num_,
                                   solver__pred_decrease_tot_,
                                   PinholeCalib_num_);
  PinholeFocalAndExtraPredDecreaseTimesTwo(
      nodes__PinholeFocalAndExtra__step_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      solver__current_diag_,
      nodes__PinholeFocalAndExtra__r_0_,
      PinholeFocalAndExtra_num_,
      solver__pred_decrease_tot_,
      PinholeFocalAndExtra_num_);
  PinholePosePredDecreaseTimesTwo(nodes__PinholePose__step_,
                                  PinholePose_num_,
                                  nodes__PinholePose__precond_diag_,
                                  PinholePose_num_,
                                  solver__current_diag_,
                                  nodes__PinholePose__r_0_,
                                  PinholePose_num_,
                                  solver__pred_decrease_tot_,
                                  PinholePose_num_);
  PinholePrincipalPointPredDecreaseTimesTwo(
      nodes__PinholePrincipalPoint__step_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      solver__current_diag_,
      nodes__PinholePrincipalPoint__r_0_,
      PinholePrincipalPoint_num_,
      solver__pred_decrease_tot_,
      PinholePrincipalPoint_num_);
  PointPredDecreaseTimesTwo(nodes__Point__step_,
                            Point_num_,
                            nodes__Point__precond_diag_,
                            Point_num_,
                            solver__current_diag_,
                            nodes__Point__r_0_,
                            Point_num_,
                            solver__pred_decrease_tot_,
                            Point_num_);
  SimpleRadialCalibPredDecreaseTimesTwo(nodes__SimpleRadialCalib__step_,
                                        SimpleRadialCalib_num_,
                                        nodes__SimpleRadialCalib__precond_diag_,
                                        SimpleRadialCalib_num_,
                                        solver__current_diag_,
                                        nodes__SimpleRadialCalib__r_0_,
                                        SimpleRadialCalib_num_,
                                        solver__pred_decrease_tot_,
                                        SimpleRadialCalib_num_);
  SimpleRadialFocalAndExtraPredDecreaseTimesTwo(
      nodes__SimpleRadialFocalAndExtra__step_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      solver__current_diag_,
      nodes__SimpleRadialFocalAndExtra__r_0_,
      SimpleRadialFocalAndExtra_num_,
      solver__pred_decrease_tot_,
      SimpleRadialFocalAndExtra_num_);
  SimpleRadialPosePredDecreaseTimesTwo(nodes__SimpleRadialPose__step_,
                                       SimpleRadialPose_num_,
                                       nodes__SimpleRadialPose__precond_diag_,
                                       SimpleRadialPose_num_,
                                       solver__current_diag_,
                                       nodes__SimpleRadialPose__r_0_,
                                       SimpleRadialPose_num_,
                                       solver__pred_decrease_tot_,
                                       SimpleRadialPose_num_);
  SimpleRadialPrincipalPointPredDecreaseTimesTwo(
      nodes__SimpleRadialPrincipalPoint__step_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      solver__current_diag_,
      nodes__SimpleRadialPrincipalPoint__r_0_,
      SimpleRadialPrincipalPoint_num_,
      solver__pred_decrease_tot_,
      SimpleRadialPrincipalPoint_num_);
  return 0.5 * ReadCuMem(solver__pred_decrease_tot_);
}

void GraphSolver::finish_indices() { indices_valid_ = true; }

void GraphSolver::SetPinholeCalibNum(const size_t num) {
  if (num > PinholeCalib_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > PinholeCalib_num_max_");
  }
  PinholeCalib_num_ = num;
}

void GraphSolver::SetPinholeCalibNodesFromStackedHost(const double* const data,
                                                      const size_t offset,
                                                      const size_t num) {
  if (offset + num > PinholeCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeCalib_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             4 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  PinholeCalibStackedToCaspar(marker__scratch_inout_,
                              nodes__PinholeCalib__storage_current_,
                              PinholeCalib_num_max_,
                              offset,
                              num);
}

void GraphSolver::SetPinholeCalibNodesFromStackedDevice(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholeCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeCalib_num_");
  }
  PinholeCalibStackedToCaspar(data,
                              nodes__PinholeCalib__storage_current_,
                              PinholeCalib_num_max_,
                              offset,
                              num);
}

void GraphSolver::GetPinholeCalibNodesToStackedHost(double* const data,
                                                    const size_t offset,
                                                    const size_t num) {
  if (offset + num > PinholeCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeCalib_num_");
  }
  PinholeCalibCasparToStacked(nodes__PinholeCalib__storage_current_,
                              marker__scratch_inout_,
                              PinholeCalib_num_max_,
                              offset,
                              num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             4 * num * sizeof(double),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetPinholeCalibNodesToStackedDevice(double* const data,
                                                      const size_t offset,
                                                      const size_t num) {
  if (offset + num > PinholeCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeCalib_num_");
  }
  PinholeCalibCasparToStacked(nodes__PinholeCalib__storage_current_,
                              data,
                              PinholeCalib_num_max_,
                              offset,
                              num);
}

void GraphSolver::SetPinholeFocalAndExtraNum(const size_t num) {
  if (num > PinholeFocalAndExtra_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > PinholeFocalAndExtra_num_max_");
  }
  PinholeFocalAndExtra_num_ = num;
}

void GraphSolver::SetPinholeFocalAndExtraNodesFromStackedHost(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholeFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocalAndExtra_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  PinholeFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeFocalAndExtraNodesFromStackedDevice(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholeFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocalAndExtra_num_");
  }
  PinholeFocalAndExtraStackedToCaspar(
      data,
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      offset,
      num);
}

void GraphSolver::GetPinholeFocalAndExtraNodesToStackedHost(double* const data,
                                                            const size_t offset,
                                                            const size_t num) {
  if (offset + num > PinholeFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocalAndExtra_num_");
  }
  PinholeFocalAndExtraCasparToStacked(
      nodes__PinholeFocalAndExtra__storage_current_,
      marker__scratch_inout_,
      PinholeFocalAndExtra_num_max_,
      offset,
      num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             2 * num * sizeof(double),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetPinholeFocalAndExtraNodesToStackedDevice(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholeFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocalAndExtra_num_");
  }
  PinholeFocalAndExtraCasparToStacked(
      nodes__PinholeFocalAndExtra__storage_current_,
      data,
      PinholeFocalAndExtra_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholePoseNum(const size_t num) {
  if (num > PinholePose_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > PinholePose_num_max_");
  }
  PinholePose_num_ = num;
}

void GraphSolver::SetPinholePoseNodesFromStackedHost(const double* const data,
                                                     const size_t offset,
                                                     const size_t num) {
  if (offset + num > PinholePose_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholePose_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  PinholePoseStackedToCaspar(marker__scratch_inout_,
                             nodes__PinholePose__storage_current_,
                             PinholePose_num_max_,
                             offset,
                             num);
}

void GraphSolver::SetPinholePoseNodesFromStackedDevice(const double* const data,
                                                       const size_t offset,
                                                       const size_t num) {
  if (offset + num > PinholePose_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholePose_num_");
  }
  PinholePoseStackedToCaspar(data,
                             nodes__PinholePose__storage_current_,
                             PinholePose_num_max_,
                             offset,
                             num);
}

void GraphSolver::GetPinholePoseNodesToStackedHost(double* const data,
                                                   const size_t offset,
                                                   const size_t num) {
  if (offset + num > PinholePose_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholePose_num_");
  }
  PinholePoseCasparToStacked(nodes__PinholePose__storage_current_,
                             marker__scratch_inout_,
                             PinholePose_num_max_,
                             offset,
                             num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             7 * num * sizeof(double),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetPinholePoseNodesToStackedDevice(double* const data,
                                                     const size_t offset,
                                                     const size_t num) {
  if (offset + num > PinholePose_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholePose_num_");
  }
  PinholePoseCasparToStacked(nodes__PinholePose__storage_current_,
                             data,
                             PinholePose_num_max_,
                             offset,
                             num);
}

void GraphSolver::SetPinholePrincipalPointNum(const size_t num) {
  if (num > PinholePrincipalPoint_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > PinholePrincipalPoint_num_max_");
  }
  PinholePrincipalPoint_num_ = num;
}

void GraphSolver::SetPinholePrincipalPointNodesFromStackedHost(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholePrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholePrincipalPoint_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  PinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholePrincipalPointNodesFromStackedDevice(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholePrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholePrincipalPoint_num_");
  }
  PinholePrincipalPointStackedToCaspar(
      data,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      offset,
      num);
}

void GraphSolver::GetPinholePrincipalPointNodesToStackedHost(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholePrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholePrincipalPoint_num_");
  }
  PinholePrincipalPointCasparToStacked(
      nodes__PinholePrincipalPoint__storage_current_,
      marker__scratch_inout_,
      PinholePrincipalPoint_num_max_,
      offset,
      num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             2 * num * sizeof(double),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetPinholePrincipalPointNodesToStackedDevice(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholePrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholePrincipalPoint_num_");
  }
  PinholePrincipalPointCasparToStacked(
      nodes__PinholePrincipalPoint__storage_current_,
      data,
      PinholePrincipalPoint_num_max_,
      offset,
      num);
}

void GraphSolver::SetPointNum(const size_t num) {
  if (num > Point_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > Point_num_max_");
  }
  Point_num_ = num;
}

void GraphSolver::SetPointNodesFromStackedHost(const double* const data,
                                               const size_t offset,
                                               const size_t num) {
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  PointStackedToCaspar(marker__scratch_inout_,
                       nodes__Point__storage_current_,
                       Point_num_max_,
                       offset,
                       num);
}

void GraphSolver::SetPointNodesFromStackedDevice(const double* const data,
                                                 const size_t offset,
                                                 const size_t num) {
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  PointStackedToCaspar(
      data, nodes__Point__storage_current_, Point_num_max_, offset, num);
}

void GraphSolver::GetPointNodesToStackedHost(double* const data,
                                             const size_t offset,
                                             const size_t num) {
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  PointCasparToStacked(nodes__Point__storage_current_,
                       marker__scratch_inout_,
                       Point_num_max_,
                       offset,
                       num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             3 * num * sizeof(double),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetPointNodesToStackedDevice(double* const data,
                                               const size_t offset,
                                               const size_t num) {
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  PointCasparToStacked(
      nodes__Point__storage_current_, data, Point_num_max_, offset, num);
}

void GraphSolver::SetSimpleRadialCalibNum(const size_t num) {
  if (num > SimpleRadialCalib_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > SimpleRadialCalib_num_max_");
  }
  SimpleRadialCalib_num_ = num;
}

void GraphSolver::SetSimpleRadialCalibNodesFromStackedHost(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialCalib_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             4 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  SimpleRadialCalibStackedToCaspar(marker__scratch_inout_,
                                   nodes__SimpleRadialCalib__storage_current_,
                                   SimpleRadialCalib_num_max_,
                                   offset,
                                   num);
}

void GraphSolver::SetSimpleRadialCalibNodesFromStackedDevice(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialCalib_num_");
  }
  SimpleRadialCalibStackedToCaspar(data,
                                   nodes__SimpleRadialCalib__storage_current_,
                                   SimpleRadialCalib_num_max_,
                                   offset,
                                   num);
}

void GraphSolver::GetSimpleRadialCalibNodesToStackedHost(double* const data,
                                                         const size_t offset,
                                                         const size_t num) {
  if (offset + num > SimpleRadialCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialCalib_num_");
  }
  SimpleRadialCalibCasparToStacked(nodes__SimpleRadialCalib__storage_current_,
                                   marker__scratch_inout_,
                                   SimpleRadialCalib_num_max_,
                                   offset,
                                   num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             4 * num * sizeof(double),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetSimpleRadialCalibNodesToStackedDevice(double* const data,
                                                           const size_t offset,
                                                           const size_t num) {
  if (offset + num > SimpleRadialCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialCalib_num_");
  }
  SimpleRadialCalibCasparToStacked(nodes__SimpleRadialCalib__storage_current_,
                                   data,
                                   SimpleRadialCalib_num_max_,
                                   offset,
                                   num);
}

void GraphSolver::SetSimpleRadialFocalAndExtraNum(const size_t num) {
  if (num > SimpleRadialFocalAndExtra_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > SimpleRadialFocalAndExtra_num_max_");
  }
  SimpleRadialFocalAndExtra_num_ = num;
}

void GraphSolver::SetSimpleRadialFocalAndExtraNodesFromStackedHost(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialFocalAndExtra_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  SimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialFocalAndExtraNodesFromStackedDevice(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialFocalAndExtra_num_");
  }
  SimpleRadialFocalAndExtraStackedToCaspar(
      data,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      offset,
      num);
}

void GraphSolver::GetSimpleRadialFocalAndExtraNodesToStackedHost(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialFocalAndExtra_num_");
  }
  SimpleRadialFocalAndExtraCasparToStacked(
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      marker__scratch_inout_,
      SimpleRadialFocalAndExtra_num_max_,
      offset,
      num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             2 * num * sizeof(double),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetSimpleRadialFocalAndExtraNodesToStackedDevice(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialFocalAndExtra_num_");
  }
  SimpleRadialFocalAndExtraCasparToStacked(
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      data,
      SimpleRadialFocalAndExtra_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialPoseNum(const size_t num) {
  if (num > SimpleRadialPose_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > SimpleRadialPose_num_max_");
  }
  SimpleRadialPose_num_ = num;
}

void GraphSolver::SetSimpleRadialPoseNodesFromStackedHost(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialPose_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialPose_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  SimpleRadialPoseStackedToCaspar(marker__scratch_inout_,
                                  nodes__SimpleRadialPose__storage_current_,
                                  SimpleRadialPose_num_max_,
                                  offset,
                                  num);
}

void GraphSolver::SetSimpleRadialPoseNodesFromStackedDevice(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialPose_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialPose_num_");
  }
  SimpleRadialPoseStackedToCaspar(data,
                                  nodes__SimpleRadialPose__storage_current_,
                                  SimpleRadialPose_num_max_,
                                  offset,
                                  num);
}

void GraphSolver::GetSimpleRadialPoseNodesToStackedHost(double* const data,
                                                        const size_t offset,
                                                        const size_t num) {
  if (offset + num > SimpleRadialPose_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialPose_num_");
  }
  SimpleRadialPoseCasparToStacked(nodes__SimpleRadialPose__storage_current_,
                                  marker__scratch_inout_,
                                  SimpleRadialPose_num_max_,
                                  offset,
                                  num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             7 * num * sizeof(double),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetSimpleRadialPoseNodesToStackedDevice(double* const data,
                                                          const size_t offset,
                                                          const size_t num) {
  if (offset + num > SimpleRadialPose_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialPose_num_");
  }
  SimpleRadialPoseCasparToStacked(nodes__SimpleRadialPose__storage_current_,
                                  data,
                                  SimpleRadialPose_num_max_,
                                  offset,
                                  num);
}

void GraphSolver::SetSimpleRadialPrincipalPointNum(const size_t num) {
  if (num > SimpleRadialPrincipalPoint_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > SimpleRadialPrincipalPoint_num_max_");
  }
  SimpleRadialPrincipalPoint_num_ = num;
}

void GraphSolver::SetSimpleRadialPrincipalPointNodesFromStackedHost(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialPrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialPrincipalPoint_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  SimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialPrincipalPointNodesFromStackedDevice(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialPrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialPrincipalPoint_num_");
  }
  SimpleRadialPrincipalPointStackedToCaspar(
      data,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      offset,
      num);
}

void GraphSolver::GetSimpleRadialPrincipalPointNodesToStackedHost(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialPrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialPrincipalPoint_num_");
  }
  SimpleRadialPrincipalPointCasparToStacked(
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      marker__scratch_inout_,
      SimpleRadialPrincipalPoint_num_max_,
      offset,
      num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             2 * num * sizeof(double),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetSimpleRadialPrincipalPointNodesToStackedDevice(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialPrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialPrincipalPoint_num_");
  }
  SimpleRadialPrincipalPointCasparToStacked(
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      data,
      SimpleRadialPrincipalPoint_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialMergedNum(const size_t num) {
  if (num > simple_radial_merged_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > simple_radial_merged_num_max_");
  }
  simple_radial_merged_num_ = num;
}
void GraphSolver::SetSimpleRadialMergedPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_merged_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_num_. Use Setsimple_radial_mergedNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialMergedPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialMergedPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_merged_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_num_. Use Setsimple_radial_mergedNum before "
        "setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices, facs__simple_radial_merged__args__pose__idx_shared_, num);
}
void GraphSolver::SetSimpleRadialMergedCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_merged_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_num_. Use Setsimple_radial_mergedNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialMergedCalibIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialMergedCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_merged_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_num_. Use Setsimple_radial_mergedNum before "
        "setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices, facs__simple_radial_merged__args__calib__idx_shared_, num);
}
void GraphSolver::SetSimpleRadialMergedPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_merged_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_num_. Use Setsimple_radial_mergedNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialMergedPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialMergedPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_merged_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_num_. Use Setsimple_radial_mergedNum before "
        "setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices, facs__simple_radial_merged__args__point__idx_shared_, num);
}
void GraphSolver::SetSimpleRadialMergedPixelDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_merged_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(marker__scratch_inout_,
                            facs__simple_radial_merged__args__pixel__data_,
                            simple_radial_merged_num_max_,
                            offset,
                            num);
}

void GraphSolver::SetSimpleRadialMergedPixelDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_merged_num_max_");
  }
  ConstPixelStackedToCaspar(data,
                            facs__simple_radial_merged__args__pixel__data_,
                            simple_radial_merged_num_max_,
                            offset,
                            num);
}
void GraphSolver::SetSimpleRadialMergedFixedPoseNum(const size_t num) {
  if (num > simple_radial_merged_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > simple_radial_merged_fixed_pose_num_max_");
  }
  simple_radial_merged_fixed_pose_num_ = num;
}
void GraphSolver::SetSimpleRadialMergedFixedPoseCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_merged_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_fixed_pose_num_. Use "
        "Setsimple_radial_merged_fixed_poseNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialMergedFixedPoseCalibIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialMergedFixedPoseCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_merged_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_fixed_pose_num_. Use "
        "Setsimple_radial_merged_fixed_poseNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices,
                facs__simple_radial_merged_fixed_pose__args__calib__idx_shared_,
                num);
}
void GraphSolver::SetSimpleRadialMergedFixedPosePointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_merged_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_fixed_pose_num_. Use "
        "Setsimple_radial_merged_fixed_poseNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialMergedFixedPosePointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialMergedFixedPosePointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_merged_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_fixed_pose_num_. Use "
        "Setsimple_radial_merged_fixed_poseNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices,
                facs__simple_radial_merged_fixed_pose__args__point__idx_shared_,
                num);
}
void GraphSolver::SetSimpleRadialMergedFixedPosePixelDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_merged_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_merged_fixed_pose__args__pixel__data_,
      simple_radial_merged_fixed_pose_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialMergedFixedPosePixelDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_merged_fixed_pose_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_merged_fixed_pose__args__pixel__data_,
      simple_radial_merged_fixed_pose_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialMergedFixedPosePoseDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_merged_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPoseStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_merged_fixed_pose__args__pose__data_,
      simple_radial_merged_fixed_pose_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialMergedFixedPosePoseDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_merged_fixed_pose_num_max_");
  }
  ConstSimpleRadialPoseStackedToCaspar(
      data,
      facs__simple_radial_merged_fixed_pose__args__pose__data_,
      simple_radial_merged_fixed_pose_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialMergedFixedPointNum(const size_t num) {
  if (num > simple_radial_merged_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > simple_radial_merged_fixed_point_num_max_");
  }
  simple_radial_merged_fixed_point_num_ = num;
}
void GraphSolver::SetSimpleRadialMergedFixedPointPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_merged_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_fixed_point_num_. Use "
        "Setsimple_radial_merged_fixed_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialMergedFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialMergedFixedPointPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_merged_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_fixed_point_num_. Use "
        "Setsimple_radial_merged_fixed_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices,
                facs__simple_radial_merged_fixed_point__args__pose__idx_shared_,
                num);
}
void GraphSolver::SetSimpleRadialMergedFixedPointCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_merged_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_fixed_point_num_. Use "
        "Setsimple_radial_merged_fixed_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialMergedFixedPointCalibIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialMergedFixedPointCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_merged_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_fixed_point_num_. Use "
        "Setsimple_radial_merged_fixed_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_merged_fixed_point__args__calib__idx_shared_,
      num);
}
void GraphSolver::SetSimpleRadialMergedFixedPointPixelDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_merged_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_merged_fixed_point__args__pixel__data_,
      simple_radial_merged_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialMergedFixedPointPixelDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_merged_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_merged_fixed_point__args__pixel__data_,
      simple_radial_merged_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialMergedFixedPointPointDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_merged_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_merged_fixed_point__args__point__data_,
      simple_radial_merged_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialMergedFixedPointPointDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_merged_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__simple_radial_merged_fixed_point__args__point__data_,
      simple_radial_merged_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialMergedFixedPoseFixedPointNum(
    const size_t num) {
  if (num > simple_radial_merged_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_merged_fixed_pose_fixed_point_num_max_");
  }
  simple_radial_merged_fixed_pose_fixed_point_num_ = num;
}
void GraphSolver::SetSimpleRadialMergedFixedPoseFixedPointCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_merged_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_fixed_pose_fixed_point_num_. Use "
        "Setsimple_radial_merged_fixed_pose_fixed_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialMergedFixedPoseFixedPointCalibIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialMergedFixedPoseFixedPointCalibIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_merged_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_merged_fixed_pose_fixed_point_num_. Use "
        "Setsimple_radial_merged_fixed_pose_fixed_pointNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__calib__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialMergedFixedPoseFixedPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_merged_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialMergedFixedPoseFixedPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_merged_fixed_pose_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialMergedFixedPoseFixedPointPoseDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_merged_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPoseStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialMergedFixedPoseFixedPointPoseDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_merged_fixed_pose_fixed_point_num_max_");
  }
  ConstSimpleRadialPoseStackedToCaspar(
      data,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialMergedFixedPoseFixedPointPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_merged_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__point__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialMergedFixedPoseFixedPointPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_merged_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_merged_fixed_pose_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__simple_radial_merged_fixed_pose_fixed_point__args__point__data_,
      simple_radial_merged_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeMergedNum(const size_t num) {
  if (num > pinhole_merged_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_merged_num_max_");
  }
  pinhole_merged_num_ = num;
}
void GraphSolver::SetPinholeMergedPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_merged_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != pinhole_merged_num_. Use "
                             "Setpinhole_mergedNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeMergedPoseIndicesFromDevice((unsigned int*)marker__scratch_inout_,
                                        num);
}

void GraphSolver::SetPinholeMergedPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_merged_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != pinhole_merged_num_. Use "
                             "Setpinhole_mergedNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__pinhole_merged__args__pose__idx_shared_, num);
}
void GraphSolver::SetPinholeMergedCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_merged_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != pinhole_merged_num_. Use "
                             "Setpinhole_mergedNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeMergedCalibIndicesFromDevice((unsigned int*)marker__scratch_inout_,
                                         num);
}

void GraphSolver::SetPinholeMergedCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_merged_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != pinhole_merged_num_. Use "
                             "Setpinhole_mergedNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__pinhole_merged__args__calib__idx_shared_, num);
}
void GraphSolver::SetPinholeMergedPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_merged_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != pinhole_merged_num_. Use "
                             "Setpinhole_mergedNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeMergedPointIndicesFromDevice((unsigned int*)marker__scratch_inout_,
                                         num);
}

void GraphSolver::SetPinholeMergedPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_merged_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != pinhole_merged_num_. Use "
                             "Setpinhole_mergedNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__pinhole_merged__args__point__idx_shared_, num);
}
void GraphSolver::SetPinholeMergedPixelDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_merged_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(marker__scratch_inout_,
                            facs__pinhole_merged__args__pixel__data_,
                            pinhole_merged_num_max_,
                            offset,
                            num);
}

void GraphSolver::SetPinholeMergedPixelDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_merged_num_max_");
  }
  ConstPixelStackedToCaspar(data,
                            facs__pinhole_merged__args__pixel__data_,
                            pinhole_merged_num_max_,
                            offset,
                            num);
}
void GraphSolver::SetPinholeMergedFixedPoseNum(const size_t num) {
  if (num > pinhole_merged_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_merged_fixed_pose_num_max_");
  }
  pinhole_merged_fixed_pose_num_ = num;
}
void GraphSolver::SetPinholeMergedFixedPoseCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_merged_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_merged_fixed_pose_num_. Use "
        "Setpinhole_merged_fixed_poseNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeMergedFixedPoseCalibIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeMergedFixedPoseCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_merged_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_merged_fixed_pose_num_. Use "
        "Setpinhole_merged_fixed_poseNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices, facs__pinhole_merged_fixed_pose__args__calib__idx_shared_, num);
}
void GraphSolver::SetPinholeMergedFixedPosePointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_merged_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_merged_fixed_pose_num_. Use "
        "Setpinhole_merged_fixed_poseNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeMergedFixedPosePointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeMergedFixedPosePointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_merged_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_merged_fixed_pose_num_. Use "
        "Setpinhole_merged_fixed_poseNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices, facs__pinhole_merged_fixed_pose__args__point__idx_shared_, num);
}
void GraphSolver::SetPinholeMergedFixedPosePixelDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_merged_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(marker__scratch_inout_,
                            facs__pinhole_merged_fixed_pose__args__pixel__data_,
                            pinhole_merged_fixed_pose_num_max_,
                            offset,
                            num);
}

void GraphSolver::SetPinholeMergedFixedPosePixelDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_merged_fixed_pose_num_max_");
  }
  ConstPixelStackedToCaspar(data,
                            facs__pinhole_merged_fixed_pose__args__pixel__data_,
                            pinhole_merged_fixed_pose_num_max_,
                            offset,
                            num);
}
void GraphSolver::SetPinholeMergedFixedPosePoseDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_merged_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePoseStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_merged_fixed_pose__args__pose__data_,
      pinhole_merged_fixed_pose_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeMergedFixedPosePoseDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_merged_fixed_pose_num_max_");
  }
  ConstPinholePoseStackedToCaspar(
      data,
      facs__pinhole_merged_fixed_pose__args__pose__data_,
      pinhole_merged_fixed_pose_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeMergedFixedPointNum(const size_t num) {
  if (num > pinhole_merged_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_merged_fixed_point_num_max_");
  }
  pinhole_merged_fixed_point_num_ = num;
}
void GraphSolver::SetPinholeMergedFixedPointPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_merged_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_merged_fixed_point_num_. Use "
        "Setpinhole_merged_fixed_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeMergedFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeMergedFixedPointPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_merged_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_merged_fixed_point_num_. Use "
        "Setpinhole_merged_fixed_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices, facs__pinhole_merged_fixed_point__args__pose__idx_shared_, num);
}
void GraphSolver::SetPinholeMergedFixedPointCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_merged_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_merged_fixed_point_num_. Use "
        "Setpinhole_merged_fixed_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeMergedFixedPointCalibIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeMergedFixedPointCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_merged_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_merged_fixed_point_num_. Use "
        "Setpinhole_merged_fixed_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices, facs__pinhole_merged_fixed_point__args__calib__idx_shared_, num);
}
void GraphSolver::SetPinholeMergedFixedPointPixelDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_merged_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_merged_fixed_point__args__pixel__data_,
      pinhole_merged_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeMergedFixedPointPixelDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_merged_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_merged_fixed_point__args__pixel__data_,
      pinhole_merged_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeMergedFixedPointPointDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_merged_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_merged_fixed_point__args__point__data_,
      pinhole_merged_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeMergedFixedPointPointDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_merged_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__pinhole_merged_fixed_point__args__point__data_,
      pinhole_merged_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeMergedFixedPoseFixedPointNum(const size_t num) {
  if (num > pinhole_merged_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_merged_fixed_pose_fixed_point_num_max_");
  }
  pinhole_merged_fixed_pose_fixed_point_num_ = num;
}
void GraphSolver::SetPinholeMergedFixedPoseFixedPointCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_merged_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_merged_fixed_pose_fixed_point_num_. Use "
        "Setpinhole_merged_fixed_pose_fixed_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeMergedFixedPoseFixedPointCalibIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeMergedFixedPoseFixedPointCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_merged_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_merged_fixed_pose_fixed_point_num_. Use "
        "Setpinhole_merged_fixed_pose_fixed_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_merged_fixed_pose_fixed_point__args__calib__idx_shared_,
      num);
}
void GraphSolver::SetPinholeMergedFixedPoseFixedPointPixelDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_merged_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeMergedFixedPoseFixedPointPixelDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_merged_fixed_pose_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_merged_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeMergedFixedPoseFixedPointPoseDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_merged_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePoseStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__pose__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeMergedFixedPoseFixedPointPoseDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_merged_fixed_pose_fixed_point_num_max_");
  }
  ConstPinholePoseStackedToCaspar(
      data,
      facs__pinhole_merged_fixed_pose_fixed_point__args__pose__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeMergedFixedPoseFixedPointPointDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_merged_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_merged_fixed_pose_fixed_point__args__point__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeMergedFixedPoseFixedPointPointDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_merged_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_merged_fixed_pose_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__pinhole_merged_fixed_pose_fixed_point__args__point__data_,
      pinhole_merged_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedFocalAndExtraNum(const size_t num) {
  if (num > simple_radial_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > simple_radial_fixed_focal_and_extra_num_max_");
  }
  simple_radial_fixed_focal_and_extra_num_ = num;
}
void GraphSolver::SetSimpleRadialFixedFocalAndExtraPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_fixed_focal_and_extraNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedFocalAndExtraPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialFixedFocalAndExtraPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_fixed_focal_and_extraNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_focal_and_extra__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraPrincipalPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_fixed_focal_and_extraNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedFocalAndExtraPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraPrincipalPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_fixed_focal_and_extraNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_focal_and_extra__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::SetSimpleRadialFixedFocalAndExtraPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_fixed_focal_and_extraNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedFocalAndExtraPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialFixedFocalAndExtraPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_fixed_focal_and_extraNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_focal_and_extra__args__point__idx_shared_,
      num);
}
void GraphSolver::SetSimpleRadialFixedFocalAndExtraPixelDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialFixedFocalAndExtraPixelDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_focal_and_extra_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_focal_and_extra_num_max_");
  }
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      data,
      facs__simple_radial_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedPrincipalPointNum(const size_t num) {
  if (num > simple_radial_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > simple_radial_fixed_principal_point_num_max_");
  }
  simple_radial_fixed_principal_point_num_ = num;
}
void GraphSolver::SetSimpleRadialFixedPrincipalPointPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_num_. Use "
        "Setsimple_radial_fixed_principal_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPrincipalPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialFixedPrincipalPointPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_num_. Use "
        "Setsimple_radial_fixed_principal_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_principal_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPrincipalPointFocalAndExtraIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_num_. Use "
        "Setsimple_radial_fixed_principal_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPrincipalPointFocalAndExtraIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedPrincipalPointFocalAndExtraIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_num_. Use "
        "Setsimple_radial_fixed_principal_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_principal_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::SetSimpleRadialFixedPrincipalPointPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_num_. Use "
        "Setsimple_radial_fixed_principal_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialFixedPrincipalPointPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_num_. Use "
        "Setsimple_radial_fixed_principal_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::SetSimpleRadialFixedPrincipalPointPixelDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialFixedPrincipalPointPixelDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      data,
      facs__simple_radial_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedPoseFixedFocalAndExtraNum(
    const size_t num) {
  if (num > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  simple_radial_fixed_pose_fixed_focal_and_extra_num_ = num;
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_fixed_pose_fixed_focal_and_extraNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_fixed_pose_fixed_focal_and_extraNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_fixed_pose_fixed_focal_and_extraNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPoseFixedFocalAndExtraPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_fixed_pose_fixed_focal_and_extraNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraPoseDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPoseStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraPoseDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstSimpleRadialPoseStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedPoseFixedPrincipalPointNum(
    const size_t num) {
  if (num > simple_radial_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_pose_fixed_principal_point_num_max_");
  }
  simple_radial_fixed_pose_fixed_principal_point_num_ = num;
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_principal_point_num_. Use "
        "Setsimple_radial_fixed_pose_fixed_principal_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_principal_point_num_. Use "
        "Setsimple_radial_fixed_pose_fixed_principal_pointNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_principal_point_num_. Use "
        "Setsimple_radial_fixed_pose_fixed_principal_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPoseFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_principal_point_num_. Use "
        "Setsimple_radial_fixed_pose_fixed_principal_pointNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPoseStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPoseStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointNum(
    const size_t num) {
  if (num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > "
        "simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  simple_radial_fixed_focal_and_extra_fixed_principal_point_num_ = num;
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use Setsimple_radial_fixed_focal_and_extra_fixed_principal_pointNum "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use Setsimple_radial_fixed_focal_and_extra_fixed_principal_pointNum "
        "before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use Setsimple_radial_fixed_focal_and_extra_fixed_principal_pointNum "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use Setsimple_radial_fixed_focal_and_extra_fixed_principal_pointNum "
        "before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedFocalAndExtraFixedPointNum(
    const size_t num) {
  if (num > simple_radial_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_focal_and_extra_fixed_point_num_max_");
  }
  simple_radial_fixed_focal_and_extra_fixed_point_num_ = num;
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPointPoseIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_point_num_. Use "
        "Setsimple_radial_fixed_focal_and_extra_fixed_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedFocalAndExtraFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPointPoseIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_point_num_. Use "
        "Setsimple_radial_fixed_focal_and_extra_fixed_pointNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_point_num_. Use "
        "Setsimple_radial_fixed_focal_and_extra_fixed_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_point_num_. Use "
        "Setsimple_radial_fixed_focal_and_extra_fixed_pointNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPointPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedPrincipalPointFixedPointNum(
    const size_t num) {
  if (num > simple_radial_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_principal_point_fixed_point_num_max_");
  }
  simple_radial_fixed_principal_point_fixed_point_num_ = num;
}
void GraphSolver::
    SetSimpleRadialFixedPrincipalPointFixedPointPoseIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_fixed_point_num_. Use "
        "Setsimple_radial_fixed_principal_point_fixed_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedPrincipalPointFixedPointPoseIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_fixed_point_num_. Use "
        "Setsimple_radial_fixed_principal_point_fixed_pointNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_fixed_point_num_. Use "
        "Setsimple_radial_fixed_principal_point_fixed_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_fixed_point_num_. Use "
        "Setsimple_radial_fixed_principal_point_fixed_pointNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_principal_point_fixed_point_num_max_");
  }
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      data,
      facs__simple_radial_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPrincipalPointFixedPointPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPrincipalPointFixedPointPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__simple_radial_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointNum(
        const size_t num) {
  if (num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > "
                             "simple_radial_fixed_pose_fixed_focal_and_extra_"
                             "fixed_principal_point_num_max_");
  }
  simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_ =
      num;
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num !=
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
        "num_. Use "
        "Setsimple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
        "pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num !=
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
        "num_. Use "
        "Setsimple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
        "pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_pose_fixed_focal_and_extra_"
                             "fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_pose_fixed_focal_and_extra_"
                             "fixed_principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_pose_fixed_focal_and_extra_"
                             "fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPoseStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_pose_fixed_focal_and_extra_"
                             "fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPoseStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_pose_fixed_focal_and_extra_"
                             "fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_pose_fixed_focal_and_extra_"
                             "fixed_principal_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_pose_fixed_focal_and_extra_"
                             "fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_pose_fixed_focal_and_extra_"
                             "fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointNum(
    const size_t num) {
  if (num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_ = num;
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_. "
        "Use Setsimple_radial_fixed_pose_fixed_focal_and_extra_fixed_pointNum "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_. "
        "Use Setsimple_radial_fixed_pose_fixed_focal_and_extra_fixed_pointNum "
        "before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPoseStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstSimpleRadialPoseStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointNum(
    const size_t num) {
  if (num >
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > "
        "simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_ = num;
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_. "
        "Use Setsimple_radial_fixed_pose_fixed_principal_point_fixed_pointNum "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_. "
        "Use Setsimple_radial_fixed_pose_fixed_principal_point_fixed_pointNum "
        "before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPoseStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstSimpleRadialPoseStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointNum(
        const size_t num) {
  if (num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > "
                             "simple_radial_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_ =
      num;
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num !=
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
        "num_. Use "
        "Setsimple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
        "pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num !=
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
        "num_. Use "
        "Setsimple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
        "pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedFocalAndExtraNum(const size_t num) {
  if (num > pinhole_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_fixed_focal_and_extra_num_max_");
  }
  pinhole_fixed_focal_and_extra_num_ = num;
}
void GraphSolver::SetPinholeFixedFocalAndExtraPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_num_. Use "
        "Setpinhole_fixed_focal_and_extraNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedFocalAndExtraPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedFocalAndExtraPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_num_. Use "
        "Setpinhole_fixed_focal_and_extraNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices,
                facs__pinhole_fixed_focal_and_extra__args__pose__idx_shared_,
                num);
}
void GraphSolver::SetPinholeFixedFocalAndExtraPrincipalPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_num_. Use "
        "Setpinhole_fixed_focal_and_extraNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedFocalAndExtraPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedFocalAndExtraPrincipalPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_num_. Use "
        "Setpinhole_fixed_focal_and_extraNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_focal_and_extra__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::SetPinholeFixedFocalAndExtraPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_num_. Use "
        "Setpinhole_fixed_focal_and_extraNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedFocalAndExtraPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedFocalAndExtraPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_num_. Use "
        "Setpinhole_fixed_focal_and_extraNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices,
                facs__pinhole_fixed_focal_and_extra__args__point__idx_shared_,
                num);
}
void GraphSolver::SetPinholeFixedFocalAndExtraPixelDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra__args__pixel__data_,
      pinhole_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeFixedFocalAndExtraPixelDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_and_extra_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_fixed_focal_and_extra__args__pixel__data_,
      pinhole_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_and_extra_num_max_");
  }
  ConstPinholeFocalAndExtraStackedToCaspar(
      data,
      facs__pinhole_fixed_focal_and_extra__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedPrincipalPointNum(const size_t num) {
  if (num > pinhole_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_fixed_principal_point_num_max_");
  }
  pinhole_fixed_principal_point_num_ = num;
}
void GraphSolver::SetPinholeFixedPrincipalPointPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_num_. Use "
        "Setpinhole_fixed_principal_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPrincipalPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedPrincipalPointPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_num_. Use "
        "Setpinhole_fixed_principal_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices,
                facs__pinhole_fixed_principal_point__args__pose__idx_shared_,
                num);
}
void GraphSolver::SetPinholeFixedPrincipalPointFocalAndExtraIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_num_. Use "
        "Setpinhole_fixed_principal_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPrincipalPointFocalAndExtraIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedPrincipalPointFocalAndExtraIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_num_. Use "
        "Setpinhole_fixed_principal_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_principal_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::SetPinholeFixedPrincipalPointPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_num_. Use "
        "Setpinhole_fixed_principal_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedPrincipalPointPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_num_. Use "
        "Setpinhole_fixed_principal_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices,
                facs__pinhole_fixed_principal_point__args__point__idx_shared_,
                num);
}
void GraphSolver::SetPinholeFixedPrincipalPointPixelDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeFixedPrincipalPointPixelDataFromStackedDevice(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_principal_point_num_max_");
  }
  ConstPinholePrincipalPointStackedToCaspar(
      data,
      facs__pinhole_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedPoseFixedFocalAndExtraNum(const size_t num) {
  if (num > pinhole_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  pinhole_fixed_pose_fixed_focal_and_extra_num_ = num;
}
void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_and_extra_num_. Use "
        "Setpinhole_fixed_pose_fixed_focal_and_extraNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_and_extra_num_. Use "
        "Setpinhole_fixed_pose_fixed_focal_and_extraNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::SetPinholeFixedPoseFixedFocalAndExtraPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_and_extra_num_. Use "
        "Setpinhole_fixed_pose_fixed_focal_and_extraNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPoseFixedFocalAndExtraPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedPoseFixedFocalAndExtraPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_and_extra_num_. Use "
        "Setpinhole_fixed_pose_fixed_focal_and_extraNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      num);
}
void GraphSolver::SetPinholeFixedPoseFixedFocalAndExtraPixelDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedPoseFixedFocalAndExtraPoseDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePoseStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraPoseDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstPinholePoseStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstPinholeFocalAndExtraStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedPoseFixedPrincipalPointNum(const size_t num) {
  if (num > pinhole_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_pose_fixed_principal_point_num_max_");
  }
  pinhole_fixed_pose_fixed_principal_point_num_ = num;
}
void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_principal_point_num_. Use "
        "Setpinhole_fixed_pose_fixed_principal_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_principal_point_num_. Use "
        "Setpinhole_fixed_pose_fixed_principal_pointNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::SetPinholeFixedPoseFixedPrincipalPointPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_principal_point_num_. Use "
        "Setpinhole_fixed_pose_fixed_principal_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPoseFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedPoseFixedPrincipalPointPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_principal_point_num_. Use "
        "Setpinhole_fixed_pose_fixed_principal_pointNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePoseStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__pose__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstPinholePoseStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_principal_point__args__pose__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstPinholePrincipalPointStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedFocalAndExtraFixedPrincipalPointNum(
    const size_t num) {
  if (num > pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  pinhole_fixed_focal_and_extra_fixed_principal_point_num_ = num;
}
void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_principal_point_num_. Use "
        "Setpinhole_fixed_focal_and_extra_fixed_principal_pointNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_principal_point_num_. Use "
        "Setpinhole_fixed_focal_and_extra_fixed_principal_pointNum before "
        "setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_principal_point_num_. Use "
        "Setpinhole_fixed_focal_and_extra_fixed_principal_pointNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_principal_point_num_. Use "
        "Setpinhole_fixed_focal_and_extra_fixed_principal_pointNum before "
        "setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  ConstPinholeFocalAndExtraStackedToCaspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  ConstPinholePrincipalPointStackedToCaspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedFocalAndExtraFixedPointNum(const size_t num) {
  if (num > pinhole_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_focal_and_extra_fixed_point_num_max_");
  }
  pinhole_fixed_focal_and_extra_fixed_point_num_ = num;
}
void GraphSolver::SetPinholeFixedFocalAndExtraFixedPointPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_point_num_. Use "
        "Setpinhole_fixed_focal_and_extra_fixed_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedFocalAndExtraFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedFocalAndExtraFixedPointPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_point_num_. Use "
        "Setpinhole_fixed_focal_and_extra_fixed_pointNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_point_num_. Use "
        "Setpinhole_fixed_focal_and_extra_fixed_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_point_num_. Use "
        "Setpinhole_fixed_focal_and_extra_fixed_pointNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPinholeFocalAndExtraStackedToCaspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPointPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__point__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__point__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedPrincipalPointFixedPointNum(const size_t num) {
  if (num > pinhole_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_principal_point_fixed_point_num_max_");
  }
  pinhole_fixed_principal_point_fixed_point_num_ = num;
}
void GraphSolver::SetPinholeFixedPrincipalPointFixedPointPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_fixed_point_num_. Use "
        "Setpinhole_fixed_principal_point_fixed_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedPrincipalPointFixedPointPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_fixed_point_num_. Use "
        "Setpinhole_fixed_principal_point_fixed_pointNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_fixed_point_num_. Use "
        "Setpinhole_fixed_principal_point_fixed_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_fixed_point_num_. Use "
        "Setpinhole_fixed_principal_point_fixed_pointNum before setting "
        "indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPinholePrincipalPointStackedToCaspar(
      data,
      facs__pinhole_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedPrincipalPointFixedPointPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPrincipalPointFixedPointPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__pinhole_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointNum(
    const size_t num) {
  if (num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > "
                             "pinhole_fixed_pose_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_ = num;
}
void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num !=
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use "
        "Setpinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_pointNum "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num !=
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use "
        "Setpinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_pointNum "
        "before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_pose_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_pose_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_pose_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePoseStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_pose_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  ConstPinholePoseStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_pose_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_pose_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  ConstPinholeFocalAndExtraStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_pose_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_pose_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  ConstPinholePrincipalPointStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedPoseFixedFocalAndExtraFixedPointNum(
    const size_t num) {
  if (num > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_ = num;
}
void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_. Use "
        "Setpinhole_fixed_pose_fixed_focal_and_extra_fixed_pointNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_. Use "
        "Setpinhole_fixed_pose_fixed_focal_and_extra_fixed_pointNum before "
        "setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePoseStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPinholePoseStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPinholeFocalAndExtraStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedPoseFixedPrincipalPointFixedPointNum(
    const size_t num) {
  if (num > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  pinhole_fixed_pose_fixed_principal_point_fixed_point_num_ = num;
}
void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_principal_point_fixed_point_num_. Use "
        "Setpinhole_fixed_pose_fixed_principal_point_fixed_pointNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_principal_point_fixed_point_num_. Use "
        "Setpinhole_fixed_pose_fixed_principal_point_fixed_pointNum before "
        "setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePoseStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPinholePoseStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPinholePrincipalPointStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointNum(
    const size_t num) {
  if (num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_ = num;
}
void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  if (num !=
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_. Use "
                             "Setpinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num !=
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_. Use "
                             "Setpinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  ConstPinholeFocalAndExtraStackedToCaspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  ConstPinholePrincipalPointStackedToCaspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedHost(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedDevice(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

size_t GraphSolver::get_nbytes() {
  size_t offset = 0;
  size_t at_least = 0;
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<double>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<double>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 8 * PinholePose_num_, 4);
  increment_offset<double>(offset, 8 * PinholePose_num_, 4);
  increment_offset<double>(offset, 8 * PinholePose_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<double>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<double>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 8 * SimpleRadialPose_num_, 4);
  increment_offset<double>(offset, 8 * SimpleRadialPose_num_, 4);
  increment_offset<double>(offset, 8 * SimpleRadialPose_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_merged_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_merged_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_merged_num_, 4);
  increment_offset<double>(offset, 2 * simple_radial_merged_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_merged_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_merged_fixed_pose_num_, 4);
  increment_offset<double>(offset, 2 * simple_radial_merged_fixed_pose_num_, 4);
  increment_offset<double>(offset, 8 * simple_radial_merged_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_merged_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_merged_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_merged_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 4 * simple_radial_merged_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_merged_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_merged_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 8 * simple_radial_merged_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 4 * simple_radial_merged_fixed_pose_fixed_point_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_merged_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_merged_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_merged_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_merged_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_merged_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_merged_fixed_pose_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_merged_fixed_pose_num_, 4);
  increment_offset<double>(offset, 8 * pinhole_merged_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_merged_fixed_point_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_merged_fixed_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_merged_fixed_point_num_, 4);
  increment_offset<double>(offset, 4 * pinhole_merged_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_merged_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_merged_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 8 * pinhole_merged_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 4 * pinhole_merged_fixed_pose_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 8 * simple_radial_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 8 * simple_radial_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 4 * simple_radial_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 4 * simple_radial_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      8 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      8 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      4 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      8 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      4 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      4 * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 8 * pinhole_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 8 * pinhole_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 4 * pinhole_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 4 * pinhole_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset,
      1 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      8 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 8 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 4 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 8 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 4 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset,
      1 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      4 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  at_least =
      std::max(at_least,
               offset + std::max({4 * PinholeCalib_num_max_,
                                  2 * PinholeFocalAndExtra_num_max_,
                                  7 * PinholePose_num_max_,
                                  2 * PinholePrincipalPoint_num_max_,
                                  3 * Point_num_max_,
                                  4 * SimpleRadialCalib_num_max_,
                                  2 * SimpleRadialFocalAndExtra_num_max_,
                                  7 * SimpleRadialPose_num_max_,
                                  2 * SimpleRadialPrincipalPoint_num_max_}) *
                            sizeof(double));
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * simple_radial_merged_num_, 4);
  increment_offset<double>(offset, 2 * simple_radial_merged_fixed_pose_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_merged_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_merged_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_merged_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_merged_fixed_pose_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_merged_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_merged_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(offset, 2 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset,
      2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset,
      2 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(offset, 12 * simple_radial_merged_num_, 4);
  increment_offset<double>(offset, 4 * simple_radial_merged_num_, 4);
  increment_offset<double>(offset, 6 * simple_radial_merged_num_, 4);
  increment_offset<double>(offset, 4 * simple_radial_merged_fixed_pose_num_, 4);
  increment_offset<double>(offset, 6 * simple_radial_merged_fixed_pose_num_, 4);
  increment_offset<double>(
      offset, 12 * simple_radial_merged_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 4 * simple_radial_merged_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 4 * simple_radial_merged_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(offset, 10 * pinhole_merged_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_merged_num_, 4);
  increment_offset<double>(offset, 6 * pinhole_merged_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_merged_fixed_pose_num_, 4);
  increment_offset<double>(offset, 6 * pinhole_merged_fixed_pose_num_, 4);
  increment_offset<double>(offset, 10 * pinhole_merged_fixed_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_merged_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_merged_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 12 * simple_radial_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 0 * simple_radial_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 6 * simple_radial_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 12 * simple_radial_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 4 * simple_radial_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 6 * simple_radial_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 0 * simple_radial_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 6 * simple_radial_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 4 * simple_radial_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 6 * simple_radial_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset,
      12 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      6 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset, 12 * simple_radial_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 0 * simple_radial_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 12 * simple_radial_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 4 * simple_radial_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset,
      6 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      0 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      4 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      12 *
          simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(offset, 10 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(offset, 0 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(offset, 6 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(offset, 10 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<double>(offset, 6 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 0 * pinhole_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 6 * pinhole_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 6 * pinhole_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 10 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 6 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 10 * pinhole_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 0 * pinhole_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 10 * pinhole_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset,
      6 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset, 0 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset,
      10 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 6 * PinholePose_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 6 * PinholePose_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 6 * PinholePose_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 6 * PinholePose_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 1);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 6 * PinholePose_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 6 * PinholePose_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 6 * PinholePose_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<double>(offset, 6 * PinholeCalib_num_, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 1 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 6 * PinholePose_num_, 4);
  increment_offset<double>(offset, 16 * PinholePose_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 1 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<double>(offset, 6 * SimpleRadialCalib_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 1 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<double>(offset, 16 * SimpleRadialPose_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 1 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 1);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * simple_radial_merged_num_, 4);
  increment_offset<double>(offset, 2 * simple_radial_merged_fixed_pose_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_merged_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_merged_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_merged_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_merged_fixed_pose_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_merged_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_merged_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(
      offset,
      2 * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(offset, 2 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_focal_and_extra_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset,
      2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<double>(
      offset,
      2 * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<double>(offset, 0 * 0, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);
  increment_offset<double>(offset, 1 * 1, 1);

  return std::max(offset, at_least);
}

}  // namespace caspar