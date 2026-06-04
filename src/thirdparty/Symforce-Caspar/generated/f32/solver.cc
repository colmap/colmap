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
#include "kernel_PinholeFocal_alpha_denominator_or_beta_numerator.h"
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
#include "kernel_pinhole_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_point_score.h"
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
#include "kernel_pinhole_split_fixed_focal_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_split_fixed_focal_fixed_point_res_jac.h"
#include "kernel_pinhole_split_fixed_focal_fixed_point_res_jac_first.h"
#include "kernel_pinhole_split_fixed_focal_fixed_point_score.h"
#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_score.h"
#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_res_jac.h"
#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_res_jac_first.h"
#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_score.h"
#include "kernel_pinhole_split_fixed_focal_jtjnjtr_direct.h"
#include "kernel_pinhole_split_fixed_focal_res_jac.h"
#include "kernel_pinhole_split_fixed_focal_res_jac_first.h"
#include "kernel_pinhole_split_fixed_focal_score.h"
#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_point_res_jac.h"
#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_point_res_jac_first.h"
#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_point_score.h"
#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_res_jac.h"
#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_res_jac_first.h"
#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_score.h"
#include "kernel_pinhole_split_fixed_pose_fixed_focal_jtjnjtr_direct.h"
#include "kernel_pinhole_split_fixed_pose_fixed_focal_res_jac.h"
#include "kernel_pinhole_split_fixed_pose_fixed_focal_res_jac_first.h"
#include "kernel_pinhole_split_fixed_pose_fixed_focal_score.h"
#include "kernel_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_score.h"
#include "kernel_pinhole_split_fixed_pose_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_pinhole_split_fixed_pose_fixed_principal_point_res_jac.h"
#include "kernel_pinhole_split_fixed_pose_fixed_principal_point_res_jac_first.h"
#include "kernel_pinhole_split_fixed_pose_fixed_principal_point_score.h"
#include "kernel_pinhole_split_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_split_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_pinhole_split_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_pinhole_split_fixed_principal_point_fixed_point_score.h"
#include "kernel_pinhole_split_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_pinhole_split_fixed_principal_point_res_jac.h"
#include "kernel_pinhole_split_fixed_principal_point_res_jac_first.h"
#include "kernel_pinhole_split_fixed_principal_point_score.h"
#include "kernel_simple_radial_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_point_score.h"
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
#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_point_res_jac.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_point_score.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_score.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_res_jac.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_res_jac_first.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_score.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_jtjnjtr_direct.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_res_jac.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_res_jac_first.h"
#include "kernel_simple_radial_split_fixed_focal_and_extra_score.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_score.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac_first.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_score.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_jtjnjtr_direct.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_res_jac.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_res_jac_first.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_score.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_score.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_principal_point_res_jac.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_principal_point_res_jac_first.h"
#include "kernel_simple_radial_split_fixed_pose_fixed_principal_point_score.h"
#include "kernel_simple_radial_split_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_split_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_simple_radial_split_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_split_fixed_principal_point_fixed_point_score.h"
#include "kernel_simple_radial_split_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_split_fixed_principal_point_res_jac.h"
#include "kernel_simple_radial_split_fixed_principal_point_res_jac_first.h"
#include "kernel_simple_radial_split_fixed_principal_point_score.h"
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
    size_t PinholeFocal_num_max,
    size_t PinholePose_num_max,
    size_t PinholePrincipalPoint_num_max,
    size_t Point_num_max,
    size_t SimpleRadialCalib_num_max,
    size_t SimpleRadialFocalAndExtra_num_max,
    size_t SimpleRadialPose_num_max,
    size_t SimpleRadialPrincipalPoint_num_max,
    size_t simple_radial_num_max,
    size_t simple_radial_fixed_pose_num_max,
    size_t simple_radial_fixed_point_num_max,
    size_t simple_radial_fixed_pose_fixed_point_num_max,
    size_t pinhole_num_max,
    size_t pinhole_fixed_pose_num_max,
    size_t pinhole_fixed_point_num_max,
    size_t pinhole_fixed_pose_fixed_point_num_max,
    size_t simple_radial_split_fixed_focal_and_extra_num_max,
    size_t simple_radial_split_fixed_principal_point_num_max,
    size_t simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max,
    size_t simple_radial_split_fixed_pose_fixed_principal_point_num_max,
    size_t
        simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max,
    size_t simple_radial_split_fixed_focal_and_extra_fixed_point_num_max,
    size_t simple_radial_split_fixed_principal_point_fixed_point_num_max,
    size_t
        simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max,
    size_t
        simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max,
    size_t
        simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max,
    size_t
        simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max,
    size_t pinhole_split_fixed_focal_num_max,
    size_t pinhole_split_fixed_principal_point_num_max,
    size_t pinhole_split_fixed_pose_fixed_focal_num_max,
    size_t pinhole_split_fixed_pose_fixed_principal_point_num_max,
    size_t pinhole_split_fixed_focal_fixed_principal_point_num_max,
    size_t pinhole_split_fixed_focal_fixed_point_num_max,
    size_t pinhole_split_fixed_principal_point_fixed_point_num_max,
    size_t pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max,
    size_t pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max,
    size_t pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max,
    size_t pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max,
    int device_id)
    : params_(params),
      device_id_(device_id),
      PinholeCalib_num_(PinholeCalib_num_max),
      PinholeCalib_num_max_(PinholeCalib_num_max),
      PinholeFocal_num_(PinholeFocal_num_max),
      PinholeFocal_num_max_(PinholeFocal_num_max),
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
      simple_radial_num_(simple_radial_num_max),
      simple_radial_num_max_(simple_radial_num_max),
      simple_radial_fixed_pose_num_(simple_radial_fixed_pose_num_max),
      simple_radial_fixed_pose_num_max_(simple_radial_fixed_pose_num_max),
      simple_radial_fixed_point_num_(simple_radial_fixed_point_num_max),
      simple_radial_fixed_point_num_max_(simple_radial_fixed_point_num_max),
      simple_radial_fixed_pose_fixed_point_num_(
          simple_radial_fixed_pose_fixed_point_num_max),
      simple_radial_fixed_pose_fixed_point_num_max_(
          simple_radial_fixed_pose_fixed_point_num_max),
      pinhole_num_(pinhole_num_max),
      pinhole_num_max_(pinhole_num_max),
      pinhole_fixed_pose_num_(pinhole_fixed_pose_num_max),
      pinhole_fixed_pose_num_max_(pinhole_fixed_pose_num_max),
      pinhole_fixed_point_num_(pinhole_fixed_point_num_max),
      pinhole_fixed_point_num_max_(pinhole_fixed_point_num_max),
      pinhole_fixed_pose_fixed_point_num_(
          pinhole_fixed_pose_fixed_point_num_max),
      pinhole_fixed_pose_fixed_point_num_max_(
          pinhole_fixed_pose_fixed_point_num_max),
      simple_radial_split_fixed_focal_and_extra_num_(
          simple_radial_split_fixed_focal_and_extra_num_max),
      simple_radial_split_fixed_focal_and_extra_num_max_(
          simple_radial_split_fixed_focal_and_extra_num_max),
      simple_radial_split_fixed_principal_point_num_(
          simple_radial_split_fixed_principal_point_num_max),
      simple_radial_split_fixed_principal_point_num_max_(
          simple_radial_split_fixed_principal_point_num_max),
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_(
          simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max),
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_(
          simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max),
      simple_radial_split_fixed_pose_fixed_principal_point_num_(
          simple_radial_split_fixed_pose_fixed_principal_point_num_max),
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_(
          simple_radial_split_fixed_pose_fixed_principal_point_num_max),
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_(
          simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max),
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_(
          simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max),
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_(
          simple_radial_split_fixed_focal_and_extra_fixed_point_num_max),
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_(
          simple_radial_split_fixed_focal_and_extra_fixed_point_num_max),
      simple_radial_split_fixed_principal_point_fixed_point_num_(
          simple_radial_split_fixed_principal_point_fixed_point_num_max),
      simple_radial_split_fixed_principal_point_fixed_point_num_max_(
          simple_radial_split_fixed_principal_point_fixed_point_num_max),
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_(
          simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max),
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_(
          simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max),
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_(
          simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max),
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_(
          simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max),
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_(
          simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max),
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_(
          simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max),
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_(
          simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max),
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_(
          simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max),
      pinhole_split_fixed_focal_num_(pinhole_split_fixed_focal_num_max),
      pinhole_split_fixed_focal_num_max_(pinhole_split_fixed_focal_num_max),
      pinhole_split_fixed_principal_point_num_(
          pinhole_split_fixed_principal_point_num_max),
      pinhole_split_fixed_principal_point_num_max_(
          pinhole_split_fixed_principal_point_num_max),
      pinhole_split_fixed_pose_fixed_focal_num_(
          pinhole_split_fixed_pose_fixed_focal_num_max),
      pinhole_split_fixed_pose_fixed_focal_num_max_(
          pinhole_split_fixed_pose_fixed_focal_num_max),
      pinhole_split_fixed_pose_fixed_principal_point_num_(
          pinhole_split_fixed_pose_fixed_principal_point_num_max),
      pinhole_split_fixed_pose_fixed_principal_point_num_max_(
          pinhole_split_fixed_pose_fixed_principal_point_num_max),
      pinhole_split_fixed_focal_fixed_principal_point_num_(
          pinhole_split_fixed_focal_fixed_principal_point_num_max),
      pinhole_split_fixed_focal_fixed_principal_point_num_max_(
          pinhole_split_fixed_focal_fixed_principal_point_num_max),
      pinhole_split_fixed_focal_fixed_point_num_(
          pinhole_split_fixed_focal_fixed_point_num_max),
      pinhole_split_fixed_focal_fixed_point_num_max_(
          pinhole_split_fixed_focal_fixed_point_num_max),
      pinhole_split_fixed_principal_point_fixed_point_num_(
          pinhole_split_fixed_principal_point_fixed_point_num_max),
      pinhole_split_fixed_principal_point_fixed_point_num_max_(
          pinhole_split_fixed_principal_point_fixed_point_num_max),
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_(
          pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max),
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_(
          pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max),
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_(
          pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max),
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_(
          pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max),
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_(
          pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max),
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_(
          pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max),
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_(
          pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max),
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_(
          pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max) {
  indices_valid_ = false;
  if (params.pcg_rel_error_exit <= 0.0f) {
    throw std::runtime_error("params.pcg_rel_error_exit must be positive");
  }
  if (params.diag_init < 0.0f) {
    throw std::runtime_error("params.diag_init must be positive");
  }
  allocation_size_ = get_nbytes();

  if (device_id_ < 0) {
    throw std::runtime_error("Invalid CUDA device id: " +
                             std::to_string(device_id_));
  }
  if (device_id_ != 0) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount <= device_id_) {
      throw std::runtime_error("CUDA detected " + std::to_string(deviceCount) +
                               " devices, but device " +
                               std::to_string(device_id_) +
                               " was requested (0-indexed)");
    }
  }
  cudaSetDevice(device_id_);
  cudaMalloc(&origin_ptr_, allocation_size_);

  size_t offset = 0;
  marker__start_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__storage_current_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeCalib__storage_check_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeCalib__storage_new_best_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeFocal__storage_current_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholeFocal__storage_check_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholeFocal__storage_new_best_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholePose__storage_current_ =
      assign_and_increment<float>(origin_ptr_, offset, 8 * PinholePose_num_, 4);
  nodes__PinholePose__storage_check_ =
      assign_and_increment<float>(origin_ptr_, offset, 8 * PinholePose_num_, 4);
  nodes__PinholePose__storage_new_best_ =
      assign_and_increment<float>(origin_ptr_, offset, 8 * PinholePose_num_, 4);
  nodes__PinholePrincipalPoint__storage_current_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__storage_check_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__storage_new_best_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__Point__storage_current_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__storage_check_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__storage_new_best_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__SimpleRadialCalib__storage_current_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialCalib__storage_check_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialCalib__storage_new_best_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialFocalAndExtra__storage_current_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__storage_check_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__storage_new_best_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPose__storage_current_ = assign_and_increment<float>(
      origin_ptr_, offset, 8 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPose__storage_check_ = assign_and_increment<float>(
      origin_ptr_, offset, 8 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPose__storage_new_best_ = assign_and_increment<float>(
      origin_ptr_, offset, 8 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPrincipalPoint__storage_current_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__storage_check_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__storage_new_best_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  facs__simple_radial__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_num_, 4);
  facs__simple_radial__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 8 * simple_radial_num_, 4);
  facs__simple_radial__args__calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_num_, 4);
  facs__simple_radial__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_num_, 4);
  facs__simple_radial__args__pixel__data_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_num_, 4);
  facs__simple_radial_fixed_pose__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 8 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_pose__args__calib__idx_shared_ =
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
  facs__simple_radial_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 8 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 4 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_pose_fixed_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  8 * simple_radial_fixed_pose_fixed_point_num_,
                                  4);
  facs__simple_radial_fixed_pose_fixed_point__args__calib__idx_shared_ =
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
  facs__pinhole__args__pose__idx_shared_ = assign_and_increment<SharedIndex>(
      origin_ptr_, offset, 1 * pinhole_num_, 4);
  facs__pinhole__args__sensor_from_rig__data_ =
      assign_and_increment<float>(origin_ptr_, offset, 8 * pinhole_num_, 4);
  facs__pinhole__args__calib__idx_shared_ = assign_and_increment<SharedIndex>(
      origin_ptr_, offset, 1 * pinhole_num_, 4);
  facs__pinhole__args__point__idx_shared_ = assign_and_increment<SharedIndex>(
      origin_ptr_, offset, 1 * pinhole_num_, 4);
  facs__pinhole__args__pixel__data_ =
      assign_and_increment<float>(origin_ptr_, offset, 2 * pinhole_num_, 4);
  facs__pinhole_fixed_pose__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 8 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__pixel__data_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__pose__data_ = assign_and_increment<float>(
      origin_ptr_, offset, 8 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 8 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__calib__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__pixel__data_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__point__data_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 8 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__args__calib__idx_shared_ =
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
  facs__simple_radial_split_fixed_focal_and_extra__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra__args__focal_and_extra__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_principal_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point_fixed_point__args__principal_point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_split_fixed_focal_num_, 4);
  facs__pinhole_split_fixed_focal__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 8 * pinhole_split_fixed_focal_num_, 4);
  facs__pinhole_split_fixed_focal__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_split_fixed_focal_num_, 4);
  facs__pinhole_split_fixed_focal__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_split_fixed_focal_num_, 4);
  facs__pinhole_split_fixed_focal__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_split_fixed_focal_num_, 4);
  facs__pinhole_split_fixed_focal__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_split_fixed_focal_num_, 4);
  facs__pinhole_split_fixed_principal_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_split_fixed_principal_point_num_, 4);
  facs__pinhole_split_fixed_principal_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 8 * pinhole_split_fixed_principal_point_num_, 4);
  facs__pinhole_split_fixed_principal_point__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_split_fixed_principal_point_num_, 4);
  facs__pinhole_split_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_split_fixed_principal_point_num_, 4);
  facs__pinhole_split_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_split_fixed_principal_point_num_, 4);
  facs__pinhole_split_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_split_fixed_principal_point_num_, 4);
  facs__pinhole_split_fixed_pose_fixed_focal__args__sensor_from_rig__data_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  8 * pinhole_split_fixed_pose_fixed_focal_num_,
                                  4);
  facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_split_fixed_pose_fixed_focal_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_split_fixed_pose_fixed_focal_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal__args__pixel__data_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * pinhole_split_fixed_pose_fixed_focal_num_,
                                  4);
  facs__pinhole_split_fixed_pose_fixed_focal__args__pose__data_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  8 * pinhole_split_fixed_pose_fixed_focal_num_,
                                  4);
  facs__pinhole_split_fixed_pose_fixed_focal__args__focal__data_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * pinhole_split_fixed_pose_fixed_focal_num_,
                                  4);
  facs__pinhole_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_split_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_split_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_split_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_split_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_split_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_split_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_point__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * pinhole_split_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_principal_point_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_principal_point_fixed_point__args__principal_point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_principal_point_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * pinhole_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__principal_point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__focal__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__sensor_from_rig__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          8 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pixel__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__focal__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__principal_point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__point__data_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
          4);
  marker__scratch_inout_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  facs__simple_radial__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_num_, 4);
  facs__simple_radial_fixed_pose__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_point__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_pose_fixed_point__res_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * simple_radial_fixed_pose_fixed_point_num_,
                                  4);
  facs__pinhole__res_ =
      assign_and_increment<float>(origin_ptr_, offset, 2 * pinhole_num_, 4);
  facs__pinhole_fixed_pose__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_point__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__simple_radial_split_fixed_focal_and_extra__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_principal_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_split_fixed_focal_num_, 4);
  facs__pinhole_split_fixed_principal_point__res_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_split_fixed_principal_point_num_, 4);
  facs__pinhole_split_fixed_pose_fixed_focal__res_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  2 * pinhole_split_fixed_pose_fixed_focal_num_,
                                  4);
  facs__pinhole_split_fixed_pose_fixed_principal_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_principal_point_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__res_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial__args__pose__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 12 * simple_radial_num_, 4);
  facs__simple_radial__args__calib__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * simple_radial_num_, 4);
  facs__simple_radial__args__point__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * simple_radial_num_, 4);
  facs__simple_radial_fixed_pose__args__calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 4 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_pose__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 6 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 12 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 4 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_pose_fixed_point__args__calib__jac_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  4 * simple_radial_fixed_pose_fixed_point_num_,
                                  4);
  facs__pinhole__args__pose__jac_ =
      assign_and_increment<float>(origin_ptr_, offset, 12 * pinhole_num_, 4);
  facs__pinhole__args__calib__jac_ =
      assign_and_increment<float>(origin_ptr_, offset, 2 * pinhole_num_, 4);
  facs__pinhole__args__point__jac_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * pinhole_num_, 4);
  facs__pinhole_fixed_pose__args__calib__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__point__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_point__args__pose__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 12 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__calib__jac_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__args__calib__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__simple_radial_split_fixed_focal_and_extra__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          12 * simple_radial_split_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra__args__principal_point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          0 * simple_radial_split_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          6 * simple_radial_split_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_principal_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          12 * simple_radial_split_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point__args__focal_and_extra__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_split_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          6 * simple_radial_split_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          0 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          6 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          6 * simple_radial_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          12 *
              simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          6 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          12 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__principal_point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          0 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          12 * simple_radial_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_extra__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          6 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          0 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          4 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          12 *
              simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 12 * pinhole_split_fixed_focal_num_, 4);
  facs__pinhole_split_fixed_focal__args__principal_point__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 0 * pinhole_split_fixed_focal_num_, 4);
  facs__pinhole_split_fixed_focal__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 6 * pinhole_split_fixed_focal_num_, 4);
  facs__pinhole_split_fixed_principal_point__args__pose__jac_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  12 * pinhole_split_fixed_principal_point_num_,
                                  4);
  facs__pinhole_split_fixed_principal_point__args__focal__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * pinhole_split_fixed_principal_point_num_, 4);
  facs__pinhole_split_fixed_principal_point__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 6 * pinhole_split_fixed_principal_point_num_, 4);
  facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__jac_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  0 * pinhole_split_fixed_pose_fixed_focal_num_,
                                  4);
  facs__pinhole_split_fixed_pose_fixed_focal__args__point__jac_ =
      assign_and_increment<float>(origin_ptr_,
                                  offset,
                                  6 * pinhole_split_fixed_pose_fixed_focal_num_,
                                  4);
  facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          6 * pinhole_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          12 * pinhole_split_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          6 * pinhole_split_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          12 * pinhole_split_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          0 * pinhole_split_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          12 * pinhole_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          6 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__principal_point__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          0 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__focal__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pose__jac_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          12 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
          4);
  nodes__PinholeCalib__z_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeCalib__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocal__z_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholeFocal__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePose__z_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePose__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePrincipalPoint__z_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__z_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialCalib__z_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialCalib__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialFocalAndExtra__z_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPose__z_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPose__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPrincipalPoint__z_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__z_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__p_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeCalib__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocal__p_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholeFocal__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePose__p_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePose__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePrincipalPoint__p_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__p_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialCalib__p_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialCalib__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialFocalAndExtra__p_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPose__p_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPose__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPrincipalPoint__p_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__p_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__step_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeCalib__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocal__step_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholeFocal__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePose__step_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePose__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePrincipalPoint__step_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__step_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialCalib__step_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialCalib__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialFocalAndExtra__step_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPose__step_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPose__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPrincipalPoint__step_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__step_end__ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  marker__w_start_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__w_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeFocal__w_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholePose__w_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePrincipalPoint__w_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__Point__w_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__SimpleRadialCalib__w_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialFocalAndExtra__w_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPose__w_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPrincipalPoint__w_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  marker__w_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 1);
  marker__r_0_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__r_0_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeFocal__r_0_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholePose__r_0_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePrincipalPoint__r_0_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__Point__r_0_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__SimpleRadialCalib__r_0_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialFocalAndExtra__r_0_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPose__r_0_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPrincipalPoint__r_0_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  marker__r_0_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  marker__r_k_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__r_k_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeFocal__r_k_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholePose__r_k_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePrincipalPoint__r_k_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__Point__r_k_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__SimpleRadialCalib__r_k_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialFocalAndExtra__r_k_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPose__r_k_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPrincipalPoint__r_k_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  marker__r_k_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  marker__Mp_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__Mp_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeFocal__Mp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholePose__Mp_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePrincipalPoint__Mp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__Point__Mp_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__SimpleRadialCalib__Mp_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialFocalAndExtra__Mp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPose__Mp_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPrincipalPoint__Mp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  marker__Mp_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  marker__precond_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeCalib__precond_diag_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * PinholeCalib_num_, 4);
  nodes__PinholeCalib__precond_tril_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * PinholeCalib_num_, 4);
  nodes__PinholeFocal__precond_diag_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholeFocal_num_, 4);
  nodes__PinholeFocal__precond_tril_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * PinholeFocal_num_, 4);
  nodes__PinholePose__precond_diag_ =
      assign_and_increment<float>(origin_ptr_, offset, 6 * PinholePose_num_, 4);
  nodes__PinholePose__precond_tril_ = assign_and_increment<float>(
      origin_ptr_, offset, 16 * PinholePose_num_, 4);
  nodes__PinholePrincipalPoint__precond_diag_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__precond_tril_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * PinholePrincipalPoint_num_, 4);
  nodes__Point__precond_diag_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__precond_tril_ =
      assign_and_increment<float>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__SimpleRadialCalib__precond_diag_ = assign_and_increment<float>(
      origin_ptr_, offset, 4 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialCalib__precond_tril_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * SimpleRadialCalib_num_, 4);
  nodes__SimpleRadialFocalAndExtra__precond_diag_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__precond_tril_ = assign_and_increment<float>(
      origin_ptr_, offset, 1 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPose__precond_diag_ = assign_and_increment<float>(
      origin_ptr_, offset, 6 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPose__precond_tril_ = assign_and_increment<float>(
      origin_ptr_, offset, 16 * SimpleRadialPose_num_, 4);
  nodes__SimpleRadialPrincipalPoint__precond_diag_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__precond_tril_ =
      assign_and_increment<float>(
          origin_ptr_, offset, 1 * SimpleRadialPrincipalPoint_num_, 4);
  marker__precond_end_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 1);
  marker__jp_start_ =
      assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);
  facs__simple_radial__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_num_, 4);
  facs__simple_radial_fixed_pose__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_point__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_pose_fixed_point__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * simple_radial_fixed_pose_fixed_point_num_, 4);
  facs__pinhole__jp_ =
      assign_and_increment<float>(origin_ptr_, offset, 2 * pinhole_num_, 4);
  facs__pinhole_fixed_pose__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_point__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__simple_radial_split_fixed_focal_and_extra__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_principal_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_principal_point_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_split_fixed_focal_num_, 4);
  facs__pinhole_split_fixed_principal_point__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_split_fixed_principal_point_num_, 4);
  facs__pinhole_split_fixed_pose_fixed_focal__jp_ = assign_and_increment<float>(
      origin_ptr_, offset, 2 * pinhole_split_fixed_pose_fixed_focal_num_, 4);
  facs__pinhole_split_fixed_pose_fixed_principal_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_principal_point_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
          4);
  facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__jp_ =
      assign_and_increment<float>(
          origin_ptr_,
          offset,
          2 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
          4);
  marker__jp_end_ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 1);
  solver__current_diag_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__alpha_numerator_ =
      assign_and_increment<float>(origin_ptr_, offset, 1 * 1, 1);
  solver__alpha_denominator_ =
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

GraphSolver::~GraphSolver() {
  cudaSetDevice(device_id_);
  cudaFree(origin_ptr_);
}

void GraphSolver::set_params(const SolverParams<double>& params) {
  this->params_ = params;
}

size_t GraphSolver::get_allocation_size() { return allocation_size_; }

SolveResult GraphSolver::solve(bool print_progress, bool verbose_logging) {
  cudaSetDevice(device_id_);
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
        float score_new_pcg = DoRetractScore();
        if (!(score_new_pcg <= score_best_pcg * params_.pcg_rel_decrease_min)) {
          break;
        }
        std::swap(nodes__PinholeCalib__storage_check_,
                  nodes__PinholeCalib__storage_new_best_);
        std::swap(nodes__PinholeFocal__storage_check_,
                  nodes__PinholeFocal__storage_new_best_);
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
      std::swap(nodes__PinholeFocal__storage_check_,
                nodes__PinholeFocal__storage_new_best_);
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

    const float diag_current = diag;
    bool step_accepted = false;
    if (score_best_pcg < score_best * params_.solver_rel_decrease_min) {
      quality = (score_best - score_best_pcg) / GetPredDecrease();
      const float quality_tmp = 2 * quality - 1;
      float scale = std::max(params_.diag_scaling_down,
                             1.0f - quality_tmp * quality_tmp * quality_tmp);
      diag = std::max(params_.diag_min, diag * scale);
      cudaMemcpy(
          solver__current_diag_, &diag, sizeof(float), cudaMemcpyHostToDevice);
      up_scale = params_.diag_scaling_up;
      score_best = score_best_pcg;
      std::swap(nodes__PinholeCalib__storage_current_,
                nodes__PinholeCalib__storage_new_best_);
      std::swap(nodes__PinholeFocal__storage_current_,
                nodes__PinholeFocal__storage_new_best_);
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

float GraphSolver::DoResJacFirst() {
  Zero(solver__res_tot_, solver__res_tot_ + 1);
  Zero(marker__r_0_start_, marker__precond_end_);

  SimpleRadialResJacFirst(nodes__SimpleRadialPose__storage_current_,
                          SimpleRadialPose_num_max_,
                          facs__simple_radial__args__pose__idx_shared_,
                          facs__simple_radial__args__sensor_from_rig__data_,
                          simple_radial_num_max_,
                          nodes__SimpleRadialCalib__storage_current_,
                          SimpleRadialCalib_num_max_,
                          facs__simple_radial__args__calib__idx_shared_,
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
                          nodes__SimpleRadialPose__r_k_,
                          SimpleRadialPose_num_,
                          nodes__SimpleRadialPose__precond_diag_,
                          SimpleRadialPose_num_,
                          nodes__SimpleRadialPose__precond_tril_,
                          SimpleRadialPose_num_,
                          facs__simple_radial__args__calib__jac_,
                          simple_radial_num_,
                          nodes__SimpleRadialCalib__r_k_,
                          SimpleRadialCalib_num_,
                          nodes__SimpleRadialCalib__precond_diag_,
                          SimpleRadialCalib_num_,
                          nodes__SimpleRadialCalib__precond_tril_,
                          SimpleRadialCalib_num_,
                          facs__simple_radial__args__point__jac_,
                          simple_radial_num_,
                          nodes__Point__r_k_,
                          Point_num_,
                          nodes__Point__precond_diag_,
                          Point_num_,
                          nodes__Point__precond_tril_,
                          Point_num_,
                          simple_radial_num_);

  SimpleRadialFixedPoseResJacFirst(
      facs__simple_radial_fixed_pose__args__sensor_from_rig__data_,
      simple_radial_fixed_pose_num_max_,
      nodes__SimpleRadialCalib__storage_current_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_fixed_pose__args__calib__idx_shared_,
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
      facs__simple_radial_fixed_pose__args__calib__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialCalib__r_k_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_diag_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_tril_,
      SimpleRadialCalib_num_,
      facs__simple_radial_fixed_pose__args__point__jac_,
      simple_radial_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_num_);

  SimpleRadialFixedPointResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_point__args__sensor_from_rig__data_,
      simple_radial_fixed_point_num_max_,
      nodes__SimpleRadialCalib__storage_current_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_fixed_point__args__calib__idx_shared_,
      facs__simple_radial_fixed_point__args__pixel__data_,
      simple_radial_fixed_point_num_max_,
      facs__simple_radial_fixed_point__args__point__data_,
      simple_radial_fixed_point_num_max_,

      facs__simple_radial_fixed_point__res_,
      simple_radial_fixed_point_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_point__args__pose__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_point__args__calib__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialCalib__r_k_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_diag_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_tril_,
      SimpleRadialCalib_num_,
      simple_radial_fixed_point_num_);

  SimpleRadialFixedPoseFixedPointResJacFirst(
      facs__simple_radial_fixed_pose_fixed_point__args__sensor_from_rig__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      nodes__SimpleRadialCalib__storage_current_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__calib__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_point__res_,
      simple_radial_fixed_pose_fixed_point_num_,
      solver__res_tot_,
      nodes__SimpleRadialCalib__r_k_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_diag_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_tril_,
      SimpleRadialCalib_num_,
      simple_radial_fixed_pose_fixed_point_num_);

  PinholeResJacFirst(nodes__PinholePose__storage_current_,
                     PinholePose_num_max_,
                     facs__pinhole__args__pose__idx_shared_,
                     facs__pinhole__args__sensor_from_rig__data_,
                     pinhole_num_max_,
                     nodes__PinholeCalib__storage_current_,
                     PinholeCalib_num_max_,
                     facs__pinhole__args__calib__idx_shared_,
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
                     nodes__PinholePose__r_k_,
                     PinholePose_num_,
                     nodes__PinholePose__precond_diag_,
                     PinholePose_num_,
                     nodes__PinholePose__precond_tril_,
                     PinholePose_num_,
                     facs__pinhole__args__calib__jac_,
                     pinhole_num_,
                     nodes__PinholeCalib__r_k_,
                     PinholeCalib_num_,
                     nodes__PinholeCalib__precond_diag_,
                     PinholeCalib_num_,
                     nodes__PinholeCalib__precond_tril_,
                     PinholeCalib_num_,
                     facs__pinhole__args__point__jac_,
                     pinhole_num_,
                     nodes__Point__r_k_,
                     Point_num_,
                     nodes__Point__precond_diag_,
                     Point_num_,
                     nodes__Point__precond_tril_,
                     Point_num_,
                     pinhole_num_);

  PinholeFixedPoseResJacFirst(
      facs__pinhole_fixed_pose__args__sensor_from_rig__data_,
      pinhole_fixed_pose_num_max_,
      nodes__PinholeCalib__storage_current_,
      PinholeCalib_num_max_,
      facs__pinhole_fixed_pose__args__calib__idx_shared_,
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
      facs__pinhole_fixed_pose__args__calib__jac_,
      pinhole_fixed_pose_num_,
      nodes__PinholeCalib__r_k_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_diag_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_tril_,
      PinholeCalib_num_,
      facs__pinhole_fixed_pose__args__point__jac_,
      pinhole_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_num_);

  PinholeFixedPointResJacFirst(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_point__args__sensor_from_rig__data_,
      pinhole_fixed_point_num_max_,
      nodes__PinholeCalib__storage_current_,
      PinholeCalib_num_max_,
      facs__pinhole_fixed_point__args__calib__idx_shared_,
      facs__pinhole_fixed_point__args__pixel__data_,
      pinhole_fixed_point_num_max_,
      facs__pinhole_fixed_point__args__point__data_,
      pinhole_fixed_point_num_max_,

      facs__pinhole_fixed_point__res_,
      pinhole_fixed_point_num_,
      solver__res_tot_,
      facs__pinhole_fixed_point__args__pose__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_fixed_point__args__calib__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholeCalib__r_k_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_diag_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_tril_,
      PinholeCalib_num_,
      pinhole_fixed_point_num_);

  PinholeFixedPoseFixedPointResJacFirst(
      facs__pinhole_fixed_pose_fixed_point__args__sensor_from_rig__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      nodes__PinholeCalib__storage_current_,
      PinholeCalib_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__calib__idx_shared_,
      facs__pinhole_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_point_num_max_,

      facs__pinhole_fixed_pose_fixed_point__res_,
      pinhole_fixed_pose_fixed_point_num_,
      solver__res_tot_,
      nodes__PinholeCalib__r_k_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_diag_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_tril_,
      PinholeCalib_num_,
      pinhole_fixed_pose_fixed_point_num_);

  SimpleRadialSplitFixedFocalAndExtraResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,

      facs__simple_radial_split_fixed_focal_and_extra__res_,
      simple_radial_split_fixed_focal_and_extra_num_,
      solver__res_tot_,
      facs__simple_radial_split_fixed_focal_and_extra__args__pose__jac_,
      simple_radial_split_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_focal_and_extra__args__principal_point__jac_,
      simple_radial_split_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_split_fixed_focal_and_extra__args__point__jac_,
      simple_radial_split_fixed_focal_and_extra_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_split_fixed_focal_and_extra_num_);

  SimpleRadialSplitFixedPrincipalPointResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_principal_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_principal_point_num_max_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_split_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_split_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_principal_point_num_max_,

      facs__simple_radial_split_fixed_principal_point__res_,
      simple_radial_split_fixed_principal_point_num_,
      solver__res_tot_,
      facs__simple_radial_split_fixed_principal_point__args__pose__jac_,
      simple_radial_split_fixed_principal_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_principal_point__args__focal_and_extra__jac_,
      simple_radial_split_fixed_principal_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_split_fixed_principal_point__args__point__jac_,
      simple_radial_split_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_split_fixed_principal_point_num_);

  SimpleRadialSplitFixedPoseFixedFocalAndExtraResJacFirst(
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,

      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__res_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
      solver__res_tot_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__point__jac_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_);

  SimpleRadialSplitFixedPoseFixedPrincipalPointResJacFirst(
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,

      facs__simple_radial_split_fixed_pose_fixed_principal_point__res_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_,
      solver__res_tot_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__jac_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_);

  SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,

      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__res_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      solver__res_tot_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_);

  SimpleRadialSplitFixedFocalAndExtraFixedPointResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,

      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__res_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      solver__res_tot_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pose__jac_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_);

  SimpleRadialSplitFixedPrincipalPointFixedPointResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,

      facs__simple_radial_split_fixed_principal_point_fixed_point__res_,
      simple_radial_split_fixed_principal_point_fixed_point_num_,
      solver__res_tot_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__jac_,
      simple_radial_split_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      simple_radial_split_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      simple_radial_split_fixed_principal_point_fixed_point_num_);

  SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointResJacFirst(
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,

      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__res_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      solver__res_tot_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_);

  SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointResJacFirst(
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,

      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__res_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      solver__res_tot_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_);

  SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointResJacFirst(
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,

      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__res_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      solver__res_tot_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_);

  SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacFirst(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,

      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__res_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      solver__res_tot_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_);

  PinholeSplitFixedFocalResJacFirst(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_focal__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_num_max_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_split_fixed_focal__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_split_fixed_focal__args__point__idx_shared_,
      facs__pinhole_split_fixed_focal__args__pixel__data_,
      pinhole_split_fixed_focal_num_max_,
      facs__pinhole_split_fixed_focal__args__focal__data_,
      pinhole_split_fixed_focal_num_max_,

      facs__pinhole_split_fixed_focal__res_,
      pinhole_split_fixed_focal_num_,
      solver__res_tot_,
      facs__pinhole_split_fixed_focal__args__pose__jac_,
      pinhole_split_fixed_focal_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_split_fixed_focal__args__principal_point__jac_,
      pinhole_split_fixed_focal_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      facs__pinhole_split_fixed_focal__args__point__jac_,
      pinhole_split_fixed_focal_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_split_fixed_focal_num_);

  PinholeSplitFixedPrincipalPointResJacFirst(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_principal_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_principal_point_num_max_,
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_split_fixed_principal_point__args__focal__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_split_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_principal_point_num_max_,

      facs__pinhole_split_fixed_principal_point__res_,
      pinhole_split_fixed_principal_point_num_,
      solver__res_tot_,
      facs__pinhole_split_fixed_principal_point__args__pose__jac_,
      pinhole_split_fixed_principal_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_split_fixed_principal_point__args__focal__jac_,
      pinhole_split_fixed_principal_point_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      facs__pinhole_split_fixed_principal_point__args__point__jac_,
      pinhole_split_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_split_fixed_principal_point_num_);

  PinholeSplitFixedPoseFixedFocalResJacFirst(
      facs__pinhole_split_fixed_pose_fixed_focal__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,

      facs__pinhole_split_fixed_pose_fixed_focal__res_,
      pinhole_split_fixed_pose_fixed_focal_num_,
      solver__res_tot_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__jac_,
      pinhole_split_fixed_pose_fixed_focal_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__point__jac_,
      pinhole_split_fixed_pose_fixed_focal_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_split_fixed_pose_fixed_focal_num_);

  PinholeSplitFixedPoseFixedPrincipalPointResJacFirst(
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,

      facs__pinhole_split_fixed_pose_fixed_principal_point__res_,
      pinhole_split_fixed_pose_fixed_principal_point_num_,
      solver__res_tot_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__jac_,
      pinhole_split_fixed_pose_fixed_principal_point_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__jac_,
      pinhole_split_fixed_pose_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_split_fixed_pose_fixed_principal_point_num_);

  PinholeSplitFixedFocalFixedPrincipalPointResJacFirst(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,

      facs__pinhole_split_fixed_focal_fixed_principal_point__res_,
      pinhole_split_fixed_focal_fixed_principal_point_num_,
      solver__res_tot_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__jac_,
      pinhole_split_fixed_focal_fixed_principal_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__jac_,
      pinhole_split_fixed_focal_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_split_fixed_focal_fixed_principal_point_num_);

  PinholeSplitFixedFocalFixedPointResJacFirst(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_focal_fixed_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_point__args__point__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,

      facs__pinhole_split_fixed_focal_fixed_point__res_,
      pinhole_split_fixed_focal_fixed_point_num_,
      solver__res_tot_,
      facs__pinhole_split_fixed_focal_fixed_point__args__pose__jac_,
      pinhole_split_fixed_focal_fixed_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__jac_,
      pinhole_split_fixed_focal_fixed_point_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      pinhole_split_fixed_focal_fixed_point_num_);

  PinholeSplitFixedPrincipalPointFixedPointResJacFirst(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__idx_shared_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,

      facs__pinhole_split_fixed_principal_point_fixed_point__res_,
      pinhole_split_fixed_principal_point_fixed_point_num_,
      solver__res_tot_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__jac_,
      pinhole_split_fixed_principal_point_fixed_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__jac_,
      pinhole_split_fixed_principal_point_fixed_point_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      pinhole_split_fixed_principal_point_fixed_point_num_);

  PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJacFirst(
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,

      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__res_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
      solver__res_tot_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_);

  PinholeSplitFixedPoseFixedFocalFixedPointResJacFirst(
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,

      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__res_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_,
      solver__res_tot_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_);

  PinholeSplitFixedPoseFixedPrincipalPointFixedPointResJacFirst(
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__focal__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,

      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__res_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      solver__res_tot_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_);

  PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJacFirst(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,

      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__res_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
      solver__res_tot_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__r_0_start_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
  return 0.5 * ReadCuMem(solver__res_tot_);
}
void GraphSolver::DoResJac() {
  Zero(solver__res_tot_, solver__res_tot_ + 1);
  Zero(marker__r_0_start_, marker__precond_end_);

  SimpleRadialResJac(nodes__SimpleRadialPose__storage_current_,
                     SimpleRadialPose_num_max_,
                     facs__simple_radial__args__pose__idx_shared_,
                     facs__simple_radial__args__sensor_from_rig__data_,
                     simple_radial_num_max_,
                     nodes__SimpleRadialCalib__storage_current_,
                     SimpleRadialCalib_num_max_,
                     facs__simple_radial__args__calib__idx_shared_,
                     nodes__Point__storage_current_,
                     Point_num_max_,
                     facs__simple_radial__args__point__idx_shared_,
                     facs__simple_radial__args__pixel__data_,
                     simple_radial_num_max_,

                     facs__simple_radial__res_,
                     simple_radial_num_,

                     facs__simple_radial__args__pose__jac_,
                     simple_radial_num_,
                     nodes__SimpleRadialPose__r_k_,
                     SimpleRadialPose_num_,
                     nodes__SimpleRadialPose__precond_diag_,
                     SimpleRadialPose_num_,
                     nodes__SimpleRadialPose__precond_tril_,
                     SimpleRadialPose_num_,
                     facs__simple_radial__args__calib__jac_,
                     simple_radial_num_,
                     nodes__SimpleRadialCalib__r_k_,
                     SimpleRadialCalib_num_,
                     nodes__SimpleRadialCalib__precond_diag_,
                     SimpleRadialCalib_num_,
                     nodes__SimpleRadialCalib__precond_tril_,
                     SimpleRadialCalib_num_,
                     facs__simple_radial__args__point__jac_,
                     simple_radial_num_,
                     nodes__Point__r_k_,
                     Point_num_,
                     nodes__Point__precond_diag_,
                     Point_num_,
                     nodes__Point__precond_tril_,
                     Point_num_,
                     simple_radial_num_);

  SimpleRadialFixedPoseResJac(
      facs__simple_radial_fixed_pose__args__sensor_from_rig__data_,
      simple_radial_fixed_pose_num_max_,
      nodes__SimpleRadialCalib__storage_current_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_fixed_pose__args__calib__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose__args__point__idx_shared_,
      facs__simple_radial_fixed_pose__args__pixel__data_,
      simple_radial_fixed_pose_num_max_,
      facs__simple_radial_fixed_pose__args__pose__data_,
      simple_radial_fixed_pose_num_max_,

      facs__simple_radial_fixed_pose__res_,
      simple_radial_fixed_pose_num_,

      facs__simple_radial_fixed_pose__args__calib__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialCalib__r_k_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_diag_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_tril_,
      SimpleRadialCalib_num_,
      facs__simple_radial_fixed_pose__args__point__jac_,
      simple_radial_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_num_);

  SimpleRadialFixedPointResJac(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_point__args__sensor_from_rig__data_,
      simple_radial_fixed_point_num_max_,
      nodes__SimpleRadialCalib__storage_current_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_fixed_point__args__calib__idx_shared_,
      facs__simple_radial_fixed_point__args__pixel__data_,
      simple_radial_fixed_point_num_max_,
      facs__simple_radial_fixed_point__args__point__data_,
      simple_radial_fixed_point_num_max_,

      facs__simple_radial_fixed_point__res_,
      simple_radial_fixed_point_num_,

      facs__simple_radial_fixed_point__args__pose__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_point__args__calib__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialCalib__r_k_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_diag_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_tril_,
      SimpleRadialCalib_num_,
      simple_radial_fixed_point_num_);

  SimpleRadialFixedPoseFixedPointResJac(
      facs__simple_radial_fixed_pose_fixed_point__args__sensor_from_rig__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      nodes__SimpleRadialCalib__storage_current_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__calib__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_point__res_,
      simple_radial_fixed_pose_fixed_point_num_,

      nodes__SimpleRadialCalib__r_k_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_diag_,
      SimpleRadialCalib_num_,
      nodes__SimpleRadialCalib__precond_tril_,
      SimpleRadialCalib_num_,
      simple_radial_fixed_pose_fixed_point_num_);

  PinholeResJac(nodes__PinholePose__storage_current_,
                PinholePose_num_max_,
                facs__pinhole__args__pose__idx_shared_,
                facs__pinhole__args__sensor_from_rig__data_,
                pinhole_num_max_,
                nodes__PinholeCalib__storage_current_,
                PinholeCalib_num_max_,
                facs__pinhole__args__calib__idx_shared_,
                nodes__Point__storage_current_,
                Point_num_max_,
                facs__pinhole__args__point__idx_shared_,
                facs__pinhole__args__pixel__data_,
                pinhole_num_max_,

                facs__pinhole__res_,
                pinhole_num_,

                facs__pinhole__args__pose__jac_,
                pinhole_num_,
                nodes__PinholePose__r_k_,
                PinholePose_num_,
                nodes__PinholePose__precond_diag_,
                PinholePose_num_,
                nodes__PinholePose__precond_tril_,
                PinholePose_num_,
                facs__pinhole__args__calib__jac_,
                pinhole_num_,
                nodes__PinholeCalib__r_k_,
                PinholeCalib_num_,
                nodes__PinholeCalib__precond_diag_,
                PinholeCalib_num_,
                nodes__PinholeCalib__precond_tril_,
                PinholeCalib_num_,
                facs__pinhole__args__point__jac_,
                pinhole_num_,
                nodes__Point__r_k_,
                Point_num_,
                nodes__Point__precond_diag_,
                Point_num_,
                nodes__Point__precond_tril_,
                Point_num_,
                pinhole_num_);

  PinholeFixedPoseResJac(facs__pinhole_fixed_pose__args__sensor_from_rig__data_,
                         pinhole_fixed_pose_num_max_,
                         nodes__PinholeCalib__storage_current_,
                         PinholeCalib_num_max_,
                         facs__pinhole_fixed_pose__args__calib__idx_shared_,
                         nodes__Point__storage_current_,
                         Point_num_max_,
                         facs__pinhole_fixed_pose__args__point__idx_shared_,
                         facs__pinhole_fixed_pose__args__pixel__data_,
                         pinhole_fixed_pose_num_max_,
                         facs__pinhole_fixed_pose__args__pose__data_,
                         pinhole_fixed_pose_num_max_,

                         facs__pinhole_fixed_pose__res_,
                         pinhole_fixed_pose_num_,

                         facs__pinhole_fixed_pose__args__calib__jac_,
                         pinhole_fixed_pose_num_,
                         nodes__PinholeCalib__r_k_,
                         PinholeCalib_num_,
                         nodes__PinholeCalib__precond_diag_,
                         PinholeCalib_num_,
                         nodes__PinholeCalib__precond_tril_,
                         PinholeCalib_num_,
                         facs__pinhole_fixed_pose__args__point__jac_,
                         pinhole_fixed_pose_num_,
                         nodes__Point__r_k_,
                         Point_num_,
                         nodes__Point__precond_diag_,
                         Point_num_,
                         nodes__Point__precond_tril_,
                         Point_num_,
                         pinhole_fixed_pose_num_);

  PinholeFixedPointResJac(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_point__args__sensor_from_rig__data_,
      pinhole_fixed_point_num_max_,
      nodes__PinholeCalib__storage_current_,
      PinholeCalib_num_max_,
      facs__pinhole_fixed_point__args__calib__idx_shared_,
      facs__pinhole_fixed_point__args__pixel__data_,
      pinhole_fixed_point_num_max_,
      facs__pinhole_fixed_point__args__point__data_,
      pinhole_fixed_point_num_max_,

      facs__pinhole_fixed_point__res_,
      pinhole_fixed_point_num_,

      facs__pinhole_fixed_point__args__pose__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_fixed_point__args__calib__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholeCalib__r_k_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_diag_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_tril_,
      PinholeCalib_num_,
      pinhole_fixed_point_num_);

  PinholeFixedPoseFixedPointResJac(
      facs__pinhole_fixed_pose_fixed_point__args__sensor_from_rig__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      nodes__PinholeCalib__storage_current_,
      PinholeCalib_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__calib__idx_shared_,
      facs__pinhole_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_point_num_max_,

      facs__pinhole_fixed_pose_fixed_point__res_,
      pinhole_fixed_pose_fixed_point_num_,

      nodes__PinholeCalib__r_k_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_diag_,
      PinholeCalib_num_,
      nodes__PinholeCalib__precond_tril_,
      PinholeCalib_num_,
      pinhole_fixed_pose_fixed_point_num_);

  SimpleRadialSplitFixedFocalAndExtraResJac(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,

      facs__simple_radial_split_fixed_focal_and_extra__res_,
      simple_radial_split_fixed_focal_and_extra_num_,

      facs__simple_radial_split_fixed_focal_and_extra__args__pose__jac_,
      simple_radial_split_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_focal_and_extra__args__principal_point__jac_,
      simple_radial_split_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_split_fixed_focal_and_extra__args__point__jac_,
      simple_radial_split_fixed_focal_and_extra_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_split_fixed_focal_and_extra_num_);

  SimpleRadialSplitFixedPrincipalPointResJac(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_principal_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_principal_point_num_max_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_split_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_split_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_principal_point_num_max_,

      facs__simple_radial_split_fixed_principal_point__res_,
      simple_radial_split_fixed_principal_point_num_,

      facs__simple_radial_split_fixed_principal_point__args__pose__jac_,
      simple_radial_split_fixed_principal_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_principal_point__args__focal_and_extra__jac_,
      simple_radial_split_fixed_principal_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_split_fixed_principal_point__args__point__jac_,
      simple_radial_split_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_split_fixed_principal_point_num_);

  SimpleRadialSplitFixedPoseFixedFocalAndExtraResJac(
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,

      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__res_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,

      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__point__jac_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_);

  SimpleRadialSplitFixedPoseFixedPrincipalPointResJac(
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,

      facs__simple_radial_split_fixed_pose_fixed_principal_point__res_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_,

      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__jac_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_);

  SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointResJac(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,

      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__res_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,

      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_);

  SimpleRadialSplitFixedFocalAndExtraFixedPointResJac(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,

      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__res_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_,

      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pose__jac_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_);

  SimpleRadialSplitFixedPrincipalPointFixedPointResJac(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,

      facs__simple_radial_split_fixed_principal_point_fixed_point__res_,
      simple_radial_split_fixed_principal_point_fixed_point_num_,

      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__jac_,
      simple_radial_split_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      simple_radial_split_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      simple_radial_split_fixed_principal_point_fixed_point_num_);

  SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointResJac(
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,

      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__res_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,

      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_);

  SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointResJac(
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,

      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__res_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,

      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_);

  SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointResJac(
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,

      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__res_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,

      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_);

  SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJac(
      nodes__SimpleRadialPose__storage_current_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,

      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__res_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,

      nodes__SimpleRadialPose__r_k_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_diag_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPose__precond_tril_,
      SimpleRadialPose_num_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_);

  PinholeSplitFixedFocalResJac(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_focal__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_num_max_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_split_fixed_focal__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_split_fixed_focal__args__point__idx_shared_,
      facs__pinhole_split_fixed_focal__args__pixel__data_,
      pinhole_split_fixed_focal_num_max_,
      facs__pinhole_split_fixed_focal__args__focal__data_,
      pinhole_split_fixed_focal_num_max_,

      facs__pinhole_split_fixed_focal__res_,
      pinhole_split_fixed_focal_num_,

      facs__pinhole_split_fixed_focal__args__pose__jac_,
      pinhole_split_fixed_focal_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_split_fixed_focal__args__principal_point__jac_,
      pinhole_split_fixed_focal_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      facs__pinhole_split_fixed_focal__args__point__jac_,
      pinhole_split_fixed_focal_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_split_fixed_focal_num_);

  PinholeSplitFixedPrincipalPointResJac(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_principal_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_principal_point_num_max_,
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_split_fixed_principal_point__args__focal__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_split_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_principal_point_num_max_,

      facs__pinhole_split_fixed_principal_point__res_,
      pinhole_split_fixed_principal_point_num_,

      facs__pinhole_split_fixed_principal_point__args__pose__jac_,
      pinhole_split_fixed_principal_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_split_fixed_principal_point__args__focal__jac_,
      pinhole_split_fixed_principal_point_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      facs__pinhole_split_fixed_principal_point__args__point__jac_,
      pinhole_split_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_split_fixed_principal_point_num_);

  PinholeSplitFixedPoseFixedFocalResJac(
      facs__pinhole_split_fixed_pose_fixed_focal__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,

      facs__pinhole_split_fixed_pose_fixed_focal__res_,
      pinhole_split_fixed_pose_fixed_focal_num_,

      facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__jac_,
      pinhole_split_fixed_pose_fixed_focal_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__point__jac_,
      pinhole_split_fixed_pose_fixed_focal_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_split_fixed_pose_fixed_focal_num_);

  PinholeSplitFixedPoseFixedPrincipalPointResJac(
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,

      facs__pinhole_split_fixed_pose_fixed_principal_point__res_,
      pinhole_split_fixed_pose_fixed_principal_point_num_,

      facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__jac_,
      pinhole_split_fixed_pose_fixed_principal_point_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__jac_,
      pinhole_split_fixed_pose_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_split_fixed_pose_fixed_principal_point_num_);

  PinholeSplitFixedFocalFixedPrincipalPointResJac(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,

      facs__pinhole_split_fixed_focal_fixed_principal_point__res_,
      pinhole_split_fixed_focal_fixed_principal_point_num_,

      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__jac_,
      pinhole_split_fixed_focal_fixed_principal_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__jac_,
      pinhole_split_fixed_focal_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_split_fixed_focal_fixed_principal_point_num_);

  PinholeSplitFixedFocalFixedPointResJac(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_focal_fixed_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_point__args__point__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,

      facs__pinhole_split_fixed_focal_fixed_point__res_,
      pinhole_split_fixed_focal_fixed_point_num_,

      facs__pinhole_split_fixed_focal_fixed_point__args__pose__jac_,
      pinhole_split_fixed_focal_fixed_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__jac_,
      pinhole_split_fixed_focal_fixed_point_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      pinhole_split_fixed_focal_fixed_point_num_);

  PinholeSplitFixedPrincipalPointFixedPointResJac(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__idx_shared_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,

      facs__pinhole_split_fixed_principal_point_fixed_point__res_,
      pinhole_split_fixed_principal_point_fixed_point_num_,

      facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__jac_,
      pinhole_split_fixed_principal_point_fixed_point_num_,
      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__jac_,
      pinhole_split_fixed_principal_point_fixed_point_num_,
      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      pinhole_split_fixed_principal_point_fixed_point_num_);

  PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJac(
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,

      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__res_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,

      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_);

  PinholeSplitFixedPoseFixedFocalFixedPointResJac(
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,

      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__res_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_,

      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_);

  PinholeSplitFixedPoseFixedPrincipalPointFixedPointResJac(
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      nodes__PinholeFocal__storage_current_,
      PinholeFocal_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__focal__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,

      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__res_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,

      nodes__PinholeFocal__r_k_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_diag_,
      PinholeFocal_num_,
      nodes__PinholeFocal__precond_tril_,
      PinholeFocal_num_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_);

  PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJac(
      nodes__PinholePose__storage_current_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,

      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__res_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,

      nodes__PinholePose__r_k_,
      PinholePose_num_,
      nodes__PinholePose__precond_diag_,
      PinholePose_num_,
      nodes__PinholePose__precond_tril_,
      PinholePose_num_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__r_0_start_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
}

void GraphSolver::DoNormalize() {
  float* r_k;
  float* z;
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
  z = pcg_iter_ == 0 ? nodes__PinholeFocal__p_ : nodes__PinholeFocal__z_;
  PinholeFocalNormalize(nodes__PinholeFocal__precond_diag_,
                        PinholeFocal_num_,
                        nodes__PinholeFocal__precond_tril_,
                        PinholeFocal_num_,
                        nodes__PinholeFocal__r_k_,
                        PinholeFocal_num_,
                        solver__current_diag_,
                        z,
                        PinholeFocal_num_,
                        PinholeFocal_num_);
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
  PinholeFocalUpdateMp(nodes__PinholeFocal__r_k_,
                       PinholeFocal_num_,
                       nodes__PinholeFocal__Mp_,
                       PinholeFocal_num_,
                       solver__beta_,
                       nodes__PinholeFocal__Mp_,
                       PinholeFocal_num_,
                       nodes__PinholeFocal__w_,
                       PinholeFocal_num_,
                       PinholeFocal_num_);
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
  SimpleRadialJtjnjtrDirect(nodes__SimpleRadialPose__p_,
                            SimpleRadialPose_num_,
                            facs__simple_radial__args__pose__idx_shared_,
                            facs__simple_radial__args__pose__jac_,
                            simple_radial_num_,
                            nodes__SimpleRadialCalib__p_,
                            SimpleRadialCalib_num_,
                            facs__simple_radial__args__calib__idx_shared_,
                            facs__simple_radial__args__calib__jac_,
                            simple_radial_num_,
                            nodes__Point__p_,
                            Point_num_,
                            facs__simple_radial__args__point__idx_shared_,
                            facs__simple_radial__args__point__jac_,
                            simple_radial_num_,
                            nodes__SimpleRadialPose__w_,
                            SimpleRadialPose_num_,
                            nodes__SimpleRadialCalib__w_,
                            SimpleRadialCalib_num_,
                            nodes__Point__w_,
                            Point_num_,
                            simple_radial_num_);
  SimpleRadialFixedPoseJtjnjtrDirect(
      nodes__SimpleRadialCalib__p_,
      SimpleRadialCalib_num_,
      facs__simple_radial_fixed_pose__args__calib__idx_shared_,
      facs__simple_radial_fixed_pose__args__calib__jac_,
      simple_radial_fixed_pose_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_fixed_pose__args__point__idx_shared_,
      facs__simple_radial_fixed_pose__args__point__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialCalib__w_,
      SimpleRadialCalib_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_pose_num_);
  SimpleRadialFixedPointJtjnjtrDirect(
      nodes__SimpleRadialPose__p_,
      SimpleRadialPose_num_,
      facs__simple_radial_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_point__args__pose__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialCalib__p_,
      SimpleRadialCalib_num_,
      facs__simple_radial_fixed_point__args__calib__idx_shared_,
      facs__simple_radial_fixed_point__args__calib__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialPose__w_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialCalib__w_,
      SimpleRadialCalib_num_,
      simple_radial_fixed_point_num_);
  PinholeJtjnjtrDirect(nodes__PinholePose__p_,
                       PinholePose_num_,
                       facs__pinhole__args__pose__idx_shared_,
                       facs__pinhole__args__pose__jac_,
                       pinhole_num_,
                       nodes__PinholeCalib__p_,
                       PinholeCalib_num_,
                       facs__pinhole__args__calib__idx_shared_,
                       facs__pinhole__args__calib__jac_,
                       pinhole_num_,
                       nodes__Point__p_,
                       Point_num_,
                       facs__pinhole__args__point__idx_shared_,
                       facs__pinhole__args__point__jac_,
                       pinhole_num_,
                       nodes__PinholePose__w_,
                       PinholePose_num_,
                       nodes__PinholeCalib__w_,
                       PinholeCalib_num_,
                       nodes__Point__w_,
                       Point_num_,
                       pinhole_num_);
  PinholeFixedPoseJtjnjtrDirect(
      nodes__PinholeCalib__p_,
      PinholeCalib_num_,
      facs__pinhole_fixed_pose__args__calib__idx_shared_,
      facs__pinhole_fixed_pose__args__calib__jac_,
      pinhole_fixed_pose_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_fixed_pose__args__point__idx_shared_,
      facs__pinhole_fixed_pose__args__point__jac_,
      pinhole_fixed_pose_num_,
      nodes__PinholeCalib__w_,
      PinholeCalib_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_pose_num_);
  PinholeFixedPointJtjnjtrDirect(
      nodes__PinholePose__p_,
      PinholePose_num_,
      facs__pinhole_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_point__args__pose__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholeCalib__p_,
      PinholeCalib_num_,
      facs__pinhole_fixed_point__args__calib__idx_shared_,
      facs__pinhole_fixed_point__args__calib__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholePose__w_,
      PinholePose_num_,
      nodes__PinholeCalib__w_,
      PinholeCalib_num_,
      pinhole_fixed_point_num_);
  SimpleRadialSplitFixedFocalAndExtraJtjnjtrDirect(
      nodes__SimpleRadialPose__p_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_focal_and_extra__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra__args__pose__jac_,
      simple_radial_split_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_split_fixed_focal_and_extra__args__principal_point__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra__args__principal_point__jac_,
      simple_radial_split_fixed_focal_and_extra_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_split_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra__args__point__jac_,
      simple_radial_split_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPose__w_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_split_fixed_focal_and_extra_num_);
  SimpleRadialSplitFixedPrincipalPointJtjnjtrDirect(
      nodes__SimpleRadialPose__p_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_principal_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_principal_point__args__pose__jac_,
      simple_radial_split_fixed_principal_point_num_,
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_split_fixed_principal_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_split_fixed_principal_point__args__focal_and_extra__jac_,
      simple_radial_split_fixed_principal_point_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_split_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_principal_point__args__point__jac_,
      simple_radial_split_fixed_principal_point_num_,
      nodes__SimpleRadialPose__w_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_split_fixed_principal_point_num_);
  SimpleRadialSplitFixedPoseFixedFocalAndExtraJtjnjtrDirect(
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__point__jac_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_);
  SimpleRadialSplitFixedPoseFixedPrincipalPointJtjnjtrDirect(
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__jac_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_);
  SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointJtjnjtrDirect(
      nodes__SimpleRadialPose__p_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__SimpleRadialPose__w_,
      SimpleRadialPose_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_);
  SimpleRadialSplitFixedFocalAndExtraFixedPointJtjnjtrDirect(
      nodes__SimpleRadialPose__p_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pose__jac_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPose__w_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_);
  SimpleRadialSplitFixedPrincipalPointFixedPointJtjnjtrDirect(
      nodes__SimpleRadialPose__p_,
      SimpleRadialPose_num_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__jac_,
      simple_radial_split_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      simple_radial_split_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialPose__w_,
      SimpleRadialPose_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      simple_radial_split_fixed_principal_point_fixed_point_num_);
  PinholeSplitFixedFocalJtjnjtrDirect(
      nodes__PinholePose__p_,
      PinholePose_num_,
      facs__pinhole_split_fixed_focal__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal__args__pose__jac_,
      pinhole_split_fixed_focal_num_,
      nodes__PinholePrincipalPoint__p_,
      PinholePrincipalPoint_num_,
      facs__pinhole_split_fixed_focal__args__principal_point__idx_shared_,
      facs__pinhole_split_fixed_focal__args__principal_point__jac_,
      pinhole_split_fixed_focal_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_split_fixed_focal__args__point__idx_shared_,
      facs__pinhole_split_fixed_focal__args__point__jac_,
      pinhole_split_fixed_focal_num_,
      nodes__PinholePose__w_,
      PinholePose_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_split_fixed_focal_num_);
  PinholeSplitFixedPrincipalPointJtjnjtrDirect(
      nodes__PinholePose__p_,
      PinholePose_num_,
      facs__pinhole_split_fixed_principal_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_principal_point__args__pose__jac_,
      pinhole_split_fixed_principal_point_num_,
      nodes__PinholeFocal__p_,
      PinholeFocal_num_,
      facs__pinhole_split_fixed_principal_point__args__focal__idx_shared_,
      facs__pinhole_split_fixed_principal_point__args__focal__jac_,
      pinhole_split_fixed_principal_point_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_split_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_principal_point__args__point__jac_,
      pinhole_split_fixed_principal_point_num_,
      nodes__PinholePose__w_,
      PinholePose_num_,
      nodes__PinholeFocal__w_,
      PinholeFocal_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_split_fixed_principal_point_num_);
  PinholeSplitFixedPoseFixedFocalJtjnjtrDirect(
      nodes__PinholePrincipalPoint__p_,
      PinholePrincipalPoint_num_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__jac_,
      pinhole_split_fixed_pose_fixed_focal_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__point__jac_,
      pinhole_split_fixed_pose_fixed_focal_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_split_fixed_pose_fixed_focal_num_);
  PinholeSplitFixedPoseFixedPrincipalPointJtjnjtrDirect(
      nodes__PinholeFocal__p_,
      PinholeFocal_num_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__jac_,
      pinhole_split_fixed_pose_fixed_principal_point_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__jac_,
      pinhole_split_fixed_pose_fixed_principal_point_num_,
      nodes__PinholeFocal__w_,
      PinholeFocal_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_split_fixed_pose_fixed_principal_point_num_);
  PinholeSplitFixedFocalFixedPrincipalPointJtjnjtrDirect(
      nodes__PinholePose__p_,
      PinholePose_num_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__jac_,
      pinhole_split_fixed_focal_fixed_principal_point_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__jac_,
      pinhole_split_fixed_focal_fixed_principal_point_num_,
      nodes__PinholePose__w_,
      PinholePose_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_split_fixed_focal_fixed_principal_point_num_);
  PinholeSplitFixedFocalFixedPointJtjnjtrDirect(
      nodes__PinholePose__p_,
      PinholePose_num_,
      facs__pinhole_split_fixed_focal_fixed_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_point__args__pose__jac_,
      pinhole_split_fixed_focal_fixed_point_num_,
      nodes__PinholePrincipalPoint__p_,
      PinholePrincipalPoint_num_,
      facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__jac_,
      pinhole_split_fixed_focal_fixed_point_num_,
      nodes__PinholePose__w_,
      PinholePose_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      pinhole_split_fixed_focal_fixed_point_num_);
  PinholeSplitFixedPrincipalPointFixedPointJtjnjtrDirect(
      nodes__PinholePose__p_,
      PinholePose_num_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__jac_,
      pinhole_split_fixed_principal_point_fixed_point_num_,
      nodes__PinholeFocal__p_,
      PinholeFocal_num_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__idx_shared_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__jac_,
      pinhole_split_fixed_principal_point_fixed_point_num_,
      nodes__PinholePose__w_,
      PinholePose_num_,
      nodes__PinholeFocal__w_,
      PinholeFocal_num_,
      pinhole_split_fixed_principal_point_fixed_point_num_);
}

void GraphSolver::DoAlphaFirst() {
  Zero(solver__alpha_numerator_, solver__alpha_denominator_ + 1);
  float* p_kp1;
  float* r_k;
  PinholeCalibAlphaNumeratorDenominator(nodes__PinholeCalib__p_,
                                        PinholeCalib_num_,
                                        nodes__PinholeCalib__r_k_,
                                        PinholeCalib_num_,
                                        nodes__PinholeCalib__w_,
                                        PinholeCalib_num_,
                                        solver__alpha_numerator_,
                                        solver__alpha_denominator_,
                                        PinholeCalib_num_);
  PinholeFocalAlphaNumeratorDenominator(nodes__PinholeFocal__p_,
                                        PinholeFocal_num_,
                                        nodes__PinholeFocal__r_k_,
                                        PinholeFocal_num_,
                                        nodes__PinholeFocal__w_,
                                        PinholeFocal_num_,
                                        solver__alpha_numerator_,
                                        solver__alpha_denominator_,
                                        PinholeFocal_num_);
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
  PinholeFocalAlphaDenominatorOrBetaNumerator(nodes__PinholeFocal__p_,
                                              PinholeFocal_num_,
                                              nodes__PinholeFocal__w_,
                                              PinholeFocal_num_,
                                              solver__alpha_denominator_,
                                              PinholeFocal_num_);
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
  PinholeFocalUpdateStepFirst(nodes__PinholeFocal__p_,
                              PinholeFocal_num_,
                              solver__alpha_,
                              nodes__PinholeFocal__step_,
                              PinholeFocal_num_,
                              PinholeFocal_num_);
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
  PinholeFocalUpdateStep(nodes__PinholeFocal__step_,
                         PinholeFocal_num_,
                         nodes__PinholeFocal__p_,
                         PinholeFocal_num_,
                         solver__alpha_,
                         nodes__PinholeFocal__step_,
                         PinholeFocal_num_,
                         PinholeFocal_num_);
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

  PinholeFocalUpdateRFirst(nodes__PinholeFocal__r_k_,
                           PinholeFocal_num_,
                           nodes__PinholeFocal__w_,
                           PinholeFocal_num_,
                           solver__neg_alpha_,
                           nodes__PinholeFocal__r_k_,
                           PinholeFocal_num_,
                           solver__r_0_norm2_tot_,
                           solver__r_kp1_norm2_tot_,
                           PinholeFocal_num_);

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
  PinholeFocalUpdateR(nodes__PinholeFocal__r_k_,
                      PinholeFocal_num_,
                      nodes__PinholeFocal__w_,
                      PinholeFocal_num_,
                      solver__neg_alpha_,
                      nodes__PinholeFocal__r_k_,
                      PinholeFocal_num_,
                      solver__r_kp1_norm2_tot_,
                      PinholeFocal_num_);
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

float GraphSolver::DoRetractScore() {
  PinholeCalibRetract(nodes__PinholeCalib__storage_current_,
                      PinholeCalib_num_max_,
                      nodes__PinholeCalib__step_,
                      PinholeCalib_num_,
                      nodes__PinholeCalib__storage_check_,
                      PinholeCalib_num_max_,
                      PinholeCalib_num_);
  PinholeFocalRetract(nodes__PinholeFocal__storage_current_,
                      PinholeFocal_num_max_,
                      nodes__PinholeFocal__step_,
                      PinholeFocal_num_,
                      nodes__PinholeFocal__storage_check_,
                      PinholeFocal_num_max_,
                      PinholeFocal_num_);
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
  SimpleRadialScore(nodes__SimpleRadialPose__storage_check_,
                    SimpleRadialPose_num_max_,
                    facs__simple_radial__args__pose__idx_shared_,
                    facs__simple_radial__args__sensor_from_rig__data_,
                    simple_radial_num_max_,
                    nodes__SimpleRadialCalib__storage_check_,
                    SimpleRadialCalib_num_max_,
                    facs__simple_radial__args__calib__idx_shared_,
                    nodes__Point__storage_check_,
                    Point_num_max_,
                    facs__simple_radial__args__point__idx_shared_,
                    facs__simple_radial__args__pixel__data_,
                    simple_radial_num_max_,
                    solver__res_tot_,
                    simple_radial_num_);
  SimpleRadialFixedPoseScore(
      facs__simple_radial_fixed_pose__args__sensor_from_rig__data_,
      simple_radial_fixed_pose_num_max_,
      nodes__SimpleRadialCalib__storage_check_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_fixed_pose__args__calib__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_pose__args__point__idx_shared_,
      facs__simple_radial_fixed_pose__args__pixel__data_,
      simple_radial_fixed_pose_num_max_,
      facs__simple_radial_fixed_pose__args__pose__data_,
      simple_radial_fixed_pose_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_num_);
  SimpleRadialFixedPointScore(
      nodes__SimpleRadialPose__storage_check_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_point__args__sensor_from_rig__data_,
      simple_radial_fixed_point_num_max_,
      nodes__SimpleRadialCalib__storage_check_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_fixed_point__args__calib__idx_shared_,
      facs__simple_radial_fixed_point__args__pixel__data_,
      simple_radial_fixed_point_num_max_,
      facs__simple_radial_fixed_point__args__point__data_,
      simple_radial_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_point_num_);
  SimpleRadialFixedPoseFixedPointScore(
      facs__simple_radial_fixed_pose_fixed_point__args__sensor_from_rig__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      nodes__SimpleRadialCalib__storage_check_,
      SimpleRadialCalib_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__calib__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_fixed_point_num_);
  PinholeScore(nodes__PinholePose__storage_check_,
               PinholePose_num_max_,
               facs__pinhole__args__pose__idx_shared_,
               facs__pinhole__args__sensor_from_rig__data_,
               pinhole_num_max_,
               nodes__PinholeCalib__storage_check_,
               PinholeCalib_num_max_,
               facs__pinhole__args__calib__idx_shared_,
               nodes__Point__storage_check_,
               Point_num_max_,
               facs__pinhole__args__point__idx_shared_,
               facs__pinhole__args__pixel__data_,
               pinhole_num_max_,
               solver__res_tot_,
               pinhole_num_);
  PinholeFixedPoseScore(facs__pinhole_fixed_pose__args__sensor_from_rig__data_,
                        pinhole_fixed_pose_num_max_,
                        nodes__PinholeCalib__storage_check_,
                        PinholeCalib_num_max_,
                        facs__pinhole_fixed_pose__args__calib__idx_shared_,
                        nodes__Point__storage_check_,
                        Point_num_max_,
                        facs__pinhole_fixed_pose__args__point__idx_shared_,
                        facs__pinhole_fixed_pose__args__pixel__data_,
                        pinhole_fixed_pose_num_max_,
                        facs__pinhole_fixed_pose__args__pose__data_,
                        pinhole_fixed_pose_num_max_,
                        solver__res_tot_,
                        pinhole_fixed_pose_num_);
  PinholeFixedPointScore(
      nodes__PinholePose__storage_check_,
      PinholePose_num_max_,
      facs__pinhole_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_point__args__sensor_from_rig__data_,
      pinhole_fixed_point_num_max_,
      nodes__PinholeCalib__storage_check_,
      PinholeCalib_num_max_,
      facs__pinhole_fixed_point__args__calib__idx_shared_,
      facs__pinhole_fixed_point__args__pixel__data_,
      pinhole_fixed_point_num_max_,
      facs__pinhole_fixed_point__args__point__data_,
      pinhole_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_point_num_);
  PinholeFixedPoseFixedPointScore(
      facs__pinhole_fixed_pose_fixed_point__args__sensor_from_rig__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      nodes__PinholeCalib__storage_check_,
      PinholeCalib_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__calib__idx_shared_,
      facs__pinhole_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_fixed_point_num_);
  SimpleRadialSplitFixedFocalAndExtraScore(
      nodes__SimpleRadialPose__storage_check_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,
      nodes__SimpleRadialPrincipalPoint__storage_check_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,
      solver__res_tot_,
      simple_radial_split_fixed_focal_and_extra_num_);
  SimpleRadialSplitFixedPrincipalPointScore(
      nodes__SimpleRadialPose__storage_check_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_principal_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_principal_point_num_max_,
      nodes__SimpleRadialFocalAndExtra__storage_check_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_split_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_split_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_principal_point_num_max_,
      solver__res_tot_,
      simple_radial_split_fixed_principal_point_num_);
  SimpleRadialSplitFixedPoseFixedFocalAndExtraScore(
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      nodes__SimpleRadialPrincipalPoint__storage_check_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      solver__res_tot_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_);
  SimpleRadialSplitFixedPoseFixedPrincipalPointScore(
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      nodes__SimpleRadialFocalAndExtra__storage_check_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      solver__res_tot_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_);
  SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointScore(
      nodes__SimpleRadialPose__storage_check_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      solver__res_tot_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_);
  SimpleRadialSplitFixedFocalAndExtraFixedPointScore(
      nodes__SimpleRadialPose__storage_check_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      nodes__SimpleRadialPrincipalPoint__storage_check_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_);
  SimpleRadialSplitFixedPrincipalPointFixedPointScore(
      nodes__SimpleRadialPose__storage_check_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      nodes__SimpleRadialFocalAndExtra__storage_check_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_split_fixed_principal_point_fixed_point_num_);
  SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointScore(
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      solver__res_tot_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_);
  SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointScore(
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      nodes__SimpleRadialPrincipalPoint__storage_check_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_);
  SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointScore(
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      nodes__SimpleRadialFocalAndExtra__storage_check_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_);
  SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointScore(
      nodes__SimpleRadialPose__storage_check_,
      SimpleRadialPose_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_);
  PinholeSplitFixedFocalScore(
      nodes__PinholePose__storage_check_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_focal__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_num_max_,
      nodes__PinholePrincipalPoint__storage_check_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_split_fixed_focal__args__principal_point__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_split_fixed_focal__args__point__idx_shared_,
      facs__pinhole_split_fixed_focal__args__pixel__data_,
      pinhole_split_fixed_focal_num_max_,
      facs__pinhole_split_fixed_focal__args__focal__data_,
      pinhole_split_fixed_focal_num_max_,
      solver__res_tot_,
      pinhole_split_fixed_focal_num_);
  PinholeSplitFixedPrincipalPointScore(
      nodes__PinholePose__storage_check_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_principal_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_principal_point_num_max_,
      nodes__PinholeFocal__storage_check_,
      PinholeFocal_num_max_,
      facs__pinhole_split_fixed_principal_point__args__focal__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_split_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_principal_point_num_max_,
      solver__res_tot_,
      pinhole_split_fixed_principal_point_num_);
  PinholeSplitFixedPoseFixedFocalScore(
      facs__pinhole_split_fixed_pose_fixed_focal__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      nodes__PinholePrincipalPoint__storage_check_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      solver__res_tot_,
      pinhole_split_fixed_pose_fixed_focal_num_);
  PinholeSplitFixedPoseFixedPrincipalPointScore(
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      nodes__PinholeFocal__storage_check_,
      PinholeFocal_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      solver__res_tot_,
      pinhole_split_fixed_pose_fixed_principal_point_num_);
  PinholeSplitFixedFocalFixedPrincipalPointScore(
      nodes__PinholePose__storage_check_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      solver__res_tot_,
      pinhole_split_fixed_focal_fixed_principal_point_num_);
  PinholeSplitFixedFocalFixedPointScore(
      nodes__PinholePose__storage_check_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_focal_fixed_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      nodes__PinholePrincipalPoint__storage_check_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_point__args__point__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_split_fixed_focal_fixed_point_num_);
  PinholeSplitFixedPrincipalPointFixedPointScore(
      nodes__PinholePose__storage_check_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      nodes__PinholeFocal__storage_check_,
      PinholeFocal_num_max_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__idx_shared_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_split_fixed_principal_point_fixed_point_num_);
  PinholeSplitFixedPoseFixedFocalFixedPrincipalPointScore(
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      solver__res_tot_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_);
  PinholeSplitFixedPoseFixedFocalFixedPointScore(
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      nodes__PinholePrincipalPoint__storage_check_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_);
  PinholeSplitFixedPoseFixedPrincipalPointFixedPointScore(
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      nodes__PinholeFocal__storage_check_,
      PinholeFocal_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__focal__idx_shared_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_);
  PinholeSplitFixedFocalFixedPrincipalPointFixedPointScore(
      nodes__PinholePose__storage_check_,
      PinholePose_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_);
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

  PinholeFocalAlphaDenominatorOrBetaNumerator(nodes__PinholeFocal__r_k_,
                                              PinholeFocal_num_,
                                              nodes__PinholeFocal__z_,
                                              PinholeFocal_num_,
                                              solver__beta_numerator_,
                                              PinholeFocal_num_);

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
  PinholeFocalUpdateP(nodes__PinholeFocal__z_,
                      PinholeFocal_num_,
                      nodes__PinholeFocal__p_,
                      PinholeFocal_num_,
                      solver__beta_,
                      nodes__PinholeFocal__p_,
                      PinholeFocal_num_,
                      PinholeFocal_num_);
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

float GraphSolver::GetPredDecrease() {
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
  PinholeFocalPredDecreaseTimesTwo(nodes__PinholeFocal__step_,
                                   PinholeFocal_num_,
                                   nodes__PinholeFocal__precond_diag_,
                                   PinholeFocal_num_,
                                   solver__current_diag_,
                                   nodes__PinholeFocal__r_0_,
                                   PinholeFocal_num_,
                                   solver__pred_decrease_tot_,
                                   PinholeFocal_num_);
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
  cudaSetDevice(device_id_);
  if (num > PinholeCalib_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > PinholeCalib_num_max_");
  }
  PinholeCalib_num_ = num;
}

void GraphSolver::SetPinholeCalibNodesFromStackedHost(const float* const data,
                                                      const size_t offset,
                                                      const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > PinholeCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeCalib_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             4 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  PinholeCalibStackedToCaspar(marker__scratch_inout_,
                              nodes__PinholeCalib__storage_current_,
                              PinholeCalib_num_max_,
                              offset,
                              num);
}

void GraphSolver::SetPinholeCalibNodesFromStackedDevice(const float* const data,
                                                        const size_t offset,
                                                        const size_t num) {
  cudaSetDevice(device_id_);
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

void GraphSolver::GetPinholeCalibNodesToStackedHost(float* const data,
                                                    const size_t offset,
                                                    const size_t num) {
  cudaSetDevice(device_id_);
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
             4 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetPinholeCalibNodesToStackedDevice(float* const data,
                                                      const size_t offset,
                                                      const size_t num) {
  cudaSetDevice(device_id_);
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

void GraphSolver::SetPinholeFocalNum(const size_t num) {
  cudaSetDevice(device_id_);
  if (num > PinholeFocal_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > PinholeFocal_num_max_");
  }
  PinholeFocal_num_ = num;
}

void GraphSolver::SetPinholeFocalNodesFromStackedHost(const float* const data,
                                                      const size_t offset,
                                                      const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > PinholeFocal_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocal_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  PinholeFocalStackedToCaspar(marker__scratch_inout_,
                              nodes__PinholeFocal__storage_current_,
                              PinholeFocal_num_max_,
                              offset,
                              num);
}

void GraphSolver::SetPinholeFocalNodesFromStackedDevice(const float* const data,
                                                        const size_t offset,
                                                        const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > PinholeFocal_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocal_num_");
  }
  PinholeFocalStackedToCaspar(data,
                              nodes__PinholeFocal__storage_current_,
                              PinholeFocal_num_max_,
                              offset,
                              num);
}

void GraphSolver::GetPinholeFocalNodesToStackedHost(float* const data,
                                                    const size_t offset,
                                                    const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > PinholeFocal_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocal_num_");
  }
  PinholeFocalCasparToStacked(nodes__PinholeFocal__storage_current_,
                              marker__scratch_inout_,
                              PinholeFocal_num_max_,
                              offset,
                              num);
  cudaMemcpy(data,
             marker__scratch_inout_,
             2 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetPinholeFocalNodesToStackedDevice(float* const data,
                                                      const size_t offset,
                                                      const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > PinholeFocal_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocal_num_");
  }
  PinholeFocalCasparToStacked(nodes__PinholeFocal__storage_current_,
                              data,
                              PinholeFocal_num_max_,
                              offset,
                              num);
}

void GraphSolver::SetPinholePoseNum(const size_t num) {
  cudaSetDevice(device_id_);
  if (num > PinholePose_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > PinholePose_num_max_");
  }
  PinholePose_num_ = num;
}

void GraphSolver::SetPinholePoseNodesFromStackedHost(const float* const data,
                                                     const size_t offset,
                                                     const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > PinholePose_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholePose_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  PinholePoseStackedToCaspar(marker__scratch_inout_,
                             nodes__PinholePose__storage_current_,
                             PinholePose_num_max_,
                             offset,
                             num);
}

void GraphSolver::SetPinholePoseNodesFromStackedDevice(const float* const data,
                                                       const size_t offset,
                                                       const size_t num) {
  cudaSetDevice(device_id_);
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

void GraphSolver::GetPinholePoseNodesToStackedHost(float* const data,
                                                   const size_t offset,
                                                   const size_t num) {
  cudaSetDevice(device_id_);
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
             7 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetPinholePoseNodesToStackedDevice(float* const data,
                                                     const size_t offset,
                                                     const size_t num) {
  cudaSetDevice(device_id_);
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
  cudaSetDevice(device_id_);
  if (num > PinholePrincipalPoint_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > PinholePrincipalPoint_num_max_");
  }
  PinholePrincipalPoint_num_ = num;
}

void GraphSolver::SetPinholePrincipalPointNodesFromStackedHost(
    const float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > PinholePrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholePrincipalPoint_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  PinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholePrincipalPointNodesFromStackedDevice(
    const float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
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
    float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
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
             2 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetPinholePrincipalPointNodesToStackedDevice(
    float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
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
  cudaSetDevice(device_id_);
  if (num > Point_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > Point_num_max_");
  }
  Point_num_ = num;
}

void GraphSolver::SetPointNodesFromStackedHost(const float* const data,
                                               const size_t offset,
                                               const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  PointStackedToCaspar(marker__scratch_inout_,
                       nodes__Point__storage_current_,
                       Point_num_max_,
                       offset,
                       num);
}

void GraphSolver::SetPointNodesFromStackedDevice(const float* const data,
                                                 const size_t offset,
                                                 const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  PointStackedToCaspar(
      data, nodes__Point__storage_current_, Point_num_max_, offset, num);
}

void GraphSolver::GetPointNodesToStackedHost(float* const data,
                                             const size_t offset,
                                             const size_t num) {
  cudaSetDevice(device_id_);
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
             3 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetPointNodesToStackedDevice(float* const data,
                                               const size_t offset,
                                               const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  PointCasparToStacked(
      nodes__Point__storage_current_, data, Point_num_max_, offset, num);
}

void GraphSolver::SetSimpleRadialCalibNum(const size_t num) {
  cudaSetDevice(device_id_);
  if (num > SimpleRadialCalib_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > SimpleRadialCalib_num_max_");
  }
  SimpleRadialCalib_num_ = num;
}

void GraphSolver::SetSimpleRadialCalibNodesFromStackedHost(
    const float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > SimpleRadialCalib_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialCalib_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             4 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  SimpleRadialCalibStackedToCaspar(marker__scratch_inout_,
                                   nodes__SimpleRadialCalib__storage_current_,
                                   SimpleRadialCalib_num_max_,
                                   offset,
                                   num);
}

void GraphSolver::SetSimpleRadialCalibNodesFromStackedDevice(
    const float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
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

void GraphSolver::GetSimpleRadialCalibNodesToStackedHost(float* const data,
                                                         const size_t offset,
                                                         const size_t num) {
  cudaSetDevice(device_id_);
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
             4 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetSimpleRadialCalibNodesToStackedDevice(float* const data,
                                                           const size_t offset,
                                                           const size_t num) {
  cudaSetDevice(device_id_);
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
  cudaSetDevice(device_id_);
  if (num > SimpleRadialFocalAndExtra_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > SimpleRadialFocalAndExtra_num_max_");
  }
  SimpleRadialFocalAndExtra_num_ = num;
}

void GraphSolver::SetSimpleRadialFocalAndExtraNodesFromStackedHost(
    const float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > SimpleRadialFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialFocalAndExtra_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  SimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialFocalAndExtraNodesFromStackedDevice(
    const float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
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
    float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
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
             2 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetSimpleRadialFocalAndExtraNodesToStackedDevice(
    float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
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
  cudaSetDevice(device_id_);
  if (num > SimpleRadialPose_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > SimpleRadialPose_num_max_");
  }
  SimpleRadialPose_num_ = num;
}

void GraphSolver::SetSimpleRadialPoseNodesFromStackedHost(
    const float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > SimpleRadialPose_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialPose_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  SimpleRadialPoseStackedToCaspar(marker__scratch_inout_,
                                  nodes__SimpleRadialPose__storage_current_,
                                  SimpleRadialPose_num_max_,
                                  offset,
                                  num);
}

void GraphSolver::SetSimpleRadialPoseNodesFromStackedDevice(
    const float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
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

void GraphSolver::GetSimpleRadialPoseNodesToStackedHost(float* const data,
                                                        const size_t offset,
                                                        const size_t num) {
  cudaSetDevice(device_id_);
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
             7 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetSimpleRadialPoseNodesToStackedDevice(float* const data,
                                                          const size_t offset,
                                                          const size_t num) {
  cudaSetDevice(device_id_);
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
  cudaSetDevice(device_id_);
  if (num > SimpleRadialPrincipalPoint_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > SimpleRadialPrincipalPoint_num_max_");
  }
  SimpleRadialPrincipalPoint_num_ = num;
}

void GraphSolver::SetSimpleRadialPrincipalPointNodesFromStackedHost(
    const float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > SimpleRadialPrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialPrincipalPoint_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  SimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialPrincipalPointNodesFromStackedDevice(
    const float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
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
    float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
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
             2 * num * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::GetSimpleRadialPrincipalPointNodesToStackedDevice(
    float* const data, const size_t offset, const size_t num) {
  cudaSetDevice(device_id_);
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

void GraphSolver::SetSimpleRadialNum(const size_t num) {
  if (num > simple_radial_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > simple_radial_num_max_");
  }
  simple_radial_num_ = num;
}
void GraphSolver::SetSimpleRadialPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != simple_radial_num_. Use Setsimple_radialNum "
                             "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialPoseIndicesFromDevice((unsigned int*)marker__scratch_inout_,
                                       num);
}

void GraphSolver::SetSimpleRadialPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != simple_radial_num_. Use Setsimple_radialNum "
                             "before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__simple_radial__args__pose__idx_shared_, num);
}
void GraphSolver::SetSimpleRadialCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != simple_radial_num_. Use Setsimple_radialNum "
                             "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialCalibIndicesFromDevice((unsigned int*)marker__scratch_inout_,
                                        num);
}

void GraphSolver::SetSimpleRadialCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != simple_radial_num_. Use Setsimple_radialNum "
                             "before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__simple_radial__args__calib__idx_shared_, num);
}
void GraphSolver::SetSimpleRadialPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != simple_radial_num_. Use Setsimple_radialNum "
                             "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialPointIndicesFromDevice((unsigned int*)marker__scratch_inout_,
                                        num);
}

void GraphSolver::SetSimpleRadialPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_num_) {
    throw std::runtime_error(std::to_string(num) +
                             " != simple_radial_num_. Use Setsimple_radialNum "
                             "before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__simple_radial__args__point__idx_shared_, num);
}
void GraphSolver::SetSimpleRadialSensorFromRigDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial__args__sensor_from_rig__data_,
      simple_radial_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialSensorFromRigDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial__args__sensor_from_rig__data_,
      simple_radial_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialPixelDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(marker__scratch_inout_,
                            facs__simple_radial__args__pixel__data_,
                            simple_radial_num_max_,
                            offset,
                            num);
}

void GraphSolver::SetSimpleRadialPixelDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_num_max_");
  }
  ConstPixelStackedToCaspar(data,
                            facs__simple_radial__args__pixel__data_,
                            simple_radial_num_max_,
                            offset,
                            num);
}
void GraphSolver::SetSimpleRadialFixedPoseNum(const size_t num) {
  if (num > simple_radial_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > simple_radial_fixed_pose_num_max_");
  }
  simple_radial_fixed_pose_num_ = num;
}
void GraphSolver::SetSimpleRadialFixedPoseCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_num_. Use Setsimple_radial_fixed_poseNum "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPoseCalibIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialFixedPoseCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_num_. Use Setsimple_radial_fixed_poseNum "
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
      indices, facs__simple_radial_fixed_pose__args__calib__idx_shared_, num);
}
void GraphSolver::SetSimpleRadialFixedPosePointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_num_. Use Setsimple_radial_fixed_poseNum "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPosePointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialFixedPosePointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_num_. Use Setsimple_radial_fixed_poseNum "
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
      indices, facs__simple_radial_fixed_pose__args__point__idx_shared_, num);
}
void GraphSolver::SetSimpleRadialFixedPoseSensorFromRigDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose__args__sensor_from_rig__data_,
      simple_radial_fixed_pose_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialFixedPoseSensorFromRigDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_pose_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose__args__sensor_from_rig__data_,
      simple_radial_fixed_pose_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedPosePixelDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(marker__scratch_inout_,
                            facs__simple_radial_fixed_pose__args__pixel__data_,
                            simple_radial_fixed_pose_num_max_,
                            offset,
                            num);
}

void GraphSolver::SetSimpleRadialFixedPosePixelDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_pose_num_max_");
  }
  ConstPixelStackedToCaspar(data,
                            facs__simple_radial_fixed_pose__args__pixel__data_,
                            simple_radial_fixed_pose_num_max_,
                            offset,
                            num);
}
void GraphSolver::SetSimpleRadialFixedPosePoseDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPoseStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose__args__pose__data_,
      simple_radial_fixed_pose_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialFixedPosePoseDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_pose_num_max_");
  }
  ConstSimpleRadialPoseStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose__args__pose__data_,
      simple_radial_fixed_pose_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedPointNum(const size_t num) {
  if (num > simple_radial_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > simple_radial_fixed_point_num_max_");
  }
  simple_radial_fixed_point_num_ = num;
}
void GraphSolver::SetSimpleRadialFixedPointPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_point_num_. Use "
        "Setsimple_radial_fixed_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialFixedPointPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_point_num_. Use "
        "Setsimple_radial_fixed_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices, facs__simple_radial_fixed_point__args__pose__idx_shared_, num);
}
void GraphSolver::SetSimpleRadialFixedPointCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_point_num_. Use "
        "Setsimple_radial_fixed_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPointCalibIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialFixedPointCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_point_num_. Use "
        "Setsimple_radial_fixed_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices, facs__simple_radial_fixed_point__args__calib__idx_shared_, num);
}
void GraphSolver::SetSimpleRadialFixedPointSensorFromRigDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_point__args__sensor_from_rig__data_,
      simple_radial_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialFixedPointSensorFromRigDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_point_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial_fixed_point__args__sensor_from_rig__data_,
      simple_radial_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedPointPixelDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(marker__scratch_inout_,
                            facs__simple_radial_fixed_point__args__pixel__data_,
                            simple_radial_fixed_point_num_max_,
                            offset,
                            num);
}

void GraphSolver::SetSimpleRadialFixedPointPixelDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(data,
                            facs__simple_radial_fixed_point__args__pixel__data_,
                            simple_radial_fixed_point_num_max_,
                            offset,
                            num);
}
void GraphSolver::SetSimpleRadialFixedPointPointDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(marker__scratch_inout_,
                            facs__simple_radial_fixed_point__args__point__data_,
                            simple_radial_fixed_point_num_max_,
                            offset,
                            num);
}

void GraphSolver::SetSimpleRadialFixedPointPointDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(data,
                            facs__simple_radial_fixed_point__args__point__data_,
                            simple_radial_fixed_point_num_max_,
                            offset,
                            num);
}
void GraphSolver::SetSimpleRadialFixedPoseFixedPointNum(const size_t num) {
  if (num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  simple_radial_fixed_pose_fixed_point_num_ = num;
}
void GraphSolver::SetSimpleRadialFixedPoseFixedPointCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_point_num_. Use "
        "Setsimple_radial_fixed_pose_fixed_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialFixedPoseFixedPointCalibIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialFixedPoseFixedPointCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_point_num_. Use "
        "Setsimple_radial_fixed_pose_fixed_pointNum before setting indices.");
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
      facs__simple_radial_fixed_pose_fixed_point__args__calib__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialFixedPoseFixedPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_point__args__sensor_from_rig__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialFixedPoseFixedPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_point__args__sensor_from_rig__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedPoseFixedPointPixelDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialFixedPoseFixedPointPixelDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedPoseFixedPointPoseDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPoseStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialFixedPoseFixedPointPoseDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  ConstSimpleRadialPoseStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialFixedPoseFixedPointPointDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetSimpleRadialFixedPoseFixedPointPointDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__simple_radial_fixed_pose_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeNum(const size_t num) {
  if (num > pinhole_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > pinhole_num_max_");
  }
  pinhole_num_ = num;
}
void GraphSolver::SetPinholePoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_num_. Use SetpinholeNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholePoseIndicesFromDevice((unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholePoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_num_. Use SetpinholeNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__pinhole__args__pose__idx_shared_, num);
}
void GraphSolver::SetPinholeCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_num_. Use SetpinholeNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeCalibIndicesFromDevice((unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_num_. Use SetpinholeNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__pinhole__args__calib__idx_shared_, num);
}
void GraphSolver::SetPinholePointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_num_. Use SetpinholeNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholePointIndicesFromDevice((unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholePointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_num_. Use SetpinholeNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices, facs__pinhole__args__point__idx_shared_, num);
}
void GraphSolver::SetPinholeSensorFromRigDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole__args__sensor_from_rig__data_,
      pinhole_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeSensorFromRigDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole__args__sensor_from_rig__data_,
      pinhole_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholePixelDataFromStackedHost(const float* const data,
                                                     size_t offset,
                                                     size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(marker__scratch_inout_,
                            facs__pinhole__args__pixel__data_,
                            pinhole_num_max_,
                            offset,
                            num);
}

void GraphSolver::SetPinholePixelDataFromStackedDevice(const float* const data,
                                                       size_t offset,
                                                       size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_num_max_");
  }
  ConstPixelStackedToCaspar(
      data, facs__pinhole__args__pixel__data_, pinhole_num_max_, offset, num);
}
void GraphSolver::SetPinholeFixedPoseNum(const size_t num) {
  if (num > pinhole_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_fixed_pose_num_max_");
  }
  pinhole_fixed_pose_num_ = num;
}
void GraphSolver::SetPinholeFixedPoseCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_num_. Use Setpinhole_fixed_poseNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPoseCalibIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedPoseCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_num_. Use Setpinhole_fixed_poseNum before "
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
      indices, facs__pinhole_fixed_pose__args__calib__idx_shared_, num);
}
void GraphSolver::SetPinholeFixedPosePointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_num_. Use Setpinhole_fixed_poseNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPosePointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedPosePointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_fixed_pose_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_num_. Use Setpinhole_fixed_poseNum before "
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
      indices, facs__pinhole_fixed_pose__args__point__idx_shared_, num);
}
void GraphSolver::SetPinholeFixedPoseSensorFromRigDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose__args__sensor_from_rig__data_,
      pinhole_fixed_pose_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeFixedPoseSensorFromRigDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole_fixed_pose__args__sensor_from_rig__data_,
      pinhole_fixed_pose_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedPosePixelDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(marker__scratch_inout_,
                            facs__pinhole_fixed_pose__args__pixel__data_,
                            pinhole_fixed_pose_num_max_,
                            offset,
                            num);
}

void GraphSolver::SetPinholeFixedPosePixelDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_num_max_");
  }
  ConstPixelStackedToCaspar(data,
                            facs__pinhole_fixed_pose__args__pixel__data_,
                            pinhole_fixed_pose_num_max_,
                            offset,
                            num);
}
void GraphSolver::SetPinholeFixedPosePoseDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholePoseStackedToCaspar(marker__scratch_inout_,
                                  facs__pinhole_fixed_pose__args__pose__data_,
                                  pinhole_fixed_pose_num_max_,
                                  offset,
                                  num);
}

void GraphSolver::SetPinholeFixedPosePoseDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_num_max_");
  }
  ConstPinholePoseStackedToCaspar(data,
                                  facs__pinhole_fixed_pose__args__pose__data_,
                                  pinhole_fixed_pose_num_max_,
                                  offset,
                                  num);
}
void GraphSolver::SetPinholeFixedPointNum(const size_t num) {
  if (num > pinhole_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_fixed_point_num_max_");
  }
  pinhole_fixed_point_num_ = num;
}
void GraphSolver::SetPinholeFixedPointPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_point_num_. Use Setpinhole_fixed_pointNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedPointPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_point_num_. Use Setpinhole_fixed_pointNum before "
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
      indices, facs__pinhole_fixed_point__args__pose__idx_shared_, num);
}
void GraphSolver::SetPinholeFixedPointCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_point_num_. Use Setpinhole_fixed_pointNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPointCalibIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedPointCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_point_num_. Use Setpinhole_fixed_pointNum before "
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
      indices, facs__pinhole_fixed_point__args__calib__idx_shared_, num);
}
void GraphSolver::SetPinholeFixedPointSensorFromRigDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_point__args__sensor_from_rig__data_,
      pinhole_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeFixedPointSensorFromRigDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_point_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole_fixed_point__args__sensor_from_rig__data_,
      pinhole_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedPointPixelDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(marker__scratch_inout_,
                            facs__pinhole_fixed_point__args__pixel__data_,
                            pinhole_fixed_point_num_max_,
                            offset,
                            num);
}

void GraphSolver::SetPinholeFixedPointPixelDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(data,
                            facs__pinhole_fixed_point__args__pixel__data_,
                            pinhole_fixed_point_num_max_,
                            offset,
                            num);
}
void GraphSolver::SetPinholeFixedPointPointDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(marker__scratch_inout_,
                            facs__pinhole_fixed_point__args__point__data_,
                            pinhole_fixed_point_num_max_,
                            offset,
                            num);
}

void GraphSolver::SetPinholeFixedPointPointDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(data,
                            facs__pinhole_fixed_point__args__point__data_,
                            pinhole_fixed_point_num_max_,
                            offset,
                            num);
}
void GraphSolver::SetPinholeFixedPoseFixedPointNum(const size_t num) {
  if (num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  pinhole_fixed_pose_fixed_point_num_ = num;
}
void GraphSolver::SetPinholeFixedPoseFixedPointCalibIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_point_num_. Use "
        "Setpinhole_fixed_pose_fixed_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeFixedPoseFixedPointCalibIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeFixedPoseFixedPointCalibIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_fixed_pose_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_point_num_. Use "
        "Setpinhole_fixed_pose_fixed_pointNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(indices,
                facs__pinhole_fixed_pose_fixed_point__args__calib__idx_shared_,
                num);
}
void GraphSolver::SetPinholeFixedPoseFixedPointSensorFromRigDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_point__args__sensor_from_rig__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeFixedPoseFixedPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_point__args__sensor_from_rig__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedPoseFixedPointPixelDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeFixedPoseFixedPointPixelDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedPoseFixedPointPoseDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholePoseStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeFixedPoseFixedPointPoseDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  ConstPinholePoseStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeFixedPoseFixedPointPointDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeFixedPoseFixedPointPointDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__pinhole_fixed_pose_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialSplitFixedFocalAndExtraNum(const size_t num) {
  if (num > simple_radial_split_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_split_fixed_focal_and_extra_num_max_");
  }
  simple_radial_split_fixed_focal_and_extra_num_ = num;
}
void GraphSolver::SetSimpleRadialSplitFixedFocalAndExtraPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_split_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_split_fixed_focal_and_extraNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedFocalAndExtraPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialSplitFixedFocalAndExtraPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_split_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_split_fixed_focal_and_extraNum before setting "
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
      facs__simple_radial_split_fixed_focal_and_extra__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraPrincipalPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_split_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_split_fixed_focal_and_extraNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedFocalAndExtraPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraPrincipalPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_split_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_split_fixed_focal_and_extraNum before setting "
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
      facs__simple_radial_split_fixed_focal_and_extra__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::SetSimpleRadialSplitFixedFocalAndExtraPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_split_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_split_fixed_focal_and_extraNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedFocalAndExtraPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialSplitFixedFocalAndExtraPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_split_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_split_fixed_focal_and_extraNum before setting "
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
      facs__simple_radial_split_fixed_focal_and_extra__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_split_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_split_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_focal_and_extra_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_split_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_split_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_focal_and_extra_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_split_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_split_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_focal_and_extra_num_max_");
  }
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialSplitFixedPrincipalPointNum(const size_t num) {
  if (num > simple_radial_split_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_split_fixed_principal_point_num_max_");
  }
  simple_radial_split_fixed_principal_point_num_ = num;
}
void GraphSolver::SetSimpleRadialSplitFixedPrincipalPointPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_split_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_principal_point_num_. Use "
        "Setsimple_radial_split_fixed_principal_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedPrincipalPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialSplitFixedPrincipalPointPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_split_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_principal_point_num_. Use "
        "Setsimple_radial_split_fixed_principal_pointNum before setting "
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
      facs__simple_radial_split_fixed_principal_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointFocalAndExtraIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_split_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_principal_point_num_. Use "
        "Setsimple_radial_split_fixed_principal_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedPrincipalPointFocalAndExtraIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointFocalAndExtraIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_split_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_principal_point_num_. Use "
        "Setsimple_radial_split_fixed_principal_pointNum before setting "
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
      facs__simple_radial_split_fixed_principal_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::SetSimpleRadialSplitFixedPrincipalPointPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_split_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_principal_point_num_. Use "
        "Setsimple_radial_split_fixed_principal_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetSimpleRadialSplitFixedPrincipalPointPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_split_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_principal_point_num_. Use "
        "Setsimple_radial_split_fixed_principal_pointNum before setting "
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
      facs__simple_radial_split_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_split_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_split_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_split_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_split_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_split_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > simple_radial_split_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialSplitFixedPoseFixedFocalAndExtraNum(
    const size_t num) {
  if (num > simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  simple_radial_split_fixed_pose_fixed_focal_and_extra_num_ = num;
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_split_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_pose_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_split_fixed_pose_fixed_focal_and_extraNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_split_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_pose_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_split_fixed_pose_fixed_focal_and_extraNum before "
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
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_split_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_pose_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_split_fixed_pose_fixed_focal_and_extraNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_split_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_pose_fixed_focal_and_extra_num_. Use "
        "Setsimple_radial_split_fixed_pose_fixed_focal_and_extraNum before "
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
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPoseDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPoseStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPoseDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstSimpleRadialPoseStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialSplitFixedPoseFixedPrincipalPointNum(
    const size_t num) {
  if (num > simple_radial_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_split_fixed_pose_fixed_principal_point_num_max_");
  }
  simple_radial_split_fixed_pose_fixed_principal_point_num_ = num;
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_split_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_pose_fixed_principal_point_num_. Use "
        "Setsimple_radial_split_fixed_pose_fixed_principal_pointNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_split_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_pose_fixed_principal_point_num_. Use "
        "Setsimple_radial_split_fixed_pose_fixed_principal_pointNum before "
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
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_split_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_pose_fixed_principal_point_num_. Use "
        "Setsimple_radial_split_fixed_pose_fixed_principal_pointNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_split_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_pose_fixed_principal_point_num_. Use "
        "Setsimple_radial_split_fixed_pose_fixed_principal_pointNum before "
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
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPoseStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPoseStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointNum(
    const size_t num) {
  if (num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_ = num;
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num !=
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use "
        "Setsimple_radial_split_fixed_focal_and_extra_fixed_principal_pointNum "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num !=
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use "
        "Setsimple_radial_split_fixed_focal_and_extra_fixed_principal_pointNum "
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
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num !=
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use "
        "Setsimple_radial_split_fixed_focal_and_extra_fixed_principal_pointNum "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num !=
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use "
        "Setsimple_radial_split_fixed_focal_and_extra_fixed_principal_pointNum "
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
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialSplitFixedFocalAndExtraFixedPointNum(
    const size_t num) {
  if (num > simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_");
  }
  simple_radial_split_fixed_focal_and_extra_fixed_point_num_ = num;
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPointPoseIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_split_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_focal_and_extra_fixed_point_num_. Use "
        "Setsimple_radial_split_fixed_focal_and_extra_fixed_pointNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPointPoseIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_split_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_focal_and_extra_fixed_point_num_. Use "
        "Setsimple_radial_split_fixed_focal_and_extra_fixed_pointNum before "
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
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_split_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_focal_and_extra_fixed_point_num_. Use "
        "Setsimple_radial_split_fixed_focal_and_extra_fixed_pointNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_split_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_focal_and_extra_fixed_point_num_. Use "
        "Setsimple_radial_split_fixed_focal_and_extra_fixed_pointNum before "
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
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPointPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialSplitFixedPrincipalPointFixedPointNum(
    const size_t num) {
  if (num > simple_radial_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_split_fixed_principal_point_fixed_point_num_max_");
  }
  simple_radial_split_fixed_principal_point_fixed_point_num_ = num;
}
void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointFixedPointPoseIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_split_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_principal_point_fixed_point_num_. Use "
        "Setsimple_radial_split_fixed_principal_point_fixed_pointNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointFixedPointPoseIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_split_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_principal_point_fixed_point_num_. Use "
        "Setsimple_radial_split_fixed_principal_point_fixed_pointNum before "
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
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != simple_radial_split_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_principal_point_fixed_point_num_. Use "
        "Setsimple_radial_split_fixed_principal_point_fixed_pointNum before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != simple_radial_split_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_split_fixed_principal_point_fixed_point_num_. Use "
        "Setsimple_radial_split_fixed_principal_point_fixed_pointNum before "
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
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_principal_point_fixed_point_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_principal_point_fixed_point_num_max_");
  }
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPrincipalPointFixedPointPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_split_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointNum(
        const size_t num) {
  if (num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_principal_point_num_max_");
  }
  simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_ =
      num;
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num !=
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_"
        "point_num_. Use "
        "Setsimple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_"
        "principal_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num !=
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_"
        "point_num_. Use "
        "Setsimple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_"
        "principal_pointNum before setting indices.");
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
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPoseStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPoseStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointNum(
    const size_t num) {
  if (num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_point_num_max_");
  }
  simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_ = num;
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num !=
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_."
        " Use "
        "Setsimple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_"
        "pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num !=
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_."
        " Use "
        "Setsimple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_"
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
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_point_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPoseStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_point_num_max_");
  }
  ConstSimpleRadialPoseStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_focal_and_"
                             "extra_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointNum(
    const size_t num) {
  if (num >
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_ = num;
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num !=
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_."
        " Use "
        "Setsimple_radial_split_fixed_pose_fixed_principal_point_fixed_"
        "pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num !=
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_."
        " Use "
        "Setsimple_radial_split_fixed_pose_fixed_principal_point_fixed_"
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
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPoseStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  ConstSimpleRadialPoseStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_pose_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointNum(
        const size_t num) {
  if (num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_ =
      num;
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num !=
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_"
        "point_num_. Use "
        "Setsimple_radial_split_fixed_focal_and_extra_fixed_principal_point_"
        "fixed_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num !=
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_"
        "point_num_. Use "
        "Setsimple_radial_split_fixed_focal_and_extra_fixed_principal_point_"
        "fixed_pointNum before setting indices.");
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
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  ConstSimpleRadialSensorFromRigStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtraStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  ConstSimpleRadialPrincipalPointStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_split_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedFocalNum(const size_t num) {
  if (num > pinhole_split_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_split_fixed_focal_num_max_");
  }
  pinhole_split_fixed_focal_num_ = num;
}
void GraphSolver::SetPinholeSplitFixedFocalPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_num_. Use "
        "Setpinhole_split_fixed_focalNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedFocalPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeSplitFixedFocalPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_num_. Use "
        "Setpinhole_split_fixed_focalNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices, facs__pinhole_split_fixed_focal__args__pose__idx_shared_, num);
}
void GraphSolver::SetPinholeSplitFixedFocalPrincipalPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_num_. Use "
        "Setpinhole_split_fixed_focalNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedFocalPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeSplitFixedFocalPrincipalPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_num_. Use "
        "Setpinhole_split_fixed_focalNum before setting indices.");
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
      facs__pinhole_split_fixed_focal__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::SetPinholeSplitFixedFocalPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_num_. Use "
        "Setpinhole_split_fixed_focalNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedFocalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeSplitFixedFocalPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_num_. Use "
        "Setpinhole_split_fixed_focalNum before setting indices.");
  }

  size_t tmp_size = SortIndicesGetTmpNbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  SharedIndices(
      indices, facs__pinhole_split_fixed_focal__args__point__idx_shared_, num);
}
void GraphSolver::SetPinholeSplitFixedFocalSensorFromRigDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_split_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeSplitFixedFocalSensorFromRigDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_split_fixed_focal_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedFocalPixelDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_split_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(marker__scratch_inout_,
                            facs__pinhole_split_fixed_focal__args__pixel__data_,
                            pinhole_split_fixed_focal_num_max_,
                            offset,
                            num);
}

void GraphSolver::SetPinholeSplitFixedFocalPixelDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_split_fixed_focal_num_max_");
  }
  ConstPixelStackedToCaspar(data,
                            facs__pinhole_split_fixed_focal__args__pixel__data_,
                            pinhole_split_fixed_focal_num_max_,
                            offset,
                            num);
}
void GraphSolver::SetPinholeSplitFixedFocalFocalDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_split_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal__args__focal__data_,
      pinhole_split_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeSplitFixedFocalFocalDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_split_fixed_focal_num_max_");
  }
  ConstPinholeFocalStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal__args__focal__data_,
      pinhole_split_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedPrincipalPointNum(const size_t num) {
  if (num > pinhole_split_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_split_fixed_principal_point_num_max_");
  }
  pinhole_split_fixed_principal_point_num_ = num;
}
void GraphSolver::SetPinholeSplitFixedPrincipalPointPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_principal_point_num_. Use "
        "Setpinhole_split_fixed_principal_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedPrincipalPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeSplitFixedPrincipalPointPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_principal_point_num_. Use "
        "Setpinhole_split_fixed_principal_pointNum before setting indices.");
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
      facs__pinhole_split_fixed_principal_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::SetPinholeSplitFixedPrincipalPointFocalIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_principal_point_num_. Use "
        "Setpinhole_split_fixed_principal_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedPrincipalPointFocalIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeSplitFixedPrincipalPointFocalIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_principal_point_num_. Use "
        "Setpinhole_split_fixed_principal_pointNum before setting indices.");
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
      facs__pinhole_split_fixed_principal_point__args__focal__idx_shared_,
      num);
}
void GraphSolver::SetPinholeSplitFixedPrincipalPointPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_principal_point_num_. Use "
        "Setpinhole_split_fixed_principal_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeSplitFixedPrincipalPointPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_principal_point_num_. Use "
        "Setpinhole_split_fixed_principal_pointNum before setting indices.");
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
      facs__pinhole_split_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPrincipalPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_split_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPrincipalPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_split_fixed_principal_point_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole_split_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedPrincipalPointPixelDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_split_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeSplitFixedPrincipalPointPixelDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_split_fixed_principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_split_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_split_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_split_fixed_principal_point_num_max_");
  }
  ConstPinholePrincipalPointStackedToCaspar(
      data,
      facs__pinhole_split_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedPoseFixedFocalNum(const size_t num) {
  if (num > pinhole_split_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_split_fixed_pose_fixed_focal_num_max_");
  }
  pinhole_split_fixed_pose_fixed_focal_num_ = num;
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalPrincipalPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_pose_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_pose_fixed_focal_num_. Use "
        "Setpinhole_split_fixed_pose_fixed_focalNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedPoseFixedFocalPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalPrincipalPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_pose_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_pose_fixed_focal_num_. Use "
        "Setpinhole_split_fixed_pose_fixed_focalNum before setting indices.");
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
      facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::SetPinholeSplitFixedPoseFixedFocalPointIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_pose_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_pose_fixed_focal_num_. Use "
        "Setpinhole_split_fixed_pose_fixed_focalNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedPoseFixedFocalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeSplitFixedPoseFixedFocalPointIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_pose_fixed_focal_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_pose_fixed_focal_num_. Use "
        "Setpinhole_split_fixed_pose_fixed_focalNum before setting indices.");
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
      facs__pinhole_split_fixed_pose_fixed_focal__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_focal__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedPoseFixedFocalPixelDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeSplitFixedPoseFixedFocalPixelDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_focal__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedPoseFixedFocalPoseDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholePoseStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeSplitFixedPoseFixedFocalPoseDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_num_max_");
  }
  ConstPinholePoseStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_focal__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedPoseFixedFocalFocalDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_focal__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeSplitFixedPoseFixedFocalFocalDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_focal_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_num_max_");
  }
  ConstPinholeFocalStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_focal__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedPoseFixedPrincipalPointNum(
    const size_t num) {
  if (num > pinhole_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_split_fixed_pose_fixed_principal_point_num_max_");
  }
  pinhole_split_fixed_pose_fixed_principal_point_num_ = num;
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointFocalIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_pose_fixed_principal_point_num_. Use "
        "Setpinhole_split_fixed_pose_fixed_principal_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedPoseFixedPrincipalPointFocalIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointFocalIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_pose_fixed_principal_point_num_. Use "
        "Setpinhole_split_fixed_pose_fixed_principal_pointNum before setting "
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
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_pose_fixed_principal_point_num_. Use "
        "Setpinhole_split_fixed_pose_fixed_principal_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedPoseFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_pose_fixed_principal_point_num_. Use "
        "Setpinhole_split_fixed_pose_fixed_principal_pointNum before setting "
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
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholePoseStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstPinholePoseStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstPinholePrincipalPointStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedFocalFixedPrincipalPointNum(
    const size_t num) {
  if (num > pinhole_split_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_split_fixed_focal_fixed_principal_point_num_max_");
  }
  pinhole_split_fixed_focal_fixed_principal_point_num_ = num;
}
void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointPoseIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_focal_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_fixed_principal_point_num_. Use "
        "Setpinhole_split_fixed_focal_fixed_principal_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedFocalFixedPrincipalPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointPoseIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_focal_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_fixed_principal_point_num_. Use "
        "Setpinhole_split_fixed_focal_fixed_principal_pointNum before setting "
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
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_focal_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_fixed_principal_point_num_. Use "
        "Setpinhole_split_fixed_focal_fixed_principal_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedFocalFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_focal_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_fixed_principal_point_num_. Use "
        "Setpinhole_split_fixed_focal_fixed_principal_pointNum before setting "
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
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_principal_point_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointFocalDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_principal_point_num_max_");
  }
  ConstPinholeFocalStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_principal_point_num_max_");
  }
  ConstPinholePrincipalPointStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedFocalFixedPointNum(const size_t num) {
  if (num > pinhole_split_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_split_fixed_focal_fixed_point_num_max_");
  }
  pinhole_split_fixed_focal_fixed_point_num_ = num;
}
void GraphSolver::SetPinholeSplitFixedFocalFixedPointPoseIndicesFromHost(
    const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_fixed_point_num_. Use "
        "Setpinhole_split_fixed_focal_fixed_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedFocalFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::SetPinholeSplitFixedFocalFixedPointPoseIndicesFromDevice(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_fixed_point_num_. Use "
        "Setpinhole_split_fixed_focal_fixed_pointNum before setting indices.");
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
      facs__pinhole_split_fixed_focal_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedFocalFixedPointPrincipalPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_fixed_point_num_. Use "
        "Setpinhole_split_fixed_focal_fixed_pointNum before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedFocalFixedPointPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeSplitFixedFocalFixedPointPrincipalPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_fixed_point_num_. Use "
        "Setpinhole_split_fixed_focal_fixed_pointNum before setting indices.");
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
      facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedFocalFixedPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedFocalFixedPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_point_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedFocalFixedPointPixelDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeSplitFixedFocalFixedPointPixelDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedFocalFixedPointFocalDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal_fixed_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeSplitFixedFocalFixedPointFocalDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_point_num_max_");
  }
  ConstPinholeFocalStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal_fixed_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedFocalFixedPointPointDataFromStackedHost(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal_fixed_point__args__point__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::SetPinholeSplitFixedFocalFixedPointPointDataFromStackedDevice(
    const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_focal_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal_fixed_point__args__point__data_,
      pinhole_split_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedPrincipalPointFixedPointNum(
    const size_t num) {
  if (num > pinhole_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_split_fixed_principal_point_fixed_point_num_max_");
  }
  pinhole_split_fixed_principal_point_fixed_point_num_ = num;
}
void GraphSolver::
    SetPinholeSplitFixedPrincipalPointFixedPointPoseIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_principal_point_fixed_point_num_. Use "
        "Setpinhole_split_fixed_principal_point_fixed_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeSplitFixedPrincipalPointFixedPointPoseIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_principal_point_fixed_point_num_. Use "
        "Setpinhole_split_fixed_principal_point_fixed_pointNum before setting "
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
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPrincipalPointFixedPointFocalIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_principal_point_fixed_point_num_. Use "
        "Setpinhole_split_fixed_principal_point_fixed_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedPrincipalPointFixedPointFocalIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeSplitFixedPrincipalPointFixedPointFocalIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_principal_point_fixed_point_num_. Use "
        "Setpinhole_split_fixed_principal_point_fixed_pointNum before setting "
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
      facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPinholePrincipalPointStackedToCaspar(
      data,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPrincipalPointFixedPointPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num > pinhole_split_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__pinhole_split_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointNum(
    const size_t num) {
  if (num >
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > "
        "pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_");
  }
  pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_ = num;
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_. "
        "Use Setpinhole_split_fixed_pose_fixed_focal_fixed_principal_pointNum "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_. "
        "Use Setpinhole_split_fixed_pose_fixed_focal_fixed_principal_pointNum "
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
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPoseDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholePoseStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPoseDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_");
  }
  ConstPinholePoseStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointFocalDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_");
  }
  ConstPinholeFocalStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_");
  }
  ConstPinholePrincipalPointStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedPoseFixedFocalFixedPointNum(
    const size_t num) {
  if (num > pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  pinhole_split_fixed_pose_fixed_focal_fixed_point_num_ = num;
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPointPrincipalPointIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_pose_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_pose_fixed_focal_fixed_point_num_. Use "
        "Setpinhole_split_fixed_pose_fixed_focal_fixed_pointNum before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedPoseFixedFocalFixedPointPrincipalPointIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPointPrincipalPointIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_pose_fixed_focal_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_pose_fixed_focal_fixed_point_num_. Use "
        "Setpinhole_split_fixed_pose_fixed_focal_fixed_pointNum before setting "
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
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPointPoseDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholePoseStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPointPoseDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  ConstPinholePoseStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPointFocalDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPointFocalDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  ConstPinholeFocalStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__focal__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPointPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedFocalFixedPointPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__point__data_,
      pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointNum(
    const size_t num) {
  if (num >
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > "
        "pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_ = num;
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointFocalIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_. "
        "Use Setpinhole_split_fixed_pose_fixed_principal_point_fixed_pointNum "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointFocalIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointFocalIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_. "
        "Use Setpinhole_split_fixed_pose_fixed_principal_point_fixed_pointNum "
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
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__focal__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholePoseStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPinholePoseStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPinholePrincipalPointStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointNum(
    const size_t num) {
  if (num >
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > "
        "pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_");
  }
  pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_ = num;
}
void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPoseIndicesFromHost(
        const unsigned int* const indices, size_t num) {
  cudaSetDevice(device_id_);
  if (num != pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_. "
        "Use Setpinhole_split_fixed_focal_fixed_principal_point_fixed_pointNum "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPoseIndicesFromDevice(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;
  cudaSetDevice(device_id_);

  if (num != pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_. "
        "Use Setpinhole_split_fixed_focal_fixed_principal_point_fixed_pointNum "
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
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeSensorFromRigStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPinholeSensorFromRigStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__sensor_from_rig__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPixelStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPixelStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointFocalDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointFocalDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPinholeFocalStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__focal__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPinholePrincipalPointStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPointDataFromStackedHost(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(float),
             cudaMemcpyHostToDevice);
  ConstPointStackedToCaspar(
      marker__scratch_inout_,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPointDataFromStackedDevice(
        const float* const data, size_t offset, size_t num) {
  cudaSetDevice(device_id_);
  if (offset + num >
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPointStackedToCaspar(
      data,
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

size_t GraphSolver::get_nbytes() {
  size_t offset = 0;
  size_t at_least = 0;
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<float>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<float>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 8 * PinholePose_num_, 4);
  increment_offset<float>(offset, 8 * PinholePose_num_, 4);
  increment_offset<float>(offset, 8 * PinholePose_num_, 4);
  increment_offset<float>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<float>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<float>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<float>(offset, 8 * SimpleRadialPose_num_, 4);
  increment_offset<float>(offset, 8 * SimpleRadialPose_num_, 4);
  increment_offset<float>(offset, 8 * SimpleRadialPose_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_num_, 4);
  increment_offset<float>(offset, 8 * simple_radial_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_num_, 4);
  increment_offset<float>(offset, 8 * simple_radial_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_pose_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_pose_num_, 4);
  increment_offset<float>(offset, 8 * simple_radial_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(offset, 8 * simple_radial_fixed_point_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(offset, 4 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 8 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 8 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 4 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_num_, 4);
  increment_offset<float>(offset, 8 * pinhole_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_num_, 4);
  increment_offset<float>(offset, 8 * pinhole_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_pose_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_num_, 4);
  increment_offset<float>(offset, 8 * pinhole_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 8 * pinhole_fixed_point_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 4 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 8 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(offset, 8 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(offset, 4 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_split_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 8 * simple_radial_split_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_split_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_split_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 8 * simple_radial_split_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_split_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 8 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 8 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 8 * simple_radial_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 8 * simple_radial_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      8 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      8 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      4 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_split_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      8 * simple_radial_split_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_split_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      4 * simple_radial_split_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      8 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      8 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      8 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      8 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      4 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      8 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      8 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      4 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      8 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      4 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_split_fixed_focal_num_, 4);
  increment_offset<float>(offset, 8 * pinhole_split_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_split_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_split_fixed_focal_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_split_fixed_focal_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_split_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 8 * pinhole_split_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_split_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 8 * pinhole_split_fixed_pose_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_split_fixed_pose_fixed_focal_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_split_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 8 * pinhole_split_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 8 * pinhole_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 8 * pinhole_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_split_fixed_focal_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 8 * pinhole_split_fixed_focal_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_split_fixed_focal_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_focal_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_focal_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_focal_fixed_principal_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_split_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 8 * pinhole_split_fixed_focal_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_split_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 4 * pinhole_split_fixed_focal_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_split_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 8 * pinhole_split_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_split_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 4 * pinhole_split_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<float>(
      offset,
      8 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      8 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset, 8 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 8 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 4 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset,
      8 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      8 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      4 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<SharedIndex>(
      offset,
      1 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      8 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      4 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
      4);
  at_least =
      std::max(at_least,
               offset + std::max({4 * PinholeCalib_num_max_,
                                  2 * PinholeFocal_num_max_,
                                  7 * PinholePose_num_max_,
                                  2 * PinholePrincipalPoint_num_max_,
                                  3 * Point_num_max_,
                                  4 * SimpleRadialCalib_num_max_,
                                  2 * SimpleRadialFocalAndExtra_num_max_,
                                  7 * SimpleRadialPose_num_max_,
                                  2 * SimpleRadialPrincipalPoint_num_max_}) *
                            sizeof(float));
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * simple_radial_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_pose_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(offset, 2 * pinhole_split_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_focal_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(offset, 12 * simple_radial_num_, 4);
  increment_offset<float>(offset, 4 * simple_radial_num_, 4);
  increment_offset<float>(offset, 6 * simple_radial_num_, 4);
  increment_offset<float>(offset, 4 * simple_radial_fixed_pose_num_, 4);
  increment_offset<float>(offset, 6 * simple_radial_fixed_pose_num_, 4);
  increment_offset<float>(offset, 12 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(offset, 4 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 4 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(offset, 12 * pinhole_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_num_, 4);
  increment_offset<float>(offset, 6 * pinhole_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_num_, 4);
  increment_offset<float>(offset, 6 * pinhole_fixed_pose_num_, 4);
  increment_offset<float>(offset, 12 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 12 * simple_radial_split_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 0 * simple_radial_split_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 6 * simple_radial_split_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 12 * simple_radial_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 4 * simple_radial_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 6 * simple_radial_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 0 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 6 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 4 * simple_radial_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 6 * simple_radial_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset,
      12 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      6 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      12 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      0 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      12 * simple_radial_split_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      4 * simple_radial_split_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      6 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      0 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      4 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      12 *
          simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(offset, 12 * pinhole_split_fixed_focal_num_, 4);
  increment_offset<float>(offset, 0 * pinhole_split_fixed_focal_num_, 4);
  increment_offset<float>(offset, 6 * pinhole_split_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 12 * pinhole_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 6 * pinhole_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 0 * pinhole_split_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 6 * pinhole_split_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 6 * pinhole_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 12 * pinhole_split_fixed_focal_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 6 * pinhole_split_fixed_focal_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 12 * pinhole_split_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 0 * pinhole_split_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 12 * pinhole_split_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<float>(
      offset,
      6 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset, 0 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      12 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 6 * PinholePose_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 6 * PinholePose_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 6 * PinholePose_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 6 * PinholePose_num_, 4);
  increment_offset<float>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<float>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<float>(offset, 0 * 0, 1);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 6 * PinholePose_num_, 4);
  increment_offset<float>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<float>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 6 * PinholePose_num_, 4);
  increment_offset<float>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<float>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 6 * PinholePose_num_, 4);
  increment_offset<float>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<float>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 4 * PinholeCalib_num_, 4);
  increment_offset<float>(offset, 6 * PinholeCalib_num_, 4);
  increment_offset<float>(offset, 2 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 1 * PinholeFocal_num_, 4);
  increment_offset<float>(offset, 6 * PinholePose_num_, 4);
  increment_offset<float>(offset, 16 * PinholePose_num_, 4);
  increment_offset<float>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<float>(offset, 1 * PinholePrincipalPoint_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * Point_num_, 4);
  increment_offset<float>(offset, 4 * SimpleRadialCalib_num_, 4);
  increment_offset<float>(offset, 6 * SimpleRadialCalib_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<float>(offset, 1 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<float>(offset, 6 * SimpleRadialPose_num_, 4);
  increment_offset<float>(offset, 16 * SimpleRadialPose_num_, 4);
  increment_offset<float>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<float>(offset, 1 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<float>(offset, 0 * 0, 1);
  increment_offset<float>(offset, 0 * 0, 4);
  increment_offset<float>(offset, 2 * simple_radial_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_pose_num_, 4);
  increment_offset<float>(offset, 2 * simple_radial_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_point_num_, 4);
  increment_offset<float>(offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<float>(
      offset, 2 * simple_radial_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(offset, 2 * pinhole_split_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_pose_fixed_focal_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_focal_fixed_principal_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_principal_point_fixed_point_num_, 4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_,
      4);
  increment_offset<float>(
      offset, 2 * pinhole_split_fixed_pose_fixed_focal_fixed_point_num_, 4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_,
      4);
  increment_offset<float>(
      offset,
      2 * pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_,
      4);
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