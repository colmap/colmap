#include "solver.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "caspar_mappings.h"
#include "kernel_PinholeFocalAndExtra_alpha_denumerator_or_beta_nummerator.h"
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
#include "kernel_PinholePrincipalPoint_alpha_denumerator_or_beta_nummerator.h"
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
#include "kernel_SimpleRadialFocalAndExtra_alpha_denumerator_or_beta_nummerator.h"
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
#include "kernel_SimpleRadialPrincipalPoint_alpha_denumerator_or_beta_nummerator.h"
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
#include "kernel_pinhole_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_point_score.h"
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
#include "kernel_pinhole_fixed_pose_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_pose_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_fixed_point_score.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_fixed_point_score.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_res_jac.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_fixed_principal_point_score.h"
#include "kernel_pinhole_fixed_pose_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_pose_res_jac.h"
#include "kernel_pinhole_fixed_pose_res_jac_first.h"
#include "kernel_pinhole_fixed_pose_score.h"
#include "kernel_pinhole_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_pinhole_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_pinhole_fixed_principal_point_fixed_point_score.h"
#include "kernel_pinhole_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_pinhole_fixed_principal_point_res_jac.h"
#include "kernel_pinhole_fixed_principal_point_res_jac_first.h"
#include "kernel_pinhole_fixed_principal_point_score.h"
#include "kernel_pinhole_jtjnjtr_direct.h"
#include "kernel_pinhole_res_jac.h"
#include "kernel_pinhole_res_jac_first.h"
#include "kernel_pinhole_score.h"
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
#include "kernel_simple_radial_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_point_score.h"
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
#include "kernel_simple_radial_fixed_pose_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_pose_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_fixed_point_score.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_fixed_point_score.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_res_jac.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_fixed_principal_point_score.h"
#include "kernel_simple_radial_fixed_pose_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_pose_res_jac.h"
#include "kernel_simple_radial_fixed_pose_res_jac_first.h"
#include "kernel_simple_radial_fixed_pose_score.h"
#include "kernel_simple_radial_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_principal_point_fixed_point_res_jac.h"
#include "kernel_simple_radial_fixed_principal_point_fixed_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_principal_point_fixed_point_score.h"
#include "kernel_simple_radial_fixed_principal_point_jtjnjtr_direct.h"
#include "kernel_simple_radial_fixed_principal_point_res_jac.h"
#include "kernel_simple_radial_fixed_principal_point_res_jac_first.h"
#include "kernel_simple_radial_fixed_principal_point_score.h"
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
    size_t PinholeFocalAndExtra_num_max,
    size_t PinholePrincipalPoint_num_max,
    size_t Point_num_max,
    size_t Pose_num_max,
    size_t SimpleRadialFocalAndExtra_num_max,
    size_t SimpleRadialPrincipalPoint_num_max,
    size_t simple_radial_num_max,
    size_t simple_radial_fixed_pose_num_max,
    size_t simple_radial_fixed_focal_and_extra_num_max,
    size_t simple_radial_fixed_principal_point_num_max,
    size_t simple_radial_fixed_point_num_max,
    size_t simple_radial_fixed_pose_fixed_focal_and_extra_num_max,
    size_t simple_radial_fixed_pose_fixed_principal_point_num_max,
    size_t simple_radial_fixed_pose_fixed_point_num_max,
    size_t simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max,
    size_t simple_radial_fixed_focal_and_extra_fixed_point_num_max,
    size_t simple_radial_fixed_principal_point_fixed_point_num_max,
    size_t
        simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max,
    size_t simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max,
    size_t simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max,
    size_t
        simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max,
    size_t pinhole_num_max,
    size_t pinhole_fixed_pose_num_max,
    size_t pinhole_fixed_focal_and_extra_num_max,
    size_t pinhole_fixed_principal_point_num_max,
    size_t pinhole_fixed_point_num_max,
    size_t pinhole_fixed_pose_fixed_focal_and_extra_num_max,
    size_t pinhole_fixed_pose_fixed_principal_point_num_max,
    size_t pinhole_fixed_pose_fixed_point_num_max,
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
      PinholeFocalAndExtra_num_(PinholeFocalAndExtra_num_max),
      PinholeFocalAndExtra_num_max_(PinholeFocalAndExtra_num_max),
      PinholePrincipalPoint_num_(PinholePrincipalPoint_num_max),
      PinholePrincipalPoint_num_max_(PinholePrincipalPoint_num_max),
      Point_num_(Point_num_max),
      Point_num_max_(Point_num_max),
      Pose_num_(Pose_num_max),
      Pose_num_max_(Pose_num_max),
      SimpleRadialFocalAndExtra_num_(SimpleRadialFocalAndExtra_num_max),
      SimpleRadialFocalAndExtra_num_max_(SimpleRadialFocalAndExtra_num_max),
      SimpleRadialPrincipalPoint_num_(SimpleRadialPrincipalPoint_num_max),
      SimpleRadialPrincipalPoint_num_max_(SimpleRadialPrincipalPoint_num_max),
      simple_radial_num_(simple_radial_num_max),
      simple_radial_num_max_(simple_radial_num_max),
      simple_radial_fixed_pose_num_(simple_radial_fixed_pose_num_max),
      simple_radial_fixed_pose_num_max_(simple_radial_fixed_pose_num_max),
      simple_radial_fixed_focal_and_extra_num_(
          simple_radial_fixed_focal_and_extra_num_max),
      simple_radial_fixed_focal_and_extra_num_max_(
          simple_radial_fixed_focal_and_extra_num_max),
      simple_radial_fixed_principal_point_num_(
          simple_radial_fixed_principal_point_num_max),
      simple_radial_fixed_principal_point_num_max_(
          simple_radial_fixed_principal_point_num_max),
      simple_radial_fixed_point_num_(simple_radial_fixed_point_num_max),
      simple_radial_fixed_point_num_max_(simple_radial_fixed_point_num_max),
      simple_radial_fixed_pose_fixed_focal_and_extra_num_(
          simple_radial_fixed_pose_fixed_focal_and_extra_num_max),
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_(
          simple_radial_fixed_pose_fixed_focal_and_extra_num_max),
      simple_radial_fixed_pose_fixed_principal_point_num_(
          simple_radial_fixed_pose_fixed_principal_point_num_max),
      simple_radial_fixed_pose_fixed_principal_point_num_max_(
          simple_radial_fixed_pose_fixed_principal_point_num_max),
      simple_radial_fixed_pose_fixed_point_num_(
          simple_radial_fixed_pose_fixed_point_num_max),
      simple_radial_fixed_pose_fixed_point_num_max_(
          simple_radial_fixed_pose_fixed_point_num_max),
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
      pinhole_num_(pinhole_num_max),
      pinhole_num_max_(pinhole_num_max),
      pinhole_fixed_pose_num_(pinhole_fixed_pose_num_max),
      pinhole_fixed_pose_num_max_(pinhole_fixed_pose_num_max),
      pinhole_fixed_focal_and_extra_num_(pinhole_fixed_focal_and_extra_num_max),
      pinhole_fixed_focal_and_extra_num_max_(
          pinhole_fixed_focal_and_extra_num_max),
      pinhole_fixed_principal_point_num_(pinhole_fixed_principal_point_num_max),
      pinhole_fixed_principal_point_num_max_(
          pinhole_fixed_principal_point_num_max),
      pinhole_fixed_point_num_(pinhole_fixed_point_num_max),
      pinhole_fixed_point_num_max_(pinhole_fixed_point_num_max),
      pinhole_fixed_pose_fixed_focal_and_extra_num_(
          pinhole_fixed_pose_fixed_focal_and_extra_num_max),
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_(
          pinhole_fixed_pose_fixed_focal_and_extra_num_max),
      pinhole_fixed_pose_fixed_principal_point_num_(
          pinhole_fixed_pose_fixed_principal_point_num_max),
      pinhole_fixed_pose_fixed_principal_point_num_max_(
          pinhole_fixed_pose_fixed_principal_point_num_max),
      pinhole_fixed_pose_fixed_point_num_(
          pinhole_fixed_pose_fixed_point_num_max),
      pinhole_fixed_pose_fixed_point_num_max_(
          pinhole_fixed_pose_fixed_point_num_max),
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
  nodes__PinholeFocalAndExtra__storage_current_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholeFocalAndExtra__storage_check_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholeFocalAndExtra__storage_new_best_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
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
  nodes__Pose__storage_current_ =
      assign_and_increment<double>(origin_ptr_, offset, 8 * Pose_num_, 4);
  nodes__Pose__storage_check_ =
      assign_and_increment<double>(origin_ptr_, offset, 8 * Pose_num_, 4);
  nodes__Pose__storage_new_best_ =
      assign_and_increment<double>(origin_ptr_, offset, 8 * Pose_num_, 4);
  nodes__SimpleRadialFocalAndExtra__storage_current_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__storage_check_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__storage_new_best_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPrincipalPoint__storage_current_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__storage_check_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__storage_new_best_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  facs__simple_radial__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_num_, 4);
  facs__simple_radial__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_num_, 4);
  facs__simple_radial__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_num_, 4);
  facs__simple_radial__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_num_, 4);
  facs__simple_radial__args__pixel__data_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_num_, 4);
  facs__simple_radial_fixed_pose__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_pose__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_pose__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_pose__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_pose__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 8 * simple_radial_fixed_pose_num_, 4);
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
  facs__simple_radial_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 4 * simple_radial_fixed_point_num_, 4);
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
  facs__simple_radial_fixed_pose_fixed_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_point__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_,
          offset,
          1 * simple_radial_fixed_pose_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_point__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          8 * simple_radial_fixed_pose_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_pose_fixed_point_num_,
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
  facs__pinhole__args__pose__idx_shared_ = assign_and_increment<SharedIndex>(
      origin_ptr_, offset, 1 * pinhole_num_, 4);
  facs__pinhole__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_num_, 4);
  facs__pinhole__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_num_, 4);
  facs__pinhole__args__point__idx_shared_ = assign_and_increment<SharedIndex>(
      origin_ptr_, offset, 1 * pinhole_num_, 4);
  facs__pinhole__args__pixel__data_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * pinhole_num_, 4);
  facs__pinhole_fixed_pose__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__pixel__data_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__pose__data_ = assign_and_increment<double>(
      origin_ptr_, offset, 8 * pinhole_fixed_pose_num_, 4);
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
  facs__pinhole_fixed_point__args__pose__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__pixel__data_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__point__data_ = assign_and_increment<double>(
      origin_ptr_, offset, 4 * pinhole_fixed_point_num_, 4);
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
  facs__pinhole_fixed_pose_fixed_point__args__focal_and_extra__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__args__principal_point__idx_shared_ =
      assign_and_increment<SharedIndex>(
          origin_ptr_, offset, 1 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__args__pixel__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__args__pose__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 8 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__args__point__data_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 4 * pinhole_fixed_pose_fixed_point_num_, 4);
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
  facs__simple_radial__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_num_, 4);
  facs__simple_radial_fixed_pose__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_focal_and_extra__res_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * simple_radial_fixed_focal_and_extra_num_, 4);
  facs__simple_radial_fixed_principal_point__res_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * simple_radial_fixed_principal_point_num_, 4);
  facs__simple_radial_fixed_point__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_fixed_point_num_, 4);
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
  facs__simple_radial_fixed_pose_fixed_point__res_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_point_num_,
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
  facs__pinhole__res_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * pinhole_num_, 4);
  facs__pinhole_fixed_pose__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_focal_and_extra__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_focal_and_extra_num_, 4);
  facs__pinhole_fixed_principal_point__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_principal_point_num_, 4);
  facs__pinhole_fixed_point__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_point_num_, 4);
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
  facs__pinhole_fixed_pose_fixed_point__res_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
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
  facs__simple_radial__args__pose__jac_ = assign_and_increment<double>(
      origin_ptr_, offset, 12 * simple_radial_num_, 4);
  facs__simple_radial__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 4 * simple_radial_num_, 4);
  facs__simple_radial__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 0 * simple_radial_num_, 4);
  facs__simple_radial__args__point__jac_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * simple_radial_num_, 4);
  facs__simple_radial_fixed_pose__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 4 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_pose__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 0 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_pose__args__point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 6 * simple_radial_fixed_pose_num_, 4);
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
  facs__simple_radial_fixed_point__args__pose__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 12 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 4 * simple_radial_fixed_point_num_, 4);
  facs__simple_radial_fixed_point__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 0 * simple_radial_fixed_point_num_, 4);
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
  facs__simple_radial_fixed_pose_fixed_point__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          4 * simple_radial_fixed_pose_fixed_point_num_,
          4);
  facs__simple_radial_fixed_pose_fixed_point__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          0 * simple_radial_fixed_pose_fixed_point_num_,
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
  facs__pinhole__args__pose__jac_ =
      assign_and_increment<double>(origin_ptr_, offset, 10 * pinhole_num_, 4);
  facs__pinhole__args__focal_and_extra__jac_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * pinhole_num_, 4);
  facs__pinhole__args__principal_point__jac_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * pinhole_num_, 4);
  facs__pinhole__args__point__jac_ =
      assign_and_increment<double>(origin_ptr_, offset, 6 * pinhole_num_, 4);
  facs__pinhole_fixed_pose__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 0 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_pose__args__point__jac_ = assign_and_increment<double>(
      origin_ptr_, offset, 6 * pinhole_fixed_pose_num_, 4);
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
  facs__pinhole_fixed_point__args__pose__jac_ = assign_and_increment<double>(
      origin_ptr_, offset, 10 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * pinhole_fixed_point_num_, 4);
  facs__pinhole_fixed_point__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 0 * pinhole_fixed_point_num_, 4);
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
  facs__pinhole_fixed_pose_fixed_point__args__focal_and_extra__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  facs__pinhole_fixed_pose_fixed_point__args__principal_point__jac_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 0 * pinhole_fixed_pose_fixed_point_num_, 4);
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
  nodes__PinholeFocalAndExtra__z_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholeFocalAndExtra__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePrincipalPoint__z_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__z_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Pose__z_ =
      assign_and_increment<double>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__Pose__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialFocalAndExtra__z_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPrincipalPoint__z_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__z_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocalAndExtra__p_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholeFocalAndExtra__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePrincipalPoint__p_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__p_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Pose__p_ =
      assign_and_increment<double>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__Pose__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialFocalAndExtra__p_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPrincipalPoint__p_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__p_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocalAndExtra__step_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholeFocalAndExtra__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholePrincipalPoint__step_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Point__step_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__Pose__step_ =
      assign_and_increment<double>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__Pose__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialFocalAndExtra__step_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__SimpleRadialPrincipalPoint__step_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  nodes__SimpleRadialPrincipalPoint__step_end__ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__w_start_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocalAndExtra__w_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholePrincipalPoint__w_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__Point__w_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Pose__w_ =
      assign_and_increment<double>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__SimpleRadialFocalAndExtra__w_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPrincipalPoint__w_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  marker__w_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 1);
  marker__r_0_start_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocalAndExtra__r_0_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholePrincipalPoint__r_0_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__Point__r_0_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Pose__r_0_ =
      assign_and_increment<double>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__SimpleRadialFocalAndExtra__r_0_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPrincipalPoint__r_0_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  marker__r_0_end_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__r_k_start_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocalAndExtra__r_k_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholePrincipalPoint__r_k_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__Point__r_k_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Pose__r_k_ =
      assign_and_increment<double>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__SimpleRadialFocalAndExtra__r_k_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPrincipalPoint__r_k_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  marker__r_k_end_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__Mp_start_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocalAndExtra__Mp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholePrincipalPoint__Mp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__Point__Mp_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Pose__Mp_ =
      assign_and_increment<double>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__SimpleRadialFocalAndExtra__Mp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialPrincipalPoint__Mp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  marker__Mp_end_ = assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  marker__precond_start_ =
      assign_and_increment<double>(origin_ptr_, offset, 0 * 0, 4);
  nodes__PinholeFocalAndExtra__precond_diag_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholeFocalAndExtra__precond_tril_ = assign_and_increment<double>(
      origin_ptr_, offset, 1 * PinholeFocalAndExtra_num_, 4);
  nodes__PinholePrincipalPoint__precond_diag_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * PinholePrincipalPoint_num_, 4);
  nodes__PinholePrincipalPoint__precond_tril_ = assign_and_increment<double>(
      origin_ptr_, offset, 1 * PinholePrincipalPoint_num_, 4);
  nodes__Point__precond_diag_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Point__precond_tril_ =
      assign_and_increment<double>(origin_ptr_, offset, 4 * Point_num_, 4);
  nodes__Pose__precond_diag_ =
      assign_and_increment<double>(origin_ptr_, offset, 6 * Pose_num_, 4);
  nodes__Pose__precond_tril_ =
      assign_and_increment<double>(origin_ptr_, offset, 16 * Pose_num_, 4);
  nodes__SimpleRadialFocalAndExtra__precond_diag_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  nodes__SimpleRadialFocalAndExtra__precond_tril_ =
      assign_and_increment<double>(
          origin_ptr_, offset, 1 * SimpleRadialFocalAndExtra_num_, 4);
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
  facs__simple_radial__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_num_, 4);
  facs__simple_radial_fixed_pose__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_fixed_pose_num_, 4);
  facs__simple_radial_fixed_focal_and_extra__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_fixed_focal_and_extra_num_, 4);
  facs__simple_radial_fixed_principal_point__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_fixed_principal_point_num_, 4);
  facs__simple_radial_fixed_point__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * simple_radial_fixed_point_num_, 4);
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
  facs__simple_radial_fixed_pose_fixed_point__jp_ =
      assign_and_increment<double>(
          origin_ptr_,
          offset,
          2 * simple_radial_fixed_pose_fixed_point_num_,
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
  facs__pinhole__jp_ =
      assign_and_increment<double>(origin_ptr_, offset, 2 * pinhole_num_, 4);
  facs__pinhole_fixed_pose__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_num_, 4);
  facs__pinhole_fixed_focal_and_extra__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_focal_and_extra_num_, 4);
  facs__pinhole_fixed_principal_point__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_principal_point_num_, 4);
  facs__pinhole_fixed_point__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_point_num_, 4);
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
  facs__pinhole_fixed_pose_fixed_point__jp_ = assign_and_increment<double>(
      origin_ptr_, offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
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
  solver__alpha_denumerator_ =
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
        double score_new_pcg = do_retract_score();
        if (!(score_new_pcg <= score_best_pcg * params_.pcg_rel_decrease_min)) {
          break;
        }
        std::swap(nodes__PinholeFocalAndExtra__storage_check_,
                  nodes__PinholeFocalAndExtra__storage_new_best_);
        std::swap(nodes__PinholePrincipalPoint__storage_check_,
                  nodes__PinholePrincipalPoint__storage_new_best_);
        std::swap(nodes__Point__storage_check_,
                  nodes__Point__storage_new_best_);
        std::swap(nodes__Pose__storage_check_, nodes__Pose__storage_new_best_);
        std::swap(nodes__SimpleRadialFocalAndExtra__storage_check_,
                  nodes__SimpleRadialFocalAndExtra__storage_new_best_);
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
      score_best_pcg = do_retract_score();
      std::swap(nodes__PinholeFocalAndExtra__storage_check_,
                nodes__PinholeFocalAndExtra__storage_new_best_);
      std::swap(nodes__PinholePrincipalPoint__storage_check_,
                nodes__PinholePrincipalPoint__storage_new_best_);
      std::swap(nodes__Point__storage_check_, nodes__Point__storage_new_best_);
      std::swap(nodes__Pose__storage_check_, nodes__Pose__storage_new_best_);
      std::swap(nodes__SimpleRadialFocalAndExtra__storage_check_,
                nodes__SimpleRadialFocalAndExtra__storage_new_best_);
      std::swap(nodes__SimpleRadialPrincipalPoint__storage_check_,
                nodes__SimpleRadialPrincipalPoint__storage_new_best_);
    }

    const double diag_current = diag;
    bool step_accepted = false;
    if (score_best_pcg < score_best * params_.solver_rel_decrease_min) {
      step_accepted = true;
      quality = (score_best - score_best_pcg) / get_pred_decrease();
      const double quality_tmp = 2 * quality - 1;
      double scale = std::max(params_.diag_scaling_down,
                              1.0f - quality_tmp * quality_tmp * quality_tmp);
      diag = std::max(params_.diag_min, diag * scale);
      cudaMemcpy(
          solver__current_diag_, &diag, sizeof(double), cudaMemcpyHostToDevice);
      up_scale = params_.diag_scaling_up;
      score_best = score_best_pcg;
      std::swap(nodes__PinholeFocalAndExtra__storage_current_,
                nodes__PinholeFocalAndExtra__storage_new_best_);
      std::swap(nodes__PinholePrincipalPoint__storage_current_,
                nodes__PinholePrincipalPoint__storage_new_best_);
      std::swap(nodes__Point__storage_current_,
                nodes__Point__storage_new_best_);
      std::swap(nodes__Pose__storage_current_, nodes__Pose__storage_new_best_);
      std::swap(nodes__SimpleRadialFocalAndExtra__storage_current_,
                nodes__SimpleRadialFocalAndExtra__storage_new_best_);
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

double GraphSolver::do_res_jac_first() {
  zero(solver__res_tot_, solver__res_tot_ + 1);
  zero(marker__r_0_start_, marker__precond_end_);

  simple_radial_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial__args__pose__idx_shared_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial__args__focal_and_extra__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial__args__principal_point__idx_shared_,
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
      facs__simple_radial__args__focal_and_extra__jac_,
      simple_radial_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial__args__principal_point__jac_,
      simple_radial_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
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
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_pose__args__focal_and_extra__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_pose__args__principal_point__idx_shared_,
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
      facs__simple_radial_fixed_pose__args__focal_and_extra__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_pose__args__principal_point__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_fixed_pose__args__point__jac_,
      simple_radial_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_num_);

  simple_radial_fixed_focal_and_extra_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
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

  simple_radial_fixed_principal_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
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

  simple_radial_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_point__args__focal_and_extra__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_point__args__principal_point__idx_shared_,
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
      facs__simple_radial_fixed_point__args__focal_and_extra__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_point__args__principal_point__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_fixed_point_num_);

  simple_radial_fixed_pose_fixed_focal_and_extra_res_jac_first(
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

  simple_radial_fixed_pose_fixed_principal_point_res_jac_first(
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

  simple_radial_fixed_pose_fixed_point_res_jac_first(
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__focal_and_extra__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_point__res_,
      simple_radial_fixed_pose_fixed_point_num_,
      solver__res_tot_,
      facs__simple_radial_fixed_pose_fixed_point__args__focal_and_extra__jac_,
      simple_radial_fixed_pose_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_pose_fixed_point__args__principal_point__jac_,
      simple_radial_fixed_pose_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_fixed_pose_fixed_point_num_);

  simple_radial_fixed_focal_and_extra_fixed_principal_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_);

  simple_radial_fixed_focal_and_extra_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_);

  simple_radial_fixed_principal_point_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      simple_radial_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      simple_radial_fixed_principal_point_fixed_point_num_);

  simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac_first(
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

  simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac_first(
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

  simple_radial_fixed_pose_fixed_principal_point_fixed_point_res_jac_first(
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

  simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_);

  pinhole_res_jac_first(nodes__Pose__storage_current_,
                        Pose_num_max_,
                        facs__pinhole__args__pose__idx_shared_,
                        nodes__PinholeFocalAndExtra__storage_current_,
                        PinholeFocalAndExtra_num_max_,
                        facs__pinhole__args__focal_and_extra__idx_shared_,
                        nodes__PinholePrincipalPoint__storage_current_,
                        PinholePrincipalPoint_num_max_,
                        facs__pinhole__args__principal_point__idx_shared_,
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
                        facs__pinhole__args__focal_and_extra__jac_,
                        pinhole_num_,
                        nodes__PinholeFocalAndExtra__r_k_,
                        PinholeFocalAndExtra_num_,
                        nodes__PinholeFocalAndExtra__precond_diag_,
                        PinholeFocalAndExtra_num_,
                        nodes__PinholeFocalAndExtra__precond_tril_,
                        PinholeFocalAndExtra_num_,
                        facs__pinhole__args__principal_point__jac_,
                        pinhole_num_,
                        nodes__PinholePrincipalPoint__r_k_,
                        PinholePrincipalPoint_num_,
                        nodes__PinholePrincipalPoint__precond_diag_,
                        PinholePrincipalPoint_num_,
                        nodes__PinholePrincipalPoint__precond_tril_,
                        PinholePrincipalPoint_num_,
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
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_pose__args__focal_and_extra__idx_shared_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_pose__args__principal_point__idx_shared_,
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
      facs__pinhole_fixed_pose__args__focal_and_extra__jac_,
      pinhole_fixed_pose_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_pose__args__principal_point__jac_,
      pinhole_fixed_pose_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      facs__pinhole_fixed_pose__args__point__jac_,
      pinhole_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_num_);

  pinhole_fixed_focal_and_extra_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
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

  pinhole_fixed_principal_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
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

  pinhole_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_point__args__pose__idx_shared_,
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_point__args__focal_and_extra__idx_shared_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_point__args__principal_point__idx_shared_,
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
      facs__pinhole_fixed_point__args__focal_and_extra__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_point__args__principal_point__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      pinhole_fixed_point_num_);

  pinhole_fixed_pose_fixed_focal_and_extra_res_jac_first(
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

  pinhole_fixed_pose_fixed_principal_point_res_jac_first(
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

  pinhole_fixed_pose_fixed_point_res_jac_first(
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__focal_and_extra__idx_shared_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_point_num_max_,

      facs__pinhole_fixed_pose_fixed_point__res_,
      pinhole_fixed_pose_fixed_point_num_,
      solver__res_tot_,
      facs__pinhole_fixed_pose_fixed_point__args__focal_and_extra__jac_,
      pinhole_fixed_pose_fixed_point_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_pose_fixed_point__args__principal_point__jac_,
      pinhole_fixed_pose_fixed_point_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      pinhole_fixed_pose_fixed_point_num_);

  pinhole_fixed_focal_and_extra_fixed_principal_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_);

  pinhole_fixed_focal_and_extra_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      pinhole_fixed_focal_and_extra_fixed_point_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      pinhole_fixed_focal_and_extra_fixed_point_num_);

  pinhole_fixed_principal_point_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      pinhole_fixed_principal_point_fixed_point_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      pinhole_fixed_principal_point_fixed_point_num_);

  pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac_first(
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

  pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac_first(
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

  pinhole_fixed_pose_fixed_principal_point_fixed_point_res_jac_first(
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

  pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_first(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_);
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
                        nodes__SimpleRadialFocalAndExtra__storage_current_,
                        SimpleRadialFocalAndExtra_num_max_,
                        facs__simple_radial__args__focal_and_extra__idx_shared_,
                        nodes__SimpleRadialPrincipalPoint__storage_current_,
                        SimpleRadialPrincipalPoint_num_max_,
                        facs__simple_radial__args__principal_point__idx_shared_,
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
                        facs__simple_radial__args__focal_and_extra__jac_,
                        simple_radial_num_,
                        nodes__SimpleRadialFocalAndExtra__r_k_,
                        SimpleRadialFocalAndExtra_num_,
                        nodes__SimpleRadialFocalAndExtra__precond_diag_,
                        SimpleRadialFocalAndExtra_num_,
                        nodes__SimpleRadialFocalAndExtra__precond_tril_,
                        SimpleRadialFocalAndExtra_num_,
                        facs__simple_radial__args__principal_point__jac_,
                        simple_radial_num_,
                        nodes__SimpleRadialPrincipalPoint__r_k_,
                        SimpleRadialPrincipalPoint_num_,
                        nodes__SimpleRadialPrincipalPoint__precond_diag_,
                        SimpleRadialPrincipalPoint_num_,
                        nodes__SimpleRadialPrincipalPoint__precond_tril_,
                        SimpleRadialPrincipalPoint_num_,
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
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_pose__args__focal_and_extra__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_pose__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__simple_radial_fixed_pose__args__point__idx_shared_,
      facs__simple_radial_fixed_pose__args__pixel__data_,
      simple_radial_fixed_pose_num_max_,
      facs__simple_radial_fixed_pose__args__pose__data_,
      simple_radial_fixed_pose_num_max_,

      facs__simple_radial_fixed_pose__res_,
      simple_radial_fixed_pose_num_,

      facs__simple_radial_fixed_pose__args__focal_and_extra__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_pose__args__principal_point__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_fixed_pose__args__point__jac_,
      simple_radial_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_pose_num_);

  simple_radial_fixed_focal_and_extra_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
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

  simple_radial_fixed_principal_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
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

  simple_radial_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__simple_radial_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_point__args__focal_and_extra__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_point__args__principal_point__idx_shared_,
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
      facs__simple_radial_fixed_point__args__focal_and_extra__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_point__args__principal_point__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_fixed_point_num_);

  simple_radial_fixed_pose_fixed_focal_and_extra_res_jac(
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

  simple_radial_fixed_pose_fixed_principal_point_res_jac(
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

  simple_radial_fixed_pose_fixed_point_res_jac(
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__focal_and_extra__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,

      facs__simple_radial_fixed_pose_fixed_point__res_,
      simple_radial_fixed_pose_fixed_point_num_,

      facs__simple_radial_fixed_pose_fixed_point__args__focal_and_extra__jac_,
      simple_radial_fixed_pose_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_pose_fixed_point__args__principal_point__jac_,
      simple_radial_fixed_pose_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_fixed_pose_fixed_point_num_);

  simple_radial_fixed_focal_and_extra_fixed_principal_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_);

  simple_radial_fixed_focal_and_extra_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_tril_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_);

  simple_radial_fixed_principal_point_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      simple_radial_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_tril_,
      SimpleRadialFocalAndExtra_num_,
      simple_radial_fixed_principal_point_fixed_point_num_);

  simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac(
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

  simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac(
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

  simple_radial_fixed_pose_fixed_principal_point_fixed_point_res_jac(
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

  simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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

      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_);

  pinhole_res_jac(nodes__Pose__storage_current_,
                  Pose_num_max_,
                  facs__pinhole__args__pose__idx_shared_,
                  nodes__PinholeFocalAndExtra__storage_current_,
                  PinholeFocalAndExtra_num_max_,
                  facs__pinhole__args__focal_and_extra__idx_shared_,
                  nodes__PinholePrincipalPoint__storage_current_,
                  PinholePrincipalPoint_num_max_,
                  facs__pinhole__args__principal_point__idx_shared_,
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
                  facs__pinhole__args__focal_and_extra__jac_,
                  pinhole_num_,
                  nodes__PinholeFocalAndExtra__r_k_,
                  PinholeFocalAndExtra_num_,
                  nodes__PinholeFocalAndExtra__precond_diag_,
                  PinholeFocalAndExtra_num_,
                  nodes__PinholeFocalAndExtra__precond_tril_,
                  PinholeFocalAndExtra_num_,
                  facs__pinhole__args__principal_point__jac_,
                  pinhole_num_,
                  nodes__PinholePrincipalPoint__r_k_,
                  PinholePrincipalPoint_num_,
                  nodes__PinholePrincipalPoint__precond_diag_,
                  PinholePrincipalPoint_num_,
                  nodes__PinholePrincipalPoint__precond_tril_,
                  PinholePrincipalPoint_num_,
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
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_pose__args__focal_and_extra__idx_shared_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_pose__args__principal_point__idx_shared_,
      nodes__Point__storage_current_,
      Point_num_max_,
      facs__pinhole_fixed_pose__args__point__idx_shared_,
      facs__pinhole_fixed_pose__args__pixel__data_,
      pinhole_fixed_pose_num_max_,
      facs__pinhole_fixed_pose__args__pose__data_,
      pinhole_fixed_pose_num_max_,

      facs__pinhole_fixed_pose__res_,
      pinhole_fixed_pose_num_,

      facs__pinhole_fixed_pose__args__focal_and_extra__jac_,
      pinhole_fixed_pose_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_pose__args__principal_point__jac_,
      pinhole_fixed_pose_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      facs__pinhole_fixed_pose__args__point__jac_,
      pinhole_fixed_pose_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_pose_num_);

  pinhole_fixed_focal_and_extra_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
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

  pinhole_fixed_principal_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
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

  pinhole_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
      facs__pinhole_fixed_point__args__pose__idx_shared_,
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_point__args__focal_and_extra__idx_shared_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_point__args__principal_point__idx_shared_,
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
      facs__pinhole_fixed_point__args__focal_and_extra__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_point__args__principal_point__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      pinhole_fixed_point_num_);

  pinhole_fixed_pose_fixed_focal_and_extra_res_jac(
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

  pinhole_fixed_pose_fixed_principal_point_res_jac(
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

  pinhole_fixed_pose_fixed_point_res_jac(
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__focal_and_extra__idx_shared_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_point_num_max_,

      facs__pinhole_fixed_pose_fixed_point__res_,
      pinhole_fixed_pose_fixed_point_num_,

      facs__pinhole_fixed_pose_fixed_point__args__focal_and_extra__jac_,
      pinhole_fixed_pose_fixed_point_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_pose_fixed_point__args__principal_point__jac_,
      pinhole_fixed_pose_fixed_point_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      pinhole_fixed_pose_fixed_point_num_);

  pinhole_fixed_focal_and_extra_fixed_principal_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__r_k_,
      Point_num_,
      nodes__Point__precond_diag_,
      Point_num_,
      nodes__Point__precond_tril_,
      Point_num_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_);

  pinhole_fixed_focal_and_extra_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      pinhole_fixed_focal_and_extra_fixed_point_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_tril_,
      PinholePrincipalPoint_num_,
      pinhole_fixed_focal_and_extra_fixed_point_num_);

  pinhole_fixed_principal_point_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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
      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      pinhole_fixed_principal_point_fixed_point_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_tril_,
      PinholeFocalAndExtra_num_,
      pinhole_fixed_principal_point_fixed_point_num_);

  pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac(
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

  pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac(
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

  pinhole_fixed_pose_fixed_principal_point_fixed_point_res_jac(
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

  pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac(
      nodes__Pose__storage_current_,
      Pose_num_max_,
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

      nodes__Pose__r_k_,
      Pose_num_,
      nodes__Pose__precond_diag_,
      Pose_num_,
      nodes__Pose__precond_tril_,
      Pose_num_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_);
  copy(marker__r_k_start_, marker__r_k_end_, marker__r_0_start_);
  copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
}

void GraphSolver::do_normalize() {
  double* r_k;
  double* z;
  z = pcg_iter_ == 0 ? nodes__PinholeFocalAndExtra__p_
                     : nodes__PinholeFocalAndExtra__z_;
  PinholeFocalAndExtra_normalize(nodes__PinholeFocalAndExtra__precond_diag_,
                                 PinholeFocalAndExtra_num_,
                                 nodes__PinholeFocalAndExtra__precond_tril_,
                                 PinholeFocalAndExtra_num_,
                                 nodes__PinholeFocalAndExtra__r_k_,
                                 PinholeFocalAndExtra_num_,
                                 solver__current_diag_,
                                 z,
                                 PinholeFocalAndExtra_num_,
                                 PinholeFocalAndExtra_num_);
  z = pcg_iter_ == 0 ? nodes__PinholePrincipalPoint__p_
                     : nodes__PinholePrincipalPoint__z_;
  PinholePrincipalPoint_normalize(nodes__PinholePrincipalPoint__precond_diag_,
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
  z = pcg_iter_ == 0 ? nodes__SimpleRadialFocalAndExtra__p_
                     : nodes__SimpleRadialFocalAndExtra__z_;
  SimpleRadialFocalAndExtra_normalize(
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
  z = pcg_iter_ == 0 ? nodes__SimpleRadialPrincipalPoint__p_
                     : nodes__SimpleRadialPrincipalPoint__z_;
  SimpleRadialPrincipalPoint_normalize(
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

void GraphSolver::do_update_Mp() {
  PinholeFocalAndExtra_update_Mp(nodes__PinholeFocalAndExtra__r_k_,
                                 PinholeFocalAndExtra_num_,
                                 nodes__PinholeFocalAndExtra__Mp_,
                                 PinholeFocalAndExtra_num_,
                                 solver__beta_,
                                 nodes__PinholeFocalAndExtra__Mp_,
                                 PinholeFocalAndExtra_num_,
                                 nodes__PinholeFocalAndExtra__w_,
                                 PinholeFocalAndExtra_num_,
                                 PinholeFocalAndExtra_num_);
  PinholePrincipalPoint_update_Mp(nodes__PinholePrincipalPoint__r_k_,
                                  PinholePrincipalPoint_num_,
                                  nodes__PinholePrincipalPoint__Mp_,
                                  PinholePrincipalPoint_num_,
                                  solver__beta_,
                                  nodes__PinholePrincipalPoint__Mp_,
                                  PinholePrincipalPoint_num_,
                                  nodes__PinholePrincipalPoint__w_,
                                  PinholePrincipalPoint_num_,
                                  PinholePrincipalPoint_num_);
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
  SimpleRadialFocalAndExtra_update_Mp(nodes__SimpleRadialFocalAndExtra__r_k_,
                                      SimpleRadialFocalAndExtra_num_,
                                      nodes__SimpleRadialFocalAndExtra__Mp_,
                                      SimpleRadialFocalAndExtra_num_,
                                      solver__beta_,
                                      nodes__SimpleRadialFocalAndExtra__Mp_,
                                      SimpleRadialFocalAndExtra_num_,
                                      nodes__SimpleRadialFocalAndExtra__w_,
                                      SimpleRadialFocalAndExtra_num_,
                                      SimpleRadialFocalAndExtra_num_);
  SimpleRadialPrincipalPoint_update_Mp(nodes__SimpleRadialPrincipalPoint__r_k_,
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

void GraphSolver::do_jtjp_direct() {
  simple_radial_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__simple_radial__args__pose__idx_shared_,
      facs__simple_radial__args__pose__jac_,
      simple_radial_num_,
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial__args__focal_and_extra__idx_shared_,
      facs__simple_radial__args__focal_and_extra__jac_,
      simple_radial_num_,
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial__args__principal_point__idx_shared_,
      facs__simple_radial__args__principal_point__jac_,
      simple_radial_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial__args__point__idx_shared_,
      facs__simple_radial__args__point__jac_,
      simple_radial_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_num_);
  simple_radial_fixed_pose_jtjnjtr_direct(
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_pose__args__focal_and_extra__idx_shared_,
      facs__simple_radial_fixed_pose__args__focal_and_extra__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_fixed_pose__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_pose__args__principal_point__jac_,
      simple_radial_fixed_pose_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_fixed_pose__args__point__idx_shared_,
      facs__simple_radial_fixed_pose__args__point__jac_,
      simple_radial_fixed_pose_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_pose_num_);
  simple_radial_fixed_focal_and_extra_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
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
      nodes__Pose__w_,
      Pose_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_focal_and_extra_num_);
  simple_radial_fixed_principal_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
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
      nodes__Pose__w_,
      Pose_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_principal_point_num_);
  simple_radial_fixed_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__simple_radial_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_point__args__pose__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_fixed_point__args__focal_and_extra__jac_,
      simple_radial_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_point__args__principal_point__jac_,
      simple_radial_fixed_point_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_fixed_point_num_);
  simple_radial_fixed_pose_fixed_focal_and_extra_jtjnjtr_direct(
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
  simple_radial_fixed_pose_fixed_principal_point_jtjnjtr_direct(
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
  simple_radial_fixed_pose_fixed_point_jtjnjtr_direct(
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_pose_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_point__args__focal_and_extra__jac_,
      simple_radial_fixed_pose_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_fixed_pose_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_point__args__principal_point__jac_,
      simple_radial_fixed_pose_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_fixed_pose_fixed_point_num_);
  simple_radial_fixed_focal_and_extra_fixed_principal_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__p_,
      Point_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__Point__w_,
      Point_num_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_);
  simple_radial_fixed_focal_and_extra_fixed_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__jac_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_,
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_);
  simple_radial_fixed_principal_point_fixed_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pose__jac_,
      simple_radial_fixed_principal_point_fixed_point_num_,
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      simple_radial_fixed_principal_point_fixed_point_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      simple_radial_fixed_principal_point_fixed_point_num_);
  pinhole_jtjnjtr_direct(nodes__Pose__p_,
                         Pose_num_,
                         facs__pinhole__args__pose__idx_shared_,
                         facs__pinhole__args__pose__jac_,
                         pinhole_num_,
                         nodes__PinholeFocalAndExtra__p_,
                         PinholeFocalAndExtra_num_,
                         facs__pinhole__args__focal_and_extra__idx_shared_,
                         facs__pinhole__args__focal_and_extra__jac_,
                         pinhole_num_,
                         nodes__PinholePrincipalPoint__p_,
                         PinholePrincipalPoint_num_,
                         facs__pinhole__args__principal_point__idx_shared_,
                         facs__pinhole__args__principal_point__jac_,
                         pinhole_num_,
                         nodes__Point__p_,
                         Point_num_,
                         facs__pinhole__args__point__idx_shared_,
                         facs__pinhole__args__point__jac_,
                         pinhole_num_,
                         nodes__Pose__w_,
                         Pose_num_,
                         nodes__PinholeFocalAndExtra__w_,
                         PinholeFocalAndExtra_num_,
                         nodes__PinholePrincipalPoint__w_,
                         PinholePrincipalPoint_num_,
                         nodes__Point__w_,
                         Point_num_,
                         pinhole_num_);
  pinhole_fixed_pose_jtjnjtr_direct(
      nodes__PinholeFocalAndExtra__p_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_pose__args__focal_and_extra__idx_shared_,
      facs__pinhole_fixed_pose__args__focal_and_extra__jac_,
      pinhole_fixed_pose_num_,
      nodes__PinholePrincipalPoint__p_,
      PinholePrincipalPoint_num_,
      facs__pinhole_fixed_pose__args__principal_point__idx_shared_,
      facs__pinhole_fixed_pose__args__principal_point__jac_,
      pinhole_fixed_pose_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_fixed_pose__args__point__idx_shared_,
      facs__pinhole_fixed_pose__args__point__jac_,
      pinhole_fixed_pose_num_,
      nodes__PinholeFocalAndExtra__w_,
      PinholeFocalAndExtra_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_pose_num_);
  pinhole_fixed_focal_and_extra_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
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
      nodes__Pose__w_,
      Pose_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_focal_and_extra_num_);
  pinhole_fixed_principal_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
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
      nodes__Pose__w_,
      Pose_num_,
      nodes__PinholeFocalAndExtra__w_,
      PinholeFocalAndExtra_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_principal_point_num_);
  pinhole_fixed_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__pinhole_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_point__args__pose__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholeFocalAndExtra__p_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_point__args__focal_and_extra__idx_shared_,
      facs__pinhole_fixed_point__args__focal_and_extra__jac_,
      pinhole_fixed_point_num_,
      nodes__PinholePrincipalPoint__p_,
      PinholePrincipalPoint_num_,
      facs__pinhole_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_fixed_point__args__principal_point__jac_,
      pinhole_fixed_point_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__PinholeFocalAndExtra__w_,
      PinholeFocalAndExtra_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      pinhole_fixed_point_num_);
  pinhole_fixed_pose_fixed_focal_and_extra_jtjnjtr_direct(
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
  pinhole_fixed_pose_fixed_principal_point_jtjnjtr_direct(
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
  pinhole_fixed_pose_fixed_point_jtjnjtr_direct(
      nodes__PinholeFocalAndExtra__p_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_pose_fixed_point__args__focal_and_extra__idx_shared_,
      facs__pinhole_fixed_pose_fixed_point__args__focal_and_extra__jac_,
      pinhole_fixed_pose_fixed_point_num_,
      nodes__PinholePrincipalPoint__p_,
      PinholePrincipalPoint_num_,
      facs__pinhole_fixed_pose_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_point__args__principal_point__jac_,
      pinhole_fixed_pose_fixed_point_num_,
      nodes__PinholeFocalAndExtra__w_,
      PinholeFocalAndExtra_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      pinhole_fixed_pose_fixed_point_num_);
  pinhole_fixed_focal_and_extra_fixed_principal_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Point__p_,
      Point_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__jac_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__Point__w_,
      Point_num_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_);
  pinhole_fixed_focal_and_extra_fixed_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__jac_,
      pinhole_fixed_focal_and_extra_fixed_point_num_,
      nodes__PinholePrincipalPoint__p_,
      PinholePrincipalPoint_num_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__jac_,
      pinhole_fixed_focal_and_extra_fixed_point_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      pinhole_fixed_focal_and_extra_fixed_point_num_);
  pinhole_fixed_principal_point_fixed_point_jtjnjtr_direct(
      nodes__Pose__p_,
      Pose_num_,
      facs__pinhole_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      facs__pinhole_fixed_principal_point_fixed_point__args__pose__jac_,
      pinhole_fixed_principal_point_fixed_point_num_,
      nodes__PinholeFocalAndExtra__p_,
      PinholeFocalAndExtra_num_,
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__jac_,
      pinhole_fixed_principal_point_fixed_point_num_,
      nodes__Pose__w_,
      Pose_num_,
      nodes__PinholeFocalAndExtra__w_,
      PinholeFocalAndExtra_num_,
      pinhole_fixed_principal_point_fixed_point_num_);
}

void GraphSolver::do_alpha_first() {
  zero(solver__alpha_numerator_, solver__alpha_denumerator_ + 1);
  double* p_kp1;
  double* r_k;
  PinholeFocalAndExtra_alpha_numerator_denominator(
      nodes__PinholeFocalAndExtra__p_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__w_,
      PinholeFocalAndExtra_num_,
      solver__alpha_numerator_,
      solver__alpha_denumerator_,
      PinholeFocalAndExtra_num_);
  PinholePrincipalPoint_alpha_numerator_denominator(
      nodes__PinholePrincipalPoint__p_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      solver__alpha_numerator_,
      solver__alpha_denumerator_,
      PinholePrincipalPoint_num_);
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
  SimpleRadialFocalAndExtra_alpha_numerator_denominator(
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      solver__alpha_numerator_,
      solver__alpha_denumerator_,
      SimpleRadialFocalAndExtra_num_);
  SimpleRadialPrincipalPoint_alpha_numerator_denominator(
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      solver__alpha_numerator_,
      solver__alpha_denumerator_,
      SimpleRadialPrincipalPoint_num_);

  alpha_from_num_denum(solver__alpha_numerator_,
                       solver__alpha_denumerator_,
                       solver__alpha_,
                       solver__neg_alpha_);
}

void GraphSolver::do_alpha() {
  zero(solver__alpha_denumerator_, solver__alpha_denumerator_ + 1);
  PinholeFocalAndExtra_alpha_denumerator_or_beta_nummerator(
      nodes__PinholeFocalAndExtra__p_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__w_,
      PinholeFocalAndExtra_num_,
      solver__alpha_denumerator_,
      PinholeFocalAndExtra_num_);
  PinholePrincipalPoint_alpha_denumerator_or_beta_nummerator(
      nodes__PinholePrincipalPoint__p_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__w_,
      PinholePrincipalPoint_num_,
      solver__alpha_denumerator_,
      PinholePrincipalPoint_num_);
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
  SimpleRadialFocalAndExtra_alpha_denumerator_or_beta_nummerator(
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      solver__alpha_denumerator_,
      SimpleRadialFocalAndExtra_num_);
  SimpleRadialPrincipalPoint_alpha_denumerator_or_beta_nummerator(
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__w_,
      SimpleRadialPrincipalPoint_num_,
      solver__alpha_denumerator_,
      SimpleRadialPrincipalPoint_num_);

  alpha_from_num_denum(solver__beta_numerator_,
                       solver__alpha_denumerator_,
                       solver__alpha_,
                       solver__neg_alpha_);
}

void GraphSolver::do_update_step_first() {
  PinholeFocalAndExtra_update_step_first(nodes__PinholeFocalAndExtra__p_,
                                         PinholeFocalAndExtra_num_,
                                         solver__alpha_,
                                         nodes__PinholeFocalAndExtra__step_,
                                         PinholeFocalAndExtra_num_,
                                         PinholeFocalAndExtra_num_);
  PinholePrincipalPoint_update_step_first(nodes__PinholePrincipalPoint__p_,
                                          PinholePrincipalPoint_num_,
                                          solver__alpha_,
                                          nodes__PinholePrincipalPoint__step_,
                                          PinholePrincipalPoint_num_,
                                          PinholePrincipalPoint_num_);
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
  SimpleRadialFocalAndExtra_update_step_first(
      nodes__SimpleRadialFocalAndExtra__p_,
      SimpleRadialFocalAndExtra_num_,
      solver__alpha_,
      nodes__SimpleRadialFocalAndExtra__step_,
      SimpleRadialFocalAndExtra_num_,
      SimpleRadialFocalAndExtra_num_);
  SimpleRadialPrincipalPoint_update_step_first(
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      solver__alpha_,
      nodes__SimpleRadialPrincipalPoint__step_,
      SimpleRadialPrincipalPoint_num_,
      SimpleRadialPrincipalPoint_num_);
}

void GraphSolver::do_update_step() {
  PinholeFocalAndExtra_update_step(nodes__PinholeFocalAndExtra__step_,
                                   PinholeFocalAndExtra_num_,
                                   nodes__PinholeFocalAndExtra__p_,
                                   PinholeFocalAndExtra_num_,
                                   solver__alpha_,
                                   nodes__PinholeFocalAndExtra__step_,
                                   PinholeFocalAndExtra_num_,
                                   PinholeFocalAndExtra_num_);
  PinholePrincipalPoint_update_step(nodes__PinholePrincipalPoint__step_,
                                    PinholePrincipalPoint_num_,
                                    nodes__PinholePrincipalPoint__p_,
                                    PinholePrincipalPoint_num_,
                                    solver__alpha_,
                                    nodes__PinholePrincipalPoint__step_,
                                    PinholePrincipalPoint_num_,
                                    PinholePrincipalPoint_num_);
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
  SimpleRadialFocalAndExtra_update_step(nodes__SimpleRadialFocalAndExtra__step_,
                                        SimpleRadialFocalAndExtra_num_,
                                        nodes__SimpleRadialFocalAndExtra__p_,
                                        SimpleRadialFocalAndExtra_num_,
                                        solver__alpha_,
                                        nodes__SimpleRadialFocalAndExtra__step_,
                                        SimpleRadialFocalAndExtra_num_,
                                        SimpleRadialFocalAndExtra_num_);
  SimpleRadialPrincipalPoint_update_step(
      nodes__SimpleRadialPrincipalPoint__step_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__p_,
      SimpleRadialPrincipalPoint_num_,
      solver__alpha_,
      nodes__SimpleRadialPrincipalPoint__step_,
      SimpleRadialPrincipalPoint_num_,
      SimpleRadialPrincipalPoint_num_);
}

void GraphSolver::do_update_r_first() {
  zero(solver__r_0_norm2_tot_, solver__r_0_norm2_tot_ + 1);

  PinholeFocalAndExtra_update_r_first(nodes__PinholeFocalAndExtra__r_k_,
                                      PinholeFocalAndExtra_num_,
                                      nodes__PinholeFocalAndExtra__w_,
                                      PinholeFocalAndExtra_num_,
                                      solver__neg_alpha_,
                                      nodes__PinholeFocalAndExtra__r_k_,
                                      PinholeFocalAndExtra_num_,
                                      solver__r_0_norm2_tot_,
                                      solver__r_kp1_norm2_tot_,
                                      PinholeFocalAndExtra_num_);

  PinholePrincipalPoint_update_r_first(nodes__PinholePrincipalPoint__r_k_,
                                       PinholePrincipalPoint_num_,
                                       nodes__PinholePrincipalPoint__w_,
                                       PinholePrincipalPoint_num_,
                                       solver__neg_alpha_,
                                       nodes__PinholePrincipalPoint__r_k_,
                                       PinholePrincipalPoint_num_,
                                       solver__r_0_norm2_tot_,
                                       solver__r_kp1_norm2_tot_,
                                       PinholePrincipalPoint_num_);

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

  SimpleRadialFocalAndExtra_update_r_first(
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__w_,
      SimpleRadialFocalAndExtra_num_,
      solver__neg_alpha_,
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      solver__r_0_norm2_tot_,
      solver__r_kp1_norm2_tot_,
      SimpleRadialFocalAndExtra_num_);

  SimpleRadialPrincipalPoint_update_r_first(
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

  pcg_r_0_norm2_ = read_cumem(solver__r_0_norm2_tot_);
  pcg_r_kp1_norm2_ = read_cumem(solver__r_kp1_norm2_tot_);
}

void GraphSolver::do_update_r() {
  zero(solver__r_kp1_norm2_tot_, solver__r_kp1_norm2_tot_ + 1);

  PinholeFocalAndExtra_update_r(nodes__PinholeFocalAndExtra__r_k_,
                                PinholeFocalAndExtra_num_,
                                nodes__PinholeFocalAndExtra__w_,
                                PinholeFocalAndExtra_num_,
                                solver__neg_alpha_,
                                nodes__PinholeFocalAndExtra__r_k_,
                                PinholeFocalAndExtra_num_,
                                solver__r_kp1_norm2_tot_,
                                PinholeFocalAndExtra_num_);
  PinholePrincipalPoint_update_r(nodes__PinholePrincipalPoint__r_k_,
                                 PinholePrincipalPoint_num_,
                                 nodes__PinholePrincipalPoint__w_,
                                 PinholePrincipalPoint_num_,
                                 solver__neg_alpha_,
                                 nodes__PinholePrincipalPoint__r_k_,
                                 PinholePrincipalPoint_num_,
                                 solver__r_kp1_norm2_tot_,
                                 PinholePrincipalPoint_num_);
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
  SimpleRadialFocalAndExtra_update_r(nodes__SimpleRadialFocalAndExtra__r_k_,
                                     SimpleRadialFocalAndExtra_num_,
                                     nodes__SimpleRadialFocalAndExtra__w_,
                                     SimpleRadialFocalAndExtra_num_,
                                     solver__neg_alpha_,
                                     nodes__SimpleRadialFocalAndExtra__r_k_,
                                     SimpleRadialFocalAndExtra_num_,
                                     solver__r_kp1_norm2_tot_,
                                     SimpleRadialFocalAndExtra_num_);
  SimpleRadialPrincipalPoint_update_r(nodes__SimpleRadialPrincipalPoint__r_k_,
                                      SimpleRadialPrincipalPoint_num_,
                                      nodes__SimpleRadialPrincipalPoint__w_,
                                      SimpleRadialPrincipalPoint_num_,
                                      solver__neg_alpha_,
                                      nodes__SimpleRadialPrincipalPoint__r_k_,
                                      SimpleRadialPrincipalPoint_num_,
                                      solver__r_kp1_norm2_tot_,
                                      SimpleRadialPrincipalPoint_num_);
  pcg_r_kp1_norm2_ = read_cumem(solver__r_kp1_norm2_tot_);
}

double GraphSolver::do_retract_score() {
  PinholeFocalAndExtra_retract(nodes__PinholeFocalAndExtra__storage_current_,
                               PinholeFocalAndExtra_num_max_,
                               nodes__PinholeFocalAndExtra__step_,
                               PinholeFocalAndExtra_num_,
                               nodes__PinholeFocalAndExtra__storage_check_,
                               PinholeFocalAndExtra_num_max_,
                               PinholeFocalAndExtra_num_);
  PinholePrincipalPoint_retract(nodes__PinholePrincipalPoint__storage_current_,
                                PinholePrincipalPoint_num_max_,
                                nodes__PinholePrincipalPoint__step_,
                                PinholePrincipalPoint_num_,
                                nodes__PinholePrincipalPoint__storage_check_,
                                PinholePrincipalPoint_num_max_,
                                PinholePrincipalPoint_num_);
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
  SimpleRadialFocalAndExtra_retract(
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      nodes__SimpleRadialFocalAndExtra__step_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__storage_check_,
      SimpleRadialFocalAndExtra_num_max_,
      SimpleRadialFocalAndExtra_num_);
  SimpleRadialPrincipalPoint_retract(
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      nodes__SimpleRadialPrincipalPoint__step_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__storage_check_,
      SimpleRadialPrincipalPoint_num_max_,
      SimpleRadialPrincipalPoint_num_);

  zero(solver__res_tot_, solver__res_tot_ + 1);
  simple_radial_score(nodes__Pose__storage_check_,
                      Pose_num_max_,
                      facs__simple_radial__args__pose__idx_shared_,
                      nodes__SimpleRadialFocalAndExtra__storage_check_,
                      SimpleRadialFocalAndExtra_num_max_,
                      facs__simple_radial__args__focal_and_extra__idx_shared_,
                      nodes__SimpleRadialPrincipalPoint__storage_check_,
                      SimpleRadialPrincipalPoint_num_max_,
                      facs__simple_radial__args__principal_point__idx_shared_,
                      nodes__Point__storage_check_,
                      Point_num_max_,
                      facs__simple_radial__args__point__idx_shared_,
                      facs__simple_radial__args__pixel__data_,
                      simple_radial_num_max_,
                      solver__res_tot_,
                      simple_radial_num_);
  simple_radial_fixed_pose_score(
      nodes__SimpleRadialFocalAndExtra__storage_check_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_pose__args__focal_and_extra__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_check_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_pose__args__principal_point__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__simple_radial_fixed_pose__args__point__idx_shared_,
      facs__simple_radial_fixed_pose__args__pixel__data_,
      simple_radial_fixed_pose_num_max_,
      facs__simple_radial_fixed_pose__args__pose__data_,
      simple_radial_fixed_pose_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_num_);
  simple_radial_fixed_focal_and_extra_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
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
  simple_radial_fixed_principal_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
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
  simple_radial_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__simple_radial_fixed_point__args__pose__idx_shared_,
      nodes__SimpleRadialFocalAndExtra__storage_check_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_point__args__focal_and_extra__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_check_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_point__args__pixel__data_,
      simple_radial_fixed_point_num_max_,
      facs__simple_radial_fixed_point__args__point__data_,
      simple_radial_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_point_num_);
  simple_radial_fixed_pose_fixed_focal_and_extra_score(
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
  simple_radial_fixed_pose_fixed_principal_point_score(
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
  simple_radial_fixed_pose_fixed_point_score(
      nodes__SimpleRadialFocalAndExtra__storage_check_,
      SimpleRadialFocalAndExtra_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__focal_and_extra__idx_shared_,
      nodes__SimpleRadialPrincipalPoint__storage_check_,
      SimpleRadialPrincipalPoint_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__principal_point__idx_shared_,
      facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      facs__simple_radial_fixed_pose_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_point_num_max_,
      solver__res_tot_,
      simple_radial_fixed_pose_fixed_point_num_);
  simple_radial_fixed_focal_and_extra_fixed_principal_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
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
  simple_radial_fixed_focal_and_extra_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
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
  simple_radial_fixed_principal_point_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
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
  simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_score(
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
  simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_score(
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
  simple_radial_fixed_pose_fixed_principal_point_fixed_point_score(
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
  simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
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
  pinhole_score(nodes__Pose__storage_check_,
                Pose_num_max_,
                facs__pinhole__args__pose__idx_shared_,
                nodes__PinholeFocalAndExtra__storage_check_,
                PinholeFocalAndExtra_num_max_,
                facs__pinhole__args__focal_and_extra__idx_shared_,
                nodes__PinholePrincipalPoint__storage_check_,
                PinholePrincipalPoint_num_max_,
                facs__pinhole__args__principal_point__idx_shared_,
                nodes__Point__storage_check_,
                Point_num_max_,
                facs__pinhole__args__point__idx_shared_,
                facs__pinhole__args__pixel__data_,
                pinhole_num_max_,
                solver__res_tot_,
                pinhole_num_);
  pinhole_fixed_pose_score(
      nodes__PinholeFocalAndExtra__storage_check_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_pose__args__focal_and_extra__idx_shared_,
      nodes__PinholePrincipalPoint__storage_check_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_pose__args__principal_point__idx_shared_,
      nodes__Point__storage_check_,
      Point_num_max_,
      facs__pinhole_fixed_pose__args__point__idx_shared_,
      facs__pinhole_fixed_pose__args__pixel__data_,
      pinhole_fixed_pose_num_max_,
      facs__pinhole_fixed_pose__args__pose__data_,
      pinhole_fixed_pose_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_num_);
  pinhole_fixed_focal_and_extra_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
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
  pinhole_fixed_principal_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
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
  pinhole_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
      facs__pinhole_fixed_point__args__pose__idx_shared_,
      nodes__PinholeFocalAndExtra__storage_check_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_point__args__focal_and_extra__idx_shared_,
      nodes__PinholePrincipalPoint__storage_check_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_fixed_point__args__pixel__data_,
      pinhole_fixed_point_num_max_,
      facs__pinhole_fixed_point__args__point__data_,
      pinhole_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_point_num_);
  pinhole_fixed_pose_fixed_focal_and_extra_score(
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
  pinhole_fixed_pose_fixed_principal_point_score(
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
  pinhole_fixed_pose_fixed_point_score(
      nodes__PinholeFocalAndExtra__storage_check_,
      PinholeFocalAndExtra_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__focal_and_extra__idx_shared_,
      nodes__PinholePrincipalPoint__storage_check_,
      PinholePrincipalPoint_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__principal_point__idx_shared_,
      facs__pinhole_fixed_pose_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      facs__pinhole_fixed_pose_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_point_num_max_,
      solver__res_tot_,
      pinhole_fixed_pose_fixed_point_num_);
  pinhole_fixed_focal_and_extra_fixed_principal_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
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
  pinhole_fixed_focal_and_extra_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
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
  pinhole_fixed_principal_point_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
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
  pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_score(
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
  pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_score(
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
  pinhole_fixed_pose_fixed_principal_point_fixed_point_score(
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
  pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_score(
      nodes__Pose__storage_check_,
      Pose_num_max_,
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
  return 0.5 * read_cumem(solver__res_tot_);
}

void GraphSolver::do_beta() {
  zero(solver__beta_numerator_, solver__beta_numerator_ + 1);

  PinholeFocalAndExtra_alpha_denumerator_or_beta_nummerator(
      nodes__PinholeFocalAndExtra__r_k_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__z_,
      PinholeFocalAndExtra_num_,
      solver__beta_numerator_,
      PinholeFocalAndExtra_num_);

  PinholePrincipalPoint_alpha_denumerator_or_beta_nummerator(
      nodes__PinholePrincipalPoint__r_k_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__z_,
      PinholePrincipalPoint_num_,
      solver__beta_numerator_,
      PinholePrincipalPoint_num_);

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

  SimpleRadialFocalAndExtra_alpha_denumerator_or_beta_nummerator(
      nodes__SimpleRadialFocalAndExtra__r_k_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__z_,
      SimpleRadialFocalAndExtra_num_,
      solver__beta_numerator_,
      SimpleRadialFocalAndExtra_num_);

  SimpleRadialPrincipalPoint_alpha_denumerator_or_beta_nummerator(
      nodes__SimpleRadialPrincipalPoint__r_k_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__z_,
      SimpleRadialPrincipalPoint_num_,
      solver__beta_numerator_,
      SimpleRadialPrincipalPoint_num_);
  beta_from_num_denum(
      solver__beta_numerator_, solver__alpha_numerator_, solver__beta_);
}

void GraphSolver::do_update_p() {
  PinholeFocalAndExtra_update_p(nodes__PinholeFocalAndExtra__z_,
                                PinholeFocalAndExtra_num_,
                                nodes__PinholeFocalAndExtra__p_,
                                PinholeFocalAndExtra_num_,
                                solver__beta_,
                                nodes__PinholeFocalAndExtra__p_,
                                PinholeFocalAndExtra_num_,
                                PinholeFocalAndExtra_num_);
  PinholePrincipalPoint_update_p(nodes__PinholePrincipalPoint__z_,
                                 PinholePrincipalPoint_num_,
                                 nodes__PinholePrincipalPoint__p_,
                                 PinholePrincipalPoint_num_,
                                 solver__beta_,
                                 nodes__PinholePrincipalPoint__p_,
                                 PinholePrincipalPoint_num_,
                                 PinholePrincipalPoint_num_);
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
  SimpleRadialFocalAndExtra_update_p(nodes__SimpleRadialFocalAndExtra__z_,
                                     SimpleRadialFocalAndExtra_num_,
                                     nodes__SimpleRadialFocalAndExtra__p_,
                                     SimpleRadialFocalAndExtra_num_,
                                     solver__beta_,
                                     nodes__SimpleRadialFocalAndExtra__p_,
                                     SimpleRadialFocalAndExtra_num_,
                                     SimpleRadialFocalAndExtra_num_);
  SimpleRadialPrincipalPoint_update_p(nodes__SimpleRadialPrincipalPoint__z_,
                                      SimpleRadialPrincipalPoint_num_,
                                      nodes__SimpleRadialPrincipalPoint__p_,
                                      SimpleRadialPrincipalPoint_num_,
                                      solver__beta_,
                                      nodes__SimpleRadialPrincipalPoint__p_,
                                      SimpleRadialPrincipalPoint_num_,
                                      SimpleRadialPrincipalPoint_num_);
}

double GraphSolver::get_pred_decrease() {
  zero(solver__pred_decrease_tot_, solver__pred_decrease_tot_ + 1);
  PinholeFocalAndExtra_pred_decrease_times_two(
      nodes__PinholeFocalAndExtra__step_,
      PinholeFocalAndExtra_num_,
      nodes__PinholeFocalAndExtra__precond_diag_,
      PinholeFocalAndExtra_num_,
      solver__current_diag_,
      nodes__PinholeFocalAndExtra__r_0_,
      PinholeFocalAndExtra_num_,
      solver__pred_decrease_tot_,
      PinholeFocalAndExtra_num_);
  PinholePrincipalPoint_pred_decrease_times_two(
      nodes__PinholePrincipalPoint__step_,
      PinholePrincipalPoint_num_,
      nodes__PinholePrincipalPoint__precond_diag_,
      PinholePrincipalPoint_num_,
      solver__current_diag_,
      nodes__PinholePrincipalPoint__r_0_,
      PinholePrincipalPoint_num_,
      solver__pred_decrease_tot_,
      PinholePrincipalPoint_num_);
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
  SimpleRadialFocalAndExtra_pred_decrease_times_two(
      nodes__SimpleRadialFocalAndExtra__step_,
      SimpleRadialFocalAndExtra_num_,
      nodes__SimpleRadialFocalAndExtra__precond_diag_,
      SimpleRadialFocalAndExtra_num_,
      solver__current_diag_,
      nodes__SimpleRadialFocalAndExtra__r_0_,
      SimpleRadialFocalAndExtra_num_,
      solver__pred_decrease_tot_,
      SimpleRadialFocalAndExtra_num_);
  SimpleRadialPrincipalPoint_pred_decrease_times_two(
      nodes__SimpleRadialPrincipalPoint__step_,
      SimpleRadialPrincipalPoint_num_,
      nodes__SimpleRadialPrincipalPoint__precond_diag_,
      SimpleRadialPrincipalPoint_num_,
      solver__current_diag_,
      nodes__SimpleRadialPrincipalPoint__r_0_,
      SimpleRadialPrincipalPoint_num_,
      solver__pred_decrease_tot_,
      SimpleRadialPrincipalPoint_num_);
  return 0.5 * read_cumem(solver__pred_decrease_tot_);
}

void GraphSolver::finish_indices() { indices_valid_ = true; }

void GraphSolver::set_PinholeFocalAndExtra_num(const size_t num) {
  if (num > PinholeFocalAndExtra_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > PinholeFocalAndExtra_num_max_");
  }
  PinholeFocalAndExtra_num_ = num;
}

void GraphSolver::set_PinholeFocalAndExtra_nodes_from_stacked_host(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholeFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocalAndExtra_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  PinholeFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      offset,
      num);
}

void GraphSolver::set_PinholeFocalAndExtra_nodes_from_stacked_device(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholeFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocalAndExtra_num_");
  }
  PinholeFocalAndExtra_stacked_to_caspar(
      data,
      nodes__PinholeFocalAndExtra__storage_current_,
      PinholeFocalAndExtra_num_max_,
      offset,
      num);
}

void GraphSolver::get_PinholeFocalAndExtra_nodes_to_stacked_host(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholeFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocalAndExtra_num_");
  }
  PinholeFocalAndExtra_caspar_to_stacked(
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

void GraphSolver::get_PinholeFocalAndExtra_nodes_to_stacked_device(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholeFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholeFocalAndExtra_num_");
  }
  PinholeFocalAndExtra_caspar_to_stacked(
      nodes__PinholeFocalAndExtra__storage_current_,
      data,
      PinholeFocalAndExtra_num_max_,
      offset,
      num);
}

void GraphSolver::set_PinholePrincipalPoint_num(const size_t num) {
  if (num > PinholePrincipalPoint_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > PinholePrincipalPoint_num_max_");
  }
  PinholePrincipalPoint_num_ = num;
}

void GraphSolver::set_PinholePrincipalPoint_nodes_from_stacked_host(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholePrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholePrincipalPoint_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  PinholePrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      offset,
      num);
}

void GraphSolver::set_PinholePrincipalPoint_nodes_from_stacked_device(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholePrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholePrincipalPoint_num_");
  }
  PinholePrincipalPoint_stacked_to_caspar(
      data,
      nodes__PinholePrincipalPoint__storage_current_,
      PinholePrincipalPoint_num_max_,
      offset,
      num);
}

void GraphSolver::get_PinholePrincipalPoint_nodes_to_stacked_host(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholePrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholePrincipalPoint_num_");
  }
  PinholePrincipalPoint_caspar_to_stacked(
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

void GraphSolver::get_PinholePrincipalPoint_nodes_to_stacked_device(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > PinholePrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > PinholePrincipalPoint_num_");
  }
  PinholePrincipalPoint_caspar_to_stacked(
      nodes__PinholePrincipalPoint__storage_current_,
      data,
      PinholePrincipalPoint_num_max_,
      offset,
      num);
}

void GraphSolver::set_Point_num(const size_t num) {
  if (num > Point_num_max_) {
    throw std::runtime_error(std::to_string(num) + " > Point_num_max_");
  }
  Point_num_ = num;
}

void GraphSolver::set_Point_nodes_from_stacked_host(const double* const data,
                                                    const size_t offset,
                                                    const size_t num) {
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  Point_stacked_to_caspar(marker__scratch_inout_,
                          nodes__Point__storage_current_,
                          Point_num_max_,
                          offset,
                          num);
}

void GraphSolver::set_Point_nodes_from_stacked_device(const double* const data,
                                                      const size_t offset,
                                                      const size_t num) {
  if (offset + num > Point_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Point_num_");
  }
  Point_stacked_to_caspar(
      data, nodes__Point__storage_current_, Point_num_max_, offset, num);
}

void GraphSolver::get_Point_nodes_to_stacked_host(double* const data,
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
             3 * num * sizeof(double),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::get_Point_nodes_to_stacked_device(double* const data,
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

void GraphSolver::set_Pose_nodes_from_stacked_host(const double* const data,
                                                   const size_t offset,
                                                   const size_t num) {
  if (offset + num > Pose_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Pose_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  Pose_stacked_to_caspar(marker__scratch_inout_,
                         nodes__Pose__storage_current_,
                         Pose_num_max_,
                         offset,
                         num);
}

void GraphSolver::set_Pose_nodes_from_stacked_device(const double* const data,
                                                     const size_t offset,
                                                     const size_t num) {
  if (offset + num > Pose_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Pose_num_");
  }
  Pose_stacked_to_caspar(
      data, nodes__Pose__storage_current_, Pose_num_max_, offset, num);
}

void GraphSolver::get_Pose_nodes_to_stacked_host(double* const data,
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
             7 * num * sizeof(double),
             cudaMemcpyDeviceToHost);
}

void GraphSolver::get_Pose_nodes_to_stacked_device(double* const data,
                                                   const size_t offset,
                                                   const size_t num) {
  if (offset + num > Pose_num_) {
    throw std::runtime_error(std::to_string(offset + num) + " > Pose_num_");
  }
  Pose_caspar_to_stacked(
      nodes__Pose__storage_current_, data, Pose_num_max_, offset, num);
}

void GraphSolver::set_SimpleRadialFocalAndExtra_num(const size_t num) {
  if (num > SimpleRadialFocalAndExtra_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > SimpleRadialFocalAndExtra_num_max_");
  }
  SimpleRadialFocalAndExtra_num_ = num;
}

void GraphSolver::set_SimpleRadialFocalAndExtra_nodes_from_stacked_host(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialFocalAndExtra_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  SimpleRadialFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      offset,
      num);
}

void GraphSolver::set_SimpleRadialFocalAndExtra_nodes_from_stacked_device(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialFocalAndExtra_num_");
  }
  SimpleRadialFocalAndExtra_stacked_to_caspar(
      data,
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      SimpleRadialFocalAndExtra_num_max_,
      offset,
      num);
}

void GraphSolver::get_SimpleRadialFocalAndExtra_nodes_to_stacked_host(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialFocalAndExtra_num_");
  }
  SimpleRadialFocalAndExtra_caspar_to_stacked(
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

void GraphSolver::get_SimpleRadialFocalAndExtra_nodes_to_stacked_device(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialFocalAndExtra_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialFocalAndExtra_num_");
  }
  SimpleRadialFocalAndExtra_caspar_to_stacked(
      nodes__SimpleRadialFocalAndExtra__storage_current_,
      data,
      SimpleRadialFocalAndExtra_num_max_,
      offset,
      num);
}

void GraphSolver::set_SimpleRadialPrincipalPoint_num(const size_t num) {
  if (num > SimpleRadialPrincipalPoint_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > SimpleRadialPrincipalPoint_num_max_");
  }
  SimpleRadialPrincipalPoint_num_ = num;
}

void GraphSolver::set_SimpleRadialPrincipalPoint_nodes_from_stacked_host(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialPrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialPrincipalPoint_num_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  SimpleRadialPrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      offset,
      num);
}

void GraphSolver::set_SimpleRadialPrincipalPoint_nodes_from_stacked_device(
    const double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialPrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialPrincipalPoint_num_");
  }
  SimpleRadialPrincipalPoint_stacked_to_caspar(
      data,
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      SimpleRadialPrincipalPoint_num_max_,
      offset,
      num);
}

void GraphSolver::get_SimpleRadialPrincipalPoint_nodes_to_stacked_host(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialPrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialPrincipalPoint_num_");
  }
  SimpleRadialPrincipalPoint_caspar_to_stacked(
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

void GraphSolver::get_SimpleRadialPrincipalPoint_nodes_to_stacked_device(
    double* const data, const size_t offset, const size_t num) {
  if (offset + num > SimpleRadialPrincipalPoint_num_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > SimpleRadialPrincipalPoint_num_");
  }
  SimpleRadialPrincipalPoint_caspar_to_stacked(
      nodes__SimpleRadialPrincipalPoint__storage_current_,
      data,
      SimpleRadialPrincipalPoint_num_max_,
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
void GraphSolver::set_simple_radial_focal_and_extra_indices_from_host(
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
  set_simple_radial_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_focal_and_extra_indices_from_device(
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
      indices, facs__simple_radial__args__focal_and_extra__idx_shared_, num);
}
void GraphSolver::set_simple_radial_principal_point_indices_from_host(
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
  set_simple_radial_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_simple_radial_principal_point_indices_from_device(
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
      indices, facs__simple_radial__args__principal_point__idx_shared_, num);
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
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(marker__scratch_inout_,
                               facs__simple_radial__args__pixel__data_,
                               simple_radial_num_max_,
                               offset,
                               num);
}

void GraphSolver::set_simple_radial_pixel_data_from_stacked_device(
    const double* const data, size_t offset, size_t num) {
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
void GraphSolver::
    set_simple_radial_fixed_pose_focal_and_extra_indices_from_host(
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
  set_simple_radial_fixed_pose_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_focal_and_extra_indices_from_device(
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
      indices,
      facs__simple_radial_fixed_pose__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_principal_point_indices_from_host(
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
  set_simple_radial_fixed_pose_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_principal_point_indices_from_device(
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
      indices,
      facs__simple_radial_fixed_pose__args__principal_point__idx_shared_,
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
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose__args__pixel__data_,
      simple_radial_fixed_pose_num_max_,
      offset,
      num);
}

void GraphSolver::set_simple_radial_fixed_pose_pixel_data_from_stacked_device(
    const double* const data, size_t offset, size_t num) {
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
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(marker__scratch_inout_,
                              facs__simple_radial_fixed_pose__args__pose__data_,
                              simple_radial_fixed_pose_num_max_,
                              offset,
                              num);
}

void GraphSolver::set_simple_radial_fixed_pose_pose_data_from_stacked_device(
    const double* const data, size_t offset, size_t num) {
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
void GraphSolver::set_simple_radial_fixed_focal_and_extra_num(
    const size_t num) {
  if (num > simple_radial_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > simple_radial_fixed_focal_and_extra_num_max_");
  }
  simple_radial_fixed_focal_and_extra_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_num_. Use "
        "set_simple_radial_fixed_focal_and_extra_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_and_extra_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_num_. Use "
        "set_simple_radial_fixed_focal_and_extra_num before setting indices.");
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
      facs__simple_radial_fixed_focal_and_extra__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_principal_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_num_. Use "
        "set_simple_radial_fixed_focal_and_extra_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_and_extra_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_principal_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_num_. Use "
        "set_simple_radial_fixed_focal_and_extra_num before setting indices.");
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
      facs__simple_radial_fixed_focal_and_extra__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_num_. Use "
        "set_simple_radial_fixed_focal_and_extra_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_and_extra_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_num_. Use "
        "set_simple_radial_fixed_focal_and_extra_num before setting indices.");
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
      facs__simple_radial_fixed_focal_and_extra__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_pixel_data_from_stacked_host(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_focal_and_extra_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_focal_and_extra_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_focal_and_extra_num_max_");
  }
  ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_principal_point_num(
    const size_t num) {
  if (num > simple_radial_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > simple_radial_fixed_principal_point_num_max_");
  }
  simple_radial_fixed_principal_point_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_principal_point_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_num_. Use "
        "set_simple_radial_fixed_principal_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_principal_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_principal_point_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_num_. Use "
        "set_simple_radial_fixed_principal_point_num before setting indices.");
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
      facs__simple_radial_fixed_principal_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_principal_point_focal_and_extra_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_num_. Use "
        "set_simple_radial_fixed_principal_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_principal_point_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_principal_point_focal_and_extra_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_num_. Use "
        "set_simple_radial_fixed_principal_point_num before setting indices.");
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
      facs__simple_radial_fixed_principal_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_principal_point_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_num_. Use "
        "set_simple_radial_fixed_principal_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_principal_point_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_principal_point_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_num_. Use "
        "set_simple_radial_fixed_principal_point_num before setting indices.");
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
      facs__simple_radial_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_principal_point_pixel_data_from_stacked_host(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_principal_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_principal_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_principal_point_principal_point_data_from_stacked_host(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_principal_point_principal_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_principal_point_num_max_,
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
void GraphSolver::
    set_simple_radial_fixed_point_focal_and_extra_indices_from_host(
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
  set_simple_radial_fixed_point_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_point_focal_and_extra_indices_from_device(
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
      facs__simple_radial_fixed_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_point_principal_point_indices_from_host(
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
  set_simple_radial_fixed_point_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_point_principal_point_indices_from_device(
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
      facs__simple_radial_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::set_simple_radial_fixed_point_pixel_data_from_stacked_host(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_point__args__pixel__data_,
      simple_radial_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::set_simple_radial_fixed_point_pixel_data_from_stacked_device(
    const double* const data, size_t offset, size_t num) {
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
    const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > simple_radial_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_point__args__point__data_,
      simple_radial_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::set_simple_radial_fixed_point_point_data_from_stacked_device(
    const double* const data, size_t offset, size_t num) {
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
void GraphSolver::set_simple_radial_fixed_pose_fixed_focal_and_extra_num(
    const size_t num) {
  if (num > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  simple_radial_fixed_pose_fixed_focal_and_extra_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_and_extra_num_. Use "
        "set_simple_radial_fixed_pose_fixed_focal_and_extra_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_and_extra_num_. Use "
        "set_simple_radial_fixed_pose_fixed_focal_and_extra_num before setting "
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
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_and_extra_num_. Use "
        "set_simple_radial_fixed_pose_fixed_focal_and_extra_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_focal_and_extra_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_and_extra_num_. Use "
        "set_simple_radial_fixed_pose_fixed_focal_and_extra_num before setting "
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
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_host(
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
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
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
  ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_pose_fixed_principal_point_num(
    const size_t num) {
  if (num > simple_radial_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_pose_fixed_principal_point_num_max_");
  }
  simple_radial_fixed_pose_fixed_principal_point_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_principal_point_num_. Use "
        "set_simple_radial_fixed_pose_fixed_principal_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_principal_point_num_. Use "
        "set_simple_radial_fixed_pose_fixed_principal_point_num before setting "
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
      facs__simple_radial_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_principal_point_num_. Use "
        "set_simple_radial_fixed_pose_fixed_principal_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_principal_point_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_principal_point_num_. Use "
        "set_simple_radial_fixed_pose_fixed_principal_point_num before setting "
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
      facs__simple_radial_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_pose_data_from_stacked_host(
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
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_pose_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_host(
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
  ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_principal_point_num_max_,
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
    set_simple_radial_fixed_pose_fixed_point_focal_and_extra_indices_from_host(
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
  set_simple_radial_fixed_pose_fixed_point_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_point_focal_and_extra_indices_from_device(
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
      facs__simple_radial_fixed_pose_fixed_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_point_principal_point_indices_from_host(
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
  set_simple_radial_fixed_pose_fixed_point_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_point_principal_point_indices_from_device(
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
      facs__simple_radial_fixed_pose_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_point_pixel_data_from_stacked_host(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
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
        const double* const data, size_t offset, size_t num) {
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
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
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
        const double* const data, size_t offset, size_t num) {
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
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
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
        const double* const data, size_t offset, size_t num) {
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
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_num(
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
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use set_simple_radial_fixed_focal_and_extra_fixed_principal_point_num "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use set_simple_radial_fixed_focal_and_extra_fixed_principal_point_num "
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
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use set_simple_radial_fixed_focal_and_extra_fixed_principal_point_num "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use set_simple_radial_fixed_focal_and_extra_fixed_principal_point_num "
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
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
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
  ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
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
  ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_focal_and_extra_fixed_point_num(
    const size_t num) {
  if (num > simple_radial_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_focal_and_extra_fixed_point_num_max_");
  }
  simple_radial_fixed_focal_and_extra_fixed_point_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_point_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_point_num_. Use "
        "set_simple_radial_fixed_focal_and_extra_fixed_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_and_extra_fixed_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_point_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_point_num_. Use "
        "set_simple_radial_fixed_focal_and_extra_fixed_point_num before "
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
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_point_num_. Use "
        "set_simple_radial_fixed_focal_and_extra_fixed_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_focal_and_extra_fixed_point_num_. Use "
        "set_simple_radial_fixed_focal_and_extra_fixed_point_num before "
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
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
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
  ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
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
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_point_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_simple_radial_fixed_principal_point_fixed_point_num(
    const size_t num) {
  if (num > simple_radial_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > simple_radial_fixed_principal_point_fixed_point_num_max_");
  }
  simple_radial_fixed_principal_point_fixed_point_num_ = num;
}
void GraphSolver::
    set_simple_radial_fixed_principal_point_fixed_point_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_fixed_point_num_. Use "
        "set_simple_radial_fixed_principal_point_fixed_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_principal_point_fixed_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_principal_point_fixed_point_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_fixed_point_num_. Use "
        "set_simple_radial_fixed_principal_point_fixed_point_num before "
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
      facs__simple_radial_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_fixed_point_num_. Use "
        "set_simple_radial_fixed_principal_point_fixed_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_principal_point_fixed_point_num_. Use "
        "set_simple_radial_fixed_principal_point_fixed_point_num before "
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
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
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
  ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_principal_point_fixed_point_num_max_");
  }
  ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_principal_point_fixed_point_point_data_from_stacked_host(
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
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > simple_radial_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > simple_radial_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num(
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
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num !=
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
        "num_. Use "
        "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
        "point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num !=
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
        "num_. Use "
        "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
        "point_num before setting indices.");
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
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_pose_fixed_focal_and_extra_"
                             "fixed_principal_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pose_data_from_stacked_host(
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
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pose_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_pose_fixed_focal_and_extra_"
                             "fixed_principal_point_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
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
  ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_pose_fixed_focal_and_extra_"
                             "fixed_principal_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
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
  ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_pose_fixed_focal_and_extra_"
                             "fixed_principal_point_num_max_");
  }
  ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num(
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
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_. "
        "Use "
        "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_. "
        "Use "
        "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num "
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
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_from_stacked_host(
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
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
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
  ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
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
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_num(
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
    set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_. "
        "Use "
        "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_num "
        "before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_. "
        "Use "
        "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_num "
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
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_host(
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
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
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
  ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_host(
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
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > "
        "simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num(
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
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num !=
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
        "num_. Use "
        "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
        "point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num !=
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
        "num_. Use "
        "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
        "point_num before setting indices.");
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
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_focal_and_extra_data_from_stacked_host(
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
  ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_focal_and_extra_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
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
  ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_point_data_from_stacked_host(
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
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "simple_radial_fixed_focal_and_extra_fixed_"
                             "principal_point_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
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
void GraphSolver::set_pinhole_focal_and_extra_indices_from_host(
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
  set_pinhole_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_focal_and_extra_indices_from_device(
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
  shared_indices(
      indices, facs__pinhole__args__focal_and_extra__idx_shared_, num);
}
void GraphSolver::set_pinhole_principal_point_indices_from_host(
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
  set_pinhole_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_principal_point_indices_from_device(
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
  shared_indices(
      indices, facs__pinhole__args__principal_point__idx_shared_, num);
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
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(marker__scratch_inout_,
                               facs__pinhole__args__pixel__data_,
                               pinhole_num_max_,
                               offset,
                               num);
}

void GraphSolver::set_pinhole_pixel_data_from_stacked_device(
    const double* const data, size_t offset, size_t num) {
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
void GraphSolver::set_pinhole_fixed_pose_focal_and_extra_indices_from_host(
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
  set_pinhole_fixed_pose_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_pose_focal_and_extra_indices_from_device(
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
  shared_indices(indices,
                 facs__pinhole_fixed_pose__args__focal_and_extra__idx_shared_,
                 num);
}
void GraphSolver::set_pinhole_fixed_pose_principal_point_indices_from_host(
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
  set_pinhole_fixed_pose_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_pose_principal_point_indices_from_device(
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
  shared_indices(indices,
                 facs__pinhole_fixed_pose__args__principal_point__idx_shared_,
                 num);
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
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(marker__scratch_inout_,
                               facs__pinhole_fixed_pose__args__pixel__data_,
                               pinhole_fixed_pose_num_max_,
                               offset,
                               num);
}

void GraphSolver::set_pinhole_fixed_pose_pixel_data_from_stacked_device(
    const double* const data, size_t offset, size_t num) {
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
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPose_stacked_to_caspar(marker__scratch_inout_,
                              facs__pinhole_fixed_pose__args__pose__data_,
                              pinhole_fixed_pose_num_max_,
                              offset,
                              num);
}

void GraphSolver::set_pinhole_fixed_pose_pose_data_from_stacked_device(
    const double* const data, size_t offset, size_t num) {
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
void GraphSolver::set_pinhole_fixed_focal_and_extra_num(const size_t num) {
  if (num > pinhole_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_fixed_focal_and_extra_num_max_");
  }
  pinhole_fixed_focal_and_extra_num_ = num;
}
void GraphSolver::set_pinhole_fixed_focal_and_extra_pose_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_num_. Use "
        "set_pinhole_fixed_focal_and_extra_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_and_extra_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_focal_and_extra_pose_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_num_. Use "
        "set_pinhole_fixed_focal_and_extra_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices,
                 facs__pinhole_fixed_focal_and_extra__args__pose__idx_shared_,
                 num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_principal_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_num_. Use "
        "set_pinhole_fixed_focal_and_extra_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_and_extra_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_principal_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_num_. Use "
        "set_pinhole_fixed_focal_and_extra_num before setting indices.");
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
      facs__pinhole_fixed_focal_and_extra__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::set_pinhole_fixed_focal_and_extra_point_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_num_. Use "
        "set_pinhole_fixed_focal_and_extra_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_and_extra_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_focal_and_extra_point_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_num_. Use "
        "set_pinhole_fixed_focal_and_extra_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices,
                 facs__pinhole_fixed_focal_and_extra__args__point__idx_shared_,
                 num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_pixel_data_from_stacked_host(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra__args__pixel__data_,
      pinhole_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_and_extra_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_and_extra__args__pixel__data_,
      pinhole_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_and_extra_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholeFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_focal_and_extra_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_focal_and_extra_num_max_");
  }
  ConstPinholeFocalAndExtra_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_and_extra__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_principal_point_num(const size_t num) {
  if (num > pinhole_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(num) +
                             " > pinhole_fixed_principal_point_num_max_");
  }
  pinhole_fixed_principal_point_num_ = num;
}
void GraphSolver::set_pinhole_fixed_principal_point_pose_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_num_. Use "
        "set_pinhole_fixed_principal_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_principal_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_principal_point_pose_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_num_. Use "
        "set_pinhole_fixed_principal_point_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices,
                 facs__pinhole_fixed_principal_point__args__pose__idx_shared_,
                 num);
}
void GraphSolver::
    set_pinhole_fixed_principal_point_focal_and_extra_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_num_. Use "
        "set_pinhole_fixed_principal_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_principal_point_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_principal_point_focal_and_extra_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_num_. Use "
        "set_pinhole_fixed_principal_point_num before setting indices.");
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
      facs__pinhole_fixed_principal_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::set_pinhole_fixed_principal_point_point_indices_from_host(
    const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_num_. Use "
        "set_pinhole_fixed_principal_point_num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_principal_point_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_principal_point_point_indices_from_device(
    const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_num_. Use "
        "set_pinhole_fixed_principal_point_num before setting indices.");
  }

  size_t tmp_size = sort_indices_get_tmp_nbytes(num);
  if (tmp_size + num > scratch_inout_size_) {
    throw std::runtime_error(
        "Scratch_inout_size too small. tmp_size: " + std::to_string(tmp_size) +
        ", num: " + std::to_string(num) +
        ", scratch_inout_size_: " + std::to_string(scratch_inout_size_));
  }
  shared_indices(indices,
                 facs__pinhole_fixed_principal_point__args__point__idx_shared_,
                 num);
}
void GraphSolver::
    set_pinhole_fixed_principal_point_pixel_data_from_stacked_host(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_principal_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_principal_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_principal_point_principal_point_data_from_stacked_host(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_principal_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPinholePrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_principal_point_principal_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_principal_point_num_max_");
  }
  ConstPinholePrincipalPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_principal_point_num_max_,
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
void GraphSolver::set_pinhole_fixed_point_focal_and_extra_indices_from_host(
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
  set_pinhole_fixed_point_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_point_focal_and_extra_indices_from_device(
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
  shared_indices(indices,
                 facs__pinhole_fixed_point__args__focal_and_extra__idx_shared_,
                 num);
}
void GraphSolver::set_pinhole_fixed_point_principal_point_indices_from_host(
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
  set_pinhole_fixed_point_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::set_pinhole_fixed_point_principal_point_indices_from_device(
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
  shared_indices(indices,
                 facs__pinhole_fixed_point__args__principal_point__idx_shared_,
                 num);
}
void GraphSolver::set_pinhole_fixed_point_pixel_data_from_stacked_host(
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPixel_stacked_to_caspar(marker__scratch_inout_,
                               facs__pinhole_fixed_point__args__pixel__data_,
                               pinhole_fixed_point_num_max_,
                               offset,
                               num);
}

void GraphSolver::set_pinhole_fixed_point_pixel_data_from_stacked_device(
    const double* const data, size_t offset, size_t num) {
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
    const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
             cudaMemcpyHostToDevice);
  ConstPoint_stacked_to_caspar(marker__scratch_inout_,
                               facs__pinhole_fixed_point__args__point__data_,
                               pinhole_fixed_point_num_max_,
                               offset,
                               num);
}

void GraphSolver::set_pinhole_fixed_point_point_data_from_stacked_device(
    const double* const data, size_t offset, size_t num) {
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
void GraphSolver::set_pinhole_fixed_pose_fixed_focal_and_extra_num(
    const size_t num) {
  if (num > pinhole_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  pinhole_fixed_pose_fixed_focal_and_extra_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_and_extra_num_. Use "
        "set_pinhole_fixed_pose_fixed_focal_and_extra_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_and_extra_num_. Use "
        "set_pinhole_fixed_pose_fixed_focal_and_extra_num before setting "
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
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_and_extra_num_. Use "
        "set_pinhole_fixed_pose_fixed_focal_and_extra_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_focal_and_extra_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_focal_and_extra_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_and_extra_num_. Use "
        "set_pinhole_fixed_pose_fixed_focal_and_extra_num before setting "
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
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_host(
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
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
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
  ConstPinholeFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_focal_and_extra_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_num_max_");
  }
  ConstPinholeFocalAndExtra_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_pose_fixed_principal_point_num(
    const size_t num) {
  if (num > pinhole_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_pose_fixed_principal_point_num_max_");
  }
  pinhole_fixed_pose_fixed_principal_point_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_principal_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_principal_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_principal_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_principal_point_num before setting "
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
      facs__pinhole_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_principal_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_principal_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_principal_point_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_principal_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_principal_point_num before setting "
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
      facs__pinhole_fixed_pose_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_pose_data_from_stacked_host(
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
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__pose__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_pose_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_principal_point__args__pose__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_host(
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
  ConstPinholePrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_num_max_");
  }
  ConstPinholePrincipalPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_principal_point_num_max_,
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
void GraphSolver::
    set_pinhole_fixed_pose_fixed_point_focal_and_extra_indices_from_host(
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
  set_pinhole_fixed_pose_fixed_point_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_point_focal_and_extra_indices_from_device(
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
      facs__pinhole_fixed_pose_fixed_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_point_principal_point_indices_from_host(
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
  set_pinhole_fixed_pose_fixed_point_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_point_principal_point_indices_from_device(
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
      facs__pinhole_fixed_pose_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_point_pixel_data_from_stacked_host(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             2 * num * sizeof(double),
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
        const double* const data, size_t offset, size_t num) {
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
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             7 * num * sizeof(double),
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
        const double* const data, size_t offset, size_t num) {
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
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_pose_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > pinhole_fixed_pose_fixed_point_num_max_");
  }
  cudaMemcpy(marker__scratch_inout_,
             data,
             3 * num * sizeof(double),
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
        const double* const data, size_t offset, size_t num) {
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
void GraphSolver::set_pinhole_fixed_focal_and_extra_fixed_principal_point_num(
    const size_t num) {
  if (num > pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  pinhole_fixed_focal_and_extra_fixed_principal_point_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_principal_point_num_. Use "
        "set_pinhole_fixed_focal_and_extra_fixed_principal_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_principal_point_num_. Use "
        "set_pinhole_fixed_focal_and_extra_fixed_principal_point_num before "
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
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_principal_point_num_. Use "
        "set_pinhole_fixed_focal_and_extra_fixed_principal_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_principal_point_num_. Use "
        "set_pinhole_fixed_focal_and_extra_fixed_principal_point_num before "
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
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
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
  ConstPinholeFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  ConstPinholeFocalAndExtra_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
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
  ConstPinholePrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_");
  }
  ConstPinholePrincipalPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_focal_and_extra_fixed_point_num(
    const size_t num) {
  if (num > pinhole_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_focal_and_extra_fixed_point_num_max_");
  }
  pinhole_fixed_focal_and_extra_fixed_point_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_point_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_point_num_. Use "
        "set_pinhole_fixed_focal_and_extra_fixed_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_and_extra_fixed_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_point_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_point_num_. Use "
        "set_pinhole_fixed_focal_and_extra_fixed_point_num before setting "
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
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_point_num_. Use "
        "set_pinhole_fixed_focal_and_extra_fixed_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_focal_and_extra_fixed_point_num_. Use "
        "set_pinhole_fixed_focal_and_extra_fixed_point_num before setting "
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
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
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
  ConstPinholeFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPinholeFocalAndExtra_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
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
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__point__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_point_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__point__data_,
      pinhole_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_principal_point_fixed_point_num(
    const size_t num) {
  if (num > pinhole_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_principal_point_fixed_point_num_max_");
  }
  pinhole_fixed_principal_point_fixed_point_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_principal_point_fixed_point_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_fixed_point_num_. Use "
        "set_pinhole_fixed_principal_point_fixed_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_principal_point_fixed_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_principal_point_fixed_point_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_fixed_point_num_. Use "
        "set_pinhole_fixed_principal_point_fixed_point_num before setting "
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
      facs__pinhole_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_fixed_point_num_. Use "
        "set_pinhole_fixed_principal_point_fixed_point_num before setting "
        "indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_principal_point_fixed_point_num_. Use "
        "set_pinhole_fixed_principal_point_fixed_point_num before setting "
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
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
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
  ConstPinholePrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPinholePrincipalPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_principal_point_fixed_point_point_data_from_stacked_host(
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
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num > pinhole_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num(
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
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num !=
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use "
        "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
        "num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num !=
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_. "
        "Use "
        "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
        "num before setting indices.");
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
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_pose_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pose_data_from_stacked_host(
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
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pose_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_pose_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
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
  ConstPinholeFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_pose_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  ConstPinholeFocalAndExtra_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
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
  ConstPinholePrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_pose_fixed_focal_and_extra_fixed_"
                             "principal_point_num_max_");
  }
  ConstPinholePrincipalPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num(
    const size_t num) {
  if (num > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num before "
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
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_from_stacked_host(
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
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
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
  ConstPinholeFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPinholeFocalAndExtra_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
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
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::set_pinhole_fixed_pose_fixed_principal_point_fixed_point_num(
    const size_t num) {
  if (num > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(num) +
        " > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  pinhole_fixed_pose_fixed_principal_point_fixed_point_num_ = num;
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num != pinhole_fixed_pose_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_principal_point_fixed_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_num before "
        "setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num != pinhole_fixed_pose_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != pinhole_fixed_pose_fixed_principal_point_fixed_point_num_. Use "
        "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_num before "
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
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_host(
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
  ConstPose_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPose_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
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
  ConstPinholePrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPinholePrincipalPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_host(
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
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(
        std::to_string(offset + num) +
        " > pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num(
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
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_host(
        const unsigned int* const indices, size_t num) {
  if (num !=
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_. "
        "Use "
        "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
        "num before setting indices.");
  }
  cudaMemcpy((unsigned int*)marker__scratch_inout_,
             indices,
             num * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_device(
      (unsigned int*)marker__scratch_inout_, num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_device(
        const unsigned int* const indices, size_t num) {
  indices_valid_ = false;

  if (num !=
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_) {
    throw std::runtime_error(
        std::to_string(num) +
        " != "
        "pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_. "
        "Use "
        "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
        "num before setting indices.");
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
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
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
  ConstPixel_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  ConstPixel_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_focal_and_extra_data_from_stacked_host(
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
  ConstPinholeFocalAndExtra_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_focal_and_extra_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  ConstPinholeFocalAndExtra_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
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
  ConstPinholePrincipalPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  ConstPinholePrincipalPoint_stacked_to_caspar(
      data,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}
void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_point_data_from_stacked_host(
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
  ConstPoint_stacked_to_caspar(
      marker__scratch_inout_,
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_,
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_,
      offset,
      num);
}

void GraphSolver::
    set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        const double* const data, size_t offset, size_t num) {
  if (offset + num >
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_) {
    throw std::runtime_error(std::to_string(offset + num) +
                             " > "
                             "pinhole_fixed_focal_and_extra_fixed_principal_"
                             "point_fixed_point_num_max_");
  }
  ConstPoint_stacked_to_caspar(
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
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 8 * Pose_num_, 4);
  increment_offset<double>(offset, 8 * Pose_num_, 4);
  increment_offset<double>(offset, 8 * Pose_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_num_, 4);
  increment_offset<double>(offset, 2 * simple_radial_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_pose_num_, 4);
  increment_offset<double>(offset, 2 * simple_radial_fixed_pose_num_, 4);
  increment_offset<double>(offset, 8 * simple_radial_fixed_pose_num_, 4);
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
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_point_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_point_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * simple_radial_fixed_point_num_, 4);
  increment_offset<double>(offset, 2 * simple_radial_fixed_point_num_, 4);
  increment_offset<double>(offset, 4 * simple_radial_fixed_point_num_, 4);
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
      offset, 1 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 8 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 4 * simple_radial_fixed_pose_fixed_point_num_, 4);
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
  increment_offset<SharedIndex>(offset, 1 * pinhole_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_pose_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_pose_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_pose_num_, 4);
  increment_offset<double>(offset, 8 * pinhole_fixed_pose_num_, 4);
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
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_point_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_point_num_, 4);
  increment_offset<SharedIndex>(offset, 1 * pinhole_fixed_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_point_num_, 4);
  increment_offset<double>(offset, 4 * pinhole_fixed_point_num_, 4);
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
      offset, 1 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<SharedIndex>(
      offset, 1 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(offset, 8 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(offset, 4 * pinhole_fixed_pose_fixed_point_num_, 4);
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
               offset + std::max({2 * PinholeFocalAndExtra_num_max_,
                                  2 * PinholePrincipalPoint_num_max_,
                                  3 * Point_num_max_,
                                  7 * Pose_num_max_,
                                  2 * SimpleRadialFocalAndExtra_num_max_,
                                  2 * SimpleRadialPrincipalPoint_num_max_}) *
                            sizeof(double));
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * simple_radial_num_, 4);
  increment_offset<double>(offset, 2 * simple_radial_fixed_pose_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_principal_point_num_, 4);
  increment_offset<double>(offset, 2 * simple_radial_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_point_num_, 4);
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
  increment_offset<double>(offset, 2 * pinhole_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_pose_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
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
  increment_offset<double>(offset, 12 * simple_radial_num_, 4);
  increment_offset<double>(offset, 4 * simple_radial_num_, 4);
  increment_offset<double>(offset, 0 * simple_radial_num_, 4);
  increment_offset<double>(offset, 6 * simple_radial_num_, 4);
  increment_offset<double>(offset, 4 * simple_radial_fixed_pose_num_, 4);
  increment_offset<double>(offset, 0 * simple_radial_fixed_pose_num_, 4);
  increment_offset<double>(offset, 6 * simple_radial_fixed_pose_num_, 4);
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
  increment_offset<double>(offset, 12 * simple_radial_fixed_point_num_, 4);
  increment_offset<double>(offset, 4 * simple_radial_fixed_point_num_, 4);
  increment_offset<double>(offset, 0 * simple_radial_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 0 * simple_radial_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 6 * simple_radial_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 4 * simple_radial_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 6 * simple_radial_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 4 * simple_radial_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 0 * simple_radial_fixed_pose_fixed_point_num_, 4);
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
  increment_offset<double>(offset, 10 * pinhole_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_num_, 4);
  increment_offset<double>(offset, 0 * pinhole_num_, 4);
  increment_offset<double>(offset, 6 * pinhole_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_pose_num_, 4);
  increment_offset<double>(offset, 0 * pinhole_fixed_pose_num_, 4);
  increment_offset<double>(offset, 6 * pinhole_fixed_pose_num_, 4);
  increment_offset<double>(offset, 10 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(offset, 0 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(offset, 6 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(offset, 10 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<double>(offset, 6 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<double>(offset, 10 * pinhole_fixed_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_point_num_, 4);
  increment_offset<double>(offset, 0 * pinhole_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 0 * pinhole_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 6 * pinhole_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 6 * pinhole_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
  increment_offset<double>(offset, 0 * pinhole_fixed_pose_fixed_point_num_, 4);
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
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 6 * Pose_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 6 * Pose_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 6 * Pose_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 6 * Pose_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 1);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 6 * Pose_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 6 * Pose_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 6 * Pose_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 1 * PinholeFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 1 * PinholePrincipalPoint_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 4 * Point_num_, 4);
  increment_offset<double>(offset, 6 * Pose_num_, 4);
  increment_offset<double>(offset, 16 * Pose_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 1 * SimpleRadialFocalAndExtra_num_, 4);
  increment_offset<double>(offset, 2 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 1 * SimpleRadialPrincipalPoint_num_, 4);
  increment_offset<double>(offset, 0 * 0, 1);
  increment_offset<double>(offset, 0 * 0, 4);
  increment_offset<double>(offset, 2 * simple_radial_num_, 4);
  increment_offset<double>(offset, 2 * simple_radial_fixed_pose_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_principal_point_num_, 4);
  increment_offset<double>(offset, 2 * simple_radial_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(
      offset, 2 * simple_radial_fixed_pose_fixed_point_num_, 4);
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
  increment_offset<double>(offset, 2 * pinhole_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_pose_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_principal_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_point_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_focal_and_extra_num_, 4);
  increment_offset<double>(
      offset, 2 * pinhole_fixed_pose_fixed_principal_point_num_, 4);
  increment_offset<double>(offset, 2 * pinhole_fixed_pose_fixed_point_num_, 4);
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