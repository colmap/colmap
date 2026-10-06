#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_split_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedPoseFixedFocalAndExtraFixedPointResJacKernel(
        float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        float *principal_point, unsigned int principal_point_num_alloc,
        SharedIndex *principal_point_indices, float *pixel,
        unsigned int pixel_num_alloc, float *pose, unsigned int pose_num_alloc,
        float *focal_and_extra, unsigned int focal_and_extra_num_alloc,
        float *point, unsigned int point_num_alloc, float *out_res,
        unsigned int out_res_num_alloc, float *const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float *const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float *const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48;
  LoadShared<2, float, float>(principal_point, 0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         0 * focal_and_extra_num_alloc,
                                         global_thread_idx, r2, r3, r4, r5);
    ReadIdx2<1024, float, float, float2>(focal_and_extra,
                                         4 * focal_and_extra_num_alloc,
                                         global_thread_idx, r6, r7);
    r8 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r9, r10, r11);
    ReadIdx3<1024, float, float, float4>(point, 0 * point_num_alloc,
                                         global_thread_idx, r12, r13, r14);
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r15, r16, r17, r18);
    ReadIdx4<1024, float, float, float4>(pose, 0 * pose_num_alloc,
                                         global_thread_idx, r19, r20, r21, r22);
    r23 = fmaf(r15, r20, r18 * r21);
    r24 = r16 * r19;
    r25 = -1.00000000000000000e+00;
    r23 = fmaf(r25, r24, r23);
    r23 = fmaf(r17, r22, r23);
    r24 = 2.00000000000000000e+00;
    r26 = fmaf(r15, r22, r18 * r19);
    r27 = r17 * r20;
    r26 = fmaf(r25, r27, r26);
    r26 = fmaf(r16, r21, r26);
    r27 = r24 * r26;
    r28 = r23 * r27;
    r29 = r15 * r21;
    r29 = fmaf(r25, r29, r18 * r20);
    r29 = fmaf(r16, r22, r29);
    r29 = fmaf(r17, r19, r29);
    r30 = -2.00000000000000000e+00;
    r31 = fmaf(r16, r20, r15 * r19);
    r31 = fmaf(r17, r21, r31);
    r31 = fmaf(r25, r31, r18 * r22);
    r22 = r30 * r31;
    r32 = fmaf(r29, r22, r28);
    r32 = fmaf(r12, r32, r11);
    ReadIdx3<1024, float, float, float4>(pose, 4 * pose_num_alloc,
                                         global_thread_idx, r11, r33, r34);
    r35 = r15 * r17;
    r35 = r35 * r24;
    r36 = r16 * r18;
    r37 = fmaf(r30, r36, r35);
    r38 = r15 * r15;
    r38 = r30 * r38;
    r39 = 1.00000000000000000e+00;
    r40 = r16 * r16;
    r40 = fmaf(r30, r40, r39);
    r41 = r38 + r40;
    r42 = r16 * r17;
    r42 = r42 * r24;
    r43 = r15 * r18;
    r43 = fmaf(r24, r43, r42);
    r44 = r24 * r23;
    r44 = r44 * r29;
    r45 = fmaf(r31, r27, r44);
    r46 = r29 * r29;
    r46 = r30 * r46;
    r47 = r39 + r46;
    r48 = r26 * r26;
    r48 = r48 * r30;
    r47 = r47 + r48;
    r32 = fmaf(r11, r37, r32);
    r32 = fmaf(r34, r41, r32);
    r32 = fmaf(r33, r43, r32);
    r32 = fmaf(r13, r45, r32);
    r32 = fmaf(r14, r47, r32);
    r47 = copysign(1.0, r32);
    r47 = fmaf(r8, r47, r32);
    r8 = r47 * r47;
    r8 = 1.0 / r8;
    r32 = r24 * r23;
    r27 = r29 * r27;
    r32 = fmaf(r31, r32, r27);
    r32 = fmaf(r12, r32, r10);
    r10 = r15 * r16;
    r10 = r10 * r24;
    r45 = r17 * r18;
    r45 = fmaf(r24, r45, r10);
    r43 = r17 * r17;
    r43 = r30 * r43;
    r41 = r39 + r43;
    r41 = r41 + r38;
    r38 = r15 * r18;
    r38 = fmaf(r30, r38, r42);
    r26 = fmaf(r26, r22, r44);
    r44 = r23 * r23;
    r44 = r30 * r44;
    r42 = r39 + r44;
    r42 = r42 + r48;
    r32 = fmaf(r11, r45, r32);
    r32 = fmaf(r33, r41, r32);
    r32 = fmaf(r34, r38, r32);
    r32 = fmaf(r14, r26, r32);
    r32 = fmaf(r13, r42, r32);
    r42 = r32 * r32;
    r42 = r8 * r42;
    r26 = 3.00000000000000000e+00;
    r46 = r39 + r46;
    r46 = r46 + r44;
    r46 = fmaf(r12, r46, r9);
    r22 = fmaf(r23, r22, r27);
    r27 = r24 * r29;
    r27 = fmaf(r31, r27, r28);
    r36 = fmaf(r24, r36, r35);
    r35 = r17 * r18;
    r35 = fmaf(r30, r35, r10);
    r40 = r43 + r40;
    r46 = fmaf(r13, r22, r46);
    r46 = fmaf(r14, r27, r46);
    r46 = fmaf(r34, r36, r46);
    r46 = fmaf(r33, r35, r46);
    r46 = fmaf(r11, r40, r46);
    r40 = r46 * r46;
    r40 = r8 * r40;
    r11 = fmaf(r26, r40, r42);
    r47 = 1.0 / r47;
    r11 = fmaf(r46, r47, r7 * r11);
    r35 = r40 + r42;
    r33 = r35 * r35;
    r33 = fmaf(r5, r33, r4 * r35);
    r33 = r33 * r47;
    r5 = r6 * r24;
    r8 = r46 * r8;
    r5 = r5 * r32;
    r11 = fmaf(r8, r5, r11);
    r11 = fmaf(r46, r33, r11);
    r11 = fmaf(r2, r11, r0);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r2, r0);
    r11 = fmaf(r2, r25, r11);
    r42 = fmaf(r26, r42, r40);
    r47 = fmaf(r32, r47, r6 * r42);
    r42 = r7 * r24;
    r42 = r42 * r32;
    r47 = fmaf(r8, r42, r47);
    r47 = fmaf(r32, r33, r47);
    r47 = fmaf(r3, r47, r1);
    r47 = fmaf(r0, r25, r47);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r11, r47);
    r11 = r25 * r11;
    r47 = r25 * r47;
    WriteSum2<float, float>((float *)inout_shared, r11, r47);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float *)inout_shared, r39, r39);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc, (float *)inout_shared);
}

void OpencvSplitFixedPoseFixedFocalAndExtraFixedPointResJac(
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *principal_point, unsigned int principal_point_num_alloc,
    SharedIndex *principal_point_indices, float *pixel,
    unsigned int pixel_num_alloc, float *pose, unsigned int pose_num_alloc,
    float *focal_and_extra, unsigned int focal_and_extra_num_alloc,
    float *point, unsigned int point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float *const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float *const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedPoseFixedFocalAndExtraFixedPointResJacKernel<<<n_blocks,
                                                                 1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, principal_point,
      principal_point_num_alloc, principal_point_indices, pixel,
      pixel_num_alloc, pose, pose_num_alloc, focal_and_extra,
      focal_and_extra_num_alloc, point, point_num_alloc, out_res,
      out_res_num_alloc, out_principal_point_njtr,
      out_principal_point_njtr_num_alloc, out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar