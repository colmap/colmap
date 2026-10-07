#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_fixed_pose_fixed_point_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialFixedPoseFixedPointResJacKernel(
        double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        double *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
        double *pixel, unsigned int pixel_num_alloc, double *pose,
        unsigned int pose_num_alloc, double *point,
        unsigned int point_num_alloc, double *out_res,
        unsigned int out_res_num_alloc, double *const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        double *const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        double *const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0,
         r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0,
         r17 = 0, r18 = 0, r19 = 0, r20 = 0, r21 = 0, r22 = 0, r23 = 0, r24 = 0,
         r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0, r31 = 0, r32 = 0,
         r33 = 0, r34 = 0, r35 = 0, r36 = 0, r37 = 0, r38 = 0, r39 = 0, r40 = 0,
         r41 = 0, r42 = 0, r43 = 0, r44 = 0, r45 = 0, r46 = 0, r47 = 0, r48 = 0;
  LoadShared<2, double, double>(calib, 2 * calib_num_alloc, calib_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        calib_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(pixel, 0 * pixel_num_alloc,
                                            global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r5 = fma(r2, r4, r0);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r6, r7);
    ReadIdx2<1024, double, double, double2>(point, 0 * point_num_alloc,
                                            global_thread_idx, r8, r9);
    r10 = -2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r11, r12);
    ReadIdx2<1024, double, double, double2>(pose, 2 * pose_num_alloc,
                                            global_thread_idx, r13, r14);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r15, r16);
    ReadIdx2<1024, double, double, double2>(pose, 0 * pose_num_alloc,
                                            global_thread_idx, r17, r18);
    r19 = fma(r15, r18, r12 * r13);
    r20 = r16 * r17;
    r19 = fma(r4, r20, r19);
    r19 = fma(r11, r14, r19);
    r20 = r19 * r19;
    r20 = r10 * r20;
    r21 = 1.00000000000000000e+00;
    r22 = r15 * r13;
    r22 = fma(r4, r22, r12 * r18);
    r22 = fma(r16, r14, r22);
    r22 = fma(r11, r17, r22);
    r23 = r22 * r22;
    r23 = fma(r10, r23, r21);
    r24 = r20 + r23;
    r24 = fma(r8, r24, r6);
    r25 = 2.00000000000000000e+00;
    r26 = fma(r15, r14, r12 * r17);
    r27 = r11 * r18;
    r26 = fma(r4, r27, r26);
    r26 = fma(r16, r13, r26);
    r27 = r25 * r26;
    r28 = r22 * r27;
    r29 = fma(r16, r18, r15 * r17);
    r29 = fma(r11, r13, r29);
    r29 = fma(r4, r29, r12 * r14);
    r30 = r10 * r29;
    r31 = fma(r19, r30, r28);
    ReadIdx1<1024, double, double, double>(point, 2 * point_num_alloc,
                                           global_thread_idx, r32);
    r33 = r25 * r22;
    r34 = r19 * r27;
    r33 = fma(r29, r33, r34);
    ReadIdx1<1024, double, double, double>(pose, 6 * pose_num_alloc,
                                           global_thread_idx, r35);
    r36 = r15 * r11;
    r36 = r36 * r25;
    r37 = r16 * r12;
    r38 = fma(r25, r37, r36);
    ReadIdx2<1024, double, double, double2>(pose, 4 * pose_num_alloc,
                                            global_thread_idx, r39, r40);
    r41 = r11 * r12;
    r42 = r15 * r16;
    r42 = r42 * r25;
    r41 = fma(r10, r41, r42);
    r43 = r16 * r16;
    r43 = r43 * r10;
    r44 = r21 + r43;
    r45 = r11 * r11;
    r45 = r10 * r45;
    r44 = r44 + r45;
    r24 = fma(r9, r31, r24);
    r24 = fma(r32, r33, r24);
    r24 = fma(r35, r38, r24);
    r24 = fma(r40, r41, r24);
    r24 = fma(r39, r44, r24);
  };
  LoadShared<2, double, double>(calib, 0 * calib_num_alloc, calib_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        calib_indices_loc[threadIdx.x].target, r44, r41);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r33);
    r34 = fma(r22, r30, r34);
    r34 = fma(r8, r34, r33);
    r37 = fma(r10, r37, r36);
    r43 = r21 + r43;
    r36 = r15 * r15;
    r36 = r10 * r36;
    r43 = r43 + r36;
    r31 = r16 * r11;
    r31 = r31 * r25;
    r46 = r15 * r12;
    r46 = fma(r25, r46, r31);
    r47 = r25 * r19;
    r47 = r47 * r22;
    r27 = fma(r29, r27, r47);
    r48 = r26 * r26;
    r48 = r48 * r10;
    r23 = r48 + r23;
    r34 = fma(r39, r37, r34);
    r34 = fma(r35, r43, r34);
    r34 = fma(r40, r46, r34);
    r34 = fma(r9, r27, r34);
    r34 = fma(r32, r23, r34);
    r23 = copysign(1.0, r34);
    r23 = fma(r38, r23, r34);
    r38 = r23 * r23;
    r38 = 1.0 / r38;
    r34 = r24 * r24;
    r27 = r25 * r19;
    r27 = fma(r29, r27, r28);
    r27 = fma(r8, r27, r7);
    r28 = r11 * r12;
    r28 = fma(r25, r28, r42);
    r45 = r21 + r45;
    r45 = r45 + r36;
    r36 = r15 * r12;
    r36 = fma(r10, r36, r31);
    r30 = fma(r26, r30, r47);
    r20 = r21 + r20;
    r20 = r20 + r48;
    r27 = fma(r39, r28, r27);
    r27 = fma(r40, r45, r27);
    r27 = fma(r35, r36, r27);
    r27 = fma(r32, r30, r27);
    r27 = fma(r9, r20, r27);
    r20 = r27 * r27;
    r20 = fma(r38, r20, r38 * r34);
    r20 = fma(r41, r20, r21);
    r20 = r44 * r20;
    r23 = 1.0 / r23;
    r20 = r20 * r23;
    r5 = fma(r24, r20, r5);
    r4 = fma(r3, r4, r1);
    r4 = fma(r27, r20, r4);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc,
                                             global_thread_idx, r5, r4);
    r4 = 1.00000000000000000e+00;
    r5 = 1.00000000000000008e-15;
    r20 = -2.00000000000000000e+00;
    r27 = r15 * r13;
    r24 = -1.00000000000000000e+00;
    r27 = fma(r24, r27, r12 * r18);
    r27 = fma(r16, r14, r27);
    r27 = fma(r11, r17, r27);
    r23 = r20 * r27;
    r21 = fma(r16, r18, r15 * r17);
    r21 = fma(r11, r13, r21);
    r21 = fma(r24, r21, r12 * r14);
    r38 = fma(r15, r18, r12 * r13);
    r34 = r16 * r17;
    r38 = fma(r24, r34, r38);
    r38 = fma(r11, r14, r38);
    r34 = 2.00000000000000000e+00;
    r14 = fma(r15, r14, r12 * r17);
    r30 = r11 * r18;
    r14 = fma(r24, r30, r14);
    r14 = fma(r16, r13, r14);
    r30 = r34 * r14;
    r36 = r38 * r30;
    r23 = fma(r21, r23, r36);
    r23 = fma(r8, r23, r33);
    r33 = r15 * r11;
    r33 = r33 * r34;
    r45 = r12 * r20;
    r28 = fma(r16, r45, r33);
    r48 = r16 * r16;
    r48 = r20 * r48;
    r26 = r15 * r15;
    r26 = fma(r20, r26, r4);
    r47 = r48 + r26;
    r31 = r16 * r11;
    r31 = r31 * r34;
    r10 = r15 * r12;
    r10 = fma(r34, r10, r31);
    r42 = r34 * r38;
    r42 = r42 * r27;
    r29 = fma(r21, r30, r42);
    r46 = r14 * r14;
    r46 = r46 * r20;
    r43 = r4 + r46;
    r37 = r27 * r27;
    r37 = r20 * r37;
    r43 = r43 + r37;
    r23 = fma(r39, r28, r23);
    r23 = fma(r35, r47, r23);
    r23 = fma(r40, r10, r23);
    r23 = fma(r9, r29, r23);
    r23 = fma(r32, r43, r23);
    r43 = copysign(1.0, r23);
    r43 = fma(r5, r43, r23);
    r5 = r43 * r43;
    r5 = 1.0 / r5;
    r23 = r38 * r38;
    r23 = r23 * r20;
    r29 = r4 + r23;
    r29 = r29 + r37;
    r29 = fma(r8, r29, r6);
    r30 = r27 * r30;
    r38 = r38 * r21;
    r6 = fma(r20, r38, r30);
    r37 = r34 * r27;
    r37 = fma(r21, r37, r36);
    r36 = r16 * r12;
    r36 = fma(r34, r36, r33);
    r33 = r15 * r16;
    r33 = r33 * r34;
    r10 = fma(r11, r45, r33);
    r47 = r11 * r11;
    r47 = r20 * r47;
    r28 = r4 + r47;
    r28 = r28 + r48;
    r29 = fma(r9, r6, r29);
    r29 = fma(r32, r37, r29);
    r29 = fma(r35, r36, r29);
    r29 = fma(r40, r10, r29);
    r29 = fma(r39, r28, r29);
    r28 = r29 * r29;
    r38 = fma(r34, r38, r30);
    r38 = fma(r8, r38, r7);
    r8 = r11 * r12;
    r8 = fma(r34, r8, r33);
    r26 = r47 + r26;
    r45 = fma(r15, r45, r31);
    r31 = r14 * r20;
    r31 = fma(r21, r31, r42);
    r46 = r4 + r46;
    r46 = r46 + r23;
    r38 = fma(r39, r8, r38);
    r38 = fma(r40, r26, r38);
    r38 = fma(r35, r45, r38);
    r38 = fma(r32, r31, r38);
    r38 = fma(r9, r46, r38);
    r46 = r38 * r38;
    r9 = fma(r5, r46, r5 * r28);
    r41 = fma(r41, r9, r4);
    r43 = 1.0 / r43;
    r31 = r41 * r43;
    r3 = fma(r3, r24, r1);
    r1 = r44 * r38;
    r3 = fma(r31, r1, r3);
    r3 = r24 * r3;
    r1 = r38 * r3;
    r32 = r24 * r29;
    r2 = fma(r2, r24, r0);
    r0 = r44 * r29;
    r2 = fma(r31, r0, r2);
    r32 = r32 * r2;
    r32 = fma(r31, r32, r31 * r1);
    r0 = r24 * r29;
    r45 = r44 * r9;
    r0 = r0 * r2;
    r0 = r0 * r43;
    r35 = r43 * r45;
    r35 = fma(r1, r35, r45 * r0);
    WriteSum2<double, double>((double *)inout_shared, r32, r35);
  };
  FlushSumShared<2, double>(out_calib_njtr, 0 * out_calib_njtr_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r24 * r2;
    WriteSum2<double, double>((double *)inout_shared, r2, r3);
  };
  FlushSumShared<2, double>(out_calib_njtr, 2 * out_calib_njtr_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r41 * r41;
    r3 = r3 * r5;
    r3 = fma(r46, r3, r28 * r3);
    r9 = r44 * r9;
    r5 = r5 * r45;
    r9 = r9 * r5;
    r9 = fma(r28, r9, r46 * r9);
    WriteSum2<double, double>((double *)inout_shared, r3, r9);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double *)inout_shared, r4, r4);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = r29 * r31;
    r9 = r41 * r28;
    r3 = r41 * r46;
    r3 = fma(r5, r3, r5 * r9);
    WriteSum2<double, double>((double *)inout_shared, r3, r4);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r38 * r31;
    r4 = r29 * r43;
    r4 = r4 * r45;
    WriteSum2<double, double>((double *)inout_shared, r31, r4);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = 0.00000000000000000e+00;
    r31 = r38 * r43;
    r31 = r31 * r45;
    WriteSum2<double, double>((double *)inout_shared, r31, r4);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
}

void SimpleRadialFixedPoseFixedPointResJac(
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    double *pixel, unsigned int pixel_num_alloc, double *pose,
    unsigned int pose_num_alloc, double *point, unsigned int point_num_alloc,
    double *out_res, unsigned int out_res_num_alloc,
    double *const out_calib_njtr, unsigned int out_calib_njtr_num_alloc,
    double *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFixedPoseFixedPointResJacKernel<<<n_blocks, 1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, calib, calib_num_alloc,
      calib_indices, pixel, pixel_num_alloc, pose, pose_num_alloc, point,
      point_num_alloc, out_res, out_res_num_alloc, out_calib_njtr,
      out_calib_njtr_num_alloc, out_calib_precond_diag,
      out_calib_precond_diag_num_alloc, out_calib_precond_tril,
      out_calib_precond_tril_num_alloc, problem_size);
}

} // namespace caspar