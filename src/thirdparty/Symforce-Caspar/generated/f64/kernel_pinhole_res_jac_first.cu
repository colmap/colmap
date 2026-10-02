#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_pinhole_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) PinholeResJacFirstKernel(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    double *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    double *pixel, unsigned int pixel_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr, double *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, double *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, double *out_calib_jac,
    unsigned int out_calib_jac_num_alloc, double *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, double *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, double *out_point_jac,
    unsigned int out_point_jac_num_alloc, double *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0,
         r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0,
         r17 = 0, r18 = 0, r19 = 0, r20 = 0, r21 = 0, r22 = 0, r23 = 0, r24 = 0,
         r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0, r31 = 0, r32 = 0,
         r33 = 0, r34 = 0, r35 = 0, r36 = 0, r37 = 0, r38 = 0, r39 = 0, r40 = 0,
         r41 = 0, r42 = 0, r43 = 0, r44 = 0, r45 = 0, r46 = 0, r47 = 0, r48 = 0,
         r49 = 0, r50 = 0, r51 = 0, r52 = 0, r53 = 0, r54 = 0, r55 = 0, r56 = 0,
         r57 = 0, r58 = 0, r59 = 0, r60 = 0, r61 = 0, r62 = 0, r63 = 0, r64 = 0,
         r65 = 0, r66 = 0, r67 = 0, r68 = 0, r69 = 0, r70 = 0, r71 = 0, r72 = 0,
         r73 = 0, r74 = 0, r75 = 0, r76 = 0, r77 = 0, r78 = 0, r79 = 0, r80 = 0,
         r81 = 0, r82 = 0, r83 = 0, r84 = 0, r85 = 0, r86 = 0, r87 = 0, r88 = 0,
         r89 = 0, r90 = 0, r91 = 0, r92 = 0;
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
  };
  LoadShared<2, double, double>(calib, 0 * calib_num_alloc, calib_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        calib_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r8, r9);
  };
  LoadShared<2, double, double>(point, 0 * point_num_alloc, point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        point_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = -2.00000000000000000e+00;
  };
  LoadShared<2, double, double>(pose, 2 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r13, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r15, r16);
  };
  LoadShared<2, double, double>(pose, 0 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r17, r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r19, r20);
    r21 = fma(r18, r19, r13 * r16);
    r22 = r17 * r20;
    r21 = fma(r4, r22, r21);
    r21 = fma(r14, r15, r21);
    r22 = r21 * r21;
    r22 = r12 * r22;
    r23 = 1.00000000000000000e+00;
    r24 = r13 * r19;
    r24 = fma(r4, r24, r18 * r16);
    r24 = fma(r14, r20, r24);
    r24 = fma(r17, r15, r24);
    r25 = r24 * r24;
    r25 = fma(r12, r25, r23);
    r26 = r22 + r25;
    r26 = fma(r10, r26, r8);
    r27 = 2.00000000000000000e+00;
    r28 = fma(r14, r19, r17 * r16);
    r29 = r18 * r15;
    r28 = fma(r4, r29, r28);
    r28 = fma(r13, r20, r28);
    r29 = r27 * r28;
    r30 = r24 * r29;
    r31 = fma(r18, r20, r17 * r19);
    r31 = fma(r13, r15, r31);
    r31 = fma(r4, r31, r14 * r16);
    r32 = r12 * r31;
    r33 = fma(r21, r32, r30);
  };
  LoadShared<1, double, double>(point, 2 * point_num_alloc, point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        point_indices_loc[threadIdx.x].target, r34);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r35 = r27 * r24;
    r36 = r21 * r29;
    r35 = fma(r31, r35, r36);
  };
  LoadShared<1, double, double>(pose, 6 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = r19 * r15;
    r38 = r38 * r27;
    r39 = r20 * r16;
    r40 = fma(r27, r39, r38);
  };
  LoadShared<2, double, double>(pose, 4 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r41, r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r43 = r15 * r16;
    r44 = r19 * r20;
    r44 = r44 * r27;
    r43 = fma(r12, r43, r44);
    r45 = r20 * r20;
    r45 = r45 * r12;
    r46 = r23 + r45;
    r47 = r15 * r15;
    r47 = r12 * r47;
    r46 = r46 + r47;
    r26 = fma(r11, r33, r26);
    r26 = fma(r34, r35, r26);
    r26 = fma(r37, r40, r26);
    r26 = fma(r42, r43, r26);
    r26 = fma(r41, r46, r26);
    r46 = r6 * r26;
    r43 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r40);
    r36 = fma(r24, r32, r36);
    r36 = fma(r10, r36, r40);
    r39 = fma(r12, r39, r38);
    r45 = r23 + r45;
    r38 = r19 * r19;
    r38 = r12 * r38;
    r45 = r45 + r38;
    r35 = r20 * r15;
    r35 = r35 * r27;
    r33 = r19 * r16;
    r33 = fma(r27, r33, r35);
    r48 = r27 * r21;
    r48 = r48 * r24;
    r29 = fma(r31, r29, r48);
    r49 = r28 * r28;
    r49 = r49 * r12;
    r25 = r49 + r25;
    r36 = fma(r41, r39, r36);
    r36 = fma(r37, r45, r36);
    r36 = fma(r42, r33, r36);
    r36 = fma(r11, r29, r36);
    r36 = fma(r34, r25, r36);
    r25 = copysign(1.0, r36);
    r25 = fma(r43, r25, r36);
    r25 = 1.0 / r25;
    r5 = fma(r25, r46, r5);
    r4 = fma(r3, r4, r1);
    r46 = r27 * r21;
    r46 = fma(r31, r46, r30);
    r46 = fma(r10, r46, r9);
    r30 = r15 * r16;
    r30 = fma(r27, r30, r44);
    r47 = r23 + r47;
    r47 = r47 + r38;
    r38 = r19 * r16;
    r38 = fma(r12, r38, r35);
    r32 = fma(r28, r32, r48);
    r22 = r23 + r22;
    r22 = r22 + r49;
    r46 = fma(r41, r30, r46);
    r46 = fma(r42, r47, r46);
    r46 = fma(r37, r38, r46);
    r46 = fma(r34, r32, r46);
    r46 = fma(r11, r22, r46);
    r22 = r7 * r46;
    r4 = fma(r25, r22, r4);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc,
                                             global_thread_idx, r5, r4);
    r5 = fma(r5, r5, r4 * r4);
  };
  SumStore<double>(out_rTr_local, (double *)inout_shared, 0,
                   global_thread_idx < problem_size, r5);
  if (global_thread_idx < problem_size) {
    r5 = 2.00000000000000000e+00;
    r4 = r14 * r20;
    r22 = fma(r18, r16, r4);
    r25 = r17 * r15;
    r32 = -1.00000000000000000e+00;
    r38 = r13 * r19;
    r22 = r22 + r25;
    r22 = fma(r32, r38, r22);
    r47 = r5 * r22;
    r30 = -5.00000000000000000e-01;
    r49 = r18 * r30;
    r23 = 5.00000000000000000e-01;
    r28 = fma(r23, r38, r16 * r49);
    r28 = fma(r30, r4, r28);
    r28 = fma(r30, r25, r28);
    r47 = r47 * r28;
    r48 = fma(r18, r19, r13 * r16);
    r35 = r17 * r20;
    r48 = fma(r32, r35, r48);
    r48 = fma(r14, r15, r48);
    r35 = r5 * r48;
    r12 = r18 * r19;
    r44 = r17 * r20;
    r44 = fma(r30, r44, r23 * r12);
    r12 = r14 * r15;
    r44 = fma(r23, r12, r44);
    r31 = r16 * r23;
    r44 = fma(r13, r31, r44);
    r35 = fma(r44, r35, r47);
    r12 = fma(r14, r19, r17 * r16);
    r43 = r18 * r15;
    r12 = fma(r32, r43, r12);
    r12 = fma(r13, r20, r12);
    r43 = r5 * r12;
    r36 = r17 * r16;
    r29 = r14 * r19;
    r29 = fma(r30, r29, r30 * r36);
    r36 = r13 * r20;
    r29 = fma(r30, r36, r29);
    r33 = r18 * r15;
    r29 = fma(r23, r33, r29);
    r43 = r43 * r29;
    r33 = r17 * r19;
    r36 = r13 * r15;
    r36 = fma(r30, r36, r30 * r33);
    r36 = fma(r14, r31, r36);
    r36 = fma(r20, r49, r36);
    r33 = fma(r18, r20, r17 * r19);
    r33 = fma(r13, r15, r33);
    r33 = fma(r32, r33, r14 * r16);
    r45 = r5 * r33;
    r39 = r36 * r45;
    r50 = r43 + r39;
    r51 = r35 + r50;
    r52 = -2.00000000000000000e+00;
    r53 = r52 * r33;
    r54 = r52 * r22;
    r53 = fma(r29, r54, r44 * r53);
    r55 = r5 * r12;
    r56 = r5 * r48;
    r56 = r56 * r36;
    r55 = fma(r28, r55, r56);
    r53 = r53 + r55;
    r53 = fma(r10, r53, r11 * r51);
    r51 = r22 * r44;
    r57 = -4.00000000000000000e+00;
    r51 = r51 * r57;
    r58 = r12 * r57;
    r59 = r36 * r58;
    r60 = r51 + r59;
    r53 = fma(r34, r60, r53);
    r60 = 1.00000000000000008e-15;
    r61 = r5 * r48;
    r61 = r61 * r12;
    r62 = fma(r33, r54, r61);
    r62 = fma(r10, r62, r40);
    r63 = r19 * r15;
    r63 = r63 * r5;
    r64 = r20 * r16;
    r64 = fma(r52, r64, r63);
    r65 = r19 * r19;
    r65 = r65 * r52;
    r66 = 1.00000000000000000e+00;
    r67 = r20 * r20;
    r67 = fma(r52, r67, r66);
    r68 = r65 + r67;
    r69 = r20 * r15;
    r69 = r69 * r5;
    r70 = r19 * r16;
    r70 = fma(r5, r70, r69);
    r71 = r5 * r48;
    r71 = r71 * r22;
    r72 = fma(r12, r45, r71);
    r73 = r22 * r54;
    r74 = r66 + r73;
    r75 = r12 * r12;
    r75 = r75 * r52;
    r74 = r74 + r75;
    r62 = fma(r41, r64, r62);
    r62 = fma(r37, r68, r62);
    r62 = fma(r42, r70, r62);
    r62 = fma(r11, r72, r62);
    r62 = fma(r34, r74, r62);
    r74 = copysign(1.0, r62);
    r74 = fma(r60, r74, r62);
    r60 = r74 * r74;
    r60 = 1.0 / r60;
    r60 = r32 * r60;
    r62 = r53 * r60;
    r73 = r66 + r73;
    r72 = r48 * r48;
    r72 = r72 * r52;
    r73 = r73 + r72;
    r73 = fma(r10, r73, r8);
    r76 = r5 * r12;
    r76 = r76 * r22;
    r77 = r48 * r52;
    r77 = fma(r33, r77, r76);
    r61 = fma(r22, r45, r61);
    r78 = r20 * r16;
    r78 = fma(r5, r78, r63);
    r63 = r15 * r16;
    r79 = r19 * r20;
    r79 = r79 * r5;
    r63 = fma(r52, r63, r79);
    r80 = r15 * r15;
    r80 = r80 * r52;
    r67 = r80 + r67;
    r73 = fma(r11, r77, r73);
    r73 = fma(r34, r61, r73);
    r73 = fma(r37, r78, r73);
    r73 = fma(r42, r63, r73);
    r73 = fma(r41, r67, r73);
    r73 = r6 * r73;
    r61 = r5 * r22;
    r61 = fma(r44, r45, r29 * r61);
    r61 = r61 + r55;
    r77 = r5 * r22;
    r77 = r77 * r36;
    r81 = r5 * r12;
    r81 = r81 * r44;
    r44 = r77 + r81;
    r82 = r48 * r52;
    r44 = fma(r29, r82, r44);
    r83 = r52 * r33;
    r44 = fma(r28, r83, r44);
    r44 = fma(r11, r44, r34 * r61);
    r61 = r48 * r28;
    r83 = r57 * r61;
    r51 = r51 + r83;
    r44 = fma(r10, r51, r44);
    r51 = r6 * r44;
    r74 = 1.0 / r74;
    r51 = fma(r74, r51, r73 * r62);
    r62 = r12 * r52;
    r82 = r52 * r33;
    r82 = r82 * r36;
    r62 = fma(r29, r62, r82);
    r62 = r62 + r35;
    r83 = r59 + r83;
    r83 = fma(r11, r83, r34 * r62);
    r81 = fma(r28, r45, r81);
    r62 = r5 * r48;
    r62 = fma(r29, r62, r77);
    r81 = r81 + r62;
    r83 = fma(r10, r81, r83);
    r81 = r7 * r74;
    r76 = fma(r48, r45, r76);
    r76 = fma(r10, r76, r9);
    r77 = r15 * r16;
    r77 = fma(r5, r77, r79);
    r80 = r66 + r80;
    r80 = r80 + r65;
    r65 = r19 * r16;
    r65 = fma(r52, r65, r69);
    r69 = r12 * r52;
    r69 = fma(r33, r69, r71);
    r72 = r66 + r72;
    r72 = r72 + r75;
    r76 = fma(r41, r77, r76);
    r76 = fma(r42, r80, r76);
    r76 = fma(r37, r65, r76);
    r76 = fma(r34, r69, r76);
    r76 = fma(r11, r72, r76);
    r72 = r7 * r76;
    r72 = r72 * r60;
    r83 = fma(r53, r72, r83 * r81);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r51, r83);
    r69 = fma(r28, r54, r82);
    r75 = r5 * r48;
    r66 = r13 * r16;
    r71 = r17 * r20;
    r71 = fma(r23, r71, r30 * r66);
    r66 = r14 * r15;
    r71 = fma(r30, r66, r71);
    r71 = fma(r19, r49, r71);
    r75 = r75 * r71;
    r66 = r5 * r12;
    r79 = r14 * r19;
    r59 = r13 * r20;
    r59 = fma(r23, r59, r23 * r79);
    r59 = fma(r17, r31, r59);
    r59 = fma(r15, r49, r59);
    r66 = fma(r59, r66, r75);
    r69 = r69 + r66;
    r49 = r5 * r22;
    r49 = r49 * r59;
    r79 = fma(r71, r45, r49);
    r79 = r79 + r55;
    r79 = fma(r11, r79, r10 * r69);
    r69 = r22 * r36;
    r69 = r69 * r57;
    r55 = r71 * r58;
    r35 = r69 + r55;
    r79 = fma(r34, r35, r79);
    r35 = r79 * r60;
    r39 = r47 + r39;
    r39 = r39 + r66;
    r66 = r48 * r57;
    r66 = r66 * r59;
    r69 = r69 + r66;
    r69 = fma(r10, r69, r34 * r39);
    r39 = r52 * r33;
    r39 = fma(r52, r61, r59 * r39);
    r47 = r5 * r12;
    r47 = r47 * r36;
    r84 = r5 * r22;
    r84 = fma(r71, r84, r47);
    r39 = r39 + r84;
    r69 = fma(r11, r39, r69);
    r39 = r6 * r69;
    r39 = fma(r74, r39, r73 * r35);
    r49 = r56 + r49;
    r56 = r12 * r52;
    r49 = fma(r28, r56, r49);
    r28 = r52 * r33;
    r49 = fma(r71, r28, r49);
    r59 = fma(r59, r45, r5 * r61);
    r59 = r59 + r84;
    r59 = fma(r10, r59, r34 * r49);
    r55 = r66 + r55;
    r59 = fma(r11, r55, r59);
    r59 = fma(r79, r72, r59 * r81);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r39, r59);
    r82 = r43 + r82;
    r43 = r5 * r22;
    r38 = fma(r30, r38, r18 * r31);
    r38 = fma(r23, r4, r38);
    r38 = fma(r23, r25, r38);
    r43 = r43 * r38;
    r25 = r48 * r52;
    r82 = fma(r71, r25, r82);
    r82 = r82 + r43;
    r36 = r48 * r36;
    r36 = r36 * r57;
    r25 = r22 * r29;
    r25 = r25 * r57;
    r57 = r36 + r25;
    r57 = fma(r10, r57, r11 * r82);
    r82 = r5 * r48;
    r82 = r82 * r38;
    r23 = fma(r29, r45, r82);
    r23 = r23 + r84;
    r57 = fma(r34, r23, r57);
    r23 = r6 * r57;
    r58 = r38 * r58;
    r25 = r25 + r58;
    r82 = r47 + r82;
    r47 = r52 * r33;
    r82 = fma(r29, r47, r82);
    r82 = fma(r71, r54, r82);
    r82 = fma(r10, r82, r34 * r25);
    r25 = r5 * r12;
    r45 = fma(r38, r45, r71 * r25);
    r45 = r45 + r62;
    r82 = fma(r11, r45, r82);
    r45 = r82 * r60;
    r45 = fma(r73, r45, r74 * r23);
    r43 = r75 + r43;
    r43 = r43 + r50;
    r50 = r12 * r52;
    r75 = r52 * r33;
    r75 = fma(r38, r75, r71 * r50);
    r75 = r75 + r62;
    r75 = fma(r34, r75, r10 * r43);
    r58 = r36 + r58;
    r75 = fma(r11, r58, r75);
    r75 = fma(r75, r81, r82 * r72);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r45, r75);
    r58 = r6 * r67;
    r36 = r64 * r60;
    r36 = fma(r73, r36, r74 * r58);
    r77 = fma(r64, r72, r77 * r81);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r36, r77);
    r58 = r70 * r60;
    r43 = r6 * r63;
    r43 = fma(r74, r43, r73 * r58);
    r80 = fma(r80, r81, r70 * r72);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r43, r80);
    r58 = r68 * r60;
    r62 = r6 * r78;
    r62 = fma(r74, r62, r73 * r58);
    r65 = fma(r65, r81, r68 * r72);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r62, r65);
    r72 = r32 * r83;
    r58 = fma(r3, r32, r1);
    r58 = fma(r76, r81, r58);
    r81 = r32 * r51;
    r76 = fma(r2, r32, r0);
    r76 = fma(r74, r73, r76);
    r81 = fma(r76, r81, r58 * r72);
    r72 = r32 * r39;
    r73 = r32 * r59;
    r73 = fma(r58, r73, r76 * r72);
    WriteSum2<double, double>((double *)inout_shared, r81, r73);
  };
  FlushSumShared<2, double>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r32 * r75;
    r81 = r32 * r45;
    r81 = fma(r76, r81, r58 * r73);
    r73 = r32 * r77;
    r72 = r32 * r36;
    r72 = fma(r76, r72, r58 * r73);
    WriteSum2<double, double>((double *)inout_shared, r81, r72);
  };
  FlushSumShared<2, double>(out_pose_njtr, 2 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r32 * r80;
    r81 = r32 * r43;
    r81 = fma(r76, r81, r58 * r72);
    r72 = r32 * r65;
    r73 = r32 * r62;
    r73 = fma(r76, r73, r58 * r72);
    WriteSum2<double, double>((double *)inout_shared, r81, r73);
  };
  FlushSumShared<2, double>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r83, r83, r51 * r51);
    r81 = fma(r39, r39, r59 * r59);
    WriteSum2<double, double>((double *)inout_shared, r73, r81);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r45, r45, r75 * r75);
    r73 = fma(r77, r77, r36 * r36);
    WriteSum2<double, double>((double *)inout_shared, r81, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r80, r80, r43 * r43);
    r81 = fma(r62, r62, r65 * r65);
    WriteSum2<double, double>((double *)inout_shared, r73, r81);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r51, r39, r83 * r59);
    r73 = fma(r51, r45, r83 * r75);
    WriteSum2<double, double>((double *)inout_shared, r81, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r51, r36, r83 * r77);
    r81 = fma(r51, r43, r83 * r80);
    WriteSum2<double, double>((double *)inout_shared, r73, r81);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r51, r62, r83 * r65);
    r73 = fma(r59, r75, r39 * r45);
    WriteSum2<double, double>((double *)inout_shared, r81, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r59, r77, r39 * r36);
    r81 = fma(r59, r80, r39 * r43);
    WriteSum2<double, double>((double *)inout_shared, r73, r81);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r59, r65, r39 * r62);
    r73 = fma(r75, r77, r45 * r36);
    WriteSum2<double, double>((double *)inout_shared, r81, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r75, r80, r45 * r43);
    r81 = fma(r75, r65, r45 * r62);
    WriteSum2<double, double>((double *)inout_shared, r73, r81);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r77, r80, r36 * r43);
    r73 = fma(r36, r62, r77 * r65);
    WriteSum2<double, double>((double *)inout_shared, r81, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r43, r62, r80 * r65);
    WriteSum1<double, double>((double *)inout_shared, r73);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = 1.00000000000000008e-15;
    r81 = fma(r18, r19, r13 * r16);
    r72 = r17 * r20;
    r76 = -1.00000000000000000e+00;
    r81 = fma(r76, r72, r81);
    r81 = fma(r14, r15, r81);
    r72 = 2.00000000000000000e+00;
    r58 = fma(r14, r19, r17 * r16);
    r74 = r18 * r15;
    r58 = fma(r76, r74, r58);
    r58 = fma(r13, r20, r58);
    r74 = r72 * r58;
    r50 = r81 * r74;
    r38 = r13 * r19;
    r38 = fma(r76, r38, r18 * r16);
    r38 = fma(r14, r20, r38);
    r38 = fma(r17, r15, r38);
    r71 = -2.00000000000000000e+00;
    r23 = fma(r18, r20, r17 * r19);
    r23 = fma(r13, r15, r23);
    r23 = fma(r76, r23, r14 * r16);
    r25 = r71 * r23;
    r54 = fma(r38, r25, r50);
    r54 = fma(r10, r54, r40);
    r47 = r19 * r15;
    r47 = r47 * r72;
    r29 = r20 * r16;
    r84 = fma(r71, r29, r47);
    r4 = 1.00000000000000000e+00;
    r30 = r20 * r20;
    r30 = r30 * r71;
    r31 = r4 + r30;
    r55 = r19 * r19;
    r55 = r71 * r55;
    r31 = r31 + r55;
    r66 = r20 * r15;
    r66 = r66 * r72;
    r49 = r19 * r16;
    r49 = fma(r72, r49, r66);
    r61 = r72 * r81;
    r61 = r61 * r38;
    r28 = fma(r23, r74, r61);
    r56 = r58 * r58;
    r56 = r56 * r71;
    r35 = r38 * r38;
    r35 = fma(r71, r35, r4);
    r85 = r56 + r35;
    r54 = fma(r41, r84, r54);
    r54 = fma(r37, r31, r54);
    r54 = fma(r42, r49, r54);
    r54 = fma(r11, r28, r54);
    r54 = fma(r34, r85, r54);
    r85 = copysign(1.0, r54);
    r85 = fma(r73, r85, r54);
    r73 = 1.0 / r85;
    r54 = r81 * r81;
    r54 = r71 * r54;
    r35 = r54 + r35;
    r35 = fma(r10, r35, r8);
    r74 = r38 * r74;
    r28 = fma(r81, r25, r74);
    r49 = r72 * r38;
    r49 = fma(r23, r49, r50);
    r29 = fma(r72, r29, r47);
    r47 = r15 * r16;
    r50 = r19 * r20;
    r50 = r50 * r72;
    r47 = fma(r71, r47, r50);
    r30 = r4 + r30;
    r31 = r15 * r15;
    r31 = r71 * r31;
    r30 = r30 + r31;
    r35 = fma(r11, r28, r35);
    r35 = fma(r34, r49, r35);
    r35 = fma(r37, r29, r35);
    r35 = fma(r42, r47, r35);
    r35 = fma(r41, r30, r35);
    r30 = r73 * r35;
    r47 = r72 * r81;
    r47 = fma(r23, r47, r74);
    r47 = fma(r10, r47, r9);
    r74 = r15 * r16;
    r74 = fma(r72, r74, r50);
    r31 = r4 + r31;
    r31 = r31 + r55;
    r55 = r19 * r16;
    r55 = fma(r71, r55, r66);
    r25 = fma(r58, r25, r61);
    r54 = r4 + r54;
    r54 = r54 + r56;
    r47 = fma(r41, r74, r47);
    r47 = fma(r42, r31, r47);
    r47 = fma(r37, r55, r47);
    r47 = fma(r34, r25, r47);
    r47 = fma(r11, r54, r47);
    r54 = r47 * r73;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             0 * out_calib_jac_num_alloc,
                                             global_thread_idx, r30, r54);
    r25 = fma(r2, r76, r0);
    r25 = fma(r6, r30, r25);
    r25 = r76 * r25;
    r55 = r30 * r25;
    r31 = r76 * r47;
    r74 = fma(r3, r76, r1);
    r56 = r7 * r47;
    r74 = fma(r73, r56, r74);
    r31 = r31 * r74;
    r31 = r31 * r73;
    WriteSum2<double, double>((double *)inout_shared, r55, r31);
  };
  FlushSumShared<2, double>(out_calib_njtr, 0 * out_calib_njtr_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r76 * r74;
    WriteSum2<double, double>((double *)inout_shared, r25, r74);
  };
  FlushSumShared<2, double>(out_calib_njtr, 2 * out_calib_njtr_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r35 * r35;
    r85 = r85 * r85;
    r85 = 1.0 / r85;
    r35 = r35 * r85;
    r74 = r47 * r47;
    r74 = r85 * r74;
    WriteSum2<double, double>((double *)inout_shared, r35, r74);
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
    r4 = 0.00000000000000000e+00;
    WriteSum2<double, double>((double *)inout_shared, r4, r30);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double *)inout_shared, r54, r4);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = 1.00000000000000008e-15;
    r54 = fma(r14, r19, r17 * r16);
    r30 = r18 * r15;
    r74 = -1.00000000000000000e+00;
    r54 = fma(r74, r30, r54);
    r54 = fma(r13, r20, r54);
    r30 = fma(r18, r19, r13 * r16);
    r35 = r17 * r20;
    r30 = fma(r74, r35, r30);
    r30 = fma(r14, r15, r30);
    r35 = 2.00000000000000000e+00;
    r85 = r30 * r35;
    r25 = r54 * r85;
    r76 = r13 * r19;
    r76 = fma(r74, r76, r18 * r16);
    r76 = fma(r14, r20, r76);
    r76 = fma(r17, r15, r76);
    r31 = -2.00000000000000000e+00;
    r55 = fma(r18, r20, r17 * r19);
    r55 = fma(r13, r15, r55);
    r55 = fma(r74, r55, r14 * r16);
    r73 = r31 * r55;
    r56 = fma(r76, r73, r25);
    r40 = fma(r10, r56, r40);
    r58 = r19 * r15;
    r58 = r58 * r35;
    r61 = r20 * r16;
    r66 = fma(r31, r61, r58);
    r71 = 1.00000000000000000e+00;
    r50 = r19 * r19;
    r50 = r31 * r50;
    r23 = r71 + r50;
    r29 = r20 * r20;
    r29 = r29 * r31;
    r23 = r23 + r29;
    r49 = r20 * r15;
    r49 = r49 * r35;
    r28 = r19 * r16;
    r28 = fma(r35, r28, r49);
    r84 = r35 * r54;
    r86 = r76 * r85;
    r84 = fma(r55, r84, r86);
    r87 = r76 * r76;
    r87 = r31 * r87;
    r88 = r71 + r87;
    r89 = r54 * r54;
    r89 = r31 * r89;
    r88 = r88 + r89;
    r40 = fma(r41, r66, r40);
    r40 = fma(r37, r23, r40);
    r40 = fma(r42, r28, r40);
    r40 = fma(r11, r84, r40);
    r40 = fma(r34, r88, r40);
    r28 = copysign(1.0, r40);
    r28 = fma(r4, r28, r40);
    r4 = 1.0 / r28;
    r40 = r6 * r4;
    r23 = r31 * r30;
    r23 = fma(r30, r23, r71);
    r87 = r87 + r23;
    r8 = fma(r10, r87, r8);
    r66 = r35 * r76;
    r66 = r66 * r54;
    r90 = fma(r30, r73, r66);
    r91 = r35 * r76;
    r91 = fma(r55, r91, r25);
    r61 = fma(r35, r61, r58);
    r58 = r15 * r16;
    r25 = r19 * r20;
    r25 = r25 * r35;
    r58 = fma(r31, r58, r25);
    r29 = r71 + r29;
    r92 = r15 * r15;
    r92 = r31 * r92;
    r29 = r29 + r92;
    r8 = fma(r11, r90, r8);
    r8 = fma(r34, r91, r8);
    r8 = fma(r37, r61, r8);
    r8 = fma(r42, r58, r8);
    r8 = fma(r41, r29, r8);
    r29 = r6 * r8;
    r28 = r28 * r28;
    r28 = 1.0 / r28;
    r28 = r74 * r28;
    r29 = r29 * r28;
    r87 = fma(r56, r29, r87 * r40);
    r58 = r56 * r28;
    r85 = fma(r55, r85, r66);
    r10 = fma(r10, r85, r9);
    r9 = r15 * r16;
    r9 = fma(r35, r9, r25);
    r50 = r71 + r50;
    r50 = r50 + r92;
    r92 = r19 * r16;
    r92 = fma(r31, r92, r49);
    r73 = fma(r54, r73, r86);
    r23 = r89 + r23;
    r10 = fma(r41, r9, r10);
    r10 = fma(r42, r50, r10);
    r10 = fma(r37, r92, r10);
    r10 = fma(r34, r73, r10);
    r10 = fma(r11, r23, r10);
    r10 = r7 * r10;
    r11 = r7 * r85;
    r11 = fma(r4, r11, r10 * r58);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx, r87, r11);
    r90 = fma(r90, r40, r84 * r29);
    r58 = r7 * r23;
    r34 = r84 * r28;
    r34 = fma(r10, r34, r4 * r58);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx, r90, r34);
    r91 = fma(r91, r40, r88 * r29);
    r29 = r7 * r73;
    r58 = r88 * r28;
    r58 = fma(r10, r58, r4 * r29);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx, r91, r58);
    r29 = r74 * r11;
    r3 = fma(r3, r74, r1);
    r3 = fma(r4, r10, r3);
    r10 = r74 * r87;
    r2 = fma(r2, r74, r0);
    r2 = fma(r8, r40, r2);
    r10 = fma(r2, r10, r3 * r29);
    r29 = r74 * r90;
    r40 = r74 * r34;
    r40 = fma(r3, r40, r2 * r29);
    WriteSum2<double, double>((double *)inout_shared, r10, r40);
  };
  FlushSumShared<2, double>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r74 * r91;
    r10 = r74 * r58;
    r10 = fma(r3, r10, r2 * r40);
    WriteSum1<double, double>((double *)inout_shared, r10);
  };
  FlushSumShared<1, double>(out_point_njtr, 2 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fma(r87, r87, r11 * r11);
    r40 = fma(r34, r34, r90 * r90);
    WriteSum2<double, double>((double *)inout_shared, r10, r40);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fma(r91, r91, r58 * r58);
    WriteSum1<double, double>((double *)inout_shared, r40);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fma(r11, r34, r87 * r90);
    r10 = fma(r11, r58, r87 * r91);
    WriteSum2<double, double>((double *)inout_shared, r40, r10);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fma(r34, r58, r90 * r91);
    WriteSum1<double, double>((double *)inout_shared, r10);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeResJacFirst(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    double *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    double *pixel, unsigned int pixel_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr, double *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, double *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, double *out_calib_jac,
    unsigned int out_calib_jac_num_alloc, double *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, double *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, double *out_point_jac,
    unsigned int out_point_jac_num_alloc, double *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeResJacFirstKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, calib, calib_num_alloc, calib_indices, point,
      point_num_alloc, point_indices, pixel, pixel_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_pose_jac, out_pose_jac_num_alloc,
      out_pose_njtr, out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_calib_jac, out_calib_jac_num_alloc,
      out_calib_njtr, out_calib_njtr_num_alloc, out_calib_precond_diag,
      out_calib_precond_diag_num_alloc, out_calib_precond_tril,
      out_calib_precond_tril_num_alloc, out_point_jac, out_point_jac_num_alloc,
      out_point_njtr, out_point_njtr_num_alloc, out_point_precond_diag,
      out_point_precond_diag_num_alloc, out_point_precond_tril,
      out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar