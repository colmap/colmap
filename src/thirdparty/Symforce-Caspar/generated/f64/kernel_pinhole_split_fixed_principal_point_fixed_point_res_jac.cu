#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_pinhole_split_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPrincipalPointFixedPointResJacKernel(
        double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        double *focal, unsigned int focal_num_alloc, SharedIndex *focal_indices,
        double *pixel, unsigned int pixel_num_alloc, double *principal_point,
        unsigned int principal_point_num_alloc, double *point,
        unsigned int point_num_alloc, double *out_res,
        unsigned int out_res_num_alloc, double *out_pose_jac,
        unsigned int out_pose_jac_num_alloc, double *const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc, double *out_focal_jac,
        unsigned int out_focal_jac_num_alloc, double *const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        double *const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        double *const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

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
         r81 = 0, r82 = 0, r83 = 0, r84 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(pixel, 0 * pixel_num_alloc,
                                            global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r5 = fma(r2, r4, r0);
  };
  LoadShared<2, double, double>(focal, 0 * focal_num_alloc, focal_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        focal_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r8, r9);
    ReadIdx2<1024, double, double, double2>(point, 0 * point_num_alloc,
                                            global_thread_idx, r10, r11);
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
    ReadIdx1<1024, double, double, double>(point, 2 * point_num_alloc,
                                           global_thread_idx, r34);
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
    r4 = 2.00000000000000000e+00;
    r5 = r14 * r20;
    r22 = fma(r18, r16, r5);
    r25 = r17 * r15;
    r32 = -1.00000000000000000e+00;
    r38 = r13 * r19;
    r22 = r22 + r25;
    r22 = fma(r32, r38, r22);
    r47 = r4 * r22;
    r30 = -5.00000000000000000e-01;
    r49 = r18 * r30;
    r23 = 5.00000000000000000e-01;
    r28 = fma(r23, r38, r16 * r49);
    r28 = fma(r30, r5, r28);
    r28 = fma(r30, r25, r28);
    r47 = r47 * r28;
    r48 = fma(r18, r19, r13 * r16);
    r35 = r17 * r20;
    r48 = fma(r32, r35, r48);
    r48 = fma(r14, r15, r48);
    r35 = r4 * r48;
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
    r43 = r4 * r12;
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
    r45 = r4 * r33;
    r39 = r36 * r45;
    r50 = r43 + r39;
    r51 = r35 + r50;
    r52 = -2.00000000000000000e+00;
    r53 = r52 * r33;
    r54 = r52 * r22;
    r53 = fma(r29, r54, r44 * r53);
    r55 = r4 * r12;
    r56 = r4 * r48;
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
    r61 = r4 * r48;
    r61 = r61 * r12;
    r62 = fma(r33, r54, r61);
    r62 = fma(r10, r62, r40);
    r63 = r19 * r15;
    r63 = r63 * r4;
    r64 = r20 * r16;
    r64 = fma(r52, r64, r63);
    r65 = r19 * r19;
    r65 = r65 * r52;
    r66 = 1.00000000000000000e+00;
    r67 = r20 * r20;
    r67 = fma(r52, r67, r66);
    r68 = r65 + r67;
    r69 = r20 * r15;
    r69 = r69 * r4;
    r70 = r19 * r16;
    r70 = fma(r4, r70, r69);
    r71 = r4 * r48;
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
    r76 = r4 * r12;
    r76 = r76 * r22;
    r77 = r48 * r52;
    r77 = fma(r33, r77, r76);
    r61 = fma(r22, r45, r61);
    r78 = r20 * r16;
    r78 = fma(r4, r78, r63);
    r63 = r15 * r16;
    r79 = r19 * r20;
    r79 = r79 * r4;
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
    r61 = r4 * r22;
    r61 = fma(r44, r45, r29 * r61);
    r61 = r61 + r55;
    r77 = r4 * r22;
    r77 = r77 * r36;
    r81 = r4 * r12;
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
    r62 = r4 * r48;
    r62 = fma(r29, r62, r77);
    r81 = r81 + r62;
    r83 = fma(r10, r81, r83);
    r81 = r7 * r74;
    r76 = fma(r48, r45, r76);
    r76 = fma(r10, r76, r9);
    r77 = r15 * r16;
    r77 = fma(r4, r77, r79);
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
    r75 = r4 * r48;
    r66 = r13 * r16;
    r71 = r17 * r20;
    r71 = fma(r23, r71, r30 * r66);
    r66 = r14 * r15;
    r71 = fma(r30, r66, r71);
    r71 = fma(r19, r49, r71);
    r75 = r75 * r71;
    r66 = r4 * r12;
    r79 = r14 * r19;
    r59 = r13 * r20;
    r59 = fma(r23, r59, r23 * r79);
    r59 = fma(r17, r31, r59);
    r59 = fma(r15, r49, r59);
    r66 = fma(r59, r66, r75);
    r69 = r69 + r66;
    r49 = r4 * r22;
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
    r47 = r4 * r12;
    r47 = r47 * r36;
    r84 = r4 * r22;
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
    r59 = fma(r59, r45, r4 * r61);
    r59 = r59 + r84;
    r59 = fma(r10, r59, r34 * r49);
    r55 = r66 + r55;
    r59 = fma(r11, r55, r59);
    r59 = fma(r59, r81, r79 * r72);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r39, r59);
    r55 = r22 * r29;
    r55 = r55 * r57;
    r38 = fma(r30, r38, r18 * r31);
    r38 = fma(r23, r5, r38);
    r38 = fma(r23, r25, r38);
    r58 = r38 * r58;
    r25 = r55 + r58;
    r23 = r4 * r48;
    r23 = r23 * r38;
    r47 = r47 + r23;
    r5 = r52 * r33;
    r47 = fma(r29, r5, r47);
    r47 = fma(r71, r54, r47);
    r47 = fma(r10, r47, r34 * r25);
    r25 = r4 * r12;
    r25 = fma(r38, r45, r71 * r25);
    r25 = r25 + r62;
    r47 = fma(r11, r25, r47);
    r25 = r47 * r60;
    r82 = r43 + r82;
    r43 = r4 * r22;
    r43 = r43 * r38;
    r54 = r48 * r52;
    r82 = fma(r71, r54, r82);
    r82 = r82 + r43;
    r36 = r48 * r36;
    r36 = r36 * r57;
    r55 = r55 + r36;
    r55 = fma(r10, r55, r11 * r82);
    r45 = fma(r29, r45, r23);
    r45 = r45 + r84;
    r55 = fma(r34, r45, r55);
    r45 = r6 * r55;
    r45 = fma(r74, r45, r73 * r25);
    r43 = r75 + r43;
    r43 = r43 + r50;
    r50 = r12 * r52;
    r75 = r52 * r33;
    r75 = fma(r38, r75, r71 * r50);
    r75 = r75 + r62;
    r75 = fma(r34, r75, r10 * r43);
    r58 = r36 + r58;
    r75 = fma(r11, r58, r75);
    r75 = fma(r75, r81, r47 * r72);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r45, r75);
    r58 = r6 * r67;
    r36 = r64 * r60;
    r36 = fma(r73, r36, r74 * r58);
    r77 = fma(r77, r81, r64 * r72);
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
    r72 = fma(r68, r72, r65 * r81);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r62, r72);
    r65 = r32 * r51;
    r58 = fma(r2, r32, r0);
    r58 = fma(r74, r73, r58);
    r73 = r32 * r83;
    r74 = fma(r3, r32, r1);
    r74 = fma(r76, r81, r74);
    r73 = fma(r74, r73, r58 * r65);
    r65 = r32 * r59;
    r81 = r32 * r39;
    r81 = fma(r58, r81, r74 * r65);
    WriteSum2<double, double>((double *)inout_shared, r73, r81);
  };
  FlushSumShared<2, double>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = r32 * r75;
    r73 = r32 * r45;
    r73 = fma(r58, r73, r74 * r81);
    r81 = r32 * r36;
    r65 = r32 * r77;
    r65 = fma(r74, r65, r58 * r81);
    WriteSum2<double, double>((double *)inout_shared, r73, r65);
  };
  FlushSumShared<2, double>(out_pose_njtr, 2 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r32 * r80;
    r73 = r32 * r43;
    r73 = fma(r58, r73, r74 * r65);
    r65 = r32 * r62;
    r81 = r32 * r72;
    r81 = fma(r74, r81, r58 * r65);
    WriteSum2<double, double>((double *)inout_shared, r73, r81);
  };
  FlushSumShared<2, double>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r51, r51, r83 * r83);
    r73 = fma(r59, r59, r39 * r39);
    WriteSum2<double, double>((double *)inout_shared, r81, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r45, r45, r75 * r75);
    r81 = fma(r36, r36, r77 * r77);
    WriteSum2<double, double>((double *)inout_shared, r73, r81);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r80, r80, r43 * r43);
    r73 = fma(r62, r62, r72 * r72);
    WriteSum2<double, double>((double *)inout_shared, r81, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r51, r39, r83 * r59);
    r81 = fma(r83, r75, r51 * r45);
    WriteSum2<double, double>((double *)inout_shared, r73, r81);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r83, r77, r51 * r36);
    r73 = fma(r83, r80, r51 * r43);
    WriteSum2<double, double>((double *)inout_shared, r81, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r51, r62, r83 * r72);
    r81 = fma(r59, r75, r39 * r45);
    WriteSum2<double, double>((double *)inout_shared, r73, r81);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r39, r36, r59 * r77);
    r73 = fma(r59, r80, r39 * r43);
    WriteSum2<double, double>((double *)inout_shared, r81, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r39, r62, r59 * r72);
    r81 = fma(r45, r36, r75 * r77);
    WriteSum2<double, double>((double *)inout_shared, r73, r81);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r45, r43, r75 * r80);
    r73 = fma(r45, r62, r75 * r72);
    WriteSum2<double, double>((double *)inout_shared, r81, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r77, r80, r36 * r43);
    r81 = fma(r36, r62, r77 * r72);
    WriteSum2<double, double>((double *)inout_shared, r73, r81);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r80, r72, r43 * r62);
    WriteSum1<double, double>((double *)inout_shared, r81);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = 1.00000000000000008e-15;
    r73 = fma(r18, r19, r13 * r16);
    r65 = r17 * r20;
    r74 = -1.00000000000000000e+00;
    r73 = fma(r74, r65, r73);
    r73 = fma(r14, r15, r73);
    r65 = 2.00000000000000000e+00;
    r58 = fma(r14, r19, r17 * r16);
    r76 = r18 * r15;
    r58 = fma(r74, r76, r58);
    r58 = fma(r13, r20, r58);
    r76 = r65 * r58;
    r50 = r73 * r76;
    r38 = r13 * r19;
    r38 = fma(r74, r38, r18 * r16);
    r38 = fma(r14, r20, r38);
    r38 = fma(r17, r15, r38);
    r71 = -2.00000000000000000e+00;
    r25 = fma(r18, r20, r17 * r19);
    r25 = fma(r13, r15, r25);
    r25 = fma(r74, r25, r14 * r16);
    r84 = r71 * r25;
    r29 = fma(r38, r84, r50);
    r29 = fma(r10, r29, r40);
    r40 = r19 * r15;
    r40 = r40 * r65;
    r23 = r20 * r16;
    r82 = fma(r71, r23, r40);
    r57 = 1.00000000000000000e+00;
    r54 = r20 * r20;
    r54 = r54 * r71;
    r5 = r57 + r54;
    r30 = r19 * r19;
    r30 = r71 * r30;
    r5 = r5 + r30;
    r31 = r20 * r15;
    r31 = r31 * r65;
    r66 = r19 * r16;
    r66 = fma(r65, r66, r31);
    r49 = r65 * r73;
    r49 = r49 * r38;
    r61 = fma(r25, r76, r49);
    r28 = r58 * r58;
    r28 = r28 * r71;
    r56 = r38 * r38;
    r56 = fma(r71, r56, r57);
    r35 = r28 + r56;
    r29 = fma(r41, r82, r29);
    r29 = fma(r37, r5, r29);
    r29 = fma(r42, r66, r29);
    r29 = fma(r11, r61, r29);
    r29 = fma(r34, r35, r29);
    r35 = copysign(1.0, r29);
    r35 = fma(r81, r35, r29);
    r81 = 1.0 / r35;
    r29 = r73 * r73;
    r29 = r71 * r29;
    r56 = r29 + r56;
    r56 = fma(r10, r56, r8);
    r76 = r38 * r76;
    r8 = fma(r73, r84, r76);
    r61 = r65 * r38;
    r61 = fma(r25, r61, r50);
    r23 = fma(r65, r23, r40);
    r40 = r15 * r16;
    r50 = r19 * r20;
    r50 = r50 * r65;
    r40 = fma(r71, r40, r50);
    r54 = r57 + r54;
    r66 = r15 * r15;
    r66 = r71 * r66;
    r54 = r54 + r66;
    r56 = fma(r11, r8, r56);
    r56 = fma(r34, r61, r56);
    r56 = fma(r37, r23, r56);
    r56 = fma(r42, r40, r56);
    r56 = fma(r41, r54, r56);
    r54 = r81 * r56;
    r40 = r65 * r73;
    r40 = fma(r25, r40, r76);
    r40 = fma(r10, r40, r9);
    r10 = r15 * r16;
    r10 = fma(r65, r10, r50);
    r66 = r57 + r66;
    r66 = r66 + r30;
    r30 = r19 * r16;
    r30 = fma(r71, r30, r31);
    r84 = fma(r58, r84, r49);
    r29 = r57 + r29;
    r29 = r29 + r28;
    r40 = fma(r41, r10, r40);
    r40 = fma(r42, r66, r40);
    r40 = fma(r37, r30, r40);
    r40 = fma(r34, r84, r40);
    r40 = fma(r11, r29, r40);
    r29 = r40 * r81;
    WriteIdx2<1024, double, double, double2>(out_focal_jac,
                                             0 * out_focal_jac_num_alloc,
                                             global_thread_idx, r54, r29);
    r2 = fma(r2, r74, r0);
    r2 = fma(r6, r54, r2);
    r2 = r74 * r2;
    r2 = r2 * r54;
    r54 = r74 * r40;
    r74 = fma(r3, r74, r1);
    r3 = r7 * r40;
    r74 = fma(r81, r3, r74);
    r54 = r54 * r74;
    r54 = r54 * r81;
    WriteSum2<double, double>((double *)inout_shared, r2, r54);
  };
  FlushSumShared<2, double>(out_focal_njtr, 0 * out_focal_njtr_num_alloc,
                            focal_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = r56 * r56;
    r35 = r35 * r35;
    r35 = 1.0 / r35;
    r56 = r56 * r35;
    r54 = r40 * r40;
    r54 = r35 * r54;
    WriteSum2<double, double>((double *)inout_shared, r56, r54);
  };
  FlushSumShared<2, double>(out_focal_precond_diag,
                            0 * out_focal_precond_diag_num_alloc,
                            focal_indices_loc, (double *)inout_shared);
}

void PinholeSplitFixedPrincipalPointFixedPointResJac(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *focal, unsigned int focal_num_alloc, SharedIndex *focal_indices,
    double *pixel, unsigned int pixel_num_alloc, double *principal_point,
    unsigned int principal_point_num_alloc, double *point,
    unsigned int point_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, double *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, double *out_focal_jac,
    unsigned int out_focal_jac_num_alloc, double *const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc, double *const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    double *const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPrincipalPointFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, focal, focal_num_alloc, focal_indices, pixel,
      pixel_num_alloc, principal_point, principal_point_num_alloc, point,
      point_num_alloc, out_res, out_res_num_alloc, out_pose_jac,
      out_pose_jac_num_alloc, out_pose_njtr, out_pose_njtr_num_alloc,
      out_pose_precond_diag, out_pose_precond_diag_num_alloc,
      out_pose_precond_tril, out_pose_precond_tril_num_alloc, out_focal_jac,
      out_focal_jac_num_alloc, out_focal_njtr, out_focal_njtr_num_alloc,
      out_focal_precond_diag, out_focal_precond_diag_num_alloc,
      out_focal_precond_tril, out_focal_precond_tril_num_alloc, problem_size);
}

} // namespace caspar