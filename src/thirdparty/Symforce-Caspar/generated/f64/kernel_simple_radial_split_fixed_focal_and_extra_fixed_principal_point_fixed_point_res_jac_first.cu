#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacFirstKernel(
        double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        double *pixel, unsigned int pixel_num_alloc, double *focal_and_extra,
        unsigned int focal_and_extra_num_alloc, double *principal_point,
        unsigned int principal_point_num_alloc, double *point,
        unsigned int point_num_alloc, double *out_res,
        unsigned int out_res_num_alloc, double *const out_rTr,
        double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
        double *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
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
         r81 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx, r0, r1);
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
  };
  LoadShared<2, double, double>(pose, 2 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r11, r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r13, r14);
  };
  LoadShared<2, double, double>(pose, 0 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r15, r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r17, r18);
    r19 = fma(r16, r17, r11 * r14);
    r20 = r15 * r18;
    r19 = fma(r4, r20, r19);
    r19 = fma(r12, r13, r19);
    r20 = r19 * r19;
    r20 = r10 * r20;
    r21 = 1.00000000000000000e+00;
    r22 = r11 * r17;
    r22 = fma(r4, r22, r16 * r14);
    r22 = fma(r12, r18, r22);
    r22 = fma(r15, r13, r22);
    r23 = r22 * r22;
    r23 = fma(r10, r23, r21);
    r24 = r20 + r23;
    r24 = fma(r8, r24, r6);
    r25 = 2.00000000000000000e+00;
    r26 = fma(r12, r17, r15 * r14);
    r27 = r16 * r13;
    r26 = fma(r4, r27, r26);
    r26 = fma(r11, r18, r26);
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
  };
  LoadShared<1, double, double>(pose, 6 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r35);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r36 = r17 * r13;
    r36 = r36 * r25;
    r37 = r18 * r14;
    r38 = fma(r25, r37, r36);
  };
  LoadShared<2, double, double>(pose, 4 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r39, r40);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r41 = r13 * r14;
    r42 = r17 * r18;
    r42 = r42 * r25;
    r41 = fma(r10, r41, r42);
    r43 = r18 * r18;
    r43 = r43 * r10;
    r44 = r21 + r43;
    r45 = r13 * r13;
    r45 = r10 * r45;
    r44 = r44 + r45;
    r24 = fma(r9, r31, r24);
    r24 = fma(r32, r33, r24);
    r24 = fma(r35, r38, r24);
    r24 = fma(r40, r41, r24);
    r24 = fma(r39, r44, r24);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            0 * focal_and_extra_num_alloc,
                                            global_thread_idx, r44, r41);
    r38 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r33);
    r34 = fma(r22, r30, r34);
    r34 = fma(r8, r34, r33);
    r37 = fma(r10, r37, r36);
    r43 = r21 + r43;
    r36 = r17 * r17;
    r36 = r10 * r36;
    r43 = r43 + r36;
    r31 = r18 * r13;
    r31 = r31 * r25;
    r46 = r17 * r14;
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
    r28 = r13 * r14;
    r28 = fma(r25, r28, r42);
    r45 = r21 + r45;
    r45 = r45 + r36;
    r36 = r17 * r14;
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
    r4 = fma(r4, r4, r5 * r5);
  };
  SumStore<double>(out_rTr_local, (double *)inout_shared, 0,
                   global_thread_idx < problem_size, r4);
  if (global_thread_idx < problem_size) {
    r4 = -1.00000000000000000e+00;
    r3 = fma(r3, r4, r1);
    r1 = 2.00000000000000000e+00;
    r5 = r15 * r14;
    r20 = fma(r12, r17, r5);
    r27 = r11 * r18;
    r24 = r16 * r13;
    r20 = fma(r4, r24, r20);
    r20 = r20 + r27;
    r24 = r1 * r20;
    r23 = r11 * r17;
    r23 = fma(r4, r23, r16 * r14);
    r23 = fma(r12, r18, r23);
    r23 = fma(r15, r13, r23);
    r24 = r24 * r23;
    r21 = fma(r16, r17, r11 * r14);
    r38 = r15 * r18;
    r21 = fma(r4, r38, r21);
    r21 = fma(r12, r13, r21);
    r38 = fma(r16, r18, r15 * r17);
    r38 = fma(r11, r13, r38);
    r38 = fma(r4, r38, r12 * r14);
    r34 = r1 * r38;
    r30 = fma(r21, r34, r24);
    r30 = fma(r8, r30, r7);
    r7 = r13 * r14;
    r36 = r17 * r18;
    r36 = r36 * r1;
    r7 = fma(r1, r7, r36);
    r45 = -2.00000000000000000e+00;
    r28 = r13 * r13;
    r28 = r45 * r28;
    r48 = 1.00000000000000000e+00;
    r26 = r17 * r17;
    r26 = fma(r45, r26, r48);
    r47 = r28 + r26;
    r31 = r18 * r13;
    r31 = r31 * r1;
    r10 = r17 * r14;
    r10 = fma(r45, r10, r31);
    r42 = r1 * r21;
    r42 = r42 * r23;
    r29 = r20 * r45;
    r29 = fma(r38, r29, r42);
    r46 = r20 * r20;
    r46 = r46 * r45;
    r43 = r48 + r46;
    r37 = r21 * r21;
    r37 = r37 * r45;
    r43 = r43 + r37;
    r30 = fma(r39, r7, r30);
    r30 = fma(r40, r47, r30);
    r30 = fma(r35, r10, r30);
    r30 = fma(r32, r29, r30);
    r30 = fma(r9, r43, r30);
    r43 = 1.00000000000000008e-15;
    r29 = r1 * r21;
    r29 = r29 * r20;
    r49 = r45 * r23;
    r49 = fma(r38, r49, r29);
    r49 = fma(r8, r49, r33);
    r33 = r17 * r13;
    r33 = r33 * r1;
    r50 = r18 * r14;
    r50 = fma(r45, r50, r33);
    r51 = r18 * r18;
    r51 = r51 * r45;
    r26 = r51 + r26;
    r52 = r17 * r14;
    r52 = fma(r1, r52, r31);
    r42 = fma(r20, r34, r42);
    r46 = r48 + r46;
    r31 = r45 * r23;
    r31 = r31 * r23;
    r46 = r46 + r31;
    r49 = fma(r39, r50, r49);
    r49 = fma(r35, r26, r49);
    r49 = fma(r40, r52, r49);
    r49 = fma(r9, r42, r49);
    r49 = fma(r32, r46, r49);
    r46 = copysign(1.0, r49);
    r46 = fma(r43, r46, r49);
    r43 = 1.0 / r46;
    r37 = r48 + r37;
    r37 = r37 + r31;
    r37 = fma(r8, r37, r6);
    r6 = r21 * r45;
    r6 = fma(r38, r6, r24);
    r29 = fma(r23, r34, r29);
    r24 = r18 * r14;
    r24 = fma(r1, r24, r33);
    r33 = r13 * r14;
    r33 = fma(r45, r33, r36);
    r28 = r48 + r28;
    r28 = r28 + r51;
    r37 = fma(r9, r6, r37);
    r37 = fma(r32, r29, r37);
    r37 = fma(r35, r24, r37);
    r37 = fma(r40, r33, r37);
    r37 = fma(r39, r28, r37);
    r39 = r46 * r46;
    r40 = 1.0 / r39;
    r35 = r37 * r40;
    r29 = r30 * r30;
    r6 = fma(r40, r29, r37 * r35);
    r6 = fma(r41, r6, r48);
    r6 = r44 * r6;
    r48 = r43 * r6;
    r3 = fma(r30, r48, r3);
    r51 = r4 * r3;
    r36 = r44 * r41;
    r31 = r12 * r17;
    r49 = -5.00000000000000000e-01;
    r42 = r16 * r13;
    r53 = 5.00000000000000000e-01;
    r42 = fma(r53, r42, r49 * r31);
    r42 = fma(r49, r5, r42);
    r42 = fma(r49, r27, r42);
    r31 = r23 * r42;
    r54 = r11 * r14;
    r55 = r16 * r17;
    r55 = fma(r53, r55, r53 * r54);
    r54 = r15 * r18;
    r55 = fma(r49, r54, r55);
    r56 = r12 * r53;
    r55 = fma(r13, r56, r55);
    r54 = fma(r55, r34, r1 * r31);
    r57 = r1 * r20;
    r58 = r11 * r17;
    r59 = r12 * r18;
    r59 = fma(r49, r59, r53 * r58);
    r58 = r15 * r13;
    r59 = fma(r49, r58, r59);
    r60 = r16 * r49;
    r59 = fma(r14, r60, r59);
    r58 = r1 * r21;
    r61 = r15 * r17;
    r62 = r11 * r13;
    r62 = fma(r49, r62, r49 * r61);
    r62 = fma(r14, r56, r62);
    r62 = fma(r18, r60, r62);
    r58 = r58 * r62;
    r57 = fma(r59, r57, r58);
    r54 = r54 + r57;
    r61 = r1 * r23;
    r61 = r61 * r62;
    r63 = r1 * r20;
    r63 = r63 * r55;
    r64 = r61 + r63;
    r65 = r21 * r45;
    r64 = fma(r42, r65, r64);
    r66 = r45 * r38;
    r64 = fma(r59, r66, r64);
    r64 = fma(r9, r64, r32 * r54);
    r54 = r23 * r55;
    r66 = -4.00000000000000000e+00;
    r54 = r54 * r66;
    r65 = r21 * r59;
    r67 = r66 * r65;
    r68 = r54 + r67;
    r64 = fma(r8, r68, r64);
    r68 = r1 * r64;
    r69 = r1 * r23;
    r69 = r69 * r59;
    r70 = r1 * r21;
    r70 = fma(r55, r70, r69);
    r71 = r1 * r20;
    r71 = r71 * r42;
    r72 = r62 * r34;
    r73 = r71 + r72;
    r74 = r70 + r73;
    r75 = r45 * r38;
    r75 = fma(r45, r31, r55 * r75);
    r75 = r75 + r57;
    r75 = fma(r8, r75, r9 * r74);
    r74 = r20 * r66;
    r55 = r62 * r74;
    r54 = r54 + r55;
    r75 = fma(r32, r54, r75);
    r54 = r37 * r37;
    r39 = r46 * r39;
    r39 = 1.0 / r39;
    r39 = r45 * r39;
    r54 = r54 * r39;
    r68 = fma(r75, r54, r35 * r68);
    r46 = r75 * r39;
    r68 = fma(r29, r46, r68);
    r76 = r1 * r30;
    r77 = r20 * r45;
    r78 = r45 * r38;
    r78 = r78 * r62;
    r77 = fma(r42, r77, r78);
    r77 = r77 + r70;
    r55 = r67 + r55;
    r55 = fma(r9, r55, r32 * r77);
    r63 = fma(r59, r34, r63);
    r77 = r1 * r21;
    r77 = fma(r42, r77, r61);
    r63 = r63 + r77;
    r55 = fma(r8, r63, r55);
    r76 = r76 * r55;
    r68 = fma(r40, r76, r68);
    r36 = r36 * r68;
    r36 = r36 * r43;
    r55 = fma(r55, r48, r30 * r36);
    r68 = r4 * r30;
    r68 = r68 * r75;
    r68 = r68 * r40;
    r55 = fma(r6, r68, r55);
    r2 = fma(r2, r4, r0);
    r2 = fma(r37, r48, r2);
    r0 = r4 * r2;
    r36 = fma(r64, r48, r37 * r36);
    r68 = r4 * r6;
    r68 = r68 * r35;
    r36 = fma(r75, r68, r36);
    r0 = fma(r36, r0, r55 * r51);
    r51 = r4 * r2;
    r76 = r44 * r41;
    r72 = r69 + r72;
    r69 = r1 * r21;
    r46 = r11 * r14;
    r63 = r15 * r18;
    r63 = fma(r53, r63, r49 * r46);
    r46 = r12 * r13;
    r63 = fma(r49, r46, r63);
    r63 = fma(r17, r60, r63);
    r69 = r69 * r63;
    r46 = r1 * r20;
    r5 = fma(r17, r56, r53 * r5);
    r5 = fma(r53, r27, r5);
    r5 = fma(r13, r60, r5);
    r46 = fma(r5, r46, r69);
    r72 = r72 + r46;
    r60 = r23 * r62;
    r60 = r60 * r66;
    r27 = r21 * r66;
    r27 = r27 * r5;
    r61 = r60 + r27;
    r61 = fma(r8, r61, r32 * r72);
    r72 = r45 * r38;
    r72 = fma(r45, r65, r5 * r72);
    r67 = r1 * r20;
    r67 = r67 * r62;
    r70 = r1 * r23;
    r70 = fma(r63, r70, r67);
    r72 = r72 + r70;
    r61 = fma(r9, r72, r61);
    r72 = r1 * r61;
    r79 = r1 * r30;
    r80 = r20 * r45;
    r80 = fma(r59, r80, r58);
    r58 = r1 * r23;
    r58 = r58 * r5;
    r81 = r45 * r38;
    r80 = fma(r63, r81, r80);
    r80 = r80 + r58;
    r5 = fma(r5, r34, r1 * r65);
    r5 = r5 + r70;
    r5 = fma(r8, r5, r32 * r80);
    r80 = r63 * r74;
    r27 = r27 + r80;
    r5 = fma(r9, r27, r5);
    r79 = r79 * r5;
    r79 = fma(r40, r79, r35 * r72);
    r72 = r45 * r23;
    r72 = fma(r59, r72, r78);
    r72 = r72 + r46;
    r58 = fma(r63, r34, r58);
    r58 = r58 + r57;
    r58 = fma(r9, r58, r8 * r72);
    r80 = r60 + r80;
    r58 = fma(r32, r80, r58);
    r80 = r58 * r39;
    r79 = fma(r29, r80, r79);
    r79 = fma(r58, r54, r79);
    r76 = r76 * r37;
    r76 = r76 * r79;
    r76 = fma(r58, r68, r43 * r76);
    r76 = fma(r61, r48, r76);
    r80 = r4 * r3;
    r60 = r44 * r41;
    r60 = r60 * r30;
    r60 = r60 * r79;
    r60 = fma(r43, r60, r5 * r48);
    r5 = r4 * r30;
    r5 = r5 * r58;
    r5 = r5 * r40;
    r60 = fma(r6, r5, r60);
    r80 = fma(r60, r80, r76 * r51);
    WriteSum2<double, double>((double *)inout_shared, r0, r80);
  };
  FlushSumShared<2, double>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = r4 * r3;
    r0 = r44 * r41;
    r51 = r1 * r30;
    r5 = r1 * r23;
    r79 = r16 * r14;
    r72 = r11 * r17;
    r72 = fma(r49, r72, r53 * r79);
    r79 = r15 * r13;
    r72 = fma(r53, r79, r72);
    r72 = fma(r18, r56, r72);
    r5 = r5 * r72;
    r69 = r69 + r5;
    r69 = r69 + r73;
    r73 = r20 * r45;
    r56 = r45 * r38;
    r56 = fma(r72, r56, r63 * r73);
    r56 = r56 + r77;
    r56 = fma(r32, r56, r8 * r69);
    r62 = r21 * r62;
    r62 = r62 * r66;
    r74 = r72 * r74;
    r69 = r62 + r74;
    r56 = fma(r9, r69, r56);
    r51 = r51 * r56;
    r31 = r66 * r31;
    r74 = r74 + r31;
    r66 = r1 * r21;
    r66 = r66 * r72;
    r67 = r67 + r66;
    r69 = r45 * r23;
    r67 = fma(r63, r69, r67);
    r73 = r45 * r38;
    r67 = fma(r42, r73, r67);
    r67 = fma(r8, r67, r32 * r74);
    r74 = r1 * r20;
    r72 = fma(r72, r34, r63 * r74);
    r72 = r72 + r77;
    r67 = fma(r9, r72, r67);
    r51 = fma(r67, r54, r40 * r51);
    r78 = r71 + r78;
    r71 = r21 * r45;
    r78 = fma(r63, r71, r78);
    r78 = r78 + r5;
    r31 = r62 + r31;
    r31 = fma(r8, r31, r9 * r78);
    r34 = fma(r42, r34, r66);
    r34 = r34 + r70;
    r31 = fma(r32, r34, r31);
    r34 = r1 * r31;
    r51 = fma(r35, r34, r51);
    r32 = r67 * r39;
    r51 = fma(r29, r32, r51);
    r0 = r0 * r30;
    r0 = r0 * r51;
    r32 = r4 * r30;
    r32 = r32 * r67;
    r32 = r32 * r40;
    r32 = fma(r6, r32, r43 * r0);
    r32 = fma(r56, r48, r32);
    r56 = r4 * r2;
    r0 = fma(r31, r48, r67 * r68);
    r34 = r44 * r41;
    r34 = r34 * r37;
    r34 = r34 * r51;
    r0 = fma(r43, r34, r0);
    r56 = fma(r0, r56, r32 * r80);
    r80 = r4 * r2;
    r34 = r44 * r41;
    r51 = r1 * r7;
    r51 = r51 * r30;
    r70 = r50 * r39;
    r70 = fma(r29, r70, r40 * r51);
    r51 = r1 * r28;
    r70 = fma(r35, r51, r70);
    r70 = fma(r50, r54, r70);
    r34 = r34 * r37;
    r34 = r34 * r70;
    r34 = fma(r43, r34, r50 * r68);
    r34 = fma(r28, r48, r34);
    r51 = r4 * r3;
    r42 = r44 * r41;
    r42 = r42 * r30;
    r42 = r42 * r70;
    r42 = fma(r43, r42, r7 * r48);
    r70 = r4 * r50;
    r70 = r70 * r30;
    r70 = r70 * r40;
    r42 = fma(r6, r70, r42);
    r51 = fma(r42, r51, r34 * r80);
    WriteSum2<double, double>((double *)inout_shared, r56, r51);
  };
  FlushSumShared<2, double>(out_pose_njtr, 2 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r4 * r3;
    r56 = r44 * r41;
    r80 = r52 * r39;
    r70 = r1 * r47;
    r70 = r70 * r30;
    r70 = fma(r40, r70, r29 * r80);
    r80 = r1 * r33;
    r70 = fma(r35, r80, r70);
    r70 = fma(r52, r54, r70);
    r56 = r56 * r30;
    r56 = r56 * r70;
    r56 = fma(r47, r48, r43 * r56);
    r80 = r4 * r52;
    r80 = r80 * r30;
    r80 = r80 * r40;
    r56 = fma(r6, r80, r56);
    r80 = r4 * r2;
    r66 = fma(r52, r68, r33 * r48);
    r8 = r44 * r41;
    r8 = r8 * r37;
    r8 = r8 * r70;
    r66 = fma(r43, r8, r66);
    r80 = fma(r66, r80, r56 * r51);
    r51 = r4 * r2;
    r8 = r44 * r41;
    r70 = r1 * r10;
    r70 = r70 * r30;
    r78 = r26 * r39;
    r78 = fma(r29, r78, r40 * r70);
    r70 = r1 * r24;
    r78 = fma(r35, r70, r78);
    r78 = fma(r26, r54, r78);
    r8 = r8 * r37;
    r8 = r8 * r78;
    r8 = fma(r24, r48, r43 * r8);
    r8 = fma(r26, r68, r8);
    r68 = r4 * r3;
    r37 = r4 * r26;
    r37 = r37 * r30;
    r37 = r37 * r40;
    r40 = r44 * r41;
    r40 = r40 * r30;
    r40 = r40 * r78;
    r40 = fma(r43, r40, r6 * r37);
    r40 = fma(r10, r48, r40);
    r68 = fma(r40, r68, r8 * r51);
    WriteSum2<double, double>((double *)inout_shared, r80, r68);
  };
  FlushSumShared<2, double>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fma(r36, r36, r55 * r55);
    r80 = fma(r60, r60, r76 * r76);
    WriteSum2<double, double>((double *)inout_shared, r68, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r0, r0, r32 * r32);
    r68 = fma(r34, r34, r42 * r42);
    WriteSum2<double, double>((double *)inout_shared, r80, r68);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fma(r66, r66, r56 * r56);
    r80 = fma(r8, r8, r40 * r40);
    WriteSum2<double, double>((double *)inout_shared, r68, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r36, r76, r55 * r60);
    r68 = fma(r55, r32, r36 * r0);
    WriteSum2<double, double>((double *)inout_shared, r80, r68);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fma(r55, r42, r36 * r34);
    r80 = fma(r36, r66, r55 * r56);
    WriteSum2<double, double>((double *)inout_shared, r68, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fma(r36, r8, r55 * r40);
    r55 = fma(r60, r32, r76 * r0);
    WriteSum2<double, double>((double *)inout_shared, r36, r55);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fma(r60, r42, r76 * r34);
    r36 = fma(r60, r56, r76 * r66);
    WriteSum2<double, double>((double *)inout_shared, r55, r36);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r60, r40, r76 * r8);
    r76 = fma(r0, r34, r32 * r42);
    WriteSum2<double, double>((double *)inout_shared, r60, r76);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fma(r0, r66, r32 * r56);
    r32 = fma(r32, r40, r0 * r8);
    WriteSum2<double, double>((double *)inout_shared, r76, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r42, r56, r34 * r66);
    r42 = fma(r42, r40, r34 * r8);
    WriteSum2<double, double>((double *)inout_shared, r32, r42);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fma(r56, r40, r66 * r8);
    WriteSum1<double, double>((double *)inout_shared, r40);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacFirst(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *pixel, unsigned int pixel_num_alloc, double *focal_and_extra,
    unsigned int focal_and_extra_num_alloc, double *principal_point,
    unsigned int principal_point_num_alloc, double *point,
    unsigned int point_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr,
    double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
    double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacFirstKernel<<<
      n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, pixel, pixel_num_alloc, focal_and_extra,
      focal_and_extra_num_alloc, principal_point, principal_point_num_alloc,
      point, point_num_alloc, out_res, out_res_num_alloc, out_rTr,
      out_pose_njtr, out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, problem_size);
}

} // namespace caspar