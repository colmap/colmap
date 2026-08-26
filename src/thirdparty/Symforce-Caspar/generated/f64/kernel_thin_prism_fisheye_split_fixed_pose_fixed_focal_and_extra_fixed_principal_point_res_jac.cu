#include "kernel_thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointResJacKernel(
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        double* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        double* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            0 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r2,
                                            r3);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            4 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r4,
                                            r5);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r6,
                                            r7);
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = 2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r11,
                                            r12);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r13, r14);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r15,
                                            r16);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r17, r18);
    r19 = fma(r15, r18, r12 * r13);
    r20 = r11 * r14;
    r21 = -1.00000000000000000e+00;
    r19 = fma(r21, r20, r19);
    r19 = fma(r16, r17, r19);
    r20 = r10 * r19;
    r22 = r15 * r17;
    r22 = fma(r21, r22, r12 * r14);
    r22 = fma(r16, r18, r22);
    r22 = fma(r11, r13, r22);
    r20 = r20 * r22;
    r23 = fma(r15, r14, r12 * r17);
    r24 = r16 * r13;
    r23 = fma(r21, r24, r23);
    r23 = fma(r11, r18, r23);
    r24 = r10 * r23;
    r25 = fma(r16, r14, r15 * r13);
    r25 = fma(r11, r17, r25);
    r25 = fma(r21, r25, r12 * r18);
    r24 = fma(r25, r24, r20);
    r7 = fma(r8, r24, r7);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r18, r26);
    r27 = r15 * r16;
    r27 = r27 * r10;
    r28 = r11 * r12;
    r28 = fma(r10, r28, r27);
    r29 = -2.00000000000000000e+00;
    r30 = r15 * r15;
    r30 = r29 * r30;
    r31 = 1.00000000000000000e+00;
    r32 = r11 * r11;
    r32 = fma(r29, r32, r31);
    r33 = r30 + r32;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r34);
    r35 = r16 * r11;
    r35 = r35 * r10;
    r36 = r15 * r12;
    r36 = fma(r29, r36, r35);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = r10 * r23;
    r38 = r38 * r22;
    r39 = r19 * r25;
    r40 = fma(r29, r39, r38);
    r41 = r23 * r23;
    r41 = r29 * r41;
    r42 = r31 + r41;
    r43 = r19 * r19;
    r43 = r43 * r29;
    r42 = r42 + r43;
    r7 = fma(r18, r28, r7);
    r7 = fma(r26, r33, r7);
    r7 = fma(r34, r36, r7);
    r7 = fma(r37, r40, r7);
    r7 = fma(r9, r42, r7);
    r36 = r7 * r7;
    r33 = 1.00000000000000008e-15;
    r28 = r22 * r22;
    r28 = r29 * r28;
    r44 = r31 + r28;
    r44 = r44 + r41;
    r6 = fma(r8, r44, r6);
    r41 = r23 * r29;
    r41 = fma(r25, r41, r20);
    r20 = r10 * r23;
    r20 = r20 * r19;
    r19 = r10 * r22;
    r19 = fma(r25, r19, r20);
    r45 = r15 * r11;
    r45 = r45 * r10;
    r46 = r16 * r12;
    r47 = fma(r10, r46, r45);
    r48 = r11 * r12;
    r48 = fma(r29, r48, r27);
    r27 = r16 * r16;
    r27 = r27 * r29;
    r32 = r27 + r32;
    r6 = fma(r9, r41, r6);
    r6 = fma(r37, r19, r6);
    r6 = fma(r34, r47, r6);
    r6 = fma(r26, r48, r6);
    r6 = fma(r18, r32, r6);
    r32 = r6 * r6;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r48);
    r47 = r29 * r22;
    r47 = fma(r25, r47, r20);
    r8 = fma(r8, r47, r48);
    r46 = fma(r29, r46, r45);
    r27 = r31 + r27;
    r27 = r27 + r30;
    r30 = r15 * r12;
    r30 = fma(r10, r30, r35);
    r39 = fma(r10, r39, r38);
    r28 = r31 + r28;
    r28 = r28 + r43;
    r8 = fma(r18, r46, r8);
    r8 = fma(r34, r27, r8);
    r8 = fma(r26, r30, r8);
    r8 = fma(r9, r39, r8);
    r8 = fma(r37, r28, r8);
    r37 = copysign(1.0, r8);
    r37 = fma(r33, r37, r8);
    r8 = r37 * r37;
    r9 = 1.0 / r8;
    r30 = r7 * r7;
    r30 = fma(r9, r30, r9 * r32);
    r32 = sqrt(r30);
    r26 = copysign(1.0, r32);
    r26 = fma(r33, r26, r32);
    r33 = r26 * r26;
    r27 = 1.0 / r33;
    r32 = atan(r32);
    r34 = r32 * r9;
    r46 = r32 * r34;
    r36 = r36 * r27;
    r36 = r36 * r46;
    r18 = 3.00000000000000000e+00;
    r43 = r18 * r46;
    r38 = r6 * r27;
    r35 = r6 * r38;
    r43 = fma(r35, r43, r36);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            8 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r45,
                                            r48);
    r20 = r46 * r35;
    r36 = r36 + r20;
    r43 = fma(r45, r36, r5 * r43);
    r25 = r4 * r7;
    r49 = r10 * r46;
    r25 = r25 * r38;
    r43 = fma(r49, r25, r43);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            2 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r50,
                                            r51);
    r52 = r36 * r36;
    r53 = fma(r51, r52, r50 * r36);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            6 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r54,
                                            r55);
    r56 = r36 * r52;
    r55 = r55 * r56;
    r53 = fma(r36, r55, r53);
    r53 = fma(r54, r56, r53);
    r56 = 1.0 / r37;
    r57 = 1.0 / r26;
    r58 = r56 * r57;
    r59 = r32 * r58;
    r60 = r53 * r59;
    r43 = fma(r6, r60, r43);
    r43 = fma(r6, r59, r43);
    r43 = fma(r2, r43, r0);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r0, r25);
    r43 = fma(r0, r21, r43);
    r0 = r7 * r7;
    r0 = r0 * r18;
    r0 = r0 * r27;
    r0 = fma(r46, r0, r20);
    r0 = fma(r48, r36, r4 * r0);
    r20 = r5 * r7;
    r20 = r20 * r38;
    r0 = fma(r49, r20, r0);
    r0 = fma(r7, r60, r0);
    r0 = fma(r7, r59, r0);
    r0 = fma(r3, r0, r1);
    r0 = fma(r25, r21, r0);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r43, r0);
    r25 = r3 * r21;
    r1 = r32 * r32;
    r20 = r27 * r1;
    r61 = r7 * r7;
    r8 = r37 * r8;
    r8 = 1.0 / r8;
    r37 = r29 * r8;
    r61 = r61 * r37;
    r20 = r20 * r61;
    r62 = r21 * r7;
    r63 = r10 * r44;
    r63 = r63 * r6;
    r63 = fma(r47, r61, r9 * r63);
    r64 = r47 * r6;
    r64 = r64 * r6;
    r63 = fma(r37, r64, r63);
    r65 = r10 * r24;
    r65 = r65 * r7;
    r63 = fma(r9, r65, r63);
    r33 = r26 * r33;
    r33 = 1.0 / r33;
    r26 = r33 * r46;
    r65 = rsqrt(r30);
    r64 = r7 * r65;
    r26 = r26 * r64;
    r62 = r62 * r63;
    r62 = fma(r26, r62, r47 * r20);
    r66 = r63 * r64;
    r30 = r31 + r30;
    r30 = 1.0 / r30;
    r31 = r30 * r34;
    r67 = r7 * r27;
    r66 = r66 * r31;
    r62 = fma(r67, r66, r62);
    r68 = r49 * r67;
    r62 = fma(r24, r68, r62);
    r66 = r47 * r37;
    r1 = r35 * r1;
    r69 = r63 * r65;
    r69 = r69 * r31;
    r69 = fma(r35, r69, r1 * r66);
    r66 = r44 * r38;
    r69 = fma(r49, r66, r69);
    r70 = r21 * r6;
    r70 = r70 * r6;
    r70 = r70 * r63;
    r70 = r70 * r65;
    r70 = r70 * r33;
    r69 = fma(r46, r70, r69);
    r70 = r62 + r69;
    r66 = r47 * r7;
    r71 = -6.00000000000000000e+00;
    r66 = r66 * r7;
    r66 = r66 * r32;
    r66 = r66 * r32;
    r66 = r66 * r71;
    r66 = r66 * r27;
    r72 = -3.00000000000000000e+00;
    r73 = r7 * r72;
    r73 = r73 * r63;
    r73 = fma(r26, r73, r8 * r66);
    r66 = r18 * r63;
    r66 = r66 * r64;
    r66 = r66 * r31;
    r73 = fma(r67, r66, r73);
    r74 = r24 * r7;
    r75 = 6.00000000000000000e+00;
    r74 = r74 * r75;
    r74 = r74 * r27;
    r73 = fma(r46, r74, r73);
    r73 = r73 + r69;
    r73 = fma(r4, r73, r48 * r70);
    r69 = r7 * r27;
    r74 = -5.00000000000000000e-01;
    r69 = r69 * r32;
    r69 = r69 * r56;
    r69 = r69 * r74;
    r69 = r69 * r65;
    r69 = r69 * r63;
    r66 = r51 * r10;
    r66 = r66 * r36;
    r66 = fma(r50, r70, r70 * r66);
    r76 = 4.00000000000000000e+00;
    r55 = r76 * r55;
    r54 = r54 * r18;
    r54 = r54 * r52;
    r66 = fma(r70, r55, r66);
    r66 = fma(r70, r54, r66);
    r52 = r7 * r66;
    r73 = fma(r59, r52, r73);
    r76 = 5.00000000000000000e-01;
    r77 = r76 * r63;
    r77 = r77 * r30;
    r77 = r77 * r64;
    r73 = fma(r58, r77, r73);
    r78 = r21 * r47;
    r78 = r78 * r57;
    r78 = r78 * r34;
    r79 = r5 * r63;
    r80 = r10 * r38;
    r80 = r80 * r64;
    r80 = r80 * r31;
    r73 = fma(r80, r79, r73);
    r81 = r5 * r29;
    r81 = r81 * r6;
    r81 = r81 * r63;
    r73 = fma(r26, r81, r73);
    r82 = r5 * r44;
    r73 = fma(r68, r82, r73);
    r83 = r76 * r53;
    r83 = r83 * r63;
    r83 = r83 * r30;
    r83 = r83 * r64;
    r73 = fma(r58, r83, r73);
    r84 = r7 * r53;
    r73 = fma(r78, r84, r73);
    r85 = r5 * r7;
    r86 = -4.00000000000000000e+00;
    r85 = r85 * r32;
    r85 = r85 * r32;
    r85 = r85 * r86;
    r85 = r85 * r8;
    r85 = r85 * r38;
    r87 = r5 * r24;
    r87 = r87 * r38;
    r73 = fma(r49, r87, r73);
    r73 = r73 + r69;
    r73 = fma(r24, r60, r73);
    r73 = fma(r7, r78, r73);
    r73 = fma(r24, r59, r73);
    r73 = fma(r53, r69, r73);
    r73 = fma(r47, r85, r73);
    r25 = r25 * r0;
    r87 = r2 * r21;
    r84 = r71 * r8;
    r84 = r84 * r1;
    r69 = r18 * r63;
    r69 = r69 * r65;
    r69 = r69 * r31;
    r69 = fma(r35, r69, r47 * r84);
    r83 = r44 * r75;
    r83 = r83 * r46;
    r69 = fma(r38, r83, r69);
    r82 = r6 * r6;
    r82 = r82 * r72;
    r82 = r82 * r63;
    r82 = r82 * r65;
    r82 = r82 * r33;
    r69 = fma(r46, r82, r69);
    r69 = r69 + r62;
    r69 = fma(r5, r69, r45 * r70);
    r70 = r6 * r76;
    r70 = r70 * r65;
    r70 = r70 * r30;
    r70 = r70 * r58;
    r62 = r53 * r70;
    r82 = r6 * r53;
    r69 = fma(r78, r82, r69);
    r83 = r4 * r63;
    r69 = fma(r80, r83, r69);
    r81 = r4 * r29;
    r81 = r81 * r6;
    r81 = r81 * r63;
    r69 = fma(r26, r81, r69);
    r79 = r4 * r68;
    r77 = r53 * r65;
    r52 = r32 * r74;
    r52 = r52 * r63;
    r52 = r52 * r56;
    r77 = r77 * r38;
    r69 = fma(r52, r77, r69);
    r88 = r6 * r66;
    r69 = fma(r59, r88, r69);
    r89 = r4 * r47;
    r89 = r89 * r7;
    r89 = r89 * r32;
    r89 = r89 * r32;
    r89 = r89 * r86;
    r89 = r89 * r8;
    r69 = fma(r38, r89, r69);
    r90 = r4 * r24;
    r90 = r90 * r38;
    r69 = fma(r49, r90, r69);
    r91 = r65 * r38;
    r69 = fma(r52, r91, r69);
    r69 = fma(r63, r62, r69);
    r69 = fma(r63, r70, r69);
    r69 = fma(r44, r79, r69);
    r69 = fma(r6, r78, r69);
    r69 = fma(r44, r59, r69);
    r69 = fma(r44, r60, r69);
    r87 = r87 * r43;
    r87 = fma(r69, r87, r73 * r25);
    r25 = r2 * r21;
    r91 = r41 * r75;
    r91 = r91 * r46;
    r91 = fma(r39, r84, r38 * r91);
    r90 = r6 * r6;
    r89 = r39 * r6;
    r89 = r89 * r6;
    r88 = r10 * r41;
    r88 = r88 * r6;
    r88 = fma(r9, r88, r37 * r89);
    r89 = r10 * r42;
    r89 = r89 * r7;
    r88 = fma(r9, r89, r88);
    r88 = fma(r39, r61, r88);
    r89 = r72 * r88;
    r90 = r90 * r65;
    r90 = r90 * r33;
    r90 = r90 * r46;
    r91 = fma(r89, r90, r91);
    r77 = r18 * r88;
    r77 = r77 * r65;
    r77 = r77 * r31;
    r91 = fma(r35, r77, r91);
    r78 = r88 * r64;
    r78 = r78 * r31;
    r78 = fma(r67, r78, r39 * r20);
    r81 = r21 * r7;
    r81 = r81 * r88;
    r78 = fma(r26, r81, r78);
    r78 = fma(r42, r68, r78);
    r91 = r91 + r78;
    r77 = r41 * r38;
    r90 = r39 * r37;
    r90 = fma(r1, r90, r49 * r77);
    r77 = r21 * r6;
    r77 = r77 * r6;
    r77 = r77 * r88;
    r77 = r77 * r65;
    r77 = r77 * r33;
    r90 = fma(r46, r77, r90);
    r81 = r88 * r65;
    r81 = r81 * r31;
    r90 = fma(r35, r81, r90);
    r78 = r78 + r90;
    r91 = fma(r45, r78, r5 * r91);
    r81 = r21 * r39;
    r81 = r81 * r6;
    r81 = r81 * r53;
    r81 = r81 * r57;
    r91 = fma(r34, r81, r91);
    r77 = r51 * r10;
    r77 = r77 * r36;
    r77 = fma(r78, r77, r50 * r78);
    r77 = fma(r78, r54, r77);
    r77 = fma(r78, r55, r77);
    r83 = r6 * r77;
    r91 = fma(r59, r83, r91);
    r82 = r21 * r39;
    r82 = r82 * r6;
    r82 = r82 * r57;
    r91 = fma(r34, r82, r91);
    r52 = r4 * r42;
    r52 = r52 * r38;
    r91 = fma(r49, r52, r91);
    r92 = r32 * r74;
    r92 = r92 * r88;
    r92 = r92 * r56;
    r92 = r92 * r65;
    r91 = fma(r38, r92, r91);
    r93 = r4 * r88;
    r91 = fma(r80, r93, r91);
    r94 = r4 * r29;
    r94 = r94 * r6;
    r94 = r94 * r88;
    r91 = fma(r26, r94, r91);
    r95 = r4 * r39;
    r95 = r95 * r7;
    r95 = r95 * r32;
    r95 = r95 * r32;
    r95 = r95 * r86;
    r95 = r95 * r8;
    r91 = fma(r38, r95, r91);
    r96 = r32 * r53;
    r96 = r96 * r74;
    r96 = r96 * r88;
    r96 = r96 * r56;
    r96 = r96 * r65;
    r91 = fma(r38, r96, r91);
    r91 = fma(r41, r60, r91);
    r91 = fma(r88, r70, r91);
    r91 = fma(r88, r62, r91);
    r91 = fma(r41, r59, r91);
    r91 = fma(r41, r79, r91);
    r25 = r25 * r43;
    r96 = r3 * r21;
    r95 = r39 * r7;
    r95 = r95 * r7;
    r95 = r95 * r32;
    r95 = r95 * r32;
    r95 = r95 * r71;
    r95 = r95 * r27;
    r94 = r18 * r88;
    r94 = r94 * r64;
    r94 = r94 * r31;
    r94 = fma(r67, r94, r8 * r95);
    r95 = r42 * r7;
    r95 = r95 * r75;
    r95 = r95 * r27;
    r94 = fma(r46, r95, r94);
    r93 = r7 * r26;
    r94 = fma(r89, r93, r94);
    r94 = r94 + r90;
    r78 = fma(r48, r78, r4 * r94);
    r94 = r21 * r39;
    r94 = r94 * r7;
    r94 = r94 * r57;
    r78 = fma(r34, r94, r78);
    r90 = r21 * r39;
    r90 = r90 * r7;
    r90 = r90 * r53;
    r90 = r90 * r57;
    r78 = fma(r34, r90, r78);
    r93 = r76 * r88;
    r93 = r93 * r30;
    r93 = r93 * r64;
    r78 = fma(r58, r93, r78);
    r95 = r5 * r42;
    r95 = r95 * r38;
    r78 = fma(r49, r95, r78);
    r89 = r32 * r74;
    r89 = r89 * r88;
    r89 = r89 * r27;
    r89 = r89 * r56;
    r78 = fma(r64, r89, r78);
    r92 = r5 * r41;
    r78 = fma(r68, r92, r78);
    r52 = r5 * r88;
    r78 = fma(r80, r52, r78);
    r82 = r5 * r29;
    r82 = r82 * r6;
    r82 = r82 * r88;
    r78 = fma(r26, r82, r78);
    r83 = r76 * r53;
    r83 = r83 * r88;
    r83 = r83 * r30;
    r83 = r83 * r64;
    r78 = fma(r58, r83, r78);
    r81 = r7 * r77;
    r78 = fma(r59, r81, r78);
    r97 = r32 * r53;
    r97 = r97 * r74;
    r97 = r97 * r88;
    r97 = r97 * r27;
    r97 = r97 * r56;
    r78 = fma(r64, r97, r78);
    r78 = fma(r42, r60, r78);
    r78 = fma(r39, r85, r78);
    r78 = fma(r42, r59, r78);
    r96 = r96 * r0;
    r96 = fma(r78, r96, r91 * r25);
    WriteSum2<double, double>((double*)inout_shared, r87, r96);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r96 = r3 * r21;
    r87 = r21 * r7;
    r25 = r10 * r19;
    r25 = r25 * r6;
    r61 = fma(r28, r61, r9 * r25);
    r25 = r28 * r6;
    r25 = r25 * r6;
    r61 = fma(r37, r25, r61);
    r97 = r10 * r40;
    r97 = r97 * r7;
    r61 = fma(r9, r97, r61);
    r97 = r61 * r26;
    r87 = fma(r97, r87, r40 * r68);
    r25 = r61 * r64;
    r25 = r25 * r31;
    r87 = fma(r67, r25, r87);
    r87 = fma(r28, r20, r87);
    r20 = r28 * r37;
    r25 = r21 * r6;
    r25 = r25 * r6;
    r25 = r25 * r61;
    r25 = r25 * r65;
    r25 = r25 * r33;
    r25 = fma(r46, r25, r1 * r20);
    r20 = r61 * r65;
    r20 = r20 * r31;
    r25 = fma(r35, r20, r25);
    r1 = r19 * r38;
    r25 = fma(r49, r1, r25);
    r1 = r87 + r25;
    r20 = r40 * r7;
    r20 = r20 * r75;
    r20 = r20 * r27;
    r9 = r7 * r72;
    r9 = fma(r97, r9, r46 * r20);
    r20 = r18 * r61;
    r20 = r20 * r64;
    r20 = r20 * r31;
    r9 = fma(r67, r20, r9);
    r67 = r28 * r7;
    r67 = r67 * r7;
    r67 = r67 * r32;
    r67 = r67 * r32;
    r67 = r67 * r71;
    r67 = r67 * r27;
    r9 = fma(r8, r67, r9);
    r9 = r9 + r25;
    r9 = fma(r4, r9, r48 * r1);
    r48 = r5 * r40;
    r48 = r48 * r38;
    r9 = fma(r49, r48, r9);
    r25 = r29 * r6;
    r25 = r25 * r97;
    r97 = r32 * r74;
    r97 = r97 * r61;
    r97 = r97 * r27;
    r97 = r97 * r56;
    r9 = fma(r64, r97, r9);
    r67 = r5 * r61;
    r9 = fma(r80, r67, r9);
    r20 = r32 * r53;
    r20 = r20 * r74;
    r20 = r20 * r61;
    r20 = r20 * r27;
    r20 = r20 * r56;
    r9 = fma(r64, r20, r9);
    r27 = r76 * r53;
    r27 = r27 * r61;
    r27 = r27 * r30;
    r27 = r27 * r64;
    r9 = fma(r58, r27, r9);
    r71 = r21 * r28;
    r71 = r71 * r7;
    r71 = r71 * r53;
    r71 = r71 * r57;
    r9 = fma(r34, r71, r9);
    r81 = r5 * r19;
    r9 = fma(r68, r81, r9);
    r68 = r51 * r10;
    r68 = r68 * r36;
    r68 = fma(r1, r68, r50 * r1);
    r68 = fma(r1, r55, r68);
    r68 = fma(r1, r54, r68);
    r54 = r7 * r68;
    r9 = fma(r59, r54, r9);
    r55 = r76 * r61;
    r55 = r55 * r30;
    r55 = r55 * r64;
    r9 = fma(r58, r55, r9);
    r58 = r21 * r28;
    r58 = r58 * r7;
    r58 = r58 * r57;
    r9 = fma(r34, r58, r9);
    r9 = fma(r5, r25, r9);
    r9 = fma(r28, r85, r9);
    r9 = fma(r40, r60, r9);
    r9 = fma(r40, r59, r9);
    r96 = r96 * r0;
    r0 = r2 * r21;
    r58 = r6 * r6;
    r58 = r58 * r72;
    r58 = r58 * r61;
    r58 = r58 * r65;
    r58 = r58 * r33;
    r58 = fma(r46, r58, r28 * r84);
    r84 = r18 * r61;
    r84 = r84 * r65;
    r84 = r84 * r31;
    r58 = fma(r35, r84, r58);
    r35 = r19 * r75;
    r35 = r35 * r46;
    r58 = fma(r38, r35, r58);
    r58 = r58 + r87;
    r1 = fma(r45, r1, r5 * r58);
    r45 = r21 * r28;
    r45 = r45 * r6;
    r45 = r45 * r57;
    r1 = fma(r34, r45, r1);
    r58 = r4 * r40;
    r58 = r58 * r38;
    r1 = fma(r49, r58, r1);
    r49 = r32 * r74;
    r49 = r49 * r61;
    r49 = r49 * r56;
    r49 = r49 * r65;
    r1 = fma(r38, r49, r1);
    r87 = r4 * r61;
    r1 = fma(r80, r87, r1);
    r80 = r4 * r28;
    r80 = r80 * r7;
    r80 = r80 * r32;
    r80 = r80 * r32;
    r80 = r80 * r86;
    r80 = r80 * r8;
    r1 = fma(r38, r80, r1);
    r8 = r32 * r53;
    r8 = r8 * r74;
    r8 = r8 * r61;
    r8 = r8 * r56;
    r8 = r8 * r65;
    r1 = fma(r38, r8, r1);
    r56 = r21 * r28;
    r56 = r56 * r6;
    r56 = r56 * r53;
    r56 = r56 * r57;
    r1 = fma(r34, r56, r1);
    r34 = r6 * r68;
    r1 = fma(r59, r34, r1);
    r1 = fma(r19, r60, r1);
    r1 = fma(r4, r25, r1);
    r1 = fma(r61, r70, r1);
    r1 = fma(r61, r62, r1);
    r1 = fma(r19, r59, r1);
    r1 = fma(r19, r79, r1);
    r0 = r0 * r43;
    r0 = fma(r1, r0, r9 * r96);
    WriteSum1<double, double>((double*)inout_shared, r0);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r3 * r3;
    r96 = r73 * r0;
    r43 = r2 * r2;
    r34 = r69 * r69;
    r34 = fma(r43, r34, r73 * r96);
    r73 = r78 * r78;
    r56 = r91 * r91;
    r56 = fma(r43, r56, r0 * r73);
    WriteSum2<double, double>((double*)inout_shared, r34, r56);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = r9 * r9;
    r34 = r1 * r43;
    r1 = fma(r1, r34, r0 * r56);
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r69 * r91;
    r1 = fma(r43, r1, r78 * r96);
    r96 = fma(r69, r34, r9 * r96);
    WriteSum2<double, double>((double*)inout_shared, r1, r96);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r96 = r78 * r9;
    r34 = fma(r91, r34, r0 * r96);
    WriteSum1<double, double>((double*)inout_shared, r34);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointResJac(
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    double* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointResJacKernel<<<
      n_blocks,
      1024>>>(sensor_from_rig,
              sensor_from_rig_num_alloc,
              point,
              point_num_alloc,
              point_indices,
              pixel,
              pixel_num_alloc,
              pose,
              pose_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              principal_point,
              principal_point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_point_njtr,
              out_point_njtr_num_alloc,
              out_point_precond_diag,
              out_point_precond_diag_num_alloc,
              out_point_precond_tril,
              out_point_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar