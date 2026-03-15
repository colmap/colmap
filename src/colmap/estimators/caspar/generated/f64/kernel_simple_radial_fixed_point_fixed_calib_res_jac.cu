#include "kernel_simple_radial_fixed_point_fixed_calib_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_point_fixed_calib_res_jac_kernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* calib,
        unsigned int calib_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        calib, 0 * calib_num_alloc, global_thread_idx, r0, r1);
    read_idx_2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r1);
  };
  load_shared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r1, r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r6, r7);
  };
  load_shared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = r8 * r9;
    r11 = 2.00000000000000000e+00;
    r10 = r10 * r11;
  };
  load_shared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = -2.00000000000000000e+00;
    r15 = r12 * r14;
    r16 = r13 * r15;
    r17 = r10 + r16;
    r1 = fma(r7, r17, r1);
    read_idx_1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r18);
    r19 = r8 * r12;
    r19 = r19 * r11;
    r20 = r9 * r13;
    r20 = r20 * r11;
    r21 = r19 + r20;
    r22 = r12 * r15;
    r23 = 1.00000000000000000e+00;
    r24 = r9 * r9;
    r25 = fma(r14, r24, r23);
    r26 = r22 + r25;
    r1 = fma(r18, r21, r1);
    r1 = fma(r6, r26, r1);
    r26 = r0 * r1;
    read_idx_2<1024, double, double, double2>(
        calib, 2 * calib_num_alloc, global_thread_idx, r27, r28);
    r29 = r1 * r1;
    r30 = 1.00000000000000008e-15;
  };
  load_shared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r32 = r9 * r12;
    r32 = r32 * r11;
    r33 = r8 * r13;
    r33 = r33 * r11;
    r34 = r32 + r33;
    r31 = fma(r7, r34, r31);
    r35 = r9 * r13;
    r35 = r35 * r14;
    r19 = r19 + r35;
    r36 = r8 * r8;
    r37 = r14 * r36;
    r25 = r37 + r25;
    r31 = fma(r6, r19, r31);
    r31 = fma(r18, r25, r31);
    r25 = copysign(1.0, r31);
    r25 = fma(r30, r25, r31);
    r30 = r25 * r25;
    r31 = 1.0 / r30;
    r38 = r12 * r13;
    r38 = r38 * r11;
    r10 = r10 + r38;
    r5 = fma(r6, r10, r5);
    r39 = r8 * r13;
    r39 = r39 * r14;
    r32 = r32 + r39;
    r22 = r23 + r22;
    r22 = r22 + r37;
    r5 = fma(r18, r32, r5);
    r5 = fma(r7, r22, r5);
    r22 = r5 * r5;
    r22 = fma(r31, r22, r31 * r29);
    r22 = fma(r28, r22, r23);
    r23 = 1.0 / r25;
    r29 = r22 * r23;
    r2 = fma(r26, r29, r2);
    r3 = fma(r3, r4, r27);
    r27 = r0 * r22;
    r27 = r27 * r23;
    r3 = fma(r5, r27, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r29 = r4 * r3;
    r37 = r12 * r12;
    r13 = r13 * r13;
    r40 = r4 * r13;
    r41 = r37 + r40;
    r42 = r4 * r24;
    r43 = r36 + r42;
    r44 = r41 + r43;
    r44 = fma(r18, r44, r7 * r32);
    r32 = r14 * r1;
    r45 = r9 * r15;
    r39 = r39 + r45;
    r42 = r37 + r42;
    r37 = r4 * r36;
    r46 = r13 + r37;
    r42 = r42 + r46;
    r42 = fma(r7, r42, r18 * r39);
    r30 = r25 * r30;
    r25 = 1.0 / r30;
    r32 = r32 * r1;
    r32 = r32 * r42;
    r39 = r14 * r5;
    r39 = r39 * r5;
    r39 = r39 * r42;
    r39 = fma(r25, r39, r25 * r32);
    r32 = r11 * r5;
    r32 = r32 * r44;
    r39 = fma(r31, r32, r39);
    r47 = r11 * r1;
    r9 = r8 * r9;
    r9 = r9 * r14;
    r38 = r38 + r9;
    r21 = fma(r7, r21, r18 * r38);
    r47 = r47 * r21;
    r39 = fma(r31, r47, r39);
    r47 = r0 * r39;
    r32 = r28 * r5;
    r47 = r47 * r23;
    r47 = fma(r32, r47, r44 * r27);
    r44 = r0 * r31;
    r44 = r44 * r5;
    r44 = r44 * r22;
    r44 = r44 * r4;
    r47 = fma(r42, r44, r47);
    r38 = r4 * r2;
    r48 = r28 * r39;
    r48 = r48 * r23;
    r21 = fma(r21, r27, r26 * r48);
    r48 = r42 * r26;
    r22 = r22 * r4;
    r22 = r22 * r31;
    r21 = fma(r22, r48, r21);
    r38 = fma(r21, r38, r47 * r29);
    r29 = r4 * r2;
    r15 = r8 * r15;
    r35 = r35 + r15;
    r12 = r12 * r12;
    r12 = r12 * r4;
    r13 = r13 + r12;
    r13 = r13 + r43;
    r13 = fma(r18, r13, r6 * r35);
    r35 = r14 * r5;
    r40 = r36 + r40;
    r12 = r24 + r12;
    r40 = r40 + r12;
    r40 = fma(r6, r40, r18 * r19);
    r35 = r35 * r5;
    r35 = r35 * r40;
    r19 = r11 * r5;
    r45 = r33 + r45;
    r45 = fma(r6, r45, r18 * r10);
    r19 = r19 * r45;
    r19 = fma(r31, r19, r25 * r35);
    r35 = r14 * r1;
    r35 = r35 * r1;
    r35 = r35 * r40;
    r19 = fma(r25, r35, r19);
    r10 = r11 * r1;
    r10 = r10 * r13;
    r19 = fma(r31, r10, r19);
    r10 = r28 * r19;
    r10 = r10 * r23;
    r10 = fma(r26, r10, r13 * r27);
    r13 = r40 * r26;
    r10 = fma(r22, r13, r10);
    r13 = r4 * r3;
    r45 = fma(r45, r27, r40 * r44);
    r35 = r0 * r19;
    r35 = r35 * r23;
    r45 = fma(r32, r35, r45);
    r13 = fma(r45, r13, r10 * r29);
    write_sum_2<double, double>((double*)inout_shared, r38, r13);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r4 * r3;
    r15 = r20 + r15;
    r15 = fma(r7, r15, r6 * r34);
    r16 = r9 + r16;
    r12 = r46 + r12;
    r12 = fma(r6, r12, r7 * r16);
    r16 = fma(r12, r27, r15 * r44);
    r46 = r11 * r5;
    r46 = r46 * r12;
    r12 = r14 * r1;
    r12 = r12 * r1;
    r12 = r12 * r15;
    r12 = fma(r25, r12, r31 * r46);
    r46 = r14 * r5;
    r46 = r46 * r5;
    r46 = r46 * r15;
    r12 = fma(r25, r46, r12);
    r9 = r11 * r1;
    r37 = r24 + r37;
    r37 = r37 + r41;
    r37 = fma(r7, r37, r6 * r17);
    r9 = r9 * r37;
    r12 = fma(r31, r9, r12);
    r9 = r0 * r12;
    r9 = r9 * r23;
    r16 = fma(r32, r9, r16);
    r9 = r4 * r2;
    r46 = r28 * r12;
    r46 = r46 * r23;
    r46 = fma(r26, r46, r37 * r27);
    r37 = r15 * r26;
    r46 = fma(r22, r37, r46);
    r9 = fma(r46, r9, r16 * r13);
    r13 = r4 * r2;
    r37 = r25 * r26;
    r31 = r11 * r37;
    r7 = r28 * r1;
    r31 = fma(r7, r31, r27);
    r17 = r14 * r3;
    r6 = r32 * r37;
    r17 = fma(r6, r17, r31 * r13);
    write_sum_2<double, double>((double*)inout_shared, r9, r17);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r4 * r2;
    r9 = r14 * r1;
    r9 = r9 * r1;
    r13 = r14 * r5;
    r13 = r13 * r5;
    r13 = fma(r25, r13, r25 * r9);
    r9 = r28 * r13;
    r9 = r9 * r23;
    r9 = fma(r26, r9, r26 * r22);
    r22 = r4 * r3;
    r41 = r0 * r13;
    r41 = r41 * r23;
    r41 = fma(r32, r41, r44);
    r22 = fma(r41, r22, r9 * r17);
    r17 = r4 * r3;
    r44 = r0 * r11;
    r44 = r44 * r5;
    r44 = r44 * r25;
    r44 = fma(r32, r44, r27);
    r27 = r14 * r2;
    r27 = fma(r6, r27, r44 * r17);
    write_sum_2<double, double>((double*)inout_shared, r27, r22);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fma(r47, r47, r21 * r21);
    r27 = fma(r10, r10, r45 * r45);
    write_sum_2<double, double>((double*)inout_shared, r22, r27);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r16, r16, r46 * r46);
    r22 = r0 * r5;
    r17 = 4.00000000000000000e+00;
    r30 = r30 * r30;
    r30 = 1.0 / r30;
    r22 = r22 * r17;
    r22 = r22 * r30;
    r22 = r22 * r26;
    r22 = r22 * r32;
    r22 = r22 * r7;
    r7 = fma(r31, r31, r22);
    write_sum_2<double, double>((double*)inout_shared, r27, r7);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fma(r44, r44, r22);
    r7 = fma(r9, r9, r41 * r41);
    write_sum_2<double, double>((double*)inout_shared, r22, r7);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r21, r10, r47 * r45);
    r22 = fma(r21, r46, r47 * r16);
    write_sum_2<double, double>((double*)inout_shared, r7, r22);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r11 * r6;
    r22 = fma(r47, r6, r21 * r31);
    r7 = fma(r21, r6, r47 * r44);
    write_sum_2<double, double>((double*)inout_shared, r22, r7);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = fma(r21, r9, r47 * r41);
    r47 = fma(r10, r46, r45 * r16);
    write_sum_2<double, double>((double*)inout_shared, r21, r47);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fma(r45, r6, r10 * r31);
    r21 = fma(r10, r6, r45 * r44);
    write_sum_2<double, double>((double*)inout_shared, r47, r21);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fma(r10, r9, r45 * r41);
    r45 = fma(r16, r6, r46 * r31);
    write_sum_2<double, double>((double*)inout_shared, r10, r45);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r16, r41, r46 * r9);
    r46 = fma(r46, r6, r16 * r44);
    write_sum_2<double, double>((double*)inout_shared, r46, r45);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r31, r6, r44 * r6);
    r31 = fma(r41, r6, r31 * r9);
    write_sum_2<double, double>((double*)inout_shared, r45, r31);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r9, r6, r44 * r41);
    write_sum_1<double, double>((double*)inout_shared, r6);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_fixed_point_fixed_calib_res_jac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* calib,
    unsigned int calib_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_point_fixed_calib_res_jac_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      pixel,
      pixel_num_alloc,
      calib,
      calib_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar