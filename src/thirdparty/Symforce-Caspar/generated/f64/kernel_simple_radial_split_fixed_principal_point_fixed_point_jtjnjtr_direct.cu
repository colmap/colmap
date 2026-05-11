#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_split_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPrincipalPointFixedPointJtjnjtrDirectKernel(
        double *pose_njtr, unsigned int pose_njtr_num_alloc,
        SharedIndex *pose_njtr_indices, double *pose_jac,
        unsigned int pose_jac_num_alloc, double *focal_and_distortion_njtr,
        unsigned int focal_and_distortion_njtr_num_alloc,
        SharedIndex *focal_and_distortion_njtr_indices,
        double *focal_and_distortion_jac,
        unsigned int focal_and_distortion_jac_num_alloc,
        double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
        double *const out_focal_and_distortion_njtr,
        unsigned int out_focal_and_distortion_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex focal_and_distortion_njtr_indices_loc[1024];
  focal_and_distortion_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_distortion_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(pose_jac, 0 * pose_jac_num_alloc,
                                            global_thread_idx, r0, r1);
  };
  LoadShared<2, double, double>(
      focal_and_distortion_njtr, 0 * focal_and_distortion_njtr_num_alloc,
      focal_and_distortion_njtr_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double *)inout_shared,
        focal_and_distortion_njtr_indices_loc[threadIdx.x].target, r2, r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        focal_and_distortion_jac, 0 * focal_and_distortion_jac_num_alloc,
        global_thread_idx, r4, r5);
    ReadIdx2<1024, double, double, double2>(
        focal_and_distortion_jac, 2 * focal_and_distortion_jac_num_alloc,
        global_thread_idx, r6, r7);
    r8 = fma(r3, r6, r2 * r4);
    r3 = fma(r3, r7, r2 * r5);
    r2 = fma(r1, r3, r0 * r8);
    ReadIdx2<1024, double, double, double2>(pose_jac, 2 * pose_jac_num_alloc,
                                            global_thread_idx, r9, r10);
    r11 = fma(r10, r3, r9 * r8);
    WriteSum2<double, double>((double *)inout_shared, r2, r11);
  };
  FlushSumShared<2, double>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(pose_jac, 4 * pose_jac_num_alloc,
                                            global_thread_idx, r11, r2);
    r12 = fma(r2, r3, r11 * r8);
    ReadIdx2<1024, double, double, double2>(pose_jac, 6 * pose_jac_num_alloc,
                                            global_thread_idx, r13, r14);
    r15 = fma(r14, r3, r13 * r8);
    WriteSum2<double, double>((double *)inout_shared, r12, r15);
  };
  FlushSumShared<2, double>(out_pose_njtr, 2 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(pose_jac, 8 * pose_jac_num_alloc,
                                            global_thread_idx, r15, r12);
    r16 = fma(r12, r3, r15 * r8);
    ReadIdx2<1024, double, double, double2>(pose_jac, 10 * pose_jac_num_alloc,
                                            global_thread_idx, r17, r18);
    r8 = fma(r17, r8, r18 * r3);
    WriteSum2<double, double>((double *)inout_shared, r16, r8);
  };
  FlushSumShared<2, double>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc, (double *)inout_shared);
  LoadShared<2, double, double>(pose_njtr, 4 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target, r8, r16);
  };
  __syncthreads();
  LoadShared<2, double, double>(pose_njtr, 2 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target, r3, r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = fma(r19, r14, r16 * r18);
  };
  LoadShared<2, double, double>(pose_njtr, 0 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target, r18, r20);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = fma(r3, r2, r14);
    r14 = fma(r18, r1, r14);
    r14 = fma(r20, r10, r14);
    r14 = fma(r8, r12, r14);
    r15 = fma(r8, r15, r16 * r17);
    r15 = fma(r3, r11, r15);
    r15 = fma(r19, r13, r15);
    r15 = fma(r18, r0, r15);
    r15 = fma(r20, r9, r15);
    r4 = fma(r4, r15, r5 * r14);
    r15 = fma(r6, r15, r7 * r14);
    WriteSum2<double, double>((double *)inout_shared, r4, r15);
  };
  FlushSumShared<2, double>(out_focal_and_distortion_njtr,
                            0 * out_focal_and_distortion_njtr_num_alloc,
                            focal_and_distortion_njtr_indices_loc,
                            (double *)inout_shared);
}

void SimpleRadialSplitFixedPrincipalPointFixedPointJtjnjtrDirect(
    double *pose_njtr, unsigned int pose_njtr_num_alloc,
    SharedIndex *pose_njtr_indices, double *pose_jac,
    unsigned int pose_jac_num_alloc, double *focal_and_distortion_njtr,
    unsigned int focal_and_distortion_njtr_num_alloc,
    SharedIndex *focal_and_distortion_njtr_indices,
    double *focal_and_distortion_jac,
    unsigned int focal_and_distortion_jac_num_alloc,
    double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
    double *const out_focal_and_distortion_njtr,
    unsigned int out_focal_and_distortion_njtr_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedPrincipalPointFixedPointJtjnjtrDirectKernel<<<n_blocks,
                                                                      1024>>>(
      pose_njtr, pose_njtr_num_alloc, pose_njtr_indices, pose_jac,
      pose_jac_num_alloc, focal_and_distortion_njtr,
      focal_and_distortion_njtr_num_alloc, focal_and_distortion_njtr_indices,
      focal_and_distortion_jac, focal_and_distortion_jac_num_alloc,
      out_pose_njtr, out_pose_njtr_num_alloc, out_focal_and_distortion_njtr,
      out_focal_and_distortion_njtr_num_alloc, problem_size);
}

} // namespace caspar