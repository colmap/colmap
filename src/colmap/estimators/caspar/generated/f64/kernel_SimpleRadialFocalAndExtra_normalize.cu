#include "kernel_SimpleRadialFocalAndExtra_normalize.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialFocalAndExtra_normalize_kernel(
        double* precond_diag,
        unsigned int precond_diag_num_alloc,
        double* precond_tril,
        unsigned int precond_tril_num_alloc,
        double* njtr,
        unsigned int njtr_num_alloc,
        const double* const diag,
        double* out_normalized,
        unsigned int out_normalized_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r0, r1);
    read_idx_1<1024, double, double, double>(
        precond_tril, 0 * precond_tril_num_alloc, global_thread_idx, r2);
    r3 = -1.00000000000000000e+00;
    r3 = r2 * r3;
    read_idx_2<1024, double, double, double2>(
        precond_diag, 0 * precond_diag_num_alloc, global_thread_idx, r4, r5);
  };
  load_unique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r7 = 1.00000000000000000e+00;
    r7 = r6 + r7;
    r8 = 1.00000000000000008e-15;
    r8 = r6 * r8;
    r4 = fma(r4, r7, r8);
    r4 = 1.0 / r4;
    r3 = r3 * r4;
    r1 = fma(r0, r3, r1);
    r7 = fma(r5, r7, r8);
    r7 = fma(r2, r3, r7);
    r7 = 1.0 / r7;
    r7 = r1 * r7;
    r4 = fma(r0, r4, r3 * r7);
    write_idx_2<1024, double, double, double2>(out_normalized,
                                               0 * out_normalized_num_alloc,
                                               global_thread_idx,
                                               r4,
                                               r7);
  };
}

void SimpleRadialFocalAndExtra_normalize(double* precond_diag,
                                         unsigned int precond_diag_num_alloc,
                                         double* precond_tril,
                                         unsigned int precond_tril_num_alloc,
                                         double* njtr,
                                         unsigned int njtr_num_alloc,
                                         const double* const diag,
                                         double* out_normalized,
                                         unsigned int out_normalized_num_alloc,
                                         size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFocalAndExtra_normalize_kernel<<<n_blocks, 1024>>>(
      precond_diag,
      precond_diag_num_alloc,
      precond_tril,
      precond_tril_num_alloc,
      njtr,
      njtr_num_alloc,
      diag,
      out_normalized,
      out_normalized_num_alloc,
      problem_size);
}

}  // namespace caspar