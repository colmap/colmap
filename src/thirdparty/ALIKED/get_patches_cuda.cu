#include "get_patches_cuda.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/macros/Macros.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

namespace F = torch::nn::functional;

#define CHECK_CUDA(x)       TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// CUDA: grid stride looping
//
// int64_t _i_n_d_e_x specifically prevents overflow in the loop increment.
// If input.numel() < INT_MAX, _i_n_d_e_x < INT_MAX, except after the final
// iteration of the loop where _i_n_d_e_x += blockDim.x * gridDim.x can be
// greater than INT_MAX.  But in that case _i_n_d_e_x >= n, so there are no
// further iterations and the overflowed value in i=_i_n_d_e_x is not used.
#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                 \
    int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x; \
    for (index_type i = _i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x += blockDim.x * gridDim.x, i = _i_n_d_e_x)

#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, int)

// Use 1024 threads per block, which requires cuda sm_2x or above
// constexpr int CUDA_NUM_THREADS = 1024;
constexpr int CUDA_NUM_THREADS = 16;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int64_t N, const int64_t max_threads_per_block = CUDA_NUM_THREADS) {
    TORCH_INTERNAL_ASSERT(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
    constexpr int64_t max_int = std::numeric_limits<int>::max();

    // Round up division for positive number that cannot cause integer overflow
    auto block_num = (N - 1) / max_threads_per_block + 1;
    TORCH_INTERNAL_ASSERT(block_num <= max_int, "Can't schedule too many blocks on CUDA device");

    return static_cast<int>(block_num);
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_NUM_THREADS)
__global__ void get_patches_forward_cuda_kernel(const int64_t n,
                                                const scalar_t* p_map,   // Cx(H+2*radius)x(W+2*radius)
                                                const int64_t* p_points, // Nx2
                                                int64_t n_input_plane, int64_t input_height, int64_t input_width, int64_t n_points,
                                                int64_t pad_left_top, int64_t pad_right_bottom, int64_t kernel_size,
                                                scalar_t* p_patches // NxCxkernel_sizexkernel_size
) {
    CUDA_KERNEL_LOOP(index, n) {
        int64_t n_out = index % n_points;       // point idx
        int64_t channel_idx = index / n_points; // channel idx

        int64_t w_in = *(p_points + 2 * n_out);
        int64_t h_in = *(p_points + 2 * n_out + 1);

        const scalar_t* im = p_map + (channel_idx * input_height + h_in) * input_width + w_in;
        scalar_t* dst_patches = p_patches + (n_out * n_input_plane + channel_idx) * kernel_size * kernel_size;

        // copy data
        for (int64_t i = 0; i < kernel_size; ++i)
        {
            for (int64_t j = 0; j < kernel_size; ++j)
            {
                int64_t h = h_in + i - pad_left_top;
                int64_t w = w_in + j - pad_left_top;

                *(dst_patches + i * kernel_size + j) = (h >= 0 && w >= 0 && h < input_height && w < input_width)
                                                           ? im[(i - pad_left_top) * input_width + j - pad_left_top]
                                                           : static_cast<scalar_t>(0);
            }
        }
    }
}

template <typename scalar_t>
__global__ void
get_patches_forward_cuda_kernel1(const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> map_pad, // Cx(H+2*radius)x(W+2*radius)
                                 const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> points,   // Nx2
                                 torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> patches,       // NxCxkernel_sizexkernel_size
                                 int64_t kernel_size) {
    const int in = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = points.size(0);
    const int C = map_pad.size(0);

    if (in < N)
    {
        long w_start = points[in][0];
        long h_start = points[in][1];

        // copy data
        for (long ic = 0; ic < C; ic++)
        {
            for (long ih = 0; ih < kernel_size; ih++)
            {
                for (long iw = 0; iw < kernel_size; iw++)
                {
                    patches[in][ic][ih][iw] = map_pad[ic][h_start + ih][w_start + iw];
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void
get_patches_backward_cuda_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> d_map_pad,       // Cx(H+2*radius)x(W+2*radius)
                                 const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> points,     // Nx2
                                 const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> d_patches, // NxCxkernel_sizexkernel_size
                                 int64_t kernel_size) {
    const int in = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = points.size(0);
    const int C = d_map_pad.size(0);

    if (in < N)
    {
        long w_start = points[in][0];
        long h_start = points[in][1];

        // copy data
        for (long ic = 0; ic < C; ic++)
        {
            for (long ih = 0; ih < kernel_size; ih++)
            {
                for (long iw = 0; iw < kernel_size; iw++)
                {
                    d_map_pad[ic][h_start + ih][w_start + iw] = d_patches[in][ic][ih][iw];
                }
            }
        }
    }
}

torch::Tensor get_patches_forward_cuda(const torch::Tensor& input, torch::Tensor& points, int64_t kernel_size) {
    CHECK_INPUT(input);
    CHECK_INPUT(points);

    int64_t n_input_plane = input.size(0);
    int64_t input_height = input.size(1);
    int64_t input_width = input.size(2);
    // kernel_size=2, radius=0.5, pad_left_top=0, pad_right_bottom=1
    // kernel_size=3, radius=1.0, pad_left_top=1, pad_right_bottom=1
    // kernel_size=4, radius=1.5, pad_left_top=1, pad_right_bottom=2
    // kernel_size=5, radius=2.0, pad_left_top=2, pad_right_bottom=2
    auto radius = (kernel_size - 1.0) / 2.0;
    int64_t pad_left_top = floor(radius);
    int64_t pad_right_bottom = ceil(radius);
    int64_t n_points = points.size(0);

    // create output patches
    torch::Tensor patches = torch::zeros({n_points, n_input_plane, kernel_size, kernel_size}, input.options());

    // cuda kernel
    int64_t num_kernels = n_input_plane * n_points;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(input.type(), "get_patches_forward_cuda",
                               (
                                   [&] {
                                       get_patches_forward_cuda_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
                                           num_kernels, input.data_ptr<scalar_t>(), points.data_ptr<int64_t>(), n_input_plane, input_height,
                                           input_width, n_points, pad_left_top, pad_right_bottom, kernel_size, patches.data_ptr<scalar_t>());
                                   }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return patches;
}

torch::Tensor get_patches_forward_cuda1(const torch::Tensor& map, torch::Tensor& points, int64_t kernel_size) {
    CHECK_INPUT(map);
    CHECK_INPUT(points);

    auto N = points.size(0);
    auto C = map.size(0);
    // kernel_size=2, radius=0.5, pad_left_top=0, pad_right_bottom=1
    // kernel_size=3, radius=1.0, pad_left_top=1, pad_right_bottom=1
    // kernel_size=4, radius=1.5, pad_left_top=1, pad_right_bottom=2
    // kernel_size=5, radius=2.0, pad_left_top=2, pad_right_bottom=2
    auto radius = (kernel_size - 1.0) / 2.0;
    int pad_left_top = floor(radius);
    int pad_right_bottom = ceil(radius);

    // pad map
    auto options = F::PadFuncOptions({pad_left_top, pad_right_bottom, pad_left_top, pad_right_bottom}).mode(torch::kConstant);
    auto map_pad = F::pad(map.unsqueeze(0), options).squeeze(0); // Cx(H+2*radius)x(W+2*radius)

    // create patches
    torch::Tensor patches = torch::empty({N, C, kernel_size, kernel_size}, map.options());

    // cuda kernel
    const int threads = CUDA_NUM_THREADS;
    const int blocks = (N + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(map_pad.type(), "get_patches_forward_cuda",
                               (
                                   [&] {
                                       get_patches_forward_cuda_kernel1<scalar_t>
                                           <<<blocks, threads>>>(map_pad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                 points.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                                                                 patches.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), kernel_size);
                                   }));

    // get error
    cudaDeviceSynchronize();
    cudaError_t cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    return patches;
}

torch::Tensor get_patches_backward_cuda(const torch::Tensor& d_patches, torch::Tensor& points, int64_t H, int64_t W) {
    CHECK_INPUT(d_patches);
    CHECK_INPUT(points);

    auto N = d_patches.size(0);
    auto C = d_patches.size(1);
    // kernel_size=2, radius=0.5, pad_left_top=0, pad_right_bottom=1
    // kernel_size=3, radius=1.0, pad_left_top=1, pad_right_bottom=1
    // kernel_size=4, radius=1.5, pad_left_top=1, pad_right_bottom=2
    // kernel_size=5, radius=2.0, pad_left_top=2, pad_right_bottom=2
    auto kernel_size = d_patches.size(2);
    auto radius = (kernel_size - 1.0) / 2.0;
    int pad_left_top = floor(radius);
    int pad_right_bottom = ceil(radius);

    torch::Tensor d_map_pad = torch::zeros({C, H + int(2 * radius), W + int(2 * radius)}, d_patches.options());

    // cuda kernel
    const int threads = CUDA_NUM_THREADS;
    const int blocks = (N + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(d_map_pad.type(), "get_patches_backward_cuda",
                               (
                                   [&] {
                                       get_patches_backward_cuda_kernel<scalar_t>
                                           <<<blocks, threads>>>(d_map_pad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                 points.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                                                                 d_patches.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), kernel_size);
                                   }));

    // get error
    cudaDeviceSynchronize();
    cudaError_t cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    using namespace torch::indexing;
    auto d_map = d_map_pad.index({Slice(), Slice(pad_left_top, -pad_right_bottom), Slice(pad_left_top, -pad_right_bottom)});

    return d_map;
}
