#include "get_patches_cuda.h"
#include <glog/logging.h>
#include <math.h>
#include <torch/torch.h>

// map: CxHxW
// points: Nx2
// kernel_size: int
// return: N x C x kernel_size x kernel_size
namespace custom_ops {
torch::Tensor get_patches_forward_cpu(const torch::Tensor& map,
                                      torch::Tensor& points,
                                      int64_t kernel_size) {
  namespace F = torch::nn::functional;
  using namespace torch::indexing;

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
  auto options =
      F::PadFuncOptions(
          {pad_left_top, pad_right_bottom, pad_left_top, pad_right_bottom})
          .mode(torch::kConstant);
  auto map_pad = F::pad(map.unsqueeze(0), options)
                     .squeeze(0);  // Cx(H+2*radius)x(W+2*radius)

  // get patches
  torch::Tensor patches =
      torch::zeros({N, C, kernel_size, kernel_size}, map.options());
  auto a_points = points.accessor<int, 2>();     // Nx2
  auto a_map_pad = map_pad.accessor<float, 3>();  // Cx(H+2*radius)x(W+2*radius)
  auto a_patches =
      patches.accessor<float, 4>();  // N x C x kernel_size x kernel_size

  for (auto in = 0; in < N; in++) {
    auto w_start = a_points[in][0];
    auto h_start = a_points[in][1];

    // copy data
    for (auto ic = 0; ic < C; ic++) {
      for (auto ih = 0; ih < kernel_size; ih++) {
        for (auto iw = 0; iw < kernel_size; iw++) {
          a_patches[in][ic][ih][iw] = a_map_pad[ic][ih + h_start][iw + w_start];
        }
      }
    }
  }
  return patches;
}

// patches: NxCx(2*radius+1)x(2*radius+1)
// points: Nx2
torch::Tensor get_patches_backward_cpu(const torch::Tensor& d_patches,
                                       torch::Tensor& points,
                                       int64_t H,
                                       int64_t W) {
  namespace F = torch::nn::functional;
  using namespace torch::indexing;

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
  // printf("kernel_size=%d, radius=%f, pad_left_top=%d, pad_right_bottom=%d\n",
  //        kernel_size,
  //        radius,
  //        pad_left_top,
  //        pad_right_bottom);

  torch::Tensor d_map_pad = torch::zeros(
      {C, H + int(2 * radius), W + int(2 * radius)}, d_patches.options());

  auto a_points = points.accessor<int, 2>();  // Nx2
  auto a_d_map_pad =
      d_map_pad.accessor<float, 3>();  // Cx(H+2*radius)x(W+2*radius)
  auto a_p_patches =
      d_patches.accessor<float, 4>();  // NxCxkernel_sizexkernel_size
  for (auto in = 0; in < N; in++) {
    // long w_start = static_cast<long>(*(p_points + in * 2 + 0));
    // long h_start = static_cast<long>(*(p_points + in * 2 + 1));
    auto w_start = a_points[in][0];
    auto h_start = a_points[in][1];

    // copy data
    for (auto ic = 0; ic < C; ic++) {
      for (auto ih = 0; ih < kernel_size; ih++) {
        for (auto iw = 0; iw < kernel_size; iw++) {
          a_d_map_pad[ic][ih + h_start][iw + w_start] =
              a_p_patches[in][ic][ih][iw];
        }
      }
    }
  }

  auto d_map = d_map_pad.index({Slice(),
                                Slice(pad_left_top, -pad_right_bottom),
                                Slice(pad_left_top, -pad_right_bottom)});

  return d_map;
}

torch::Tensor get_patches_forward(const torch::Tensor& map,
                                  torch::Tensor& points,
                                  int64_t kernel_size) {
  if (map.device() == torch::kCPU) {
    return get_patches_forward_cpu(map, points, kernel_size);
  } else {
#ifdef COLMAP_CUDA_ENABLED
    return get_patches_forward_cuda(map, points, kernel_size);
#else
    LOG_FIRST_N(WARNING, 1)
        << "Requested to extract patches using CUDA but CUDA is not available. "
           "Falling back to CPU based patch extraction.";
    return get_patches_forward_cpu(map, points, kernel_size);
#endif
  }
}

torch::Tensor get_patches_backward(const torch::Tensor& d_patches,
                                   torch::Tensor& points,
                                   int64_t H,
                                   int64_t W) {
  if (d_patches.device() == torch::kCPU) {
    return get_patches_backward_cpu(d_patches, points, H, W);
  } else {
#ifdef COLMAP_CUDA_ENABLED
    return get_patches_backward_cuda(d_patches, points, H, W);
#else
    LOG_FIRST_N(WARNING, 1)
        << "Requested to extract patches using CUDA but CUDA is not available. "
           "Falling back to CPU based patch extraction.";
    return get_patches_backward_cpu(d_patches, points, H, W);
#endif
  }
}
}  // namespace custom_ops