#include "dkd.hpp"

#include <torch/torch.h>

namespace F = torch::nn::functional;
using namespace torch::indexing;

DKD::DKD(int radius, int top_k, float scores_th, int n_limit)
    : radius_(radius),
      top_k_(top_k),
      scores_th_(scores_th),
      n_limit_(n_limit),
      kernel_size_(calculateKernelSize(radius)),
      temperature_(0.1f),
      unfold_(torch::nn::UnfoldOptions(kernel_size_).padding(radius)) {

    auto x = torch::linspace(-radius_, radius_, kernel_size_);
    auto meshgrid = torch::meshgrid({x, x}, "ij");
    hw_grid_ = torch::stack({meshgrid[1], meshgrid[0]}, -1)
                   .reshape({-1, 2})
                   .contiguous(); // Ensure contiguous memory layout
}

torch::Tensor DKD::simple_nms(const torch::Tensor& scores, int nms_radius) {
    auto zeros = torch::zeros_like(scores);
    auto max_pool_options = F::MaxPool2dFuncOptions(nms_radius * 2 + 1)
                                .stride(1)
                                .padding(nms_radius);

    auto max_mask = scores == F::max_pool2d(scores, max_pool_options);

    for (int i = 0; i < 2; ++i)
    {
        auto supp_mask = F::max_pool2d(max_mask.to(torch::kFloat), max_pool_options) > 0;
        auto supp_scores = torch::where(supp_mask, zeros, scores);
        auto new_max_mask = supp_scores == F::max_pool2d(supp_scores, max_pool_options);
        max_mask = max_mask | (new_max_mask & (~supp_mask));
    }

    return torch::where(max_mask, scores, zeros);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
DKD::detect_keypoints(const torch::Tensor& scores_map, bool sub_pixel) {
    const auto batch_size = scores_map.size(0);
    const auto height = scores_map.size(2);
    const auto width = scores_map.size(3);
    const auto device = scores_map.device();

    auto nms_scores = simple_nms(scores_map, 2);

    auto border_mask = torch::ones_like(nms_scores,
                                        torch::TensorOptions()
                                            .dtype(torch::kBool)
                                            .device(device));

    border_mask.index_put_({Slice(), Slice(), Slice(None, radius_), Slice()}, false);
    border_mask.index_put_({Slice(), Slice(), Slice(), Slice(None, radius_)}, false);
    border_mask.index_put_({Slice(), Slice(), Slice(-radius_, None), Slice()}, false);
    border_mask.index_put_({Slice(), Slice(), Slice(), Slice(-radius_, None)}, false);

    nms_scores = torch::where(border_mask, nms_scores, torch::zeros_like(nms_scores));

    std::vector<torch::Tensor> keypoints;
    std::vector<torch::Tensor> scoredispersitys;
    std::vector<torch::Tensor> kptscores;
    keypoints.reserve(batch_size);
    scoredispersitys.reserve(batch_size);
    kptscores.reserve(batch_size);

    // Create wh tensor on the correct device
    auto wh = torch::tensor(
        {static_cast<float>(width - 1), static_cast<float>(height - 1)},
        torch::TensorOptions().dtype(scores_map.dtype()).device(device));

    // Ensure hw_grid_ is on the correct device
    if (hw_grid_.device() != device)
    {
        hw_grid_ = hw_grid_.to(device);
    }

    if (sub_pixel)
    {
        auto patches = unfold_(scores_map);

        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx)
        {
            auto patch = patches[batch_idx].transpose(0, 1);

            torch::Tensor indices_kpt;
            if (top_k_ > 0)
            {
                auto scores_view = nms_scores[batch_idx].reshape(-1);
                auto topk = scores_view.topk(top_k_);
                indices_kpt = std::get<1>(topk);
            } else
            {
                auto scores_view = nms_scores[batch_idx].reshape(-1);
                auto mask = scores_view > scores_th_;
                indices_kpt = mask.nonzero().squeeze(1);
                if (indices_kpt.size(0) > n_limit_)
                {
                    auto kpts_sc = scores_view.index_select(0, indices_kpt);
                    auto sort_idx = kpts_sc.argsort(/*stable=*/false, /*dim=*/-1, /*descending=*/true);
                    indices_kpt = indices_kpt.index_select(0, sort_idx.slice(0, n_limit_));
                }
            }

            auto patch_scores = patch.index_select(0, indices_kpt);
            auto keypoints_xy_nms = torch::stack({indices_kpt % width,
                                                  torch::div(indices_kpt, width, /*rounding_mode=*/"floor")},
                                                 1)
                                        .to(device);

            auto [max_v, _] = patch_scores.max(1, true);
            auto x_exp = ((patch_scores - max_v) / temperature_).exp();
            auto xy_residual = (x_exp.unsqueeze(2) * hw_grid_.unsqueeze(0)).sum(1) /
                               x_exp.sum(1, true);

            auto dist2 = (hw_grid_.unsqueeze(0) - xy_residual.unsqueeze(1))
                             .div(radius_)
                             .norm(2, -1)
                             .pow(2);

            auto scoredispersity = (x_exp * dist2).sum(1) / x_exp.sum(1);
            auto keypoints_xy = keypoints_xy_nms + xy_residual;
            keypoints_xy = keypoints_xy.div(wh).mul(2).sub(1);

            auto kptscore = torch::nn::functional::grid_sample(
                scores_map[batch_idx].unsqueeze(0),
                keypoints_xy.view({1, 1, -1, 2}),
                torch::nn::functional::GridSampleFuncOptions()
                    .mode(torch::kBilinear)
                    .align_corners(true))[0][0][0];

            keypoints.push_back(std::move(keypoints_xy));
            scoredispersitys.push_back(std::move(scoredispersity));
            kptscores.push_back(std::move(kptscore));
        }
    } else
    {
        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx)
        {
            torch::Tensor indices_kpt;
            if (top_k_ > 0)
            {
                auto scores_view = nms_scores[batch_idx].reshape(-1);
                auto topk = scores_view.topk(top_k_);
                indices_kpt = std::get<1>(topk);
            } else
            {
                auto scores_view = nms_scores[batch_idx].reshape(-1);
                auto mask = scores_view > scores_th_;
                indices_kpt = mask.nonzero().squeeze(1);
                if (indices_kpt.size(0) > n_limit_)
                {
                    auto kpts_sc = scores_view.index_select(0, indices_kpt);
                    auto sort_idx = kpts_sc.argsort(/*stable=*/false, /*dim=*/-1, /*descending=*/true);
                    indices_kpt = indices_kpt.index_select(0, sort_idx.slice(0, n_limit_));
                }
            }

            auto keypoints_xy = torch::stack({indices_kpt % width,
                                              torch::div(indices_kpt, width, /*rounding_mode=*/"floor")},
                                             1)
                                    .to(device);

            keypoints_xy = keypoints_xy.div(wh).mul(2).sub(1);

            auto kptscore = torch::nn::functional::grid_sample(
                scores_map[batch_idx].unsqueeze(0),
                keypoints_xy.view({1, 1, -1, 2}),
                torch::nn::functional::GridSampleFuncOptions()
                    .mode(torch::kBilinear)
                    .align_corners(true))[0][0][0];

            keypoints.push_back(std::move(keypoints_xy));
            scoredispersitys.push_back(kptscore.clone());
            kptscores.push_back(std::move(kptscore));
        }
    }

    return std::make_tuple(std::move(keypoints),
                           std::move(scoredispersitys),
                           std::move(kptscores));
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
DKD::forward(const torch::Tensor& scores_map, bool sub_pixel) {
    return detect_keypoints(scores_map, sub_pixel);
}
