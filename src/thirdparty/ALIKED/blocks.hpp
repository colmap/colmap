#pragma once
#include <torch/torch.h>

#include <memory>
#include <span>

class DeformableConv2d : public torch::nn::Module {
public:
    DeformableConv2d(int in_channels, int out_channels,
                     int kernel_size = 3, int stride = 1,
                     int padding = 1, bool bias = false);

    torch::Tensor forward(const torch::Tensor& x) &;
    torch::Tensor forward(torch::Tensor x) &&;

private:
    torch::nn::Conv2d offset_conv_{nullptr};
    torch::nn::Conv2d regular_conv_{nullptr};
    int padding_;
    int groups_ = 1;
    int mask_offset_ = 1;
};

class ConvBlock : public torch::nn::Module {
public:
    ConvBlock(int in_channels, int out_channels,
              std::string_view conv_type = "conv",
              bool mask = false);

    torch::Tensor forward(torch::Tensor x) &&;
    torch::Tensor forward(const torch::Tensor& x) &;

private:
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr};
    std::shared_ptr<DeformableConv2d> deform1_{nullptr}, deform2_{nullptr};
    torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr};
};

class ResBlock : public torch::nn::Module {
public:
    ResBlock(int inplanes, int planes, int stride = 1,
             const torch::nn::Conv2d& downsample = nullptr,
             std::string_view conv_type = "conv");

    torch::Tensor forward(torch::Tensor x) &&;
    torch::Tensor forward(const torch::Tensor& x) &;

private:
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr};
    std::shared_ptr<DeformableConv2d> deform1_{nullptr}, deform2_{nullptr};
    torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr};
    torch::nn::Conv2d downsample_;
};