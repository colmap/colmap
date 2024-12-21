#include "blocks.hpp"

#include "deform_conv2d.h"

DeformableConv2d::DeformableConv2d(int in_channels, int out_channels,
                                   int kernel_size, int stride, int padding,
                                   bool bias) {
    padding_ = padding;
    const int channel_num = 2 * kernel_size * kernel_size;

    // Register offset conv
    offset_conv_ = register_module("offset_conv",
                                   torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, channel_num, kernel_size)
                                                         .stride(stride)
                                                         .padding(padding)
                                                         .bias(true)));

    // Register regular conv
    regular_conv_ = register_module("regular_conv",
                                    torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                                          .stride(stride)
                                                          .padding(padding)
                                                          .bias(bias)));
}

torch::Tensor DeformableConv2d::forward(const torch::Tensor& x) & {
    auto h = x.size(2);
    auto w = x.size(3);
    float max_offset = std::max(h, w) / 4.0f;

    // Offset and mask
    auto offset = offset_conv_->forward(x);
    auto mask = torch::zeros(
        {offset.size(0), 1},
        torch::TensorOptions().device(offset.device()).dtype(offset.dtype()));

    offset = offset.clamp(-max_offset, max_offset);

    if (!regular_conv_->bias.defined())
    {
        regular_conv_->bias = torch::zeros(
            {regular_conv_->weight.size(0)},
            torch::TensorOptions().device(x.device()).dtype(x.dtype()));
    }

    return vision::ops::deform_conv2d(
        x,
        regular_conv_->weight,
        offset,
        mask,
        regular_conv_->bias,
        1, 1,
        padding_, padding_,
        1, 1,
        groups_,
        mask_offset_,
        false);
}

torch::Tensor DeformableConv2d::forward(torch::Tensor x) && {
    auto h = x.size(2);
    auto w = x.size(3);
    float max_offset = std::max(h, w) / 4.0f;

    // Offset and mask
    auto offset = offset_conv_->forward(std::move(x));
    auto mask = torch::zeros(
        {offset.size(0), 1},
        torch::TensorOptions().device(offset.device()).dtype(offset.dtype()));

    offset = std::move(offset).clamp(-max_offset, max_offset);

    if (!regular_conv_->bias.defined())
    {
        regular_conv_->bias = torch::zeros(
            {regular_conv_->weight.size(0)},
            torch::TensorOptions().device(x.device()).dtype(x.dtype()));
    }

    return vision::ops::deform_conv2d(
        std::move(x),
        regular_conv_->weight,
        std::move(offset),
        std::move(mask),
        regular_conv_->bias,
        1, 1,
        padding_, padding_,
        1, 1,
        groups_,
        mask_offset_,
        false);
}

ConvBlock::ConvBlock(int in_channels, int out_channels,
                     std::string_view conv_type, bool mask) {

    if (conv_type == "conv")
    {
        auto conv1 = torch::nn::Conv2d((torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                                            .stride(1)
                                            .padding(1)
                                            .bias(false)));
        conv1_ = register_module("conv1", conv1);

        auto conv2 = torch::nn::Conv2d((torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                                            .stride(1)
                                            .padding(1)
                                            .bias(false)));
        conv2_ = register_module("conv2", conv2);

    } else
    {
        auto conv1 = std::make_shared<DeformableConv2d>(
            in_channels,
            out_channels,
            3,
            1,
            1,
            false);
        deform1_ = register_module("conv1", conv1);

        auto conv2 = std::make_shared<DeformableConv2d>(
            out_channels,
            out_channels,
            3,
            1,
            1,
            false);
        deform2_ = register_module("conv2", conv2);
    }

    bn1_ = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
    bn2_ = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
}

ResBlock::ResBlock(int inplanes, int planes, int stride,
                   const torch::nn::Conv2d& downsample,
                   std::string_view conv_type)
    : downsample_(downsample) {

    if (conv_type == "conv")
    {
        auto conv1 = torch::nn::Conv2d((torch::nn::Conv2dOptions(inplanes, planes, 3)
                                            .stride(stride)
                                            .padding(1)
                                            .bias(false)));
        conv1_ = register_module("conv1", conv1);

        auto conv2 = torch::nn::Conv2d((torch::nn::Conv2dOptions(planes, planes, 3)
                                            .stride(stride)
                                            .padding(1)
                                            .bias(false)));
        conv2_ = register_module("conv2", conv2);

    } else
    {
        auto conv1 = std::make_shared<DeformableConv2d>(
            inplanes,
            planes,
            3,
            1,
            1,
            false);
        deform1_ = register_module("conv1", conv1);

        auto conv2 = std::make_shared<DeformableConv2d>(
            planes,
            planes,
            3,
            1,
            1,
            false);
        deform2_ = register_module("conv2", conv2);
    }

    bn1_ = register_module("bn1",
                           torch::nn::BatchNorm2d(planes));
    bn2_ = register_module("bn2",
                           torch::nn::BatchNorm2d(planes));

    if (downsample)
    {
        register_module("downsample", downsample);
    }
}

torch::Tensor ConvBlock::forward(torch::Tensor x) && {
    return std::move(*this).forward(std::move(x));
}

torch::Tensor ConvBlock::forward(const torch::Tensor& x) & {
    if (conv1_ && conv2_)
    {
        auto tmp = torch::selu(bn1_->forward(conv1_->forward(x)));
        return torch::selu(bn2_->forward(conv2_->forward(std::move(tmp))));
    } else
    {
        auto tmp = torch::selu(bn1_->forward(deform1_->forward(x)));
        return torch::selu(bn2_->forward(deform2_->forward(std::move(tmp))));
    }
}

torch::Tensor ResBlock::forward(torch::Tensor x) && {
    return std::move(*this).forward(std::move(x));
}

torch::Tensor ResBlock::forward(const torch::Tensor& x) & {
    auto identity = x;

    torch::Tensor processed;
    if (conv1_ && conv2_)
    {
        auto tmp = conv1_->forward(x);
        tmp = bn1_->forward(std::move(tmp));
        tmp = torch::selu(std::move(tmp));

        processed = conv2_->forward(std::move(tmp));
        processed = bn2_->forward(std::move(processed));
    } else
    {
        auto tmp = deform1_->forward(x);
        tmp = bn1_->forward(std::move(tmp));
        tmp = torch::selu(std::move(tmp));

        processed = deform2_->forward(std::move(tmp));
        processed = bn2_->forward(std::move(processed));
    }

    if (downsample_)
    {
        identity = downsample_->as<torch::nn::Conv2d>()->forward(std::move(identity));
    }

    processed += identity;
    return torch::selu(std::move(processed));
}