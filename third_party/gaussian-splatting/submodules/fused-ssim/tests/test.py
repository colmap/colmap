import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from math import exp
from fused_ssim import fused_ssim
from pytorch_msssim import SSIM
import time

# Reference Implementation is taken from the following:
# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/loss_utils.py

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, CH, H, W = 5, 5, 1080, 1920
    pm_ssim = SSIM(data_range=1.0, channel=CH)
    iterations = 100

    for _ in range(iterations):
        with torch.no_grad():
            img1_og = nn.Parameter(torch.rand([B, CH, H, W], device="cuda"))
            img2_og = torch.rand([B, CH, H, W], device="cuda")

            img1_mine_same = nn.Parameter(img1_og.clone())
            img2_mine_same = img2_og.clone()

            img1_mine_valid = nn.Parameter(img1_og.clone())
            img2_mine_valid = img2_og.clone()

            img1_pm = nn.Parameter(img1_og.clone())
            img2_pm = img2_og.clone()

        og_ssim_val = ssim(img1_og, img2_og)
        mine_ssim_val_same = fused_ssim(img1_mine_same, img2_mine_same)
        mine_ssim_val_valid = fused_ssim(img1_mine_valid, img2_mine_valid, "valid")
        pm_ssim_val = pm_ssim(img1_pm, img2_pm)

        assert torch.isclose(og_ssim_val, mine_ssim_val_same)
        assert torch.isclose(mine_ssim_val_valid, pm_ssim_val)

        og_ssim_val.backward()
        mine_ssim_val_same.backward()
        mine_ssim_val_valid.backward()
        pm_ssim_val.backward()

        assert torch.isclose(img1_og.grad, img1_mine_same.grad).all()
        assert torch.isclose(img1_mine_valid.grad, img1_pm.grad).all()

    img1 = nn.Parameter(torch.rand([B, CH, H, W], device="cuda"))
    img2 = torch.rand([B, CH, H, W], device="cuda")

    # benchmark og
    begin = time.time()
    for _ in range(iterations):
        og_ssim_val = ssim(img1, img2)
    torch.cuda.synchronize()
    end = time.time()
    og_time_forward = (end - begin) / iterations * 1000
    print("Reference Time (Forward):", og_time_forward, "ms")

    begin = time.time()
    for _ in range(iterations):
        og_ssim_val = ssim(img1, img2)
        og_ssim_val.backward()
    torch.cuda.synchronize()
    end = time.time()
    og_time_backward = (end - begin) / iterations * 1000 - og_time_forward
    print("Reference Time (Backward):", og_time_backward, "ms")

    # benchmark pytorch_mssim (pm)
    begin = time.time()
    for _ in range(iterations):
        pm_ssim_val = pm_ssim(img1, img2)
    torch.cuda.synchronize()
    end = time.time()
    pm_time_forward = (end - begin) / iterations * 1000
    print("pytorch_mssim Time (Forward):", pm_time_forward, "ms")

    begin = time.time()
    for _ in range(iterations):
        pm_ssim_val = pm_ssim(img1, img2)
        pm_ssim_val.backward()
    torch.cuda.synchronize()
    end = time.time()
    pm_time_backward = (end - begin) / iterations * 1000 - pm_time_forward
    print("pytorch_mssim Time (Backward):", pm_time_backward, "ms")


    # benchmark mine
    begin = time.time()
    for _ in range(iterations):
        mine_ssim_val = fused_ssim(img1, img2)
    torch.cuda.synchronize()
    end = time.time()
    mine_time_forward = (end - begin) / iterations * 1000
    print("fused-ssim Time (Forward):", mine_time_forward, "ms")

    begin = time.time()
    for _ in range(iterations):
        mine_ssim_val = fused_ssim(img1, img2)
        mine_ssim_val.backward()
    torch.cuda.synchronize()
    end = time.time()
    mine_time_backward = (end - begin) / iterations * 1000 - mine_time_forward
    print("fused-ssim Time (Backward):", mine_time_backward, "ms")

    begin = time.time()
    for _ in range(iterations):
        mine_ssim_val = fused_ssim(img1, img2, train=False)
    torch.cuda.synchronize()
    end = time.time()
    mine_time_infer = (end - begin) / iterations * 1000
    print("fused-ssim Time (Inference):", mine_time_infer, "ms")
