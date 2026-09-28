import torch
import numpy as np
import os
from PIL import Image
from fused_ssim import fused_ssim

gt_image = torch.tensor(np.array(Image.open(os.path.join("..", "images", "albert.jpg"))), dtype=torch.float32, device="cuda").unsqueeze(0).unsqueeze(0) / 255.0
pred_image = torch.nn.Parameter(torch.rand_like(gt_image))

with torch.no_grad():
    ssim_value = fused_ssim(pred_image, gt_image, train=False)
    print("Starting with SSIM value:", ssim_value)


optimizer = torch.optim.Adam([pred_image])

while ssim_value < 0.9999:
    optimizer.zero_grad()
    loss = 1.0 - fused_ssim(pred_image, gt_image)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        ssim_value = fused_ssim(pred_image, gt_image, train=False)
        print("SSIM value:", ssim_value)

pred_image = (pred_image * 255.0).squeeze(0).squeeze(0)
to_save = pred_image.detach().cpu().numpy().astype(np.uint8)
Image.fromarray(to_save).save(os.path.join("..", "images", "predicted.jpg"))
