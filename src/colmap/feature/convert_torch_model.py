import torch

import kornia
from kornia.feature.laf import (
    laf_from_center_scale_ori, extract_patches_from_pyramid)

model = kornia.feature.HardNet8(pretrained=True)
inputs = [torch.rand([1, 1, 32, 32])]
traced_script_module = torch.jit.trace(model, inputs)
traced_script_module.save("hardnet8.pt")
