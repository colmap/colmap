import torch

import kornia
from kornia.feature.laf import (
    laf_from_center_scale_ori, extract_patches_from_pyramid)

def disk(image):
    model = kornia.feature.DISK()
    features = model.forward(image[None, ...])
    return (features[0].keypoints, features[0].descriptors)

def sift(image):
    model = kornia.feature.SIFTFeature()
    features = model.forward(image[None, ...])
    return (features[0], features[2])

inputs = torch.rand([3, 256, 256])
a = disk(inputs)
b = disk(inputs)
assert torch.all(a[0] == b[0])

# traced_script_module = torch.jit.trace(disk, inputs)
# traced_script_module.save("disk.pt")
