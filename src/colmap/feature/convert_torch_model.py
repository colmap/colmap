import torch
import kornia

def disk(image):
    model = kornia.feature.DISK.from_pretrained("depth")
    features = model.forward(image[None, ...])
    return (features[0].keypoints, features[0].descriptors)

def sift(image):
    model = kornia.feature.SIFTFeature()
    features = model.forward(image[None, ...])
    return (features[0], features[2])

inputs = torch.rand([3, 256, 256])
traced_script_module = torch.jit.trace(disk, inputs)
traced_script_module.save("disk.pt")
