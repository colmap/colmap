import torch
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED

# def disk(image):
#     model = kornia.feature.DISK.from_pretrained("depth")
#     features = model.forward(image[None, ...])
#     return (features[0].keypoints, features[0].descriptors)

# def sift(image):
#     model = kornia.feature.SIFTFeature()
#     features = model.forward(image[None, ...])
#     return (features[0], features[2])

def aliked(image):
    model = ALIKED()
    features = model.forward({"image": image})
    return (features["keypoints"], features["keypoint_scores"], features["descriptors"])

# inputs = torch.rand([1, 3, 256, 256])
# traced_script_module = torch.jit.trace(aliked, inputs)
# traced_script_module.save("aliked.pt")

def lightglue(keypoints0, scales0, oris0, descriptors0, image_size0, keypoints1, scales1, oris1, descriptors1, image_size1):
    # keypoints0, descriptors0, image_size0, keypoints1, descriptors1, image_size1 = inputs
    model = LightGlue(features="sift")
    output = model.forward(
        {"image0": {"keypoints": keypoints0, "scales": scales0, "oris": oris0, "descriptors": descriptors0, "image_size": image_size0},
         "image1": {"keypoints": keypoints1, "scales": scales1, "oris": oris1, "descriptors": descriptors1, "image_size": image_size1}})
    return (output["matches"],)

keypoints = torch.rand([1, 100, 2])
scales = torch.rand([1, 100])
oris = torch.rand([1, 100])
descriptors = torch.rand([1, 100, 128])
image_size = torch.rand([1, 2])
traced_script_module = torch.jit.trace(lightglue, (keypoints, scales, oris, descriptors, image_size, keypoints, scales, oris, descriptors, image_size))
traced_script_module.save("lightglue_sift.pt")
