"""
Code to use MegaLoc as a global features extractor.

MegaLoc paper: https://arxiv.org/abs/2502.17237
"""

import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel


class MegaPlaces(BaseModel):
    required_inputs = ["image"]

    def _init(self, conf):
        self.net = torch.hub.load("gmberton/MegaLoc", "get_trained_model").eval()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)

    def _forward(self, data):
        image = self.norm_rgb(data["image"])
        desc = self.net(image)
        return {
            "global_descriptor": desc,
        }
