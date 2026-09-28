import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.io import loadmat

from ..utils.base_model import BaseModel

logger = logging.getLogger(__name__)

EPS = 1e-6


class NetVLADLayer(nn.Module):
    def __init__(self, input_dim=512, K=64, score_bias=False, intranorm=True):
        super().__init__()
        self.score_proj = nn.Conv1d(input_dim, K, kernel_size=1, bias=score_bias)
        centers = nn.parameter.Parameter(torch.empty([input_dim, K]))
        nn.init.xavier_uniform_(centers)
        self.register_parameter("centers", centers)
        self.intranorm = intranorm
        self.output_dim = input_dim * K

    def forward(self, x):
        b = x.size(0)
        scores = self.score_proj(x)
        scores = F.softmax(scores, dim=1)
        diff = x.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(-1)
        desc = (scores.unsqueeze(1) * diff).sum(dim=-1)
        if self.intranorm:
            # From the official MATLAB implementation.
            desc = F.normalize(desc, dim=1)
        desc = desc.view(b, -1)
        desc = F.normalize(desc, dim=1)
        return desc


class NetVLAD(BaseModel):
    default_conf = {"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True}
    required_inputs = ["image"]

    # Models exported using
    # https://github.com/uzh-rpg/netvlad_tf_open/blob/master/matlab/net_class2struct.m.
    checkpoint_urls = {
        "VGG16-NetVLAD-Pitts30K": "https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat",  # noqa: E501
        "VGG16-NetVLAD-TokyoTM": "https://cvg-data.inf.ethz.ch/hloc/netvlad/TokyoTM_struct.mat",  # noqa: E501
    }

    def _init(self, conf):
        if conf["model_name"] not in self.checkpoint_urls:
            raise ValueError(
                f'{conf["model_name"]} not in {self.checkpoint_urls.keys()}.'
            )

        # Download the checkpoint.
        checkpoint_path = Path(
            torch.hub.get_dir(), "netvlad", conf["model_name"] + ".mat"
        )
        if not checkpoint_path.exists():
            checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            url = self.checkpoint_urls[conf["model_name"]]
            torch.hub.download_url_to_file(url, checkpoint_path)

        # Create the network.
        # Remove classification head.
        backbone = list(models.vgg16().children())[0]
        # Remove last ReLU + MaxPool2d.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.netvlad = NetVLADLayer()

        if conf["whiten"]:
            self.whiten = nn.Linear(self.netvlad.output_dim, 4096)

        # Parse MATLAB weights using https://github.com/uzh-rpg/netvlad_tf_open
        mat = loadmat(checkpoint_path, struct_as_record=False, squeeze_me=True)

        # CNN weights.
        for layer, mat_layer in zip(self.backbone.children(), mat["net"].layers):
            if isinstance(layer, nn.Conv2d):
                w = mat_layer.weights[0]  # Shape: S x S x IN x OUT
                b = mat_layer.weights[1]  # Shape: OUT
                # Prepare for PyTorch - enforce float32 and right shape.
                # w should have shape: OUT x IN x S x S
                # b should have shape: OUT
                w = torch.tensor(w).float().permute([3, 2, 0, 1])
                b = torch.tensor(b).float()
                # Update layer weights.
                layer.weight = nn.Parameter(w)
                layer.bias = nn.Parameter(b)

        # NetVLAD weights.
        score_w = mat["net"].layers[30].weights[0]  # D x K
        # centers are stored as opposite in official MATLAB code
        center_w = -mat["net"].layers[30].weights[1]  # D x K
        # Prepare for PyTorch - make sure it is float32 and has right shape.
        # score_w should have shape K x D x 1
        # center_w should have shape D x K
        score_w = torch.tensor(score_w).float().permute([1, 0]).unsqueeze(-1)
        center_w = torch.tensor(center_w).float()
        # Update layer weights.
        self.netvlad.score_proj.weight = nn.Parameter(score_w)
        self.netvlad.centers = nn.Parameter(center_w)

        # Whitening weights.
        if conf["whiten"]:
            w = mat["net"].layers[33].weights[0]  # Shape: 1 x 1 x IN x OUT
            b = mat["net"].layers[33].weights[1]  # Shape: OUT
            # Prepare for PyTorch - make sure it is float32 and has right shape
            w = torch.tensor(w).float().squeeze().permute([1, 0])  # OUT x IN
            b = torch.tensor(b.squeeze()).float()  # Shape: OUT
            # Update layer weights.
            self.whiten.weight = nn.Parameter(w)
            self.whiten.bias = nn.Parameter(b)

        # Preprocessing parameters.
        self.preprocess = {
            "mean": mat["net"].meta.normalization.averageImage[0, 0],
            "std": np.array([1, 1, 1], dtype=np.float32),
        }

    def _forward(self, data):
        image = data["image"]
        assert image.shape[1] == 3
        assert image.min() >= -EPS and image.max() <= 1 + EPS
        image = torch.clamp(image * 255, 0.0, 255.0)  # Input should be 0-255.
        mean = self.preprocess["mean"]
        std = self.preprocess["std"]
        image = image - image.new_tensor(mean).view(1, -1, 1, 1)
        image = image / image.new_tensor(std).view(1, -1, 1, 1)

        # Feature extraction.
        descriptors = self.backbone(image)
        b, c, _, _ = descriptors.size()
        descriptors = descriptors.view(b, c, -1)

        # NetVLAD layer.
        descriptors = F.normalize(descriptors, dim=1)  # Pre-normalization.
        desc = self.netvlad(descriptors)

        # Whiten if needed.
        if hasattr(self, "whiten"):
            desc = self.whiten(desc)
            desc = F.normalize(desc, dim=1)  # Final L2 normalization.

        return {"global_descriptor": desc}
