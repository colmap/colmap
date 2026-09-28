import torch
from kornia.feature.adalam import AdalamFilter
from kornia.utils.helpers import get_cuda_device_if_available

from ..utils.base_model import BaseModel


class AdaLAM(BaseModel):
    default_conf = {
        "area_ratio": 100,
        "search_expansion": 4,
        "ransac_iters": 128,
        "min_inliers": 6,
        "min_confidence": 200,
        "orientation_difference_threshold": 30,
        "scale_rate_threshold": 1.5,
        "detected_scale_rate_threshold": 5,
        "refit": True,
        "force_seed_mnn": True,
        "device": get_cuda_device_if_available(),
    }
    required_inputs = [
        "image0",
        "image1",
        "descriptors0",
        "descriptors1",
        "keypoints0",
        "keypoints1",
        "scales0",
        "scales1",
        "oris0",
        "oris1",
    ]

    def _init(self, conf):
        self.adalam = AdalamFilter(conf)

    def _forward(self, data):
        assert data["keypoints0"].size(0) == 1
        if data["keypoints0"].size(1) < 2 or data["keypoints1"].size(1) < 2:
            matches = torch.zeros(
                (0, 2), dtype=torch.int64, device=data["keypoints0"].device
            )
        else:
            matches = self.adalam.match_and_filter(
                data["keypoints0"][0],
                data["keypoints1"][0],
                data["descriptors0"][0].T,
                data["descriptors1"][0].T,
                data["image0"].shape[2:],
                data["image1"].shape[2:],
                data["oris0"][0],
                data["oris1"][0],
                data["scales0"][0],
                data["scales1"][0],
            )
        matches_new = torch.full(
            (data["keypoints0"].size(1),),
            -1,
            dtype=torch.int64,
            device=data["keypoints0"].device,
        )
        matches_new[matches[:, 0]] = matches[:, 1]
        return {
            "matches0": matches_new.unsqueeze(0),
            "matching_scores0": torch.zeros(matches_new.size(0)).unsqueeze(0),
        }
