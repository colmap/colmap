import sys
from pathlib import Path

import torchvision.transforms as tvf

from ..utils.base_model import BaseModel

r2d2_path = Path(__file__).parent / "../../third_party/r2d2"
sys.path.append(str(r2d2_path))
from extract import NonMaxSuppression, extract_multiscale, load_network  # noqa: E402


class R2D2(BaseModel):
    default_conf = {
        "model_name": "r2d2_WASF_N16.pt",
        "max_keypoints": 5000,
        "scale_factor": 2**0.25,
        "min_size": 256,
        "max_size": 1024,
        "min_scale": 0,
        "max_scale": 1,
        "reliability_threshold": 0.7,
        "repetability_threshold": 0.7,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        model_fn = r2d2_path / "models" / conf["model_name"]
        self.norm_rgb = tvf.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.net = load_network(model_fn)
        self.detector = NonMaxSuppression(
            rel_thr=conf["reliability_threshold"],
            rep_thr=conf["repetability_threshold"],
        )

    def _forward(self, data):
        img = data["image"]
        img = self.norm_rgb(img)

        xys, desc, scores = extract_multiscale(
            self.net,
            img,
            self.detector,
            scale_f=self.conf["scale_factor"],
            min_size=self.conf["min_size"],
            max_size=self.conf["max_size"],
            min_scale=self.conf["min_scale"],
            max_scale=self.conf["max_scale"],
        )
        idxs = scores.argsort()[-self.conf["max_keypoints"] or None :]
        xy = xys[idxs, :2]
        desc = desc[idxs].t()
        scores = scores[idxs]

        pred = {
            "keypoints": xy[None],
            "descriptors": desc[None],
            "scores": scores[None],
        }
        return pred
