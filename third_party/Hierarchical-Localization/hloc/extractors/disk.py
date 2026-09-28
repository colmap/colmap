import kornia

from ..utils.base_model import BaseModel


class DISK(BaseModel):
    default_conf = {
        "weights": "depth",
        "max_keypoints": None,
        "nms_window_size": 5,
        "detection_threshold": 0.0,
        "pad_if_not_divisible": True,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        self.model = kornia.feature.DISK.from_pretrained(conf["weights"])

    def _forward(self, data):
        image = data["image"]
        features = self.model(
            image,
            n=self.conf["max_keypoints"],
            window_size=self.conf["nms_window_size"],
            score_threshold=self.conf["detection_threshold"],
            pad_if_not_divisible=self.conf["pad_if_not_divisible"],
        )
        return {
            "keypoints": [f.keypoints for f in features],
            "keypoint_scores": [f.detection_scores for f in features],
            "descriptors": [f.descriptors.t() for f in features],
        }
