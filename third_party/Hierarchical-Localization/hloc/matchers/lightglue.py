from lightglue import LightGlue as LightGlue_

from ..utils.base_model import BaseModel


class LightGlue(BaseModel):
    default_conf = {
        "features": "superpoint",
        "depth_confidence": 0.95,
        "width_confidence": 0.99,
        "compile": False,
    }
    required_inputs = [
        "image0",
        "keypoints0",
        "descriptors0",
        "image1",
        "keypoints1",
        "descriptors1",
    ]

    def _init(self, conf):
        self.net = LightGlue_(conf.pop("features"), **conf)
        if conf["compile"]:
            self.net.compile()

    def _forward(self, data):
        data["descriptors0"] = data["descriptors0"].transpose(-1, -2)
        data["descriptors1"] = data["descriptors1"].transpose(-1, -2)

        return self.net(
            {
                "image0": {k[:-1]: v for k, v in data.items() if k[-1] == "0"},
                "image1": {k[:-1]: v for k, v in data.items() if k[-1] == "1"},
            }
        )
