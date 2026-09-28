# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb

from .dataset import Dataset
from .pair_dataset import SyntheticPairDataset


class ImgFolder (Dataset):
    """ load all images in a folder (no recursion).
    """
    def __init__(self, root, imgs=None, exts=('.jpg','.png','.ppm')):
        Dataset.__init__(self)
        self.root = root
        self.imgs = imgs or [f for f in os.listdir(root) if f.endswith(exts)]
        self.nimg = len(self.imgs)

    def get_key(self, idx):
        return self.imgs[idx]


