# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

from .pair_dataset import CatPairDataset, SyntheticPairDataset, TransformedPairs
from .imgfolder import ImgFolder

from .web_images import RandomWebImages
from .aachen import *

# try to instanciate datasets
import sys
try:
    web_images = RandomWebImages(0, 52)
except AssertionError as e:
    print(f"Dataset web_images not available, reason: {e}", file=sys.stderr)

try:
    aachen_db_images = AachenImages_DB()
except AssertionError as e:
    print(f"Dataset aachen_db_images not available, reason: {e}", file=sys.stderr)

try:
    aachen_style_transfer_pairs = AachenPairs_StyleTransferDayNight()
except AssertionError as e:
    print(f"Dataset aachen_style_transfer_pairs not available, reason: {e}", file=sys.stderr)

try:
    aachen_flow_pairs = AachenPairs_OpticalFlow()
except AssertionError as e:
    print(f"Dataset aachen_flow_pairs not available, reason: {e}", file=sys.stderr)


