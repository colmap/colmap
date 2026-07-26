# TartanAir V2 COLMAP benchmark subset

This package contains a modified subset of the TartanAir V2 dataset for
evaluating panoramic structure-from-motion in COLMAP. Images and poses were
selected without changing their pixel or numeric values.

Source: https://huggingface.co/datasets/theairlabcmu/tartanair2

TartanAir V2 is distributed under the BSD-3-Clause license. Cite the original
dataset when publishing results based on this subset:

> Wenshan Wang et al., "TartanAir: A Dataset to Push the Limits of Visual
> SLAM," IROS 2020.

The exact source revision, trajectories, frames, and packaging checksums are
recorded in `tartanair_v2_manifest.json`.

From the COLMAP checkout, download and evaluate both pipeline variants with:

```bash
python benchmark/reconstruction/download.py --datasets tartanair-v2
python benchmark/reconstruction/evaluate.py \
  --datasets tartanair-v2-perspective tartanair-v2-spherical \
  --colmap_path /path/to/colmap
```

The panorama reconstruction itself uses the pycolmap package imported by the
current Python interpreter. The COLMAP executable is used for model alignment.
