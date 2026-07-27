# TartanAir V2 COLMAP benchmark subset

This package contains a modified subset of the TartanAir V2 dataset for
evaluating panoramic structure-from-motion in COLMAP. Selected RGB images are
encoded as quality-97 progressive JPEG with 4:4:4 chroma sampling.
Ground-truth depth maps are min-pooled from 2048x1024 to 512x256 and stored as
lossless 16-bit PNGs. A depth value `d` represents `d / 8` meters, and zero is
invalid.

During evaluation, the depth maps are sampled every four pixels and projected
between ground-truth camera poses. A sample is considered shared when its
projected range agrees with the target depth within `max(0.25 m, 1%)`, up to a
maximum range of 256 m. Verified image pairs with no shared samples are removed
before mapping. Derived overlap counts are cached locally in
`covisibility.npz` and can be regenerated from the packaged depths.

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
