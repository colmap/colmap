#  HPatches Sequences / Image Pairs Matching Benchmark

Please check the [official repository](https://github.com/hpatches/hpatches-dataset) for more information regarding references.

The dataset can be downloaded by running `bash download.sh` - this script downloads and extracts the HPatches Sequences dataset and removes the sequences containing high resolution images (`> 1600x1200`) as mentioned in the D2-Net paper. You can also download the cache with results for all methods from the D2-Net paper by running `bash download_cache.sh`.

New methods can be added in cell 4 of the notebook. The local features are supposed to be stored in the [`npz`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) format with three fields:

- `keypoints` - `N x 2` matrix with `x, y` coordinates of each keypoint in COLMAP format (the `X` axis points to the right, the `Y` axis to the bottom)

- `scores` - `N` array with detection scores for each keypoint (higher is better) - only required for the "top K" version of the benchmark

- `descriptors` - `N x D` matrix with the descriptors (L2 normalized if you plan on using the provided mutual nearest neighbors matcher)

Moreover, the `npz` files are supposed to be saved alongside their corresponding images with the same extension as the `method` (e.g. if `method = d2-net`, the features for the image `hpatches-sequences-release/i_ajuntament/1.ppm` should be in the file `hpatches-sequences-release/i_ajuntament/1.ppm.d2-net`).

We provide a simple script to extract Hessian Affine keypoints with SIFT descriptors (`extract_hesaff.m`); this script requires MATLAB and [VLFeat](http://www.vlfeat.org/).

D2-Net features can be extracted by running:
```
python extract_features.py --image_list_file image_list_hpatches_sequences.txt
```
