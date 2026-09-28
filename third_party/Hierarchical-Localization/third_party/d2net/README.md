# D2-Net: A Trainable CNN for Joint Detection and Description of Local Features

This repository contains the implementation of the following paper:

```text
"D2-Net: A Trainable CNN for Joint Detection and Description of Local Features".
M. Dusmanu, I. Rocco, T. Pajdla, M. Pollefeys, J. Sivic, A. Torii, and T. Sattler. CVPR 2019.
```

[Paper on arXiv](https://arxiv.org/abs/1905.03561), [Project page](https://dsmn.ml/publications/d2-net.html)
    
## Getting started

Python 3.6+ is recommended for running our code. [Conda](https://docs.conda.io/en/latest/) can be used to install the required packages:

```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install h5py imageio imagesize matplotlib numpy scipy tqdm
```

## Downloading the models

The off-the-shelf **Caffe VGG16** weights and their tuned counterpart can be downloaded by running:

```bash
mkdir models
wget https://dsmn.ml/files/d2-net/d2_ots.pth -O models/d2_ots.pth
wget https://dsmn.ml/files/d2-net/d2_tf.pth -O models/d2_tf.pth
wget https://dsmn.ml/files/d2-net/d2_tf_no_phototourism.pth -O models/d2_tf_no_phototourism.pth
```

**Update - 23 May 2019** We have added a new set of weights trained on MegaDepth without the PhotoTourism scenes (sagrada_familia - 0019, lincoln_memorial_statue - 0021, british_museum - 0024, london_bridge - 0025, us_capitol - 0078, mount_rushmore - 1589). Our initial results show similar performance. In order to use these weights at test time, you should add `--model_file models/d2_tf_no_phototourism.pth`.

## Feature extraction

`extract_features.py` can be used to extract D2 features for a given list of images. The singlescale features require less than 6GB of VRAM for 1200x1600 images. The `--multiscale` flag can be used to extract multiscale features - for this, we recommend at least 12GB of VRAM. 

The output format can be either [`npz`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) or `mat`. In either case, the feature files encapsulate three arrays: 

- `keypoints` [`N x 3`] array containing the positions of keypoints `x, y` and the scales `s`. The positions follow the COLMAP format, with the `X` axis pointing to the right and the `Y` axis to the bottom.
- `scores` [`N`] array containing the activations of keypoints (higher is better).
- `descriptors` [`N x 512`] array containing the L2 normalized descriptors.

```bash
python extract_features.py --image_list_file images.txt (--multiscale)
```

## Tuning on MegaDepth

The training pipeline provided here is a PyTorch implementation of the TensorFlow code that was used to train the model available to download above.

**Update - 05 June 2019** We have fixed a bug in the dataset preprocessing - retraining now yields similar results to the original TensorFlow implementation.

**Update - 07 August 2019** We have released an updated, more accurate version of the training dataset - training is more stable and significantly faster for equal performance.

### Downloading and preprocessing the MegaDepth dataset

For this part, [COLMAP](https://colmap.github.io/) should be installed. Please refer to the official website for installation instructions.

After downloading the entire [MegaDepth](http://www.cs.cornell.edu/projects/megadepth/) dataset (including SfM models), the first step is generating the undistorted reconstructions. This can be done by calling `undistort_reconstructions.py` as follows:

```bash
python undistort_reconstructions.py --colmap_path /path/to/colmap/executable --base_path /path/to/megadepth
```

Next, `preprocess_megadepth.sh` can be used to retrieve the camera parameters and compute the overlap between images for all scenes. 

```bash
bash preprocess_undistorted_megadepth.sh /path/to/megadepth /path/to/output/folder
```

In case you prefer downloading the undistorted reconstructions and aggregated scene information folder directly, you can find them [here - Google Drive](https://drive.google.com/open?id=1hxpOsqOZefdrba_BqnW490XpNX_LgXPB). You will still need to download the depth maps ("MegaDepth v1 Dataset") from the MegaDepth website.

### Training

After downloading and preprocessing MegaDepth, the training can be started right away:

```bash
python train.py --use_validation --dataset_path /path/to/megadepth --scene_info_path /path/to/preprocessing/output
```

## BibTeX

If you use this code in your project, please cite the following paper:

```bibtex
@InProceedings{Dusmanu2019CVPR,
    author = {Dusmanu, Mihai and Rocco, Ignacio and Pajdla, Tomas and Pollefeys, Marc and Sivic, Josef and Torii, Akihiko and Sattler, Torsten},
    title = {{D2-Net: A Trainable CNN for Joint Detection and Description of Local Features}},
    booktitle = {Proceedings of the 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year = {2019},
}
```
