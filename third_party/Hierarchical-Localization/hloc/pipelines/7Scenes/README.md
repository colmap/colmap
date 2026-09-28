# 7Scenes dataset

## Installation

1.
Download the images from the [7Scenes project page](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/):
```bash
export dataset=datasets/7scenes
for scene in chess fire heads office pumpkin redkitchen stairs; \
do wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/$scene.zip -P $dataset \
&& unzip $dataset/$scene.zip -d $dataset && unzip $dataset/$scene/'*.zip' -d $dataset/$scene; done
```
2.
Download the SIFT SfM models and DenseVLAD image pairs, courtesy of Torsten Sattler:

```bash
function download {
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
unzip $2 -d $dataset && rm $2;
}
download 1cu6KUR7WHO7G4EO49Qi3HEKU6n_yYDjb $dataset/7scenes_sfm_triangulated.zip
download 1IbS2vLmxr1N0f3CEnd_wsYlgclwTyvB1 $dataset/7scenes_densevlad_retrieval_top_10.zip
```
Alternatively, if you have ```gdown``` installed:

```bash
gdown 1cu6KUR7WHO7G4EO49Qi3HEKU6n_yYDjb $dataset/7scenes_sfm_triangulated.zip
gdown 1IbS2vLmxr1N0f3CEnd_wsYlgclwTyvB1 $dataset/7scenes_densevlad_retrieval_top_10.zip
```

3.
Download the rendered depth maps, courtesy of Eric Brachmann for [DSAC\*](https://github.com/vislearn/dsacstar):
```bash
wget https://heidata.uni-heidelberg.de/api/access/datafile/4037 -O $dataset/7scenes_rendered_depth.tar.gz
mkdir $dataset/depth/
tar xzf $dataset/7scenes_rendered_depth.tar.gz -C $dataset/depth/ && rm $dataset/7scenes_rendered_depth.tar.gz
```

## Pipeline

```bash
python3 -m hloc.pipelines.7Scenes.pipeline [--use_dense_depth]
```
By default, hloc triangulates a sparse point cloud that can be noisy in indoor environements due to image noise and lack of texture. With the flag `--use_dense_depth`, the pipeline improves the accuracy of the sparse point cloud using dense depth maps provided by the dataset. The original depth maps captured by the RGBD sensor are miscalibrated, so we use depth maps rendered from the mesh obtained by fusing the RGBD data.

## Results
We report the median error in translation/rotation in cm/deg over all scenes:
| Method \ Scene                  | Chess          | Fire           | Heads          | Office         | Pumpkin        | Kitchen        | Stairs     |
| ------------------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | ---------- |
| Active Search                   | 3/0.87         | **2**/1.01     | **1**/0.82     | 4/1.15         | 7/1.69         | 5/1.72         | 4/**1.01** |
| DSAC*                           | **2**/1.10     | **2**/1.24     | **1**/1.82     | **3**/1.15     | **4**/1.34     | 4/1.68         | **3**/1.16 |
| **SuperPoint+SuperGlue** (sfm)  | **2**/0.84     | **2**/0.93     | **1**/**0.74** | **3**/0.92     | 5/1.27         | 4/1.40         | 5/1.47     |
| **SuperPoint+SuperGlue** (RGBD) | **2**/**0.80** | **2**/**0.77** | **1**/0.79     | **3**/**0.80** | **4**/**1.07** | **3**/**1.13** | 4/1.15     |

## Citation
Please cite the following paper if you use the 7Scenes dataset:
```
@inproceedings{shotton2013scene,
  title={Scene coordinate regression forests for camera relocalization in {RGB-D} images},
  author={Shotton, Jamie and Glocker, Ben and Zach, Christopher and Izadi, Shahram and Criminisi, Antonio and Fitzgibbon, Andrew},
  booktitle={CVPR},
  year={2013}
}
```

Also cite DSAC* if you use dense depth maps with the flag `--use_dense_depth`:
```
@article{brachmann2020dsacstar,
  title={Visual Camera Re-Localization from {RGB} and {RGB-D} Images Using {DSAC}},
  author={Brachmann, Eric and Rother, Carsten},
  journal={TPAMI},
  year={2021}
}
```
