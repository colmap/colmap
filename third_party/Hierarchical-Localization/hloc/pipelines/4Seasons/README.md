# 4Seasons dataset

This pipeline localizes sequences from the [4Seasons dataset](https://arxiv.org/abs/2009.06364) and can reproduce our winning submission to the challenge of the [ECCV 2020 Workshop on Map-based Localization for Autonomous Driving](https://sites.google.com/view/mlad-eccv2020/home).

## Installation

Download the sequences from the [challenge webpage](https://sites.google.com/view/mlad-eccv2020/challenge) and run:
```bash
unzip recording_2020-04-07_10-20-32.zip -d datasets/4Seasons/reference
unzip recording_2020-03-24_17-36-22.zip -d datasets/4Seasons/training
unzip recording_2020-03-03_12-03-23.zip -d datasets/4Seasons/validation
unzip recording_2020-03-24_17-45-31.zip -d datasets/4Seasons/test0
unzip recording_2020-04-23_19-37-00.zip -d datasets/4Seasons/test1
```
Note that the provided scripts might modify the dataset files by deleting unused images to speed up the feature extraction

## Pipeline

The process is presented in our workshop talk, whose recording can be found [here](https://youtu.be/M-X6HX1JxYk?t=5245).

We first triangulate a 3D model from the given poses of the reference sequence:
```bash
python3 -m hloc.pipelines.4Seasons.prepare_reference
```

We then relocalize a given sequence:
```bash
python3 -m hloc.pipelines.4Seasons.localize --sequence [training|validation|test0|test1]
```

The final submission files can be found in `outputs/4Seasons/submission_hloc+superglue/`. The script will also evaluate these results if the training or validation sequences are selected.

## Results

We evaluate the localization recall at distance thresholds 0.1m, 0.2m, and 0.5m.

| Methods              | test0                  | test1                  |
| -------------------- | ---------------------- | ---------------------- |
| **hloc + SuperGlue**     | **91.8 / 97.7 / 99.2**     | **67.3 / 93.5 / 98.7**     |
| Baseline SuperGlue   | 21.2 / 33.9 / 60.0     | 12.4 / 26.5 / 54.4     |
| Baseline R2D2        | 21.5 / 33.1 / 53.0     | 12.3 / 23.7 / 42.0     |
| Baseline D2Net       | 12.5 / 29.3 / 56.7     | 7.5 / 21.4 / 47.7      |
| Baseline SuperPoint  | 15.5 / 27.5 / 47.5     | 9.0 / 19.4 / 36.4      |
