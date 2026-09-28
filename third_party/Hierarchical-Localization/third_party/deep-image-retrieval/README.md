# Deep Image Retrieval

This repository contains the models and the evaluation scripts (in Python3 and Pytorch 1.0+) of the papers:

**[1] End-to-end Learning of Deep Visual Representations for Image Retrieval**
Albert Gordo, Jon Almazan, Jerome Revaud, Diane Larlus, IJCV 2017 [\[PDF\]](https://arxiv.org/abs/1610.07940)

**[2] Learning with Average Precision: Training Image Retrieval with a Listwise Loss**
Jerome Revaud, Jon Almazan, Rafael S. Rezende, Cesar de Souza, ICCV 2019 [\[PDF\]](https://arxiv.org/abs/1906.07589)


Both papers tackle the problem of image retrieval and explore different ways to learn deep visual representations for this task. In both cases, a CNN is used to extract a feature map that is aggregated into a compact, fixed-length representation by a global-aggregation layer*. Finally, this representation is first projected using a FC layer, and L2 normalized so images can be efficiently compared with the dot product.


![dir_network](https://user-images.githubusercontent.com/228798/59742085-aae19f80-9221-11e9-8063-e5f2528c304a.png)

All components in this network, including the aggregation layer, are differentiable, which makes it end-to-end trainable for the end task. In [1], a Siamese architecture that combines three streams with a triplet loss was proposed to train this network.  In [2], this work was extended by replacing the triplet loss with a new loss that directly optimizes for Average Precision.

![Losses](https://user-images.githubusercontent.com/228798/59742025-7a9a0100-9221-11e9-9d58-1494716e9071.png)

\* Originally, [1] used R-MAC pooling [3] as the global-aggregation layer. However, due to its efficiency and better performace we have replaced the R-MAC pooling layer with the Generalized-mean pooling layer (GeM) proposed in [4]. You can find the original implementation of [1] in Caffe following [this link](https://europe.naverlabs.com/Research/Computer-Vision/Learning-Visual-Representations/Deep-Image-Retrieval/).


## News

- **(6/9/2019)** AP loss, Tie-aware AP loss, Triplet Margin loss, and Triplet LogExp loss added for reference
- **(5/9/2019)** Update evaluation and AP numbers for all the benchmarks
- **(22/7/2019)** Paper **_Learning with Average Precision: Training Image Retrieval with a Listwise Loss_** accepted at ICCV 2019


## Pre-requisites

In order to run this toolbox you will need:

- Python3 (tested with Python 3.7.3)
- PyTorch (tested with version 1.4)
- The following packages: numpy, matplotlib, tqdm, scikit-learn

With conda you can run the following commands:

```
conda install numpy matplotlib tqdm scikit-learn
conda install pytorch torchvision -c pytorch
```

## Installation

```
# Download the code
git clone https://github.com/naver/deep-image-retrieval.git

# Create env variables
cd deep-image-retrieval
export DIR_ROOT=$PWD
export DB_ROOT=/PATH/TO/YOUR/DATASETS
# for example: export DB_ROOT=$PWD/dirtorch/data/datasets
```


## Evaluation


### Pre-trained models

The table below contains the pre-trained models that we provide with this library, together with their mAP performance on some of the most well-know image retrieval benchmakrs: [Oxford5K](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/), [Paris6K](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/), and their Revisited versions ([ROxford5K and RParis6K](https://github.com/filipradenovic/revisitop)).


| Model | Oxford5K | Paris6K |  ROxford5K (med/hard) | RParis6K (med/hard) |
|---	|:-:|:-:|:-:|:-:|
|  [Resnet101-TL-MAC](https://drive.google.com/file/d/13MUGNwn_CYGZvqDBD8FGD8fVYxThsSDg/view?usp=sharing) |  85.6	| 90.1 |  63.3 / 35.7 	|   76.6 / 55.5  |
|  [Resnet101-TL-GeM](https://drive.google.com/open?id=1vhm1GYvn8T3-1C4SPjPNJOuTU9UxKAG6) | 85.7 | **93.4** | 64.5 / 40.9 |  78.8 / 59.2  |
|  [Resnet50-AP-GeM](https://drive.google.com/file/d/1oPtE_go9tnsiDLkWjN4NMpKjh-_md1G5/view?usp=sharing) | 87.7 	| 91.9 |  65.5 / 41.0 | 77.6 / 57.1 |
|  [Resnet101-AP-GeM](https://drive.google.com/open?id=1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy) | **89.1** | **93.0** | **67.1** / **42.3** |  **80.3**/**60.9** |
|  [Resnet101-AP-GeM-LM18](https://drive.google.com/open?id=1r76NLHtJsH-Ybfda4aLkUIoW3EEsi25I)** |  88.1	| **93.1** | 66.3 / **42.5**	|   **80.2** / **60.8**  |


The name of the model encodes the backbone architecture of the network and the loss that has been used to train it (TL for triplet loss and AP for Average Precision loss). All models use **Generalized-mean pooling (GeM)** [3] as the global pooling mechanism, except for the model in the first row that uses MAC [3] \(i.e. max-pooling), and have been trained on the **Landmarks-clean** [1] dataset (the clean version of the [Landmarks dataset](http://sites.skoltech.ru/compvision/projects/neuralcodes/)) directly **fine-tuning from ImageNet**. These numbers have been obtained using a **single resolution** and applying **whitening** to the output features (which has also been learned on Landmarks-clean). For a detailed explanation of all the hyper-parameters see [1] and [2] for the triplet loss and AP loss models, respectively.

** For the sake of completeness, we have added an extra model, `Resnet101-AP-LM18`, which has been trained on the [Google-Landmarks Dataset](https://www.kaggle.com/google/google-landmarks-dataset), a large dataset consisting of more than 1M images and 15K classes.

### Reproducing the results

The script `test_dir.py` can be used to evaluate the pre-trained models provided and to reproduce the results above:

```
python -m dirtorch.test_dir --dataset DATASET --checkpoint PATH_TO_MODEL \
		[--whiten DATASET] [--whitenp POWER] [--aqe ALPHA-QEXP] \
		[--trfs TRANSFORMS] [--gpu ID] [...]
```

- `--dataset`: selects the dataset (eg.: Oxford5K, Paris6K, ROxford5K, RParis6K) [**required**]
- `--checkpoint`: path to the model weights [**required**]
- `--whiten`: applies whitening to the output features [default 'Landmarks_clean']
- `--whitenp`: whitening power [default: 0.25]
- `--aqe`: alpha-query expansion parameters [default: None]
- `--trfs`: input image transformations (can be used to apply multi-scale) [default: None]
- `--gpu`: selects the GPU ID (-1 selects the CPU)

For example, to reproduce the results of the Resnet101-AP_loss model on the RParis6K dataset download the model `Resnet-101-AP-GeM.pt` from [here](https://drive.google.com/open?id=1mi50tG6oXY1eE9yJnmGCPdTmlIjG7mr0) and run:

```
cd $DIR_ROOT
export DB_ROOT=/PATH/TO/YOUR/DATASETS

python -m dirtorch.test_dir --dataset RParis6K \
		--checkpoint dirtorch/data/Resnet101-AP-GeM.pt \
		--whiten Landmarks_clean --whitenp 0.25 --gpu 0
```

And you should see the following output:

```
>> Evaluation...
 * mAP-easy = 0.907568
 * mAP-medium = 0.803098
 * mAP-hard = 0.608556
```

**Note:** this script integrates an automatic downloader for the Oxford5K, Paris6K, ROxford5K, and RParis6K datasets (kudos to Filip Radenovic ;)). The datasets will be saved in `$DB_ROOT`.

## Feature extractor

You can also use the pre-trained models to extract features from your own datasets or collection of images. For that we provide the script `feature_extractor.py`:

```
python -m dirtorch.extract_features --dataset DATASET --checkpoint PATH_TO_MODEL \
		--output PATH_TO_FILE [--whiten DATASET] [--whitenp POWER] \
		[--trfs TRANSFORMS] [--gpu ID] [...]
```

where `--output` is used to specify the destination where the features will be saved. The rest of the parameters are the same as seen above.

For example, this is how the script can be used to extract a feature representation for each one of the images in the RParis6K dataset using the `Resnet-101-AP-GeM.pt` model, and storing them in `rparis6k_features.npy`:

```
cd $DIR_ROOT
export DB_ROOT=/PATH/TO/YOUR/DATASETS

python -m dirtorch.extract_features --dataset RParis6K \
		--checkpoint dirtorch/data/Resnet101-AP-GeM.pt \
		--output rparis6k_features.npy \
		--whiten Landmarks_clean --whitenp 0.25 --gpu 0
```

The library also provides a **generic class dataset** (`ImageList`) that allows you to specify the list of images by providing a simple text file.

```
--dataset 'ImageList("PATH_TO_TEXTFILE" [, "IMAGES_ROOT"])'
```

Each row of the text file should contain a single path to a given image:

```
/PATH/TO/YOUR/DATASET/images/image1.jpg
/PATH/TO/YOUR/DATASET/images/image2.jpg
/PATH/TO/YOUR/DATASET/images/image3.jpg
/PATH/TO/YOUR/DATASET/images/image4.jpg
/PATH/TO/YOUR/DATASET/images/image5.jpg
```

Alternatively, you can also use relative paths, and use `IMAGES_ROOT` to specify the root folder.

## Feature extraction with kapture datasets

Kapture is a pivot file format, based on text and binary files, used to describe SFM (Structure From Motion) and more generally sensor-acquired data.

It is available at https://github.com/naver/kapture.
It contains conversion tools for popular formats and several popular datasets are directly available in kapture.

It can be installed with:
```bash
pip install kapture
```

Datasets can be downloaded with:
```bash
kapture_download_dataset.py update
kapture_download_dataset.py list
# e.g.: install mapping and query of Extended-CMU-Seasons_slice22
kapture_download_dataset.py install "Extended-CMU-Seasons_slice22_*"
```
If you want to convert your own dataset into kapture, please find some examples [here](https://github.com/naver/kapture/blob/master/doc/datasets.adoc).

Once installed, you can extract global features for your kapture dataset with:
```bash
cd $DIR_ROOT
python -m dirtorch.extract_kapture --kapture-root pathto/yourkapturedataset --checkpoint dirtorch/data/Resnet101-AP-GeM-LM18.pt --gpu 0
```

Run `python -m dirtorch.extract_kapture --help` for more information on the extraction parameters. 

## Citations

Please consider citing the following papers in your publications if this helps your research.

```
@article{GARL17,
 title = {End-to-end Learning of Deep Visual Representations for Image Retrieval},
 author = {Gordo, A. and Almazan, J. and Revaud, J. and Larlus, D.}
 journal = {IJCV},
 year = {2017}
}

@inproceedings{RARS19,
 title = {Learning with Average Precision: Training Image Retrieval with a Listwise Loss},
 author = {Revaud, J. and Almazan, J. and Rezende, R.S. and de Souza, C.R.}
 booktitle = {ICCV},
 year = {2019}
}
```

## Contributors

This library has been developed by Jerome Revaud, Rafael de Rezende, Cesar de Souza, Diane Larlus, and Jon Almazan at **[Naver Labs Europe](https://europe.naverlabs.com)**.


**Special thanks to [Filip Radenovic](https://github.com/filipradenovic).** In this library, we have used the ROxford5K and RParis6K downloader from his awesome **[CNN-imageretrieval repository](https://github.com/filipradenovic/cnnimageretrieval-pytorch)**. Consider checking it out if you want to train your own models for image retrieval!

## References

[1] Gordo, A., Almazan, J., Revaud, J., Larlus, D., [End-to-end Learning of Deep Visual Representations for Image Retrieval](https://arxiv.org/abs/1610.07940). IJCV 2017

[2] Revaud, J., Almazan, J., Rezende, R.S., de Souza, C., [Learning with Average Precision: Training Image Retrieval with a Listwise Loss](https://arxiv.org/abs/1906.07589). ICCV 2019

[3] Tolias, G., Sicre, R., Jegou, H., [Particular object retrieval with integral max-pooling of CNN activations](https://arxiv.org/abs/1511.05879). ICLR 2016

[4] Radenovic, F., Tolias, G., Chum, O., [Fine-tuning CNN Image Retrieval with No Human Annotation](https://arxiv.org/pdf/1711.02512). TPAMI 2018
