# Extended CMU Seasons dataset

## Installation

Download the dataset from [visuallocalization.net](https://www.visuallocalization.net):
```bash
export dataset=datasets/cmu_extended
wget -r -np -nH -R "index.html*" --cut-dirs=4  https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/ -P $dataset
for slice in $dataset/*.tar; do tar -xf $slice -C $dataset && rm $slice; done
```

## Pipeline

```bash
python3 -m hloc.pipelines.CMU.pipeline
```
