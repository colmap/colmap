# Aachen-Day-Night dataset

## Installation

Download the dataset from [visuallocalization.net](https://www.visuallocalization.net):
```bash
export dataset=datasets/aachen
wget -r -np -nH -R "index.html*,aachen_v1_1.zip" --cut-dirs=4  https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/ -P $dataset
unzip $dataset/images/database_and_query_images.zip -d $dataset
```

## Pipeline

```bash
python3 -m hloc.pipelines.Aachen.pipeline
```
