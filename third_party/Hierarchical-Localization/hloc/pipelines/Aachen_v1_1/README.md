# Aachen-Day-Night dataset v1.1

## Installation

Download the dataset from [visuallocalization.net](https://www.visuallocalization.net):
```bash
export dataset=datasets/aachen_v1.1
wget -r -np -nH -R "index.html*" --cut-dirs=4  https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/ -P $dataset
unzip $dataset/images/database_and_query_images.zip -d $dataset
unzip $dataset/aachen_v1_1.zip -d $dataset
```

## Pipeline

```bash
python3 -m hloc.pipelines.Aachen_v1_1.pipeline
```
