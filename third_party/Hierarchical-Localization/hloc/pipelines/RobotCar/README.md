# RobotCar Seasons dataset

## Installation

Download the dataset from [visuallocalization.net](https://www.visuallocalization.net):
```bash
export dataset=datasets/robotcar
wget -r -np -nH -R "index.html*" --cut-dirs=4  https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/ -P $dataset
for condition in $dataset/images/*.zip; do unzip condition -d $dataset/images/; done
```

## Pipeline

```bash
python3 -m hloc.pipelines.RobotCar.pipeline
```
