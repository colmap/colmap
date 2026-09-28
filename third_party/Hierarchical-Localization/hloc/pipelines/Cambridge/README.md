# Cambridge Landmarks dataset

## Installation

Download the dataset from the [PoseNet project page](http://mi.eng.cam.ac.uk/projects/relocalisation/):
```bash
export dataset=datasets/cambridge
export scenes=( "KingsCollege" "OldHospital" "StMarysChurch" "ShopFacade" "GreatCourt" )
export IDs=( "251342" "251340" "251294" "251336" "251291" )
for i in "${!scenes[@]}"; do
wget https://www.repository.cam.ac.uk/bitstream/handle/1810/${IDs[i]}/${scenes[i]}.zip -P $dataset \
&& unzip $dataset/${scenes[i]}.zip -d $dataset && rm $dataset/${scenes[i]}.zip; done
```

Download the SIFT SfM models, courtesy of Torsten Sattler:
```bash
export fileid=1esqzZ1zEQlzZVic-H32V6kkZvc4NeS15
export filename=$dataset/CambridgeLandmarks_Colmap_Retriangulated_1024px.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$fileid" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$fileid" -O $filename && rm -rf /tmp/cookies.txt
unzip $filename -d $dataset
```

## Pipeline

```bash
python3 -m hloc.pipelines.Cambridge.pipeline
```

## Results
We report the median error in translation/rotation in cm/deg over all scenes:
| Method \ Scene           | Court           | King's          | Hospital        | Shop           | St. Mary's     |
| ------------------------ | --------------- | --------------- | --------------- | -------------- | -------------- |
| Active Search            | 24/0.13         | 13/0.22         | 20/0.36         | **4**/0.21     | 8/0.25         |
| DSAC*                    | 49/0.3          | 15/0.3          | 21/0.4          | 5/0.3          | 13/0.4         |
| **SuperPoint+SuperGlue** | **17**/**0.11** | **12**/**0.21** | **14**/**0.30** | **4**/**0.19** | **7**/**0.22** |

## Citation

Please cite the following paper if you use the Cambridge Landmarks dataset:
```
@inproceedings{kendall2015posenet,
  title={{PoseNet}: A convolutional network for real-time {6-DoF} camera relocalization},
  author={Kendall, Alex and Grimes, Matthew and Cipolla, Roberto},
  booktitle={ICCV},
  year={2015}
}
```
