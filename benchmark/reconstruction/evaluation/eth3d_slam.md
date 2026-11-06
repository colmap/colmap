# ETH3D-SLAM

The benchmark uses the 55 training sequences from GLOMAP Table S6. From the
COLMAP checkout, download the dataset and evaluate with:

```sh
python benchmark/reconstruction/download.py --datasets eth3d-slam
python benchmark/reconstruction/evaluate.py --datasets eth3d-slam \
    --mapper global --colmap_path /path/to/colmap
```

Use `--scenes <scene_name>` to select a sequence. The images are rectified
and calibrated. RGB-D archives are also downloaded for depth experiments;
their axial depths use units of 1/5000 m, with zero marking missing values.

GT poses are interpolated following [ETH3D's evaluator](https://github.com/ETH3D/slam-evaluation)
and [batch-script settings](https://www.eth3d.net/data/slam/evaluate_eth3d_slam_results.py):
maximum gaps of 1/75 s for motion-capture scenes and 1/11.25 s for `sfm_*` scenes.
Images without supported GT remain reconstruction inputs but are excluded from
pose evaluation. Metrics follow the COLMAP benchmark.

Sources: https://www.eth3d.net/slam_datasets and GLOMAP supplementary Table S6.
