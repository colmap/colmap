# South Building 稀疏重建

用官方 [South Building](https://demuc.de/colmap/datasets/) 数据集，在本仓库编出来的 COLMAP 上跑增量稀疏重建。128 张同一相机的照片。当前二进制开了 CUDA，特征提取和匹配走 RTX 5090 上的 SIFT GPU。

网页看结果：`demo/south_building_viewer/`（下面有打开方式）。页面加载的是这次 GPU 重建。

| | GPU `sparse_gpu/0` | 预计算库 `sparse_demo/0` | 压缩包 `sparse/` |
|--|--|--|--|
| 已注册图像 | 128 / 128 | 128 / 128 | 128 |
| 相机 | 1，`SIMPLE_RADIAL`，3072×2304 | 同左 | 同左 |
| 焦距 / k1 | 2558.23，−0.0203 | 2559.67，−0.0205 | 2559.68，−0.0205 |
| 三维点 | 84563 | 61124 | 61514 |
| 观测数 | 502450 | 326932 | — |
| 平均轨迹长度 | 5.94 | 5.35 | — |
| 每张图平均观测 | 3925 | 2554 | — |
| 平均重投影误差 | 0.61 px | 0.51 px | — |
| 提特征 | 0.15 分钟 | 跳过（库里已有） | — |
| 穷举匹配 | 0.42 分钟 | 跳过（库里已有） | — |
| `mapper` | 3.03 分钟 | 1.66 分钟 | — |

GPU 点云：`data/south-building/sparse_gpu/points.ply`（约 1.3 MB）。点数比 2016 年的参考模型多，是因为这次用 GPU SIFT 在原图尺寸上重新提了特征。

## 数据

压缩包：<https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip>（约 399 MB）

```bash
mkdir -p data
curl -L -o data/south-building.zip \
  https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip
unzip -q data/south-building.zip -d data
```

解压后：

```text
data/south-building/
  images/          128 张 JPG，P1180141.JPG 起，3072×2304
  database.db      2016 年预计算的特征与匹配（212 MB）
  sparse/          官方参考模型（cameras/images/points3D.txt），不要覆盖
  sparse_demo/     用压缩包里的 database.db 跑的 mapper
  database_gpu.db  GPU SIFT 重新提取、匹配的数据库
  sparse_gpu/      GPU 稀疏重建
```

`data/` 在 `.gitignore` 里，数据集不会进 git。

## 这次用的程序

可执行文件：

```text
build/src/colmap/exe/colmap
```

依赖在 conda 环境 `colmap-build` 里。当前这次构建打开了 CUDA（`CUDA_ENABLED=ON`，架构 `120`，对应这台 RTX 5090），Qt 仍然关掉，所以没有 `colmap gui`。SIFT 的 GPU 路径走 CUDA，不需要显示器。

重新配置时在原来的 cmake 参数上加上：

```bash
cmake -S . -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCUDA_ENABLED=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGUI_ENABLED=OFF -DONNX_ENABLED=OFF -DTESTS_ENABLED=OFF
ninja -C build colmap
```

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate colmap-build

export DATASET=/home/jesse/workspace/colmap/data/south-building
export COLMAP=/home/jesse/workspace/colmap/build/src/colmap/exe/colmap
```

## GPU 重建

换一个新数据库，避免碰到压缩包里那份 2016 年的特征。`--FeatureExtraction.use_gpu 1` 在这个二进制上会走 CUDA SIFT；日志里能看到 `Creating SIFT GPU feature extractor` 和 `Creating SIFT GPU feature matcher`。

```bash
"$COLMAP" feature_extractor \
  --database_path "$DATASET/database_gpu.db" \
  --image_path "$DATASET/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --FeatureExtraction.use_gpu 1

"$COLMAP" exhaustive_matcher \
  --database_path "$DATASET/database_gpu.db" \
  --FeatureMatching.use_gpu 1

mkdir -p "$DATASET/sparse_gpu"
"$COLMAP" mapper \
  --database_path "$DATASET/database_gpu.db" \
  --image_path "$DATASET/images" \
  --output_path "$DATASET/sparse_gpu"

"$COLMAP" model_analyzer --path "$DATASET/sparse_gpu/0"
"$COLMAP" model_converter \
  --input_path "$DATASET/sparse_gpu/0" \
  --output_path "$DATASET/sparse_gpu/points.ply" \
  --output_type PLY
```

128 张全部注册。内参：

```text
1 SIMPLE_RADIAL 3072 2304 2558.2343062265354 1536 1152 -0.020270511957636763
```

## 用压缩包里的数据库重建

三步一样，输出写到 `sparse_demo/`，官方参考模型 `sparse/` 原样留着。这份库里的特征是 CPU 时代写进去的，所以这里显式关掉 GPU。

```bash
"$COLMAP" feature_extractor \
  --database_path "$DATASET/database.db" \
  --image_path "$DATASET/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --FeatureExtraction.use_gpu 0

"$COLMAP" exhaustive_matcher \
  --database_path "$DATASET/database.db" \
  --FeatureMatching.use_gpu 0

mkdir -p "$DATASET/sparse_demo"
"$COLMAP" mapper \
  --database_path "$DATASET/database.db" \
  --image_path "$DATASET/images" \
  --output_path "$DATASET/sparse_demo"
```

压缩包里的 `database.db` 已经有特征和几何验证过的匹配。这次 `feature_extractor` 对每张图都报 `IMAGE_EXISTS` 后立刻结束（0.000 分钟），`exhaustive_matcher` 同样秒过（0.001 分钟）。真正干活的是 `mapper`：128 张全部注册，1.66 分钟，模型在 `sparse_demo/0/`（二进制 `cameras.bin`、`images.bin`、`points3D.bin`，以及新版的 `rigs.bin`、`frames.bin`）。

想从头提特征、重新匹配，换一个空数据库，不要删官方那个：

```bash
"$COLMAP" feature_extractor \
  --database_path "$DATASET/database_fresh.db" \
  --image_path "$DATASET/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --FeatureExtraction.use_gpu 0

"$COLMAP" exhaustive_matcher \
  --database_path "$DATASET/database_fresh.db" \
  --FeatureMatching.use_gpu 0

"$COLMAP" mapper \
  --database_path "$DATASET/database_fresh.db" \
  --image_path "$DATASET/images" \
  --output_path "$DATASET/sparse_fresh"
```

128 张 3072×2304、CPU SIFT、穷举匹配，这一版会比上面慢很多。同一相机所以加了 `--ImageReader.single_camera 1`；镜头是普通径向畸变，模型用 `SIMPLE_RADIAL`。

## 看结果

```bash
"$COLMAP" model_analyzer --path "$DATASET/sparse_demo/0"

"$COLMAP" model_converter \
  --input_path "$DATASET/sparse_demo/0" \
  --output_path "$DATASET/sparse_demo/points.ply" \
  --output_type PLY
```

`model_analyzer` 打出上面那张表里的数字。这个构建没有 Qt GUI。点云和相机用仓库里的网页看：

```bash
python3 demo/south_building_viewer/export_scene.py \
  --colmap "$COLMAP" \
  --model "$DATASET/sparse_gpu/0" \
  --ply "$DATASET/sparse_gpu/points.ply" \
  --label "South Building · GPU"

python3 -m http.server 8899 --bind 127.0.0.1 \
  --directory demo/south_building_viewer
```

浏览器打开 <http://127.0.0.1:8899/>。左侧是图像列表，点一张相机会把视角挪到那台相机；拖拽旋转，滚轮缩放，右键平移。也可以勾掉「显示相机」，或拖「点大小」。`points.ply` 仍可以用 MeshLab 或 CloudCompare 打开。

文本模型（和官方 `sparse/*.txt` 同一格式）：

```bash
mkdir -p "$DATASET/sparse_demo/txt"
"$COLMAP" model_converter \
  --input_path "$DATASET/sparse_demo/0" \
  --output_path "$DATASET/sparse_demo/txt" \
  --output_type TXT
```

`cameras.txt` 一行就是内参。这次是：

```text
1 SIMPLE_RADIAL 3072 2304 2559.668950374913 1536 1152 -0.02047611810686072
```

和 2016 年参考模型的焦距、主点、k1 一致到小数点后两位。点数 61124 对参考的 61514，差在增量式重建的三角化与过滤，两条管线都注册满了 128 张。
