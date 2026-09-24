# COLMAP 高级玩法

这份笔记是本地实验手册：从官方数据集或自己的照片出发，把 COLMAP 里比较有意思的管线各跑一遍。命令按当前仓库的 CLI 写，细节以 `colmap <command> -h` 和 `doc/` 为准。

官方文档：<https://colmap.github.io/>  
示例数据：<https://demuc.de/colmap/datasets/>

## 先准备什么

最小可玩数据：

| 数据 | 规模 | 适合 |
|------|------|------|
| [South Building](https://demuc.de/colmap/datasets/) | 128 张，同一相机 | 稀疏重建、全局 SfM、导出 |
| [Gerrard Hall](https://demuc.de/colmap/datasets/) | 100 张，广角、变焦 | 相机模型、稠密重建 |
| 手机绕一个物体拍 40–80 张 | 自备 | 几乎所有 demo |
| 手机视频抽帧 | 自备 | 顺序匹配、回环 |
| 带 GPS 的航拍 / 街拍 | 自备 | 空间匹配、位姿先验、地理配准 |
| 等距柱状 360° 全景 | 自备 | 全景 rig |

目录约定（后面所有命令都用这个）：

```text
/path/to/project/
  images/
    image1.jpg
    ...
```

```bash
export DATASET_PATH=/path/to/project
```

从源码构建时，可执行文件在 `build/src/colmap/exe/colmap`。没有显示器或没有 CUDA 时，特征步骤加：

```bash
--FeatureExtraction.use_gpu 0 --FeatureMatching.use_gpu 0
```

看结果：

- 稀疏模型：`colmap gui` → `File > Import Model`，选 `sparse/0`。
- 稠密点云：`File > Import Model from...`，选 `dense/fused.ply`。
- 网格和贴图用 MeshLab / CloudCompare / Blender。COLMAP 自己不显示网格。
- 统计：`colmap model_analyzer --path $DATASET_PATH/sparse/0`

`mapper` 中途 Ctrl-C 会把当前模型写到输出目录（退出码 130）。同一条命令加上 `--input_path` 可以接着跑。

## 怎么选管线

| 你手里的数据 | 先跑 |
|--------------|------|
| 几十到几百张无序照片 | Demo 1，或 Demo 2 |
| 几千张、匹配图比较干净 | Demo 3 全局 SfM |
| 很大、增量太慢 | Demo 4 层次 SfM，或词袋匹配 |
| 视频、沿路径连续拍摄 | Demo 5 |
| EXIF 里有 GPS | Demo 6 |
| 双目 / 多相机刚体 | Demo 7 |
| 360° 全景 | Demo 8 |
| 已有模型，又补了新照片 | Demo 9 |
| 要网格、贴图 | Demo 10 |
| 想在 Python 里改 BA | Demo 11 |

---

## Demo 1 — 一键重建，换质量和数据预设

`automatic_reconstructor` 把提特征、匹配、稀疏、去畸变、PatchMatch、融合、网格串成一条命令。

```bash
# 稀疏就够看相机和点云时，关掉稠密，速度快很多
colmap automatic_reconstructor \
    --workspace_path $DATASET_PATH \
    --image_path $DATASET_PATH/images \
    --quality low \
    --sparse 1 \
    --dense 0
```

值得拧的旋钮：

| 参数 | 取值 | 效果 |
|------|------|------|
| `--quality` | `low` `medium` `high` `extreme` | 特征数量、图像分辨率、BA 轮数 |
| `--data_type` | `individual` `video` `internet` | 分别偏向无序照片、视频、网络图 |
| `--feature` | `sift` `aliked` `loma` `loma128` | 后三个需要编译时打开 ONNX |
| `--mapper` | `incremental` `global` `hierarchical` | 三种 SfM |
| `--mesher` | `poisson` `delaunay` | 稠密打开时的网格方法 |
| `--Mapper.ba_backend` | `ceres` `caspar` | Caspar 是实验性 GPU BA |

输出一般在工作目录的 `sparse/`、`dense/`。`low` + `--dense 0` 适合先确认这组图能不能重建，再把质量调上去。

同一套预设也可以只生成 ini，自己改完再跑：

```bash
colmap project_generator \
    --output_path $DATASET_PATH/project.ini \
    --quality medium \
    --output_type automatic_reconstructor
```

---

## Demo 2 — 手工稀疏流水线（默认 SIFT）

这是后面所有 demo 的底子。图像少于几百张时用穷举匹配。

```bash
colmap feature_extractor \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model SIMPLE_RADIAL

colmap exhaustive_matcher \
    --database_path $DATASET_PATH/database.db

mkdir -p $DATASET_PATH/sparse
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse

colmap model_analyzer --path $DATASET_PATH/sparse/0
```

同一部相机、同一套参数时加上 `--ImageReader.single_camera 1`，内参共享，通常更稳。变焦或换镜头就不要开。

已知标定可以写死，并在 mapper 里不再优化：

```bash
colmap feature_extractor \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --ImageReader.camera_params "fx,fy,cx,cy,k1,k2,p1,p2"

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 0
```

导出成能用 MeshLab 打开的点云：

```bash
colmap model_converter \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse.ply \
    --output_type PLY
```

`output_type` 还可以是 `TXT` `BIN` `NVM` `Bundler` `VRML` `R3D` `CAM`。

换成神经网络特征（需要 ONNX）。提取器和匹配器必须成对，同一个数据库里不要混 SIFT 和 ALIKED。

```bash
# ALIKED + LightGlue
colmap feature_extractor \
    --database_path $DATASET_PATH/database_aliked.db \
    --image_path $DATASET_PATH/images \
    --FeatureExtraction.type ALIKED_N16ROT \
    --AlikedExtraction.max_num_features 2048

colmap exhaustive_matcher \
    --database_path $DATASET_PATH/database_aliked.db \
    --FeatureMatching.type ALIKED_LIGHTGLUE

# LoMa（ECCV 2026）。LOMA_B 配 LOMA_B / LOMA_R / LOMA_L / LOMA_G
colmap feature_extractor \
    --database_path $DATASET_PATH/database_loma.db \
    --image_path $DATASET_PATH/images \
    --FeatureExtraction.type LOMA_B

colmap exhaustive_matcher \
    --database_path $DATASET_PATH/database_loma.db \
    --FeatureMatching.type LOMA_R
```

权重默认会下载并缓存。视角变化大、光照差的时候，LightGlue / LoMa 往往比 SIFT 暴力匹配的内点更多。想在同一库上重跑匹配，先清掉旧结果：

```bash
colmap database_cleaner \
    --database_path $DATASET_PATH/database.db \
    --type two_view_geometries \
    --type matches
```

---

## Demo 3 — 全局 SfM（GLOMAP 那条路）

增量 `mapper` 一张张加相机，最稳。`global_mapper` 先平均旋转，再做全局定位，大场景往往更快，对匹配外点更敏感，而且依赖还不错的焦距先验。

```bash
colmap feature_extractor \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images

colmap exhaustive_matcher \
    --database_path $DATASET_PATH/database.db

# 标定会原地改数据库，先拷一份
cp $DATASET_PATH/database.db $DATASET_PATH/database_global.db
colmap view_graph_calibrator \
    --database_path $DATASET_PATH/database_global.db

mkdir -p $DATASET_PATH/sparse_global
colmap global_mapper \
    --database_path $DATASET_PATH/database_global.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse_global
```

EXIF 焦距可靠时可以跳过 `view_graph_calibrator`。想单独看旋转平均：

```bash
colmap rotation_averager \
    --database_path $DATASET_PATH/database_global.db
```

和增量结果比：

```bash
colmap model_comparer \
    --input_path1 $DATASET_PATH/sparse/0 \
    --input_path2 $DATASET_PATH/sparse_global/0
```

### 和增量建图差在哪

两条路用的是同一份匹配。差别在位姿怎么解出来。

增量 `mapper` 从一对内点多的图像起步。之后每次只加一张：用已经三角化的三维点做绝对位姿，新相机再交出新的点，接着做大约 6 张图的局部平差。已注册图像涨到一定比例，再做一次全局重三角化和全局平差。一张图对不上现有模型就先跳过，所以一对坏匹配很难把已经建好的相机拧转。代价是必须一张张来，全局平差的规模随着已注册图像变大。

`global_mapper` 一次看完全部几何验证过的图像对，顺序是：

1. 把每对相对位姿分解成相对旋转和平移。这一步用本质矩阵，焦距要先差不多对。
2. 旋转平均：所有相机的绝对旋转放进同一个最小二乘。相对旋转和当前解差过 10° 的边会被丢掉。
3. 把全部观测串成轨迹。
4. 全局定位：旋转先固定，所有相机中心和三维点一起求。
5. 几轮光束法平差，再重三角化、按重投影误差滤点。

坏的相对位姿会进同一个方程组。旋转平均没有「这张图注册失败就丢掉」这一档，一簇互相一致、但整体转了 180° 的相机可以整组留下来。图干净、图像上千张时，省掉逐步注册通常更快。一百多张、外点还在的时候，速度优势不大，姿态会裂开。

Gerrard Hall 不适合做这个对比。100 张共用一个变焦焦距，先验是折中值，全局方法又靠这个焦距分解相对位姿。South Building 是同一台相机、EXIF 焦距可用、穷举匹配已经齐，旁边又有增量模型。

### South Building 上的一次对比

数据在 `data/south-building/`。全局这条复制的是 `database_gpu.db`（128 张，几何验证通过 2525 对），输出在 `sparse_global/0`。增量对照是同一份数据库的 `sparse_gpu/0`。`view_graph_calibrator` 报了 “No cameras to optimize”：焦距先验已经标上，没有相机要再估。

`global_mapper` 重建 99.1 秒，平差退回 CPU（Ceres 没有 cuDSS）。同一份数据库上增量 `mapper`、CPU 平差是 181.74 秒，打开 GPU 平差是 129.11 秒。全局这边时间大致是：旋转平均 0.03 秒，建轨迹 0.82 秒，全局定位 29.3 秒，三轮平差 26.4 秒，重三角化 42.4 秒。旋转平均丢掉 561 对相对旋转误差超过 10° 的边，剩下仍是 1 个连通分量、128 张图。

| | 增量 `sparse_gpu/0` | 全局 `sparse_global/0` |
|--|--:|--:|
| 已注册图像 | 128 / 128 | 128 / 128 |
| 三维点 | 84563 | 83688 |
| 观测 | 502450 | 484253 |
| 平均轨迹长度 | 5.94 | 5.79 |
| 每张图平均观测 | 3925 | 3783 |
| 平均重投影误差 | 0.61 px | 0.68 px |

点数和重投影几乎一样，位姿不是。`model_comparer` 把全局模型对齐到增量模型之后：

| | 旋转（度） | 相机中心 |
|--|--:|--:|
| 最小 | 0.042 | 1.5e-5 |
| 中位数 | 0.41 | 0.021 |
| 平均 | 58.4 | 3.29 |
| 90 分位 | 178.9 | 8.56 |
| 最大 | 179.1 | 9.75 |

128 张里 75 张和增量模型重合（中位旋转约 0.3°，中位中心约 1 厘米）。另外 53 张旋转差大约 180°，中心差到数米。中位数好看，是因为多数相机是对的；平均值被这 53 张拉高。注册数 128/128 看不出这组翻转。

查看器：增量模型是 <http://127.0.0.1:8899/>。全局模型自己的点云和相机在 <http://127.0.0.1:8899/?scene=global>。把全局相机对齐到增量点云之后，重合的是绿色，转了 180° 的是橙色：<http://127.0.0.1:8899/?scene=global_cmp>。

---

## Demo 4 — 大图集：词袋匹配和层次重建

几百张以上不要穷举。词袋树官方提供预训练文件：<https://demuc.de/colmap/>。不传路径时，顺序匹配的回环检测会自动下载一棵默认树。

```bash
colmap vocab_tree_matcher \
    --database_path $DATASET_PATH/database.db \
    --VocabTreeMatching.vocab_tree_path /path/to/vocab_tree.bin \
    --VocabTreeMatching.num_images 50
```

自己建树（特征数大约是视觉单词数的 10–100 倍）：

```bash
colmap vocab_tree_builder \
    --database_path $DATASET_PATH/database.db \
    --vocab_tree_path $DATASET_PATH/vocab_tree.bin
```

只做检索、不做匹配：

```bash
colmap vocab_tree_retriever \
    --database_path $DATASET_PATH/database.db \
    --vocab_tree_path /path/to/vocab_tree.bin
```

匹配图已经比较密、增量 BA 太慢时，用层次 SfM：切成有重叠的子模型，各自重建再合并。它通常不如增量稳，合并后最好再三角化和 BA 一轮。

```bash
mkdir -p $DATASET_PATH/sparse_hier
colmap hierarchical_mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse_hier

colmap point_triangulator \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse_hier/0 \
    --output_path $DATASET_PATH/sparse_hier/0

colmap bundle_adjuster \
    --input_path $DATASET_PATH/sparse_hier/0 \
    --output_path $DATASET_PATH/sparse_hier/0
```

场景被拆成多个互不连通的模型时，有公共已注册图像才能合并：

```bash
colmap model_merger \
    --input_path1 $DATASET_PATH/sparse/0 \
    --input_path2 $DATASET_PATH/sparse/1 \
    --output_path $DATASET_PATH/sparse/merged
```

---

## Demo 5 — 视频 / 沿路径拍摄

文件名必须能按顺序排，例如 `image0001.jpg`、`image0002.jpg`。顺序匹配只比相邻帧，再用词袋做回环。

```bash
# 从视频抽帧（间隔按运动快慢调）
mkdir -p $DATASET_PATH/images
ffmpeg -i video.mp4 -vf fps=2 $DATASET_PATH/images/frame%06d.jpg

colmap feature_extractor \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --ImageReader.single_camera 1

colmap sequential_matcher \
    --database_path $DATASET_PATH/database.db \
    --SequentialMatching.overlap 10 \
    --SequentialMatching.loop_detection 1 \
    --SequentialMatching.loop_detection_period 10 \
    --SequentialMatching.loop_detection_num_images 30 \
    --SequentialMatching.loop_detection_min_index_distance 30

mkdir -p $DATASET_PATH/sparse
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse
```

`overlap` 是前后各看多少帧。纯前进、没有回到旧地方，可以把 `loop_detection` 关掉。走一圈回到起点时，回环是尺度和漂移能不能闭合的关键。

一键预设：`--data_type video`。

人、车这种会动的区域可以遮掉。掩码和图片相对路径一致，文件名多一个 `.png`：`images/abc/012.jpg` 对应 `masks/abc/012.jpg.png`。

```bash
colmap feature_extractor \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --ImageReader.mask_path $DATASET_PATH/masks
```

---

## Demo 6 — GPS：空间匹配、位姿先验、地理配准

提特征时会把 EXIF GPS 存成位姿先验。位置比较准时，用空间近邻匹配，比穷举省很多。

```bash
colmap feature_extractor \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images

colmap spatial_matcher \
    --database_path $DATASET_PATH/database.db \
    --SpatialMatching.max_num_neighbors 50 \
    --SpatialMatching.max_distance 100

mkdir -p $DATASET_PATH/sparse
colmap pose_prior_mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse \
    --overwrite_priors_covariance 1 \
    --prior_position_std_x 1.0 \
    --prior_position_std_y 1.0 \
    --prior_position_std_z 2.0
```

`pose_prior_mapper` 就是带位置约束的增量重建。手机 GPS 垂直方向通常更差，可以把 `prior_position_std_z` 设大一点。

### Caliterra：ALIKED + GPU 暴力匹配 + 位姿先验

OpenDroneMap 的 [Caliterra](https://github.com/OpenDroneMap/odm_data_caliterra)：77 张 Canon PowerShot SX260 HS，4000×3000，EXIF 里有 GPS。数据在 `data/caliterra/`。提点用 `ALIKED_N16ROT`（GPU，最多 2048 点），穷举匹配用 `ALIKED_BRUTEFORCE`（GPU，`min_cossim` 0.85），建图用 `pose_prior_mapper`，水平标准差 2 米、高程 5 米，`ba_use_gpu 1`。

| 步骤 | 墙钟 |
|--|--:|
| 提特征 | 4.99 秒 |
| 穷举匹配 | 3.92 秒 |
| `pose_prior_mapper` | 52.63 秒 |

77 张里注册了 50 张，拆成两块，都在同一套东-北-天坐标里（大约东 −20 到 80 米）：

| 模型 | 图像 | 三维点 | 重投影 |
|--|--:|--:|--:|
| `sparse_aliked_bf/0` | 17 | 4063 | 0.70 px |
| `sparse_aliked_bf/1` | 33 | 9211 | 0.61 px |

几何验证通过 253 对：块 0 内部 79 对，块 1 内部 135 对，两块之间 0 对。验证图的连通分量是 44、17、7、3、3，再加三张孤图。17 张那块就是单独的一个分量，33 张是 44 张那个分量里注册完成的部分。没有跨块内点，增量建图接不过去。`model_merger` 要靠共同图像估相似变换，这两块共同图像数是 0，合不上。GPS 已经把它们放进同一个米制坐标系，可视化是把两块画在一起，轨迹并没有连上。

页面：<http://127.0.0.1:8899/?scene=caliterra>。蓝相机是块 0，橙相机是块 1。单独看用 `?scene=caliterra0` 和 `?scene=caliterra1`。服务器在 `demo/south_building_viewer/` 里开着。

```bash
"$COLMAP" feature_extractor \
  --database_path "$DATASET/database_aliked_bf.db" \
  --image_path "$DATASET/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --FeatureExtraction.type ALIKED_N16ROT \
  --FeatureExtraction.use_gpu 1 \
  --AlikedExtraction.max_num_features 2048

"$COLMAP" exhaustive_matcher \
  --database_path "$DATASET/database_aliked_bf.db" \
  --FeatureMatching.type ALIKED_BRUTEFORCE \
  --FeatureMatching.use_gpu 1

"$COLMAP" pose_prior_mapper \
  --database_path "$DATASET/database_aliked_bf.db" \
  --image_path "$DATASET/images" \
  --output_path "$DATASET/sparse_aliked_bf" \
  --Mapper.ba_use_gpu 1 \
  --overwrite_priors_covariance 1 \
  --prior_position_std_x 2 \
  --prior_position_std_y 2 \
  --prior_position_std_z 5
```

已经有一个任意坐标系的稀疏模型、只想事后贴到地图上：至少 3 张图的相机中心。文本格式：

```text
image_name1.jpg lat1 lon1 alt1
image_name2.jpg lat2 lon2 alt2
image_name3.jpg lat3 lon3 alt3
```

```bash
colmap model_aligner \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/geo \
    --database_path $DATASET_PATH/database.db \
    --ref_is_gps 1 \
    --alignment_type enu \
    --alignment_max_error 3.0
```

`--alignment_type` 用 `enu`（东-北-天，第一张 GPS 当原点）或 `ecef`。手写坐标文件时把 `--database_path` 换成 `--ref_images_path`。

室内、建筑立面还可以按曼哈顿假设摆正坐标轴（重力方向 + 主水平方向）：

```bash
colmap model_orientation_aligner \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/aligned
```

按包围盒切开大模型：

```bash
colmap model_splitter \
    --input_path $DATASET_PATH/sparse/geo \
    --output_path $DATASET_PATH/sparse/tiles \
    --split_type extent \
    --split_params 50,50,50
```

---

## Demo 7 — 多相机 rig

同一时刻曝光的多相机（双目、车载环视）要建成一个 rig：参考相机是 rig 原点，其余相机相对它固定。同一帧的文件名必须相同：

```text
images/
  rig1/camera1/image0001.jpg
  rig1/camera2/image0001.jpg
```

```bash
colmap feature_extractor \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --ImageReader.single_camera_per_folder 1

colmap rig_configurator \
    --database_path $DATASET_PATH/database.db \
    --rig_config_path $DATASET_PATH/rig_config.json
```

外参已知时 `rig_config.json` 类似：

```json
[
  {
    "cameras": [
      {
        "image_prefix": "rig1/camera1/",
        "ref_sensor": true
      },
      {
        "image_prefix": "rig1/camera2/",
        "cam_from_rig_rotation": [0.7071067811865475, 0.0, 0.7071067811865476, 0.0],
        "cam_from_rig_translation": [0.12, 0.0, 0.0]
      }
    ]
  }
]
```

旋转是 Hamilton 四元数 `(w, x, y, z)` 的 `cam_from_rig`。配置 rig **之后**再匹配。视频式采集用 `sequential_matcher`，它会按帧去配相邻帧里的所有相机。

```bash
colmap sequential_matcher \
    --database_path $DATASET_PATH/database.db

mkdir -p $DATASET_PATH/sparse
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse \
    --Mapper.ba_refine_sensor_from_rig 0
```

外参很准时加上 `--FeatureMatching.rig_verification 1`。同一帧的相机视野完全不重叠时，用 `--FeatureMatching.skip_image_pairs_in_same_frame 1`。

外参未知：JSON 里只写 `image_prefix` 和 `ref_sensor`，先正常重建，再用重建结果估计平均 rig 外参，第二次重建时固定 `--Mapper.ba_refine_sensor_from_rig 0`。完整步骤在 `doc/rigs.rst`。

---

## Demo 8 — 360° 全景

等距柱状全景会被切成一组已知内外参的虚拟针孔相机，当成 rig 来重建。需要 pycolmap，以及可选的全景渲染依赖。

```bash
# 若仓库里已经装过 COLMAP 到 ./install
colmap_DIR=./install ./python/incremental_build.sh
pip install 'pycolmap[panorama]'

python python/examples/panorama_sfm.py \
    --input_image_path /path/to/panoramas \
    --output_path /path/to/pano_out \
    --matcher SEQUENTIAL \
    --mapper INCREMENTAL \
    --pano_render_type PERSPECTIVE_OVERLAPPING
```

无序全景把 `--matcher` 换成 `EXHAUSTIVE` 或 `VOCABTREE`，`--mapper` 可以换成 `GLOBAL`。脚本要和当前 COLMAP / pycolmap 版本一致。

---

## Demo 9 — 给已有模型补图

新照片放进 `images/`，对**同一个**数据库再提一次特征（已有图片会跳过），然后只注册新图。这一步不做 BA，也不做三角化。

```bash
colmap feature_extractor \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images

colmap exhaustive_matcher \
    --database_path $DATASET_PATH/database.db

colmap image_registrator \
    --database_path $DATASET_PATH/database.db \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/registered

colmap point_triangulator \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/registered \
    --output_path $DATASET_PATH/sparse/triangulated

colmap bundle_adjuster \
    --input_path $DATASET_PATH/sparse/triangulated \
    --output_path $DATASET_PATH/sparse/refined
```

也可以直接 `mapper --input_path $DATASET_PATH/sparse/0`，让增量流程自己注册、三角化并做 BA。

删掉某张图、或按重投影误差滤点：

```bash
colmap image_deleter \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/pruned \
    --image_names_path $DATASET_PATH/drop_list.txt

colmap point_filtering \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/filtered
```

---

## Demo 10 — 稠密点云、网格、简化、贴图

稀疏模型先去畸变，PatchMatch 需要 CUDA。

```bash
mkdir -p $DATASET_PATH/dense
colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000

colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply

colmap poisson_mesher \
    --input_path $DATASET_PATH/dense/fused.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson.ply

# 需要 CGAL
colmap delaunay_mesher \
    --input_path $DATASET_PATH/dense \
    --output_path $DATASET_PATH/dense/meshed-delaunay.ply

colmap advancing_front_mesher \
    --input_path $DATASET_PATH/dense \
    --output_path $DATASET_PATH/dense/meshed-advancing-front.ply

colmap mesh_simplifier \
    --input_path $DATASET_PATH/dense/meshed-poisson.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson-simplified.ply \
    --MeshSimplification.target_face_ratio 0.25

colmap mesh_texturer \
    --workspace_path $DATASET_PATH/dense \
    --input_path $DATASET_PATH/dense/meshed-poisson-simplified.ply \
    --output_path $DATASET_PATH/dense/textured
```

`--max_image_size 2000` 是速度和细节的折中。Poisson 表面更光滑，Delaunay 更贴观测、洞更少。贴图之前先简化，不然图集会非常大。

### Gerrard Hall：稠密重建

数据包在 `data/gerrard-hall/`，100 张 Canon EOS 5D Mark II，5616×3744。压缩包里已经有稀疏模型，100 张全部注册，这一次直接拿它做稠密，没有重跑提点和 `mapper`。

内参是估出来的，但 100 张共用一个 `OPENCV` 相机。数据库里的初值来自 EXIF：`fx = fy = 3744`，主点在图像中心 `(2808, 1872)`，畸变全是 0，`prior_focal_length = 1`。稀疏结果 `sparse/cameras.txt` 里焦距和畸变已经变了，主点没动：

```text
1 OPENCV 5616 3744 3838.27 3837.22 2808 1872 -0.110339 0.079547 0.000116211 0.00029483
```

默认光束法平差会优化焦距和畸变（`ba_refine_focal_length`、`ba_refine_extra_params`）。主点默认固定（`ba_refine_principal_point` 为 false），所以还在图像中心。这组图是变焦的，共用一个焦距是整段变焦的折中。要每张图各自的内参，提点时加 `--ImageReader.single_camera_per_image 1`。

去畸变把长边收到 2000，然后 CUDA PatchMatch（光度 + 几何一致）、融合、Poisson 网格：

| 步骤 | 墙钟 | 结果 |
|--|--:|--|
| `image_undistorter` | 8.26 秒 | `dense/images`、`dense/sparse` |
| `patch_match_stereo` | 3403 秒 | 100 张深度图和法向 |
| `stereo_fusion` | 19.48 秒 | `dense/fused.ply`，3,572,972 点 |
| `poisson_mesher` | 1612 秒 | `dense/meshed-poisson.ply`，8,601,459 顶点，17,133,369 面 |
| `mesh_simplifier` | 197 秒 | `dense/meshed-poisson-simplified.ply`，181,832 顶点，342,666 面（`target_face_ratio 0.02`） |
| `mesh_texturer` | 10.75 秒 | `dense/textured/mesh.ply` + `texture.png`，图集 16384×4283。`texture_scale_factor 1`，视角平滑 20 轮，`inpaint_radius 1`。选视角时用深度图去掉被网格挡住的面，并用多视图颜色一致性压低穿过树枝的照片 |

`fused.ply` 带法向，约 93 MB。Poisson 网格约 418 MB，简化后约 6.4 MB，贴图网格加图集约 60 MB。查看器在 <http://127.0.0.1:8899/?scene=gerrard>。有网格时默认只显示贴图表面，点云可以再打开叠上去对比。贴图关掉时网格用光照显示形状。相机来自去畸变后的稀疏模型。

```bash
"$COLMAP" image_undistorter \
  --image_path "$DATASET/images" \
  --input_path "$DATASET/sparse" \
  --output_path "$DATASET/dense" \
  --output_type COLMAP \
  --max_image_size 2000

"$COLMAP" patch_match_stereo \
  --workspace_path "$DATASET/dense" \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true

"$COLMAP" stereo_fusion \
  --workspace_path "$DATASET/dense" \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path "$DATASET/dense/fused.ply"

"$COLMAP" poisson_mesher \
  --input_path "$DATASET/dense/fused.ply" \
  --output_path "$DATASET/dense/meshed-poisson.ply"

"$COLMAP" mesh_simplifier \
  --input_path "$DATASET/dense/meshed-poisson.ply" \
  --output_path "$DATASET/dense/meshed-poisson-simplified.ply" \
  --MeshSimplification.target_face_ratio 0.02

"$COLMAP" mesh_texturer \
  --workspace_path "$DATASET/dense" \
  --input_path "$DATASET/dense/meshed-poisson-simplified.ply" \
  --output_path "$DATASET/dense/textured" \
  --MeshTextureMapping.texture_scale_factor 1 \
  --MeshTextureMapping.view_selection_smoothing_iterations 20 \
  --MeshTextureMapping.inpaint_radius 1
```

立体校正（两台已经标定的相机，给传统视差算法用）：

```bash
colmap image_rectifier \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/rectified
```

---

## Demo 11 — Python：改重建循环

仓库里这几个脚本可以直接改：

| 脚本 | 玩法 |
|------|------|
| `python/examples/custom_incremental_pipeline.py` | 用 Python 重写增量循环，中间可以插自己的逻辑 |
| `python/examples/custom_bundle_adjustment.py` | 换 BA 的残差和固定哪些参数 |
| `python/examples/visualize_model.py` | Open3D 看稀疏模型和相机 |
| `python/examples/panorama_sfm.py` | Demo 8 |
| `python/pycolmap/panorama.py` | 全景渲染和 rig 的实现 |

最小读取：

```python
import pycolmap

rec = pycolmap.Reconstruction("/path/to/project/sparse/0")
print(rec.summary())
for image_id, image in rec.images.items():
    print(image.name, image.cam_from_world.translation)
```

自己接 Ctrl-C / 抢占信号：

```python
import signal
import pycolmap

token = pycolmap.CancellationToken()
signal.signal(signal.SIGTERM, lambda *_: token.cancel())
pycolmap.incremental_mapping(
    database_path,
    image_path,
    output_path,
    cancellation_token=token,
)
```

---

## 一组可以连续做完的下午

用 South Building，或者自己绕物体拍的 60 张：

1. Demo 1，`--quality low --dense 0`，确认能出模型。
2. Demo 2，`--quality` 换成手工流水线，`model_analyzer` 看注册了多少张、平均重投影误差。
3. 拷贝数据库，跑 Demo 3，`model_comparer` 看两种 SfM 差多少。
4. 若机器有 CUDA，用 Demo 2 的 `sparse/0` 跑 Demo 10 的前半段，得到 `fused.ply`。
5. `model_converter` 导出 PLY，或 `visualize_model.py` 在 Open3D 里转一转。

视频和 GPS 各自换一组数据再跑 Demo 5、Demo 6。全景和 rig 需要对应的采集方式，硬套普通照片没有意义。

每个命令的全部参数：`colmap <command> -h`。概念和故障排查在 `doc/tutorial.rst`、`doc/faq.rst`、`doc/rigs.rst`、`doc/features.rst`。
