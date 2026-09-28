# South Building：CPU 与 GPU 加速

同一套 128 张 South Building 照片、同一个带 CUDA 的 COLMAP 二进制，把稀疏重建拆成提特征、穷举匹配、增量建图三段计时。建图再多跑一档：Ceres 稠密 Schur 放到 GPU 上。

操作步骤和网页查看在 `SOUTH_BUILDING_SPARSE.md`。这里只记这次加速实验。

## 结论

整条稀疏重建，GPU 只接管提特征和匹配时大约 **4.9 倍**（1049 秒对 216 秒）。匹配再打开 Ceres 的 GPU 稠密求解后大约 **6.4 倍**（1049 秒对 163 秒）。时间主要省在穷举匹配上。

| 阶段 | CPU | GPU | 加速比 |
|--|--:|--:|--:|
| 提特征 | 82.00 秒 | 9.18 秒 | 8.9× |
| 穷举匹配 | 826.68 秒 | 24.90 秒 | 33.2× |
| 建图 | 140.46 秒 | 181.74 秒 | 0.77× |
| 合计 | 1049.14 秒 | 215.82 秒 | 4.9× |
| 建图，同一份 GPU 特征，`ba_use_gpu 1` | — | 129.11 秒 | 相对上面那次建图 1.4× |
| 合计，含 GPU 平差 | 1049.14 秒 | 163.19 秒 | 6.4× |

提特征加匹配两段加起来是 26.7 倍（908.68 秒对 34.08 秒）。

建图那一列的 0.77× 不能当成「GPU 建图更慢」的算法结论。CPU 行用的是 CPU SIFT 的数据库，GPU 行用的是 GPU SIFT 的数据库，特征数量和匹配内点都不同。同一份特征上，把全局平差的稠密求解放到 GPU，建图从 181.74 秒降到 129.11 秒。

## 机器与程序

| | |
|--|--|
| GPU | NVIDIA GeForce RTX 5090，compute capability 12.0，驱动 595.84，32607 MiB |
| CPU | 32 核 |
| CUDA | 13.2（`/usr/local/cuda-13.2`） |
| COLMAP | `4.3.0.dev0`，commit `8f23d8ec`，`CUDA_ENABLED=ON`，`CMAKE_CUDA_ARCHITECTURES=120` |
| 二进制 | `build/src/colmap/exe/colmap` |
| Ceres | 2.2.0，装在 `/home/jesse/.local/ceres-cuda`，`USE_CUDA=ON`，架构 120。链了 cuBLAS、cuSOLVER、cuSPARSE |
| 没有的库 | cuDSS。Ceres 2.2 也没有 `CUDA_SPARSE` |

conda 环境 `colmap-build`。GUI 关掉了。SIFT 的 GPU 路径走 CUDA，不需要显示器。

提特征和匹配的对比用的是同一份二进制。CPU 建图发生在换成 CUDA Ceres 之前，平差走 CPU。`ba_use_gpu 1` 那次是换成 CUDA Ceres 之后、同一份 `database_gpu.db` 重跑的。

## 数据与开关

128 张 JPEG，Panasonic DMC-TZ3，3072×2304。压缩包和目录说明见 `SOUTH_BUILDING_SPARSE.md`。

两边都是：

- `--ImageReader.single_camera 1`
- `--ImageReader.camera_model SIMPLE_RADIAL`
- `max_image_size` 保持默认 `-1`。SIFT 的有效上限是 3200，这批图不超过它，按原分辨率提特征
- 穷举匹配。图像对数量是 \(128\times127/2 = 8128\)，数据库里的 `matches` 行数也是 8128
- `mapper` 其余选项默认。局部平差每次大约 6 张图，全局平差在已注册图像涨到上次的 1.1 倍时触发

GPU 侧显式打开 `--FeatureExtraction.use_gpu 1` 和 `--FeatureMatching.use_gpu 1`。日志分别是 `Creating SIFT GPU feature extractor` 和 `Creating SIFT GPU feature matcher`。CPU 侧这两个开关设为 0，日志是 `Creating SIFT CPU feature extractor` / `matcher`。

压缩包里那份 2016 年的 `database.db` 没有放进这张表。对它再跑提取和匹配会因 `IMAGE_EXISTS` 立刻结束，计时没有意义。

## 计时怎么读

CPU 三段和 GPU 平差那次建图用 `/usr/bin/time -f '… %e'` 的墙钟。GPU 提特征、匹配，以及第一次 GPU 特征上的建图，日志里只有 COLMAP 自己的 `Elapsed time`（分钟，三位小数），表里已换成秒：

| 阶段 | COLMAP 计时 | 墙钟 |
|--|--:|--:|
| CPU 提特征 | 1.365 分钟 | 82.00 秒 |
| CPU 匹配 | 13.776 分钟 | 826.68 秒 |
| CPU 建图 | 2.336 分钟 | 140.46 秒 |
| GPU 提特征 | 0.153 分钟 | — |
| GPU 匹配 | 0.415 分钟 | — |
| GPU 特征 + CPU 平差 | 3.029 分钟 | — |
| GPU 特征 + `ba_use_gpu 1` | 2.145 分钟 | 129.11 秒 |

有墙钟的几行，COLMAP 计时和墙钟差在 1 秒以内。

各跑一次，没有重复取平均。

## 特征不一样，所以建图不是同一道题

| | `database_cpu_bench.db` | `database_gpu.db` |
|--|--:|--:|
| 图像 | 128 | 128 |
| 关键点总数 | 1,704,094 | 1,429,051 |
| 每张平均 / 最多 | 13313 / 18728 | 11164 / 14986 |
| 匹配行（未验证） | 2,054,274 | 1,663,114 |
| 几何验证通过的图像对 | 2593 | 2525 |
| 内点总数 | 1,882,803 | 1,501,313 |
| 每对平均内点 | 726 | 595 |

默认 `SiftExtraction.max_num_features` 是 8192。一个特征可以有多个主方向，写进数据库的关键点数会高于这个上限。CPU 用 VLFeat，GPU 用 SiftGPU，检出的点和描述子不是同一套，所以关键点和内点都少一截。

重建结果：

| | CPU SIFT `sparse_cpu_bench/0` | GPU SIFT，CPU 平差 `sparse_gpu/0` | GPU SIFT，GPU 平差 `sparse_gpu_ba/0` |
|--|--:|--:|--:|
| 已注册图像 | 128 / 128 | 128 / 128 | 128 / 128 |
| 三维点 | 100428 | 84563 | 84562 |
| 观测 | 623564 | 502450 | 502418 |
| 平均轨迹长度 | 6.21 | 5.94 | 5.94 |
| 每张图平均观测 | 4872 | 3925 | 3925 |
| 平均重投影误差 | 0.57 px | 0.61 px | 0.61 px |
| 焦距 / k1 | 2558.81 / −0.02030 | 2558.23 / −0.02027 | 2558.25 / −0.02027 |

内参都是 `SIMPLE_RADIAL`，3072×2304，主点固定在 (1536, 1152)。后两列用同一份数据库，点数只差 1，是稠密求解放到 GPU 之后数值路径不同，模型规模没有变。

官方参考模型 `sparse/` 是 61514 个点、焦距 2559.68、k1 −0.0205。那是 2016 年另一套特征，不拿来比速度。

## 建图的时间花在哪

增量建图每次只注册一张图，然后做局部三角化和大约 6 张图的局部平差。已注册图像增长到一定比例后再做一次全局重三角化加全局平差。`IncrementalTriangulator` 里对点和图像对的循环是单线程的，没有 GPU kernel。下一张的位姿依赖上一张更新过的模型，128 张不能一起三角化。

下面把日志里相邻事件的时间差加起来。`Registering image` 到下一条全局平差日志算局部工作，全局平差日志到下一张注册算全局工作。相加比整段墙钟少几秒，差在启动和收尾，没有被这两类日志包住。

| | CPU SIFT，CPU 平差 | GPU SIFT，CPU 平差 | GPU SIFT，`ba_use_gpu 1` |
|--|--:|--:|--:|
| 全局平差次数 | 34 | 33 | 33 |
| 注册 + 局部三角化 / 局部平差 | 49.9 秒 | 55.9 秒 | 40.6 秒 |
| 全局重三角化 + 全局平差 | 85.9 秒 | 120.7 秒 | 84.4 秒 |
| 两段合计 | 135.8 秒 | 176.5 秒 | 125.0 秒 |
| 整段墙钟或 COLMAP 计时 | 140.46 秒 | 181.74 秒 | 129.11 秒 |

CPU SIFT 对 GPU SIFT（当时平差都在 CPU）：多出来的大约 40 秒几乎都在全局那一列（85.9 秒对 120.7 秒）。局部只多 6 秒。GPU SIFT 的点更少、内点更少，所以不是「点更多所以更慢」。全局平差把当时已注册的全部相机和观测放进 Ceres，特征分布不同，迭代和 refinement 轮数可以差一截。日志没有逐轮迭代次数。

按已注册图像数把这两段再切开（秒）：

| 已注册图像 | CPU SIFT 局部 | CPU SIFT 全局 | GPU SIFT 局部 | GPU SIFT 全局 | GPU 平差局部 | GPU 平差全局 |
|--|--:|--:|--:|--:|--:|--:|
| 0–19 | 9.9 | 5.3 | 10.5 | 6.5 | 8.4 | 5.0 |
| 20–39 | 8.0 | 6.9 | 9.0 | 10.0 | 6.7 | 7.0 |
| 40–59 | 9.3 | 17.1 | 10.0 | 21.9 | 6.9 | 16.6 |
| 60–79 | 9.7 | 19.9 | 9.3 | 39.5 | 6.2 | 26.0 |
| 80–99 | 7.1 | 13.8 | 10.2 | 11.9 | 7.3 | 8.6 |
| 100–119 | 4.4 | 9.4 | 5.1 | 12.3 | 3.8 | 8.4 |
| 120–127 | 1.4 | 13.4 | 1.6 | 18.6 | 1.3 | 12.7 |

前 40 张两边的全局步骤都还小。GPU SIFT、CPU 平差慢的那一块集中在 40–79 张，全局步骤 39.5 秒加 21.9 秒。打开 GPU 平差之后，同一段降到 26.0 秒加 16.6 秒。

## `ba_use_gpu` 实际碰到了什么

`--Mapper.ba_use_gpu 1 --Mapper.ba_gpu_index 0`。Ceres 2.2 的 CUDA 只做稠密分解（cuSOLVER / cuBLAS）。消掉三维点之后，剩下的相机方程大约是 6n 阶。n 小的时候 COLMAP 用稠密 Schur，整块矩阵做 Cholesky，计算量按 n³ 涨。

阈值在 `src/colmap/estimators/bundle_adjustment_ceres.h`：

| 条件 | 求解器 |
|--|--|
| 图像数 < 50 | 不送 GPU。局部平差大约 6 张图，始终在 CPU |
| 50–200，且 `ba_use_gpu 1` | `DENSE_SCHUR`，稠密库 `ceres::CUDA` |
| > 200 | `SPARSE_SCHUR`。GPU 上要 cuDSS（Ceres 的 `CUDA_SPARSE`），本机没有，退回 CPU 上的 SuiteSparse |
| CPU 上稠密只到 50 张，稀疏到 1000 张 | 再往上是 `ITERATIVE_SCHUR` |

South Building 一共 128 张，后半段全局平差落在 50–200，所以这次能用上 GPU 稠密求解。日志里那句

```text
Requested to use GPU for bundle adjustment, but Ceres was compiled without
cuDSS support. Falling back to CPU-based sparse solvers.
```

在 `use_gpu` 为真时就会打，不表示这次稠密求解失败。运行正常结束。

同一份 `database_gpu.db` 上，建图 181.74 秒降到 129.11 秒。省下的时间在全局那一列（120.7 秒降到 84.4 秒），也就是图像数已经超过 50、稠密 Schur 改走 CUDA 的那些步。局部那一列也从 55.9 秒降到 40.6 秒，但前 20 张（还没到 GPU 阈值）就已经更快，所以这里面有换成 CUDA 版 Ceres 之后 CPU 路径本身的差别，不能全部算成 GPU kernel。

稠密相机块在 100 张图这个量级很小。平差的墙钟里，Jacobian 和残差仍在 CPU 上组装，GPU 只吃最后那次稠密分解。

超过 200 张会改走稀疏 Schur。稀疏矩阵只保留真正共视过同一个点的相机对，分解要用 cuDSS。官方 tag 停在 Ceres 2.2.0；`master` 的版本号已经是 2.3.0 并且接了 cuDSS，但没有 2.3 的发行包，这台机器上也没装 `libcudss`。所以这一段现在上不了 GPU。

## 复现

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate colmap-build

export DATASET=/home/jesse/workspace/colmap/data/south-building
export COLMAP=/home/jesse/workspace/colmap/build/src/colmap/exe/colmap
```

CPU 基线，空数据库：

```bash
rm -f "$DATASET/database_cpu_bench.db"
/usr/bin/time -f 'EXTRACT_WALL %e' "$COLMAP" feature_extractor \
  --database_path "$DATASET/database_cpu_bench.db" \
  --image_path "$DATASET/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --FeatureExtraction.use_gpu 0

/usr/bin/time -f 'MATCH_WALL %e' "$COLMAP" exhaustive_matcher \
  --database_path "$DATASET/database_cpu_bench.db" \
  --FeatureMatching.use_gpu 0

mkdir -p "$DATASET/sparse_cpu_bench"
/usr/bin/time -f 'MAP_WALL %e' "$COLMAP" mapper \
  --database_path "$DATASET/database_cpu_bench.db" \
  --image_path "$DATASET/images" \
  --output_path "$DATASET/sparse_cpu_bench"
```

GPU 提特征、匹配、CPU 平差的建图：

```bash
rm -f "$DATASET/database_gpu.db"
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
```

同一份 GPU 特征，全局平差的稠密求解改走 GPU。输出换目录，避免覆盖 `sparse_gpu/`：

```bash
mkdir -p "$DATASET/sparse_gpu_ba"
/usr/bin/time -f 'MAP_WALL %e' "$COLMAP" mapper \
  --database_path "$DATASET/database_gpu.db" \
  --image_path "$DATASET/images" \
  --output_path "$DATASET/sparse_gpu_ba" \
  --Mapper.ba_use_gpu 1 \
  --Mapper.ba_gpu_index 0
```

看模型：

```bash
"$COLMAP" model_analyzer --path "$DATASET/sparse_cpu_bench/0"
"$COLMAP" model_analyzer --path "$DATASET/sparse_gpu/0"
"$COLMAP" model_analyzer --path "$DATASET/sparse_gpu_ba/0"
```

## 产物

```text
data/south-building/
  database_cpu_bench.db   CPU SIFT + CPU 匹配
  sparse_cpu_bench/0      对应的模型
  database_gpu.db         GPU SIFT + GPU 匹配
  sparse_gpu/0            这份库、CPU 平差
  sparse_gpu_ba/0         这份库、ba_use_gpu 1
```

`data/` 在 `.gitignore` 里。
