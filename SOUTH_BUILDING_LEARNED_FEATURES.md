# South Building：学习型检测器与匹配器

在同一套 128 张 South Building 上比较学习型特征和 GPU 暴力匹配。离线建图跑的是 **ALIKED-N16ROT 提点 + GPU 暴力匹配 + GPU 平差**。LightGlue 穷举留作对照，说明它为什么不适合这一步。SIFT 的数字来自 `SOUTH_BUILDING_CPU_GPU.md`。

离线建图用的是 ALIKED 提点加 GPU 暴力匹配，可视化在 <http://127.0.0.1:8899/?scene=aliked_bf>。LightGlue 那次在 <http://127.0.0.1:8899/?scene=aliked>，SIFT GPU 在 <http://127.0.0.1:8899/>。页面左侧可以来回切。服务器要先在 `demo/south_building_viewer/` 里开着。

## 这次的结果

ALIKED 每张图最多 2048 个点，SIFT 那次是默认预算（入库平均 11164）。两边都在 RTX 5090 上，建图的 `ba_use_gpu` 都关着。128 张全部注册。

| | GPU SIFT + 暴力匹配 | ALIKED-N16ROT + LightGlue |
|--|--:|--:|
| 提特征 | 9.18 秒 | 5.59 秒 |
| 穷举匹配 | 24.90 秒 | 163.95 秒 |
| 建图 | 181.74 秒 | 84.20 秒 |
| 合计 | 215.82 秒 | 253.74 秒 |
| 已注册图像 | 128 / 128 | 128 / 128 |
| 每张关键点（平均） | 11164 | 2046 |
| 几何验证通过的图像对 | 2525 / 8128 | 6339 / 8128 |
| 内点 / 匹配行 | 1,501,313 / 1,663,114（0.90） | 1,280,464 / 1,577,782（0.81） |
| 三维点 | 84563 | 29695 |
| 观测 | 502450 | 200408 |
| 平均轨迹长度 | 5.94 | 6.75 |
| 每张图平均观测 | 3925 | 1566 |
| 平均重投影误差 | 0.61 px | 0.73 px |
| 焦距 / k1 | 2558.23 / −0.0203 | 2557.80 / −0.0218 |

提特征快了约 1.6 倍，建图快了约 2.2 倍。穷举匹配的墙钟是 163.95 秒对 SIFT 的 24.90 秒，整条管线因此是 254 秒对 216 秒。这 164 秒里大约 38 秒是第一次下载 `aliked-lightglue.onnx`（21:04:37 到 21:05:15，之后进缓存）。九个匹配块加起来是 125.7 秒，模型已经在本地时穷举匹配仍约 126 秒，大约是 SIFT GPU 暴力匹配的 5 倍。原因在下面一节。

通过几何验证的图像对从 2525 涨到 6339。点少很多，是因为预算是 2048 而不是一万多。平均轨迹更长（6.75 对 5.94），重投影误差略高（0.73 px 对 0.61 px）。内参和 SIFT 模型差在小数点后。这套楼重叠高，两条管线都注册满了，点数差主要来自预算，不单独当精度胜负。

模型在 `data/south-building/sparse_aliked_lg/0`，数据库是 `database_aliked_lg.db`。

### 穷举匹配为什么慢

164 秒是整条 `exhaustive_matcher` 的墙钟，里面有两段：

- 21:04:37 到 21:05:15，约 38 秒在下载 `aliked-lightglue.onnx`。文件进 `~/.cache/colmap` 之后不会再下。
- 21:05:15 开始生成图像对。默认 `block_size` 是 50，128 张分成 3 块，一共 9 个块。块耗时依次是 18.922、18.958、5.786、19.690、19.111、5.826、15.774、15.866、5.748 秒，合计 125.7 秒。COLMAP 打印的 2.729 分钟从命令一开始算，下载算在里面。

模型已经在本地时，穷举匹配仍要约 126 秒。`C(128, 2) = 8128` 对，平均大约 15.5 毫秒一对。最后一块只有 28 张图，`C(28, 2) = 378` 对用了 5.748 秒，同样是大约 15 毫秒一对，所以这个速度贯穿全程，不是某一块卡住。

同一 8128 对，SIFT 的 GPU 暴力匹配是 24.90 秒，大约 3.1 毫秒一对。那次每张图平均 11164 个点。ALIKED 每张只有约 2046 个点，匹配仍然更慢。

慢的是每一对上的网络，不是点的个数：

1. SIFT GPU 匹配走 SiftGPU：一对图做一次描述子距离矩阵，再做 ratio test，一次核函数。LightGlue 对同一对图跑多层 self-attention 和 cross-attention。注意力随每张图的点数按平方增长，2048 个点已经要做很多层矩阵乘。
2. `LightGlueONNXFeatureMatcher::Match` 对每一对单独调用一次 `model_.Run`。8128 对就是 8128 次前向，没有把多对打成一个 batch。输入张量建在 CPU 内存上，CUDA execution provider 再拷到 GPU。
3. 打开 `use_gpu` 时，每个 GPU 只起一个 `FeatureMatcherWorker`。日志里是 “Bind FeatureMatcherWorker to GPU device 0”，这台机器上网络只有这一路在跑。几何验证用 CPU 线程并行着做，块时间和图像对数量成正比，和后来通过验证的内点数对不上，时间在前向里。
4. 穷举对每一对都付这笔钱，包括几乎没有重叠的对。没重叠也要跑完 LightGlue。SIFT 的距离核便宜，同样 8128 对能在 25 秒里结束。

再跑一次会少掉那 38 秒下载。剩下的大约 126 秒仍是 SIFT GPU 暴力匹配的约 5 倍。这是穷举配对加上逐对 Transformer 前向的结果。

## 离线建图用的方案

LightGlue 适合图像对很少的时候，比如里程计里相邻几帧，或者检索之后只剩几十对难例。离线穷举要跑 8128 对，逐对 Transformer 填不满 5090，时间却按对线性涨。

这套数据上换成下面三条，GPU 各干一段：

1. 提点：`ALIKED_N16ROT`，ONNX CUDA，每张图最多 2048 点。沿用上一次已经提好的点，5.59 秒。
2. 匹配：`ALIKED_BRUTEFORCE`，`use_gpu 1`。一对图一次余弦矩阵加交叉检查，默认 `min_cossim 0.85`。穷举 8128 对的墙钟是 4.54 秒（COLMAP 计时 0.073 分钟）。九个块是 0.719、0.606、0.209、0.653、0.642、0.192、0.506、0.520、0.192 秒。
3. 建图：`Mapper.ba_use_gpu 1`。图像数过 50 之后，全局平差的稠密 Schur 走 Ceres CUDA。128 张在稠密 GPU 的范围内。稀疏 GPU 仍要 cuDSS，日志里那句退回 CPU 稀疏求解指的是这一段；这次的全局步走的是稠密 CUDA。建图墙钟 29.94 秒（COLMAP 计时 0.495 分钟）。

三段合计约 40 秒。同一套点上 LightGlue 穷举是 164 秒、CPU 平差建图 84 秒，整条 254 秒。GPU SIFT 暴力匹配加 GPU 平差是 9.18 + 24.90 + 129.11 = 163 秒。

| | GPU SIFT + 暴力匹配 + GPU 平差 | ALIKED + GPU 暴力匹配 + GPU 平差 | ALIKED + LightGlue + CPU 平差 |
|--|--:|--:|--:|
| 提特征 | 9.18 秒 | 5.59 秒 | 5.59 秒 |
| 穷举匹配 | 24.90 秒 | 4.54 秒 | 163.95 秒 |
| 建图 | 129.11 秒 | 29.94 秒 | 84.20 秒 |
| 合计 | 163 秒 | 40 秒 | 254 秒 |
| 已注册图像 | 128 / 128 | 128 / 128 | 128 / 128 |
| 几何验证通过的图像对 | 2525 / 8128 | 2249 / 8128 | 6339 / 8128 |
| 内点 / 匹配行 | 1,501,313 / 1,663,114（0.90） | 609,081 / 629,819（0.97） | 1,280,464 / 1,577,782（0.81） |
| 三维点 | 84563 | 18319 | 29695 |
| 观测 | 502450 | 151433 | 200408 |
| 平均轨迹长度 | 5.94 | 8.27 | 6.75 |
| 每张图平均观测 | 3925 | 1183 | 1566 |
| 平均重投影误差 | 0.61 px | 0.46 px | 0.73 px |
| 焦距 / k1 | 2558.23 / −0.0203 | 2559.99 / −0.0226 | 2557.80 / −0.0218 |

暴力匹配通过的图像对少了（2249 对 6339），留下的内点比例更高（0.97），轨迹更长（8.27），重投影更小（0.46 px）。三维点少，是 2048 的预算再加上 `min_cossim 0.85` 把弱匹配丢掉了。128 张仍然全部注册，焦距和官方模型差在小数点后。点云密度要靠 SIFT 那一档（平均一万多个点，84563 个三维点）。

数据库 `data/south-building/database_aliked_bf.db`，模型 `sparse_aliked_bf/0`。提点从 `database_aliked_lg.db` 拷出来，匹配表清空后重跑，所以和 LightGlue 用的是同一批关键点。

### LoMa-B + GPU 暴力匹配

同一套 128 张、同样的 `min_cossim 0.85` 和 `ba_use_gpu 1`，把检测器换成 `LOMA_B`（每张 2048 点，256 维），匹配用 `LOMA_BRUTEFORCE`。模型事先放在缓存里，计时不含下载。可视化在 <http://127.0.0.1:8899/?scene=loma_bf>。

| | ALIKED + GPU 暴力匹配 | LoMa-B + GPU 暴力匹配 |
|--|--:|--:|
| 提特征 | 5.59 秒 | 39.04 秒 |
| 穷举匹配 | 4.54 秒 | 12.35 秒 |
| 建图 | 29.94 秒 | 32.85 秒 |
| 合计 | 40 秒 | 84 秒 |
| 已注册图像 | 128 / 128 | 128 / 128 |
| 每张关键点 | 2046 | 2048 |
| 几何验证通过的图像对 | 2249 / 8128 | 1930 / 8128 |
| 内点 / 匹配行 | 609,081 / 629,819（0.97） | 699,372 / 702,491（0.996） |
| 三维点 | 18319 | 18515 |
| 观测 | 151433 | 163270 |
| 平均轨迹长度 | 8.27 | 8.82 |
| 每张图平均观测 | 1183 | 1276 |
| 平均重投影误差 | 0.46 px | 0.74 px |
| 焦距 / k1 | 2559.99 / −0.0226 | 2557.39 / −0.0206 |

匹配九个块是 1.915、1.512、0.591、1.913、1.836、0.587、1.529、1.525、0.569 秒。256 维余弦比 ALIKED 的 128 维贵，穷举仍是 12 秒这一档，不是 LightGlue 的 126 秒。时间多在提点：描述子图大约 1.3 GB，每张图先在 3072×2304 上跑检测器，再把图缩到描述子网络的固定输入。128 张全部注册，三维点数和 ALIKED 暴力匹配接近，轨迹略长，重投影从 0.46 px 升到 0.74 px。这套楼上，离线速度仍是 ALIKED 暴力匹配更快。

数据库 `data/south-building/database_loma_bf.db`，模型 `sparse_loma_bf/0`。

推理走 ONNX Runtime 1.27.1 的 CUDA 13 包，执行在 5090 上。这个包还要 `libcudnn.so.9`，用的是本机 `pytorch-common` 环境里的 cuDNN，跑之前放进 `LD_LIBRARY_PATH`。COLMAP 自带的预编译包是 CUDA 12 的，和这台 CUDA 13.2 对不上，所以没有用 `FETCH_ONNX` 下的那份。

## 推荐哪条管线

这棵代码能换的特征只有三类：SIFT、ALIKED、LoMa。后两类和 LightGlue 都要 ONNX。这次实验的二进制已经打开 ONNX，推理在 5090 上。

按「这台机器上、这条 COLMAP 里、稀疏重建能直接吃进去」来排：

| 优先级 | 检测 + 描述 | 匹配 | 为什么放在这 |
|--|--|--|--|
| 质量首选 | `LOMA_B` | `LOMA_G` | LoMa（ECCV 2026）是这版里最新的局部特征。`LOMA_B` 是 256 维描述子（冻结 DINOv2 加卷积）。`LOMA_G` 是给它用的最大专用匹配器，文档里按模型尺寸和质量排在最后一档 |
| 离线建图 | `ALIKED_N16ROT` | `ALIKED_BRUTEFORCE` | GPU 上一次余弦矩阵。这套 128 张整条约 40 秒。`LOMA_B` + `LOMA_BRUTEFORCE` 已跑过，合计 84 秒，重投影 0.74 px，慢在提点 |
| 难例、少量图像对 | `ALIKED_N16ROT` | `ALIKED_LIGHTGLUE` | 逐对 Transformer。适合里程计或检索之后剩下的难对，不适合穷举 |
| 要更密的点云 | `SIFT` | `SIFT_BRUTEFORCE` | SiftGPU 暴力 L2。点数多一个量级，匹配 24.90 秒，GPU 平差后建图 129 秒 |
| 只换匹配器做对照 | `SIFT` | `SIFT_LIGHTGLUE` | 检测器仍是 SIFT，用来把匹配器拆开 |

`LOMA_B128` + `LOMA_B128` 是轻量档，文档写明精度最低，不作为主对比。`LOMA_R` 和 `LOMA_B` 一样大，多了旋转增强；South Building 相机基本直立，主实验用 `LOMA_G`。以后若加转了 90° 的图，再加一列 `LOMA_R`。

不放进第一轮的模型：SuperPoint、DISK、XFeat、RoMa、MASt3R。它们在别的基准上很强，但这版 `feature_extractor` / `exhaustive_matcher` 没有对应类型。稠密匹配器要另做采样才能进增量建图，和这次「换检测器、换匹配器、其余管线不动」不是同一件事。

检测器和匹配器必须成对。混用（SIFT 特征配 ALIKED 匹配器，或 `LOMA_B128` 特征配 `LOMA_B` 匹配器）会在运行时失败。每种管线单独一个数据库。

## 和「SIFT + FLANN」的关系

这次已经测过的传统管线是 **GPU SIFT + GPU 暴力匹配**，不是 FLANN。

- 检测：SiftGPU（CUDA）。CPU 检测是 VLFeat。
- 匹配：`FeatureMatching.type` 默认 `SIFT_BRUTEFORCE`。`use_gpu 1` 时走 SiftGPU 的暴力 L2 + ratio test。`use_gpu 0` 时走 FAISS 最近邻，不是 FLANN。
- 经典 COLMAP 的 CPU 近似匹配曾经是 FLANN。这棵代码里 CPU 最近邻已经换成 `src/colmap/feature/index.cc` 的 FAISS。

所以对比表里的传统列用已经跑完的 GPU SIFT 穷举匹配。若要一条更接近旧 FLANN 的「近似最近邻」列，另跑 `use_gpu 0` 的 `SIFT_BRUTEFORCE`（FAISS），不要把它和 GPU 暴力匹配混成一个基线。

## 点数必须分开比

默认预算不一样：

| 类型 | `max_num_features` 默认 |
|--|--:|
| SIFT | 8192 |
| ALIKED | 2048 |
| LoMa | 2048 |

SIFT 一个点可以有多个主方向，数据库里的关键点数会高于 8192。South Building 的 GPU SIFT 平均每张 11164，CPU SIFT 平均 13313。学习型模型按分数截断，2048 就是大约 2048。

若只跑默认值，点更多的 SIFT 会在轨迹长度和三维点数上占便宜，看不出检测器本身好不好。实验分两档：

1. **默认档**：各用自己的默认点数。这是用户不改参数时的结果。
2. **控制档**：全部把 `max_num_features` 设为 2048。SIFT 用 `--SiftExtraction.max_num_features 2048`。这一档才用来比检测器和匹配器。

主结论以控制档为准，默认档只说明「开箱即用差多少」。

## 要跑的六条管线

数据集、相机、建图选项与 `SOUTH_BUILDING_CPU_GPU.md` 的 GPU SIFT 那次相同：128 张、3072×2304、`--ImageReader.single_camera 1`、`SIMPLE_RADIAL`、穷举匹配、增量 `mapper`。几何验证保持打开。`mapper` 的 `ba_use_gpu` 保持关闭，和平差那次的 181.74 秒基线对齐，避免把 Ceres GPU 的差异算进特征里。

| 编号 | 角色 | 提取 | 匹配 | 数据库 |
|--|--|--|--|--|
| A | 传统基线，已有 | `SIFT`，`use_gpu 1` | `SIFT_BRUTEFORCE`，`use_gpu 1` | 已有 `database_gpu.db` |
| B | 近似最近邻，可选 | `SIFT`，`use_gpu 0` | `SIFT_BRUTEFORCE`，`use_gpu 0`（FAISS） | `database_sift_faiss.db` |
| C | 只换匹配器 | `SIFT`，`use_gpu 1` | `SIFT_LIGHTGLUE` | `database_sift_lg.db` |
| D | 只换检测器 | `ALIKED_N16ROT` | `ALIKED_BRUTEFORCE` | `database_aliked_bf.db` |
| E | 常用学习型管线 | `ALIKED_N16ROT` | `ALIKED_LIGHTGLUE` | `database_aliked_lg.db` |
| F | 这版里的质量档 | `LOMA_B` | `LOMA_G` | `database_loma_g.db` |

A 的提特征 9.18 秒、匹配 24.90 秒、建图 181.74 秒直接引用，不重跑。B 的 CPU SIFT 提特征和匹配已有（82.00 秒、826.68 秒），但那次 `max_num_features` 是默认 8192；控制档的 2048 需要新数据库。C–F 都是新跑。

默认档和控制档各跑 C、D、E、F。控制档的 A 也要新跑一份 2048 的 SIFT，因为现有 `database_gpu.db` 是默认预算。

South Building 重叠高、纹理够、光照一致。SIFT 在这种数据上已经注册满 128 张。学习型模型的优势通常在重叠少、光照差、视角差大的图上，这里可能只体现为内点比例更高或建图略稳，也可能更慢且点数更少。第一轮仍用这套图，方便和已有计时对齐。若六条都注册满、重投影误差接近，结论就是「这套简单数据分不出质量」，再换难例，而不是在这里把点数多少当成模型胜负。

## 记录什么

每条管线同一张表：

| 量 | 从哪来 |
|--|--|
| 提特征墙钟、匹配墙钟、建图墙钟 | `/usr/bin/time`，并记下 COLMAP 的 `Elapsed time` |
| 关键点总数、每张平均、最大 | `keypoints` 表 |
| 匹配行数、验证通过的图像对、内点数、内点比例 | `matches` 与 `two_view_geometries` |
| 已注册图像、三维点、观测、平均轨迹、每张观测、平均重投影误差 | `model_analyzer` |
| 焦距、k1 | `cameras.txt` |

内点比例 = 验证内点行数 / 未验证匹配行数。匹配器好不好先看这个和验证通过的图像对，再看建图是否仍是 128/128。三维点数受点数预算和过滤影响，不单独当精度。

## 前置：打开 ONNX

现有 `build/src/colmap/exe/colmap` 没有 ONNX，`ALIKED_*` 和 `LOMA_*` 会直接报缺支持。重编时在原来的 CUDA 配置上加 `-DONNX_ENABLED=ON`，`CMAKE_CUDA_ARCHITECTURES` 仍是 `120`。权重默认是 URL，第一次提取或匹配时 COLMAP 会下载并缓存。RTX 5090 上学习型模型走 `--FeatureExtraction.use_gpu 1` 和 `--FeatureMatching.use_gpu 1`。

确认日志里出现的是对应类型，而不是 `Creating SIFT GPU feature extractor`。

## 命令

下面是控制档（2048）。默认档去掉各条 `max_num_features`。每条换新的 `database_path` 和 `output_path`。

SIFT + LightGlue：

```bash
"$COLMAP" feature_extractor \
  --database_path "$DATASET/database_sift_lg.db" \
  --image_path "$DATASET/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --FeatureExtraction.type SIFT \
  --FeatureExtraction.use_gpu 1 \
  --SiftExtraction.max_num_features 2048

"$COLMAP" exhaustive_matcher \
  --database_path "$DATASET/database_sift_lg.db" \
  --FeatureMatching.type SIFT_LIGHTGLUE \
  --FeatureMatching.use_gpu 1
```

ALIKED-N16ROT + 暴力余弦，以及把匹配器换成 LightGlue 时只改 `FeatureMatching.type`：

```bash
"$COLMAP" feature_extractor \
  --database_path "$DATASET/database_aliked_lg.db" \
  --image_path "$DATASET/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --FeatureExtraction.type ALIKED_N16ROT \
  --FeatureExtraction.use_gpu 1 \
  --AlikedExtraction.max_num_features 2048

"$COLMAP" exhaustive_matcher \
  --database_path "$DATASET/database_aliked_lg.db" \
  --FeatureMatching.type ALIKED_LIGHTGLUE \
  --FeatureMatching.use_gpu 1
```

只换检测器时，同一套 ALIKED 特征配 `--FeatureMatching.type ALIKED_BRUTEFORCE`。已经写过匹配的库会跳过已有图像对，所以 D 和 E 用两个数据库，不要在同一个库里改匹配器类型。

LoMa：

```bash
"$COLMAP" feature_extractor \
  --database_path "$DATASET/database_loma_g.db" \
  --image_path "$DATASET/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --FeatureExtraction.type LOMA_B \
  --FeatureExtraction.use_gpu 1 \
  --LomaExtraction.max_num_features 2048

"$COLMAP" exhaustive_matcher \
  --database_path "$DATASET/database_loma_g.db" \
  --FeatureMatching.type LOMA_G \
  --FeatureMatching.use_gpu 1
```

建图与之前相同，输出目录按管线分开（`sparse_sift_lg/`、`sparse_aliked_lg/`、`sparse_loma_g/`）：

```bash
"$COLMAP" mapper \
  --database_path "$DATASET/database_loma_g.db" \
  --image_path "$DATASET/images" \
  --output_path "$DATASET/sparse_loma_g"
```

## 怎么读结果

- C 相对 A：检测器相同，差在 LightGlue 和暴力 L2。看内点比例和验证图像对。建图时间变长或变短都可能，因为送进几何验证的匹配数变了。
- D 相对 A 的 2048 控制档：匹配都是暴力最近邻，差在 ALIKED 和 SIFT 的点与描述子。
- E 相对 D：同一个 ALIKED，差在 LightGlue。
- F 相对 E：整条学习型管线换成 LoMa 最大匹配器。若 F 的内点比例和注册数不高于 E，而时间更长，质量档就不值得作为这套数据的默认。
- 注册数掉到 128 以下，先看是匹配内点太少，还是几何验证把对滤掉了，不要直接归因到建图。
