# 稀疏地图上的视觉定位

3DGS 那次是先渲一张图，再把特征和渲染图匹配，用逆深度抬成三维点，然后 PnP。传统做法不渲模型。地图里已经有稀疏三维点，查询照片的特征和这些点对上，同一次 PnP 出 6DoF。

这次复现的是 [hloc](https://github.com/cvg/Hierarchical-Localization)（Hierarchical Localization，Sarlin 等，CVPR 2019）。它把这件事拆成检索、局部匹配、在已知位姿上三角化、再 PnP。公开榜 [visuallocalization.net](https://www.visuallocalization.net/) 上，结构法这一档的常用基线就是它。README 里用 NetVLAD 检索、SuperPoint + SuperGlue 匹配的数字是：

| | 0.25 m、2° | 0.5 m、5° | 5 m、10° |
|--|--|--|--|
| Aachen 白天，检索 top 50 | 89.6% | 95.4% | 98.8% |
| Aachen 夜晚，检索 top 50 | 86.7% | 93.9% | 100% |

三个百分比是查询里有多少张同时满足位移和旋转阈值。SuperGlue 的权重不好取，hloc 里对应的现成配置是 `superpoint+lightglue`。LightGlue 是同一条匹配线的后续，Aachen 那张表仍是 SuperGlue 跑出来的。

Gerrard Hall 不再下 Aachen。100 张的位姿已经在 `dense/sparse` 里，而且和 3DGS 用的是同一批查询，表可以并排放。Aachen 只说明这套方法在公开数据上的位置。

## 同一次 PnP，旁边几条先不跑

位姿求解器就是 P3P 加 RANSAC。本仓库的实现是 `EstimateAbsolutePose`，默认重投影阈值 12 像素，至少 100 次、最多 10000 次采样。查询图的注册走 `mapper`，内部就是这个函数。差在点和对应从哪来。

- 本仓库的 `vocab_tree_retriever` 加 `image_registrator`：SIFT，词袋检索，查询图要先把特征和匹配写进数据库，`RegisterNextImage` 再调用上面的绝对位姿。Gerrard 的稀疏模型就是这条路建的。它当对照，不代替 hloc。
- InLoc 在 hloc 里的那条：室内有激光深度，跳过三角化。Gerrard 没有这套深度。
- MeshLoc：用网格渲染的深度把 2D-2D 抬起来。和 3DGS 用逆深度抬点是同一类，地图不是稀疏点。
- PixLoc：hloc 给出位姿之后，再对稀疏点做光度收紧。对应 3DGS 页面上那 60 次渲染。第一跑只到 PnP。
- ACE、DSAC*：直接回归每个像素的场景坐标，没有三角化出来的点。
- MegaLoc、EigenPlaces、SALAD：只换全局描述子。hloc 已经带了 `megaloc`。楼只有 87 张重建图，检索不是这组数据的瓶颈。

## 数据和划分

照片用 `data/gerrard-hall/dense/images`，长边 2000，一个 PINHOLE。原始 5616×3744 的 JPG 不参与，内参已经是去畸变之后的。

划分和 3DGS 的 `--eval` 相同。100 张按文件名排序，下标能被 8 整除的 13 张只做查询：

`IMG_2331`、`IMG_2339`、`IMG_2347`、`IMG_2355`、`IMG_2363`、`IMG_2371`、`IMG_2379`、`IMG_2387`、`IMG_2395`、`IMG_2403`、`IMG_2411`、`IMG_2419`、`IMG_2427`。

其余 87 张留在地图里。查询的 COLMAP 位姿只用来算误差，不参与三角化，也不参与检索库。

## 地图

从 `dense/sparse` 拷一份参考模型，注销那 13 张。相机内参和 87 张的 `cam_from_world` 保持不动，不再做一次增量重建，也不再跑 PatchMatch。

点不能沿用现有的 `points3D`。那些点是 SIFT 轨迹，SuperPoint 或 ALIKED 的关键点对不上。hloc 的 `triangulation` 在这 87 张的已知位姿上，用新特征重新三角化。图像对来自参考模型的共视，不用检索去猜哪两张重建图应该匹配。

第一套特征用 hloc 的配置名：

- 局部特征 `superpoint_max`：最多 4096 点，长边收到 1600。Aachen 公开配置是 `superpoint_aachen`（长边 1024）。Gerrard 去畸变图长边已经是 2000，1600 和 3DGS 训练时的缩放一致。
- 匹配 `superpoint+lightglue`。
- 建图这一步不提全局描述子。

第二套留作对照，配置名 `aliked-n16` 和 `aliked+lightglue`。本仓库的 COLMAP 二进制里已经有 ALIKED 和 LightGlue 的 ONNX，查看器里的 `?scene=aliked` 就是这套特征。第一跑仍走 hloc 自带的 SuperPoint，这样和 Aachen 那条线是同一个检测器。

## 查询

1. 全局描述子用 `netvlad`。每张查询在 87 张里取最像的 10 张。87 张的穷举匹配可以另跑一列，当作检索没有漏掉邻居时的上限。
2. 查询只和这 10 张做 LightGlue。
3. 重建图上的关键点如果属于一条三角化轨迹，就带出一个三维点。查询关键点因此得到 2D-3D 对应。
4. `localize_sfm` 做 P3P + RANSAC。SuperPoint + LightGlue 在 Aachen 的笔记本里关掉了 `covisibility_clustering`，这里同样关掉。焦距固定为参考模型里的 PINHOLE，不从 EXIF 再估。
5. 真值不进 RANSAC。误差和 3DGS 的 `pose_error` 同一套定义：旋转是 `R_cw.T @ R_gt` 的旋转角，位移是两个相机中心的距离。中心是 `-R_cw.T @ t_cw`。

3DGS 在 `IMG_2371.JPG` 上的 PnP 是 1.34°、7.3 厘米、722 个内点；后面的光度收紧收到 0.15°、5 毫米。稀疏这条没有那 60 次渲染，表停在 PnP。`IMG_2403.JPG` 在 3DGS 里是匹配不够，这里单独看它能不能从稀疏点上解出来。

## 产出

都写在 `data/gerrard-hall/hloc/`，继续被 `data/` 的忽略规则挡住，不进 git。

- 特征、匹配、三角化后的模型和 hloc 的位姿文件。
- 一张和定位页同格式的表：查询、检索到的重建图、检索位姿误差、PnP 角度和位移、内点数。
- `IMG_2371.JPG` 的匹配图。线画在查询和被检索到的重建照片之间，点是进了 PnP 的那些对应。
- 13 张里落在 0.25 m / 2° 以内的张数，以及落在 5 cm / 5° 以内的张数。前者对齐 Aachen 的紧阈值，后者是室内定位常用的一档。

## 代码和依赖

hloc 是子模块 `third_party/Hierarchical-Localization`（`https://github.com/liujiacheng1009/Hierarchical-Localization`），不接入 COLMAP 的 CMake。特征提取和匹配仍走 hloc 的 SuperPoint、LightGlue、NetVLAD。几何不再用 pip 里的 pycolmap：`hloc/colmap_backend.py` 把相机、关键点和匹配写进本仓库能读的数据库，然后调用 `build/src/colmap/exe/colmap`。

| 步骤 | 二进制子命令 |
|--|--|
| 建库 | `database_creator` |
| 用已知位姿核验重建图的匹配 | `guided_geometric_verifier` |
| 在 87 张已知位姿上三角化 | `point_triangulator`（`--clear_points 1 --refine_intrinsics 0`） |
| 查询图 P3P | `mapper --input_path`（`--Mapper.fix_existing_frames 1`，重投影 12 px，至少 15 个内点） |

查询图没有位姿，`guided_geometric_verifier` 不会处理它们。查询和检索邻居的 LightGlue 匹配直接写成 `CALIBRATED` 的双视几何，离群点交给 mapper 里的 RANSAC。关键点从 hloc 的像素角点坐标系加 0.5，对齐 COLMAP 的像素中心。`feature_importer` 只认 128 维 SIFT，SuperPoint 不走它。

Python 用已有的 `pytorch-common`。权重留在本机缓存，不提交。产出仍在 `data/gerrard-hall/hloc/`。

```bash
export COLMAP_ROOT=/home/jesse/workspace/colmap
export GS_DATA=$COLMAP_ROOT/data/gerrard-hall
# 可选。默认就是仓库里的 build/src/colmap/exe/colmap
export COLMAP_BIN=$COLMAP_ROOT/build/src/colmap/exe/colmap

~/.conda/envs/pytorch-common/bin/python "$COLMAP_ROOT/scripts/hloc_gerrard.py" \
  --images "$GS_DATA/dense/images" \
  --reference "$GS_DATA/dense/sparse" \
  --outputs "$GS_DATA/hloc"
```

`scripts/hloc_gerrard.py` 划出 13 张，按共视导出 87 张的参考模型，跑上面的配置，再把位姿换成角度和米。`IMG_2371.JPG/matches.jpg` 画查询和第一张检索图之间、落在同一条三角化轨迹上的对应。

## 这一跑

13 张都注册上了。12 张同时落在 0.25 m / 2° 和 5 cm / 5° 里。相邻帧重叠很大，PnP 之后的位姿和原来的 COLMAP 位姿差在百分之几度、不到 1 毫米。内点是注册并做完光束法平差后，查询图上还连着三维点的观测数。

| 查询 | 检索到 | 检索误差 | PnP |
|--|--|--|--|
| IMG_2371.JPG | IMG_2372.JPG | 4.15° / 0.221 m | 0.04° / <1 mm，3071 点 |
| IMG_2403.JPG | IMG_2402.JPG | 3.67° / 0.309 m | 0.03° / 1 mm，1697 点 |
| IMG_2387.JPG | IMG_2386.JPG | 33.74° / 0.030 m | 73.66° / 1.728 m，1 点 |

`IMG_2371` 的 3DGS PnP 是 1.34°、7.3 厘米、722 个内点。稀疏这条更紧，因为点来自 87 张的三角化，不是一张渲染图。`IMG_2403` 在 3DGS 里匹配不够，这里解出来了。`IMG_2387` 的最近邻转了 34°，注册时只看到 119 个三维点，平差后只剩 1 个观测，位姿是错的。完整表和每张查询的匹配图在 [http://127.0.0.1:8899/gerrard_hloc/](http://127.0.0.1:8899/gerrard_hloc/)。

## 这一轮不改的东西

不改 `EstimateAbsolutePose`，也不把查询注册进现有的 SIFT 数据库。不重跑 Gerrard 的稠密重建。不把 13 张加回地图。PnP 的表出来之前不做 PixLoc，也不做光度收紧。
