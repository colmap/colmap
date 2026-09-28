# 同一组照片：稠密表面，还是自由视角

COLMAP 稠密重建交出一个能转的表面：点云、网格、贴图。3D Gaussian Splatting 交出任意相机下的一张照片。两边用同一批去畸变后的相机，比的是离开拍照方向之后，谁还像那个地方。

Gerrard Hall 的稠密结果已经在查看器里：<http://127.0.0.1:8899/?scene=gerrard>。100 张、长边收到 2000，PatchMatch 约 57 分钟，融合 357 万点，简化后的 Poisson 网格 34 万面，再贴上一张图集。下面的 3DGS 用的就是这套 `dense/images` 和 `dense/sparse`，不再重跑 SfM。

同一组相机的对照页：<http://127.0.0.1:8899/gerrard_gs.html>。左边原照片，右边 3DGS 在这个相机下重渲的图，下面一条沿拍摄顺序插出来的自由视角。这次训练 30000 步，约 20 分钟，输入被收到长边 1600，结束时 56.4 万个高斯，训练视角 L1 0.031、PSNR 23.8。重渲图是 1600×1052。

## 对着看

| | COLMAP 稠密 | 3DGS |
|--|--|--|
| 你拿到的 | 一个表面。点云能量距离，网格能进别的软件 | 一堆高斯。只有渲出来的图像 |
| 站在原来的相机上 | 贴图就是那张照片投到面上，近看够用 | 训练目标就是贴上这些照片，重渲训练视角应和原图接近 |
| 转到两张照片中间 | 表面还在，贴图被拉过墙面，接缝和拉伸露出来 | 这是它要做的事：这个新相机下渲出一张完整的图 |
| 细的东西 | 窗框、屋檐、树枝在 Poisson 里被抹平，或在简化时丢掉 | 颜色能留下来，但没有可靠的厚度，会飘在表面前面 |
| 没拍到的背面 | Poisson 补一层光滑的假墙；Delaunay 留洞 | 没有观测就糊成一团，或者漂出浮点 |
| 拿去测量、碰撞 | 用网格 | 不要用高斯的位置当表面 |

在 Gerrard 的查看器里转一圈就能看到稠密这边的毛病：立面近看是照片，绕到屋檐和树，贴图开始糊，关掉贴图只剩一块被光照照着的光滑壳。3DGS 不会给你这个壳。它给你的是另一张图，相机可以不在原来那 100 个位置上。

## 应用 1 — 自由视角

指定一个 COLMAP 没拍过的相机，直接出图。典型用法是在两张相邻照片之间滑动，或者把相机抬高一点看立面。

`render.py` 只重渲训练相机，用来和原照片并排，确认训练没崩。自由视角是训练相机之间的位置，或者实时查看器里拖出来的位置。

Gerrard 的去畸变结果直接当 3DGS 的输入。`dense/sparse` 里的文件要放到 `sparse/0`，3DGS 只认这个目录。

```bash
export COLMAP_ROOT=/home/jesse/workspace/colmap
export GS_DATA=$COLMAP_ROOT/data/gerrard-hall/gs
export GS_ROOT=$COLMAP_ROOT/third_party/gaussian-splatting

mkdir -p "$GS_DATA/sparse/0"
ln -sfn ../dense/images "$GS_DATA/images"
for f in cameras.bin images.bin points3D.bin; do
  ln -sfn ../../../dense/sparse/$f "$GS_DATA/sparse/0/$f"
done

cd "$GS_ROOT"
python train.py -s "$GS_DATA" -m "$GS_DATA/output"
```

这里的 `python` 要用带 CUDA 版 PyTorch 的解释器。这次是 `~/.conda/envs/pytorch-common/bin/python`（torch 2.8，CUDA 12.8）。系统自带的 CUDA 13.2 和它对不上，编 `diff-gaussian-rasterization` 时要把 `CUDA_HOME` 指到这个环境，并在 `rasterizer_impl.h`、`forward.h`、`backward.h` 里补上 `#include <cstdint>`。

训完后，训练视角的对照图：

```bash
python render.py -m "$GS_DATA/output" -s "$GS_DATA" --skip_test
```

`output/train/ours_30000/renders` 是高斯渲的，旁边 `gt` 是原照片。这一步还是站在原来的相机上。

自由视角用下面的脚本。它从训练相机里取两台（默认是文件名排序后的第 0 张和第 1 张，换场景时改 `i0, i1`），在相机中心和朝向之间插 30 帧。焦距不动。输出是一条能当视频播的图序列。

```bash
cd "$GS_ROOT"
python - << 'PY'
import os
import numpy as np
import torch
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import safe_state

parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--quiet", action="store_true")
args = get_combined_args(parser)
safe_state(True)
dataset = model.extract(args)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
views = scene.getTrainCameras()
i0, i1 = 0, 1
a, b = views[i0], views[i1]
# 存下来的 R 是相机到世界的旋转，T 是世界到相机的平移。插值要在相机中心上做。
ca, cb = -a.R @ a.T, -b.R @ b.T
out = os.path.join(dataset.model_path, "flythrough")
os.makedirs(out, exist_ok=True)
bg = torch.zeros(3, device="cuda")
pipe = pipeline.extract(args)
for i, t in enumerate(np.linspace(0, 1, 30)):
    R = (1 - t) * a.R + t * b.R
    u, _, vh = np.linalg.svd(R)
    R = u @ vh
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vh
    center = (1 - t) * ca + t * cb
    T = -R.T @ center
    fovx = (1 - t) * a.FoVx + t * b.FoVx
    fovy = (1 - t) * a.FoVy + t * b.FoVy
    w2c = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
    proj = getProjectionMatrix(0.01, 100.0, fovx, fovy).transpose(0, 1).cuda()
    full = w2c.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    cam = MiniCam(a.image_width, a.image_height, fovy, fovx, 0.01, 100.0, w2c, full)
    image = render(cam, gaussians, pipe, bg)["render"]
    torchvision.utils.save_image(image, os.path.join(out, f"{i:02d}.png"))
print(a.image_name, "->", b.image_name)
print("wrote", out)
PY
```

跑的时候加上和训练相同的 `-m`、`-s`。第 0 帧和第 29 帧应接近那两张训练照片，中间帧是没有拍过的角度。网格查看器转到同样的方向，会看到贴图接缝；这条序列里不应该出现图集的缝。

## 应用 2 — 和稠密重建比效果

比的时候把三样东西放在一起：

1. 原照片。
2. 查看器里的贴图网格，相机转到同一方向。
3. 应用 1 里同一方向的高斯渲染。

站在训练相机上，两边都应该像原照片。网格靠把照片投到面上，高斯靠把颜色拟合进高斯。这一帧看不出谁更好。

拉开之后看这几处：

- **墙面和窗**：网格的贴图在斜视时发糊、错位。高斯仍是一张完整的立面，窗的边缘来自多张照片的颜色，不是一张贴图被拉开。
- **树和屋檐**：Poisson 把它们收成一块光滑的壳，简化到 34 万面之后更明显。高斯里这些结构还在，但前后会有一层漂着的点，网格上没有这种漂。
- **楼的背面和天空**：网格要么是补出来的墙，要么是洞。高斯在这里没有照片可拟合，渲出来是糊的颜色，不能当成补全的几何。
- **要量尺寸、要导进别的软件**：用 `dense/fused.ply` 或简化后的网格。高斯的 `point_cloud.ply` 是椭球中心，不是表面。

稠密这边的代价已经记在 playground 里：PatchMatch 57 分钟，Poisson 又约 27 分钟。3DGS 默认 30000 步，时间主要花在反复渲这 100 张图，不产生网格。

## 应用 3 — 只飞训练相机，检查有没有训坏

自由视角好看，不代表训练相机被拟合上了。`render.py` 的 `renders` 和 `gt` 逐张看：

- 整张发灰、窗户对不上，是相机或去畸变目录指错了。Gerrard 必须用 `dense/images` 配 `dense/sparse`，不能拿原始的 5616×3744 照片去配去畸变后的内参。
- 只有树和天空脏、墙是对的，是高斯太少或停得太早。初始点就是这 357 万融合点之前的稀疏点；要让起步更密，把 `dense/fused.ply` 拷成 `sparse/0/points3D.ply` 再训练。这个文件必须在第一次 `train.py` 之前放好。
- 墙面颜色随照片变亮变暗，是拍摄时曝光没锁。训练加上 `--exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp`，每张图学一个颜色变换。Gerrard 是同一套相机、曝光比较稳，可以先不加。
