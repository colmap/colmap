# Held-out localization against a 3DGS map.
# Coarse pose: render RGB and inverse-depth at a retrieved training view,
# match SIFT features, lift them with the rendered depth, solve PnP.
# Fine pose: photometric refinement. Matching and the loss curve are written
# next to the poses so the split can be inspected.

import json
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
from utils.loss_utils import l1_loss, ssim

SHOWCASE = "IMG_2371.JPG"
REFINE_STEPS = 60
REFINE_WIDTH = 640


def skew(w):
    x, y, z = w
    o = w.new_zeros(())
    return torch.stack(
        (torch.stack((o, -z, y)), torch.stack((z, o, -x)), torch.stack((-y, x, o)))
    )


def so3_exp(phi):
    theta = torch.linalg.norm(phi).clamp(min=1e-8)
    k = phi / theta
    K = skew(k)
    eye = torch.eye(3, device=phi.device, dtype=phi.dtype)
    return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def rotmat_to_quat(R):
    s = torch.sqrt((R.trace() + 1.0).clamp(min=1e-8)) * 2.0
    return torch.stack(
        (
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        )
    )


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def w2c_of(cam):
    R_cw = torch.tensor(cam.R.T, device="cuda", dtype=torch.float32)
    t_cw = torch.tensor(cam.T, device="cuda", dtype=torch.float32)
    return R_cw, t_cw


def pose_error(R_cw, t_cw, R_gt, t_gt):
    cosine = (((R_cw.T @ R_gt).trace() - 1.0) * 0.5).clamp(-1.0, 1.0)
    rot = torch.rad2deg(torch.acos(cosine)).item()
    center = -R_cw.T @ t_cw
    center_gt = -R_gt.T @ t_gt
    trans = torch.linalg.norm(center - center_gt).item()
    return rot, trans


def render_pose(gaussians, pipe, bg, R_cw, t_cw, width, height, fovx, fovy):
    R_c2w = R_cw.detach().float().cpu().numpy().T
    t = t_cw.detach().float().cpu().numpy()
    w2c = torch.tensor(getWorld2View2(R_c2w, t)).transpose(0, 1).cuda().float()
    proj = getProjectionMatrix(0.01, 100.0, fovx, fovy).transpose(0, 1).cuda()
    full = w2c.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    cam = MiniCam(width, height, fovy, fovx, 0.01, 100.0, w2c, full)
    out = render(cam, gaussians, pipe, bg)
    return out["render"], out["depth"]


def to_bgr(image):
    rgb = (image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def intrinsics(width, height, fovx, fovy):
    fx = width / (2.0 * math.tan(fovx / 2.0))
    fy = height / (2.0 * math.tan(fovy / 2.0))
    return fx, fy, width / 2.0, height / 2.0


def lift_and_pnp(query_bgr, render_bgr, invdepth, R_cw, t_cw, fovx, fovy):
    height, width = query_bgr.shape[:2]
    sift = cv2.SIFT_create(nfeatures=4000)
    kq, dq = sift.detectAndCompute(cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY), None)
    kr, dr = sift.detectAndCompute(cv2.cvtColor(render_bgr, cv2.COLOR_BGR2GRAY), None)
    if dq is None or dr is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dq, dr, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    depth = invdepth.detach().float().cpu().numpy()
    depth = depth[0] if depth.ndim == 3 else depth
    fx, fy, cx, cy = intrinsics(width, height, fovx, fovy)
    R_c2w = R_cw.detach().float().cpu().numpy().T
    center = (-R_cw.T @ t_cw).detach().float().cpu().numpy()
    obj, img, kept = [], [], []
    for m in good:
        uq, vq = kq[m.queryIdx].pt
        ur, vr = kr[m.trainIdx].pt
        ui, vi = int(round(ur)), int(round(vr))
        if not (0 <= ui < width and 0 <= vi < height):
            continue
        inv_z = float(depth[vi, ui])
        if inv_z < 1e-3:
            continue
        z = 1.0 / inv_z
        if not (0.3 < z < 80.0):
            continue
        p_cam = np.array([(ur - cx) / fx * z, (vr - cy) / fy * z, z], dtype=np.float64)
        obj.append(R_c2w @ p_cam + center)
        img.append([uq, vq])
        kept.append(m)
    if len(obj) < 6:
        return {"n_match": len(good), "n_lifted": len(obj), "ok": False, "good": good, "kq": kq, "kr": kr}
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(obj, np.float64),
        np.asarray(img, np.float64),
        K,
        None,
        iterationsCount=200,
        reprojectionError=max(width * 0.01, 3.0),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    inlier_set = set(inliers.reshape(-1).tolist()) if inliers is not None else set()
    inlier_ids = sorted(inlier_set)
    if len(inlier_ids) > 80:
        step = len(inlier_ids) / 80
        show = {inlier_ids[int(i * step)] for i in range(80)}
    else:
        show = inlier_set
    drawn = cv2.drawMatches(
        query_bgr,
        kq,
        render_bgr,
        kr,
        kept,
        None,
        matchColor=(80, 220, 120),
        singlePointColor=(80, 80, 80),
        matchesMask=[1 if i in show else 0 for i in range(len(kept))],
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    result = {
        "n_match": len(good),
        "n_lifted": len(obj),
        "n_inlier": len(inlier_set),
        "ok": bool(ok) and len(inlier_set) >= 6,
        "vis": drawn,
    }
    if result["ok"]:
        R, _ = cv2.Rodrigues(rvec)
        result["R_cw"] = torch.tensor(R, device="cuda", dtype=torch.float32)
        result["t_cw"] = torch.tensor(tvec.reshape(3), device="cuda", dtype=torch.float32)
    return result


def refine(gaussians, pipe, bg, xyz0, quat0, query, R_init, t_init, R_gt, t_gt):
    p_init = xyz0 @ R_init.T + t_init
    q_init = quat_mul(rotmat_to_quat(R_init), quat0)
    phi = torch.nn.Parameter(torch.zeros(3, device="cuda"))
    rho = torch.nn.Parameter(torch.zeros(3, device="cuda"))
    opt = torch.optim.Adam(({"params": [phi], "lr": 0.002}, {"params": [rho], "lr": 0.008}))
    gt = F.interpolate(
        query.original_image.unsqueeze(0),
        size=(
            max(int(round(REFINE_WIDTH * query.image_height / query.image_width)), 1),
            REFINE_WIDTH,
        ),
        mode="bilinear",
        align_corners=False,
    )[0]
    gaussians.active_sh_degree = 0
    history = []
    try:
        for _ in range(REFINE_STEPS):
            opt.zero_grad(set_to_none=True)
            R_delta = so3_exp(phi)
            gaussians._xyz = p_init @ R_delta.T + rho
            gaussians._rotation = quat_mul(rotmat_to_quat(R_delta), q_init)
            height = gt.shape[1]
            image, _ = render_pose(
                gaussians, pipe, bg, torch.eye(3, device="cuda"), torch.zeros(3, device="cuda"),
                REFINE_WIDTH, height, query.FoVx, query.FoVy,
            )
            loss = (1.0 - 0.2) * l1_loss(image, gt) + 0.2 * (1.0 - ssim(image, gt))
            loss.backward()
            opt.step()
            R_cw = R_delta.detach() @ R_init
            t_cw = (rho + R_delta @ t_init).detach()
            rot, trans = pose_error(R_cw, t_cw, R_gt, t_gt)
            history.append({"loss": float(loss.item()), "rot": rot, "trans": trans})
        return R_cw, t_cw, history
    finally:
        gaussians._xyz = xyz0
        gaussians._rotation = quat0
        gaussians.active_sh_degree = gaussians.max_sh_degree


def curve_svg(history, key, label, color):
    w, h, pad = 640, 220, 36
    values = [row[key] for row in history]
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-6)

    def xy(i, value):
        x = pad + (w - 2 * pad) * (i / max(len(values) - 1, 1))
        y = pad + (h - 2 * pad) * (1.0 - (value - lo) / span)
        return x, y

    pts = " ".join(f"{xy(i, v)[0]:.1f},{xy(i, v)[1]:.1f}" for i, v in enumerate(values))
    return f"""<svg viewBox="0 0 {w} {h}" width="100%" role="img">
      <rect width="{w}" height="{h}" fill="#12171e"/>
      <polyline fill="none" stroke="{color}" stroke-width="2.5" points="{pts}"/>
      <text x="{pad}" y="22" fill="#8b98a8" font-size="13">{label} {values[0]:.3f} → {values[-1]:.3f}</text>
    </svg>"""


def write_page(out_dir, rows, showcase):
    body = []
    for row in rows:
        mark = " style=\"background:#1c2836\"" if row["query"] == showcase else ""
        body.append(
            "<tr{mark}><td><a href=\"{query}/matches.jpg\">{query}</a> · <a href=\"{query}/loss.svg\">损失</a></td><td>{retrieved}</td>"
            "<td>{ret_rot:.2f}° / {ret_trans:.3f} m</td>"
            "<td>{pnp}</td><td>{ref}</td></tr>".format(
                mark=mark,
                query=row["query"],
                retrieved=row["retrieved"],
                ret_rot=row["retrieval_rot"],
                ret_trans=row["retrieval_trans"],
                pnp=row["pnp_text"],
                ref=row["refine_text"],
            )
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>Gerrard Hall · 定位</title>
  <style>
    body {{ margin: 0; background: #0c0f14; color: #e7edf5; font: 14px/1.5 "Segoe UI","PingFang SC",sans-serif; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 28px 20px 64px; }}
    a {{ color: #7eb6ff; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    .muted {{ color: #8b98a8; }}
    img, svg {{ display: block; width: 100%; border-radius: 10px; background: #12171e; }}
    figure {{ margin: 18px 0; }}
    figcaption {{ color: #8b98a8; margin-bottom: 6px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #2a3340; font-variant-numeric: tabular-nums; }}
    th {{ color: #8b98a8; font-weight: 500; }}
  </style>
</head>
<body>
<main>
  <h1>Gerrard Hall · 重建集和定位集</h1>
  <p class="muted">100 张按文件名排序，下标能被 8 整除的 13 张只做定位，其余 87 张参与 3DGS。查询不在重建里。初值是和查询最像的一张重建照片的位姿，不是查询自己的真值。</p>
  <p><a href="../">返回查看器</a></p>
  <figure>
    <figcaption>{showcase}：左边是查询照片，右边是用检索位姿渲出来的图。绿线是 PnP 内点里均匀抽出的 80 对，避免全部画上去糊成一片。</figcaption>
    <img src="{showcase}/matches.jpg" alt="特征匹配"/>
  </figure>
  <figure>
    <figcaption>PnP 之后接着做光度收紧。上图是渲染图和查询的 L1+SSIM，下图是相对 COLMAP 位姿的旋转误差。</figcaption>
    {open(os.path.join(out_dir, showcase, "loss.svg"), encoding="utf-8").read()}
    {open(os.path.join(out_dir, showcase, "rot.svg"), encoding="utf-8").read()}
  </figure>
  <table>
    <thead><tr><th>查询</th><th>检索到的重建图</th><th>检索位姿误差</th><th>PnP 之后</th><th>光度收紧之后</th></tr></thead>
    <tbody>
    {''.join(body)}
    </tbody>
  </table>
</main>
</body>
</html>
"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--only", default="")
    args = get_combined_args(parser)
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    pipe = pipeline.extract(args)
    bg = torch.zeros(3, device="cuda")
    xyz0 = gaussians.get_xyz.detach()
    quat0 = gaussians.get_rotation.detach()
    thumbs = torch.stack(
        [F.interpolate(cam.original_image.unsqueeze(0), size=(48, 64), mode="area")[0] for cam in train_cams]
    )

    out_dir = os.path.join(dataset.model_path, "localize_demo")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    print(f"train {len(train_cams)}  test {len(test_cams)}", flush=True)
    queries = [c for c in test_cams if args.only in c.image_name]
    for query in queries:
        R_gt, t_gt = w2c_of(query)
        qthumb = F.interpolate(query.original_image.unsqueeze(0), size=(48, 64), mode="area")[0]
        dist = (thumbs - qthumb).pow(2).mean(dim=(1, 2, 3))
        retrieved = train_cams[int(dist.argmin())]
        R_init, t_init = w2c_of(retrieved)
        ret_rot, ret_trans = pose_error(R_init, t_init, R_gt, t_gt)
        color, invdepth = render_pose(
            gaussians, pipe, bg, R_init, t_init,
            query.image_width, query.image_height, query.FoVx, query.FoVy,
        )
        matched = lift_and_pnp(
            to_bgr(query.original_image), to_bgr(color), invdepth, R_init, t_init, query.FoVx, query.FoVy
        )
        folder = os.path.join(out_dir, query.image_name)
        os.makedirs(folder, exist_ok=True)
        if matched and matched.get("vis") is not None:
            cv2.imwrite(os.path.join(folder, "matches.jpg"), matched["vis"])
        pnp_text = "匹配不够"
        R_pnp, t_pnp = R_init, t_init
        if matched and matched.get("ok"):
            R_pnp, t_pnp = matched["R_cw"], matched["t_cw"]
            pnp_rot, pnp_trans = pose_error(R_pnp, t_pnp, R_gt, t_gt)
            pnp_text = f"{pnp_rot:.2f}° / {pnp_trans:.3f} m · 内点 {matched['n_inlier']}"
        else:
            pnp_rot = pnp_trans = None
        R_ref, t_ref, history = refine(
            gaussians, pipe, bg, xyz0, quat0, query, R_pnp, t_pnp, R_gt, t_gt
        )
        ref_rot, ref_trans = pose_error(R_ref, t_ref, R_gt, t_gt)
        with open(os.path.join(folder, "loss.svg"), "w", encoding="utf-8") as f:
            f.write(curve_svg(history, "loss", "光度损失", "#7eb6ff"))
        with open(os.path.join(folder, "rot.svg"), "w", encoding="utf-8") as f:
            f.write(curve_svg(history, "rot", "旋转误差（度）", "#e2b15a"))
        row = {
            "query": query.image_name,
            "retrieved": retrieved.image_name,
            "retrieval_rot": ret_rot,
            "retrieval_trans": ret_trans,
            "pnp_rot": pnp_rot,
            "pnp_trans": pnp_trans,
            "pnp_text": pnp_text,
            "refine_rot": ref_rot,
            "refine_trans": ref_trans,
            "refine_text": f"{ref_rot:.2f}° / {ref_trans:.3f} m",
            "n_inlier": None if not matched else matched.get("n_inlier"),
        }
        rows.append(row)
        print(
            f"{query.image_name} via {retrieved.image_name}  "
            f"retrieval {ret_rot:.1f}°/{ret_trans:.2f}m  {pnp_text}  refine {ref_rot:.2f}°/{ref_trans:.3f}m",
            flush=True,
        )
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": [c.image_name for c in train_cams],
                "test": [c.image_name for c in test_cams],
                "rows": rows,
            },
            f,
            indent=2,
        )
    showcase = SHOWCASE if any(r["query"] == SHOWCASE for r in rows) else rows[0]["query"]
    write_page(out_dir, rows, showcase)
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    main()
