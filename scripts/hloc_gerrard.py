#!/usr/bin/env python3
"""Sparse-map localization on Gerrard Hall, using this repository's COLMAP.

The hold-out matches the 3DGS demo: sorted names whose index is divisible by 8
are queries. The other 87 images keep their existing poses. SuperPoint and
LightGlue stay in hloc. Verification, triangulation, and PnP call
build/src/colmap/exe/colmap.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HLOC_ROOT = ROOT / "third_party" / "Hierarchical-Localization"
sys.path.insert(0, str(HLOC_ROOT))

from hloc import extract_features, match_features, pairs_from_retrieval  # noqa: E402
from hloc import localize_sfm, triangulation  # noqa: E402
from hloc.colmap_backend import (  # noqa: E402
    model_to_txt,
    read_images_txt,
)


def holdout(image_dir):
    names = sorted(
        path.name
        for path in Path(image_dir).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    queries = [name for index, name in enumerate(names) if index % 8 == 0]
    mapping = [name for name in names if name not in set(queries)]
    return mapping, queries


def covisible_pairs(model_dir, keep_names, num_matched, output):
    """Pair map images that already share SIFT points in the reference model."""
    txt_dir = Path(output).parent / "covis_txt"
    model_to_txt(model_dir, txt_dir)
    observations = _observations(txt_dir / "images.txt")
    keep_names = set(keep_names)
    point_to_images = defaultdict(list)
    for name, points in observations.items():
        if name not in keep_names:
            continue
        for _xy, point3d_id in points:
            if point3d_id != -1:
                point_to_images[point3d_id].append(name)

    shared = defaultdict(int)
    for names in point_to_images.values():
        unique = sorted(set(names))
        for i, name_i in enumerate(unique):
            for name_j in unique[i + 1 :]:
                shared[(name_i, name_j)] += 1

    neighbors = defaultdict(list)
    for (name_i, name_j), count in shared.items():
        neighbors[name_i].append((count, name_j))
        neighbors[name_j].append((count, name_i))

    pairs = []
    seen = set()
    for name in keep_names:
        ranked = sorted(neighbors[name], reverse=True)[:num_matched]
        for _count, other in ranked:
            pair = tuple(sorted((name, other)))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("".join(f"{a} {b}\n" for a, b in pairs))
    return Path(output)


def _observations(images_txt):
    observations = {}
    header = None
    for line in Path(images_txt).read_text().splitlines():
        if line.startswith("#"):
            continue
        if header is None:
            if not line.strip():
                continue
            header = line
            continue
        points = line
        name = header.split()[-1]
        values = points.split()
        obs = []
        for index in range(0, len(values), 3):
            obs.append(
                (
                    (float(values[index]), float(values[index + 1])),
                    int(values[index + 2]),
                )
            )
        observations[name] = obs
        header = None
    if header is not None:
        raise ValueError(f"{images_txt} is not a two-line COLMAP images.txt")
    return observations


def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def pose_error(qvec, tvec, qvec_gt, tvec_gt):
    rotation = qvec_to_rotmat(qvec)
    rotation_gt = qvec_to_rotmat(qvec_gt)
    cosine = np.clip((np.trace(rotation.T @ rotation_gt) - 1.0) * 0.5, -1.0, 1.0)
    degrees = float(np.degrees(np.arccos(cosine)))
    center = -rotation.T @ tvec
    center_gt = -rotation_gt.T @ tvec_gt
    meters = float(np.linalg.norm(center - center_gt))
    return degrees, meters


def _first_retrieved(retrieval_pairs):
    retrieved = {}
    for line in Path(retrieval_pairs).read_text().splitlines():
        if not line.strip():
            continue
        query, ref = line.split()
        retrieved.setdefault(query, ref)
    return retrieved


def write_report(output_dir, reference_model, queries, retrieval_pairs, localized_txt):
    gt_txt = Path(output_dir) / "gt_txt"
    model_to_txt(reference_model, gt_txt)
    truth = read_images_txt(gt_txt / "images.txt")
    posed = read_images_txt(localized_txt)
    observations = _observations(localized_txt)
    retrieved = _first_retrieved(retrieval_pairs)

    rows = []
    for name in queries:
        gt = truth[name]
        ref_name = retrieved.get(name)
        ref = truth[ref_name]
        ret_rot, ret_trans = pose_error(ref["qvec"], ref["tvec"], gt["qvec"], gt["tvec"])
        inliers = sum(point_id != -1 for _xy, point_id in observations.get(name, []))
        if name not in posed:
            rows.append((name, ref_name, ret_rot, ret_trans, None, None, inliers))
            continue
        est = posed[name]
        rot, trans = pose_error(est["qvec"], est["tvec"], gt["qvec"], gt["tvec"])
        rows.append((name, ref_name, ret_rot, ret_trans, rot, trans, inliers))

    tight = sum(
        1
        for row in rows
        if row[4] is not None and row[4] <= 2.0 and row[5] <= 0.25
    )
    indoor = sum(
        1
        for row in rows
        if row[4] is not None and row[4] <= 5.0 and row[5] <= 0.05
    )
    lines = [
        f"registered {sum(row[4] is not None for row in rows)} / {len(rows)}",
        f"within 0.25 m and 2 deg: {tight}",
        f"within 5 cm and 5 deg: {indoor}",
        "3DGS PnP on IMG_2371.JPG was 1.34 deg / 0.073 m, 722 inliers",
        "",
    ]
    for name, ref_name, ret_rot, ret_trans, rot, trans, inliers in rows:
        if rot is None:
            posed_text = "not registered"
        else:
            posed_text = f"{rot:.2f} deg / {trans:.3f} m, {inliers} inliers"
        lines.append(
            f"{name} via {ref_name}  retrieval {ret_rot:.2f} deg / {ret_trans:.3f} m"
            f"  pnp {posed_text}"
        )
    report = Path(output_dir) / "report.txt"
    report.write_text("\n".join(lines) + "\n")
    print(report.read_text())
    return report


def draw_match(image_dir, localized_txt, retrieval_pairs, showcase, output):
    """Draw 2D-3D correspondences that share a triangulated point."""
    import cv2

    observations = _observations(localized_txt)
    if showcase not in observations:
        return None
    ref_name = _first_retrieved(retrieval_pairs).get(showcase)
    if ref_name not in observations:
        return None
    query_points = {
        point_id: xy
        for xy, point_id in observations[showcase]
        if point_id != -1
    }
    ref_points = {
        point_id: xy for xy, point_id in observations[ref_name] if point_id != -1
    }
    shared = [point_id for point_id in query_points if point_id in ref_points]
    if not shared:
        return None
    shared.sort(key=lambda point_id: query_points[point_id][0])
    if len(shared) > 80:
        step = len(shared) / 80
        shared = [shared[int(i * step)] for i in range(80)]
    query = cv2.imread(str(Path(image_dir) / showcase))
    ref = cv2.imread(str(Path(image_dir) / ref_name))
    height = 480
    def _resize(image):
        scale = height / image.shape[0]
        return cv2.resize(image, (int(image.shape[1] * scale), height)), scale

    query, query_scale = _resize(query)
    ref, ref_scale = _resize(ref)
    canvas = np.concatenate([query, ref], axis=1)
    offset = query.shape[1]
    rng = np.random.default_rng(0)
    for point_id in shared[:80]:
        qx, qy = query_points[point_id]
        rx, ry = ref_points[point_id]
        color = tuple(int(v) for v in rng.integers(40, 255, size=3))
        p0 = (int(qx * query_scale), int(qy * query_scale))
        p1 = (int(rx * ref_scale) + offset, int(ry * ref_scale))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)
    return Path(output)


def draw_lightglue(image_dir, features, matches, name, ref_name, output):
    """Draw LightGlue matches when the pair shares no triangulated point."""
    import cv2

    from hloc.utils.io import get_keypoints, get_matches

    pairs, _scores = get_matches(matches, name, ref_name)
    if len(pairs) == 0:
        return None
    query_kps = get_keypoints(features, name)
    ref_kps = get_keypoints(features, ref_name)
    order = np.argsort(query_kps[pairs[:, 0], 0])
    pairs = pairs[order]
    if len(pairs) > 80:
        step = len(pairs) / 80
        pairs = pairs[[int(i * step) for i in range(80)]]
    query = cv2.imread(str(Path(image_dir) / name))
    ref = cv2.imread(str(Path(image_dir) / ref_name))
    height = 480

    def _resize(image):
        scale = height / image.shape[0]
        return cv2.resize(image, (int(image.shape[1] * scale), height)), scale

    query, query_scale = _resize(query)
    ref, ref_scale = _resize(ref)
    canvas = np.concatenate([query, ref], axis=1)
    offset = query.shape[1]
    rng = np.random.default_rng(0)
    for query_idx, ref_idx in pairs:
        qx, qy = query_kps[int(query_idx)]
        rx, ry = ref_kps[int(ref_idx)]
        color = tuple(int(v) for v in rng.integers(40, 255, size=3))
        p0 = (int(qx * query_scale), int(qy * query_scale))
        p1 = (int(rx * ref_scale) + offset, int(ry * ref_scale))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)
    return Path(output)


def write_page(
    output_dir, image_dir, queries, retrieval_pairs, localized_txt, features=None, matches=None
):
    """Match images for every query, plus a page the local viewer can serve."""
    retrieved = _first_retrieved(retrieval_pairs)
    for name in queries:
        output = Path(output_dir) / name / "matches.jpg"
        drawn = draw_match(image_dir, localized_txt, retrieval_pairs, name, output)
        if drawn is None and features is not None and matches is not None:
            draw_lightglue(
                image_dir, features, matches, name, retrieved[name], output
            )
    report_lines = (Path(output_dir) / "report.txt").read_text().splitlines()
    body = []
    for line in report_lines:
        if " via " not in line or " pnp " not in line:
            continue
        name, rest = line.split(" via ", 1)
        ref_name, rest = rest.split("  retrieval ", 1)
        retrieval, pnp = rest.split("  pnp ", 1)
        failed = pnp.startswith("not ") or float(pnp.split(" deg", 1)[0]) > 5.0
        style = ' style="background:#3a2430"' if failed else ""
        if name == "IMG_2371.JPG":
            style = ' style="background:#1c2836"'
        body.append(
            f'<tr{style}><td><a href="{name}/matches.jpg">{name}</a></td>'
            f"<td>{ref_name}</td><td>{retrieval}</td><td>{pnp}</td></tr>"
        )
    registered = next(line for line in report_lines if line.startswith("registered"))
    tight = next(line for line in report_lines if line.startswith("within 0.25"))
    indoor = next(line for line in report_lines if line.startswith("within 5 cm"))
    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>Gerrard Hall · 稀疏定位</title>
  <style>
    body {{ margin: 0; background: #0c0f14; color: #e7edf5; font: 14px/1.5 "Segoe UI","PingFang SC",sans-serif; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 28px 20px 64px; }}
    a {{ color: #7eb6ff; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    .muted {{ color: #8b98a8; }}
    img {{ display: block; width: 100%; border-radius: 10px; background: #12171e; }}
    figure {{ margin: 18px 0; }}
    figcaption {{ color: #8b98a8; margin-bottom: 6px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #2a3340; font-variant-numeric: tabular-nums; }}
    th {{ color: #8b98a8; font-weight: 500; }}
  </style>
</head>
<body>
<main>
  <h1>Gerrard Hall · 稀疏地图定位</h1>
  <p class="muted">87 张保持已有位姿，用 SuperPoint 和 LightGlue 重新三角化。13 张查询先用 NetVLAD 找 10 张邻居，再由本仓库的 mapper 做 P3P。左边是查询照片，右边是检索到的第一张重建图。线是两边落在同一个三维点上的观测，最多 80 条。</p>
  <p class="muted">{registered}。{tight}。{indoor}。</p>
  <p><a href="../">返回查看器</a> · <a href="../gerrard_loc/">3DGS 定位</a></p>
  <figure>
    <figcaption>IMG_2371.JPG 对上 IMG_2372.JPG。PnP 是 0.04°，位移不到 1 毫米，3071 个三维点。3DGS 同一张的 PnP 是 1.34°、7.3 厘米。</figcaption>
    <img src="IMG_2371.JPG/matches.jpg" alt="IMG_2371 匹配"/>
  </figure>
  <figure>
    <figcaption>IMG_2387.JPG 对上 IMG_2386.JPG。线是 LightGlue 的二维匹配。这张重建图上没有留下三角化的点，所以这些匹配进不了 PnP。位姿是 73.66°、1.73 米。</figcaption>
    <img src="IMG_2387.JPG/matches.jpg" alt="IMG_2387 匹配"/>
  </figure>
  <table>
    <thead><tr><th>查询</th><th>检索到的重建图</th><th>检索位姿误差</th><th>PnP 之后</th></tr></thead>
    <tbody>
    {''.join(body)}
    </tbody>
  </table>
</main>
</body>
</html>
"""
    path = Path(output_dir) / "index.html"
    path.write_text(page)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "data/gerrard-hall/dense/images",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "data/gerrard-hall/dense/sparse",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=ROOT / "data/gerrard-hall/hloc",
    )
    parser.add_argument("--num-covis", type=int, default=20)
    parser.add_argument("--num-retrieval", type=int, default=10)
    args = parser.parse_args()

    outputs = args.outputs
    outputs.mkdir(parents=True, exist_ok=True)
    mapping, queries = holdout(args.images)
    (outputs / "mapping.txt").write_text("\n".join(mapping) + "\n")
    (outputs / "queries.txt").write_text("\n".join(queries) + "\n")

    feature_conf = extract_features.confs["superpoint_max"]
    matcher_conf = match_features.confs["superpoint+lightglue"]
    retrieval_conf = extract_features.confs["netvlad"]
    features = extract_features.main(feature_conf, args.images, outputs)
    global_features = extract_features.main(retrieval_conf, args.images, outputs)

    map_pairs = covisible_pairs(
        args.reference, mapping, args.num_covis, outputs / "pairs-covis.txt"
    )
    map_matches = match_features.main(
        matcher_conf, map_pairs, features, outputs, matches=outputs / "matches-covis.h5"
    )
    model = triangulation.main(
        outputs / "sfm",
        args.reference,
        args.images,
        map_pairs,
        features,
        map_matches,
        image_names=mapping,
    )

    loc_pairs = outputs / "pairs-loc.txt"
    pairs_from_retrieval.main(
        global_features,
        loc_pairs,
        args.num_retrieval,
        query_list=outputs / "queries.txt",
        db_list=outputs / "mapping.txt",
    )
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, features, outputs, matches=outputs / "matches-loc.h5"
    )
    query_list = outputs / "queries_with_intrinsics.txt"
    # localize_sfm only reads the image name. Intrinsics come from the map camera.
    query_list.write_text("".join(f"{name}\n" for name in queries))
    results = outputs / "poses.txt"
    localize_sfm.main(
        model,
        query_list,
        loc_pairs,
        features,
        loc_matches,
        results,
        image_dir=args.images,
        database_path=outputs / "sfm" / "database.db",
    )
    localized_txt = outputs / "sfm" / "localized" / "txt" / "images.txt"
    write_report(outputs, args.reference, queries, loc_pairs, localized_txt)
    write_page(
        outputs,
        args.images,
        queries,
        loc_pairs,
        localized_txt,
        features,
        loc_matches,
    )


if __name__ == "__main__":
    main()
