#!/usr/bin/env python3
"""Visualize MixVPR global descriptor retrieval results.

For a set of query images, shows the top-K most similar images from the
database, arranged in a grid: query on the left, results on the right.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import onnxruntime as ort


# ---------------------------------------------------------------------------
# MixVPR preprocessing (must match export_quant_onnx.py exactly)
# ---------------------------------------------------------------------------
INPUT_SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_path: str) -> np.ndarray:
    """Load, resize to 320x320, ImageNet-normalize → (1, 3, 320, 320) float32."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0  # HWC, [0, 1]
    # Normalize: (val - mean) / std
    arr = (arr - MEAN) / STD
    # HWC → CHW, add batch dim
    arr = np.transpose(arr, (2, 0, 1))
    return arr[np.newaxis, ...]


# ---------------------------------------------------------------------------
# Descriptor extraction
# ---------------------------------------------------------------------------
def extract_descriptors(
    image_paths: list, model_path: str, batch_size: int = 16
) -> np.ndarray:
    """Extract MixVPR descriptors for all images (batched)."""
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    N = len(image_paths)
    all_descs = np.empty((N, 4096), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_imgs = []
        for i in range(start, end):
            batch_imgs.append(preprocess(image_paths[i])[0])  # remove batch dim
        batch = np.stack(batch_imgs, axis=0).astype(np.float32)
        descs = sess.run(None, {iname: batch})[0]
        all_descs[start:end] = descs
        print(f"  Extracted [{end}/{N}]", flush=True)

    # Descriptors are already L2-normalized by the model
    return all_descs


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve_topk(
    query_idx: int,
    all_descs: np.ndarray,
    all_paths: list,
    top_k: int = 10,
) -> list[tuple[int, float, str]]:
    """Return top-K most similar images (excluding self)."""
    query = all_descs[query_idx]
    # Cosine similarity = dot product (vectors are L2-normalized)
    sims = np.dot(all_descs, query)  # (N,)
    # Sort descending, exclude self
    order = np.argsort(-sims)
    results = []
    for idx in order:
        if idx == query_idx:
            continue
        results.append((int(idx), float(sims[idx]), all_paths[idx]))
        if len(results) >= top_k:
            break
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def create_grid(
    query_path: str,
    results: list[tuple[int, float, str]],
    thumb_size: int = 256,
    cols: int = 5,
) -> Image.Image:
    """Create a visualization grid: query + top-K results."""
    n_results = len(results)
    rows = (n_results + cols - 1) // cols

    # Grid: [query label, query image, spacer row, results...]
    # Layout: 1 row for query, 1 row spacer, rows for results
    grid_h = (2 + rows) * (thumb_size + 30)  # 30px for labels
    grid_w = cols * (thumb_size + 10) + 10

    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    y = 5

    # --- Query row ---
    draw.text((10, y), f"Query: {Path(query_path).name}", fill=(255, 255, 100), font=font)
    y += 22

    qimg = Image.open(query_path).convert("RGB")
    qimg.thumbnail((thumb_size * 2, thumb_size * 2), Image.LANCZOS)
    grid.paste(qimg, (10, y))

    y += qimg.height + 15

    # --- Spacer ---
    draw.text((10, y), "Top-K Retrieval Results (cosine similarity):", fill=(200, 200, 200), font=font)
    y += 25

    # --- Results grid ---
    for i, (idx, score, path) in enumerate(results):
        col = i % cols
        row = i // cols
        x = 10 + col * (thumb_size + 10)
        yy = y + row * (thumb_size + 28)

        # Thumbnail
        try:
            thumb = Image.open(path).convert("RGB")
            thumb.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
            grid.paste(thumb, (x, yy))
        except Exception:
            draw.rectangle([x, yy, x + thumb_size, yy + thumb_size], fill=(60, 60, 60))

        # Label
        label = f"#{i+1} sim={score:.4f}  {Path(path).name}"
        draw.text((x, yy + thumb_size + 2), label, fill=(180, 180, 180), font=font_small)

    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize MixVPR image retrieval")
    parser.add_argument("--image_dir", required=True, help="Directory of images")
    parser.add_argument("--model", required=True, help="Path to MixVPR ONNX model")
    parser.add_argument("--num_query", type=int, default=3, help="Number of query images")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K results per query")
    parser.add_argument("--batch_size", type=int, default=16, help="ONNX batch size")
    parser.add_argument("--thumb_size", type=int, default=200, help="Thumbnail size")
    parser.add_argument("--output_dir", default="retrieval_viz", help="Output directory")
    parser.add_argument("--query_indices", nargs="*", type=int, default=None,
                        help="Specific image indices to use as queries (0-based)")
    args = parser.parse_args()

    # Gather images
    image_dir = Path(args.image_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths = sorted(
        [str(p) for p in image_dir.iterdir() if p.suffix.lower() in exts]
    )

    if not image_paths:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images.")

    # Choose query images
    if args.query_indices:
        query_indices = [i for i in args.query_indices if 0 <= i < len(image_paths)]
    else:
        step = max(1, len(image_paths) // args.num_query)
        query_indices = list(range(0, len(image_paths), step))[: args.num_query]

    print(f"Query indices: {query_indices}")

    # Extract descriptors
    print(f"\nLoading ONNX model: {args.model}")
    t0 = time.time()
    all_descs = extract_descriptors(image_paths, args.model, args.batch_size)
    print(f"Descriptors extracted in {time.time() - t0:.1f}s  shape={all_descs.shape}")

    # Verify L2 norm
    norms = np.linalg.norm(all_descs, axis=1)
    print(f"Descriptor L2 norms: min={norms.min():.4f} max={norms.max():.4f} mean={norms.mean():.4f}")

    # Retrieve and visualize
    os.makedirs(args.output_dir, exist_ok=True)

    for qi in query_indices:
        print(f"\n--- Query {qi}: {Path(image_paths[qi]).name} ---")
        results = retrieve_topk(qi, all_descs, image_paths, args.top_k)

        # Print to console
        for rank, (idx, score, path) in enumerate(results):
            print(f"  #{rank+1}: {Path(path).name}  sim={score:.6f}")

        # Create visualization
        grid = create_grid(image_paths[qi], results, thumb_size=args.thumb_size)
        out_path = os.path.join(args.output_dir, f"query_{qi:04d}.jpg")
        grid.save(out_path, quality=90)
        print(f"  Saved: {out_path}")

    print(f"\nDone! Output in: {args.output_dir}/")


if __name__ == "__main__":
    main()
