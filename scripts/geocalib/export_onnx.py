#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

"""Export GeoCalib's perspective field network to a standalone ONNX file.

Exports the feed-forward submodule of GeoCalib (MSCAN backbone + low-level
encoder + PerspectiveDecoder for up and latitude fields and their confidences).
The subsequent Levenberg-Marquardt camera/gravity fitting is implemented in
C++ using Ceres (`src/colmap/calibration/perspective_field_fitting.cc`).

Usage:
    python scripts/geocalib/export_onnx.py \
        --weights distorted \
        --output geocalib_distorted.onnx
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class GeoCalibPerspectiveWrapper(nn.Module):
    """Wraps GeoCalib's feed-forward perspective field estimation network.

    Takes an RGB image tensor in [0, 1] of shape (1, 3, H, W) where H and W
    are divisible by 32, and returns:
      - up_field: (1, 2, H, W) unit 2D up-vectors
      - up_confidence: (1, 1, H, W) confidence in [0, 1]
      - latitude_field: (1, 1, H, W) latitude angle in radians in (-pi/2, pi/2)
      - latitude_confidence: (1, 1, H, W) confidence in [0, 1]
    """

    def __init__(self, geocalib_model: nn.Module):
        super().__init__()
        self.backbone = geocalib_model.backbone
        self.ll_enc = geocalib_model.ll_enc
        self.perspective_decoder = geocalib_model.perspective_decoder

        # Make NMF2D basis initialization deterministic so the exported ONNX
        # graph has no RandomUniform nodes and matches PyTorch exactly.
        for name, module in self.named_modules():
            if module.__class__.__name__ == "NMF2D":
                gen = torch.Generator(device="cpu")
                gen.manual_seed(0)
                bases = torch.rand(
                    (1, module.D, module.R), generator=gen, dtype=torch.float32
                )
                bases = F.normalize(bases, dim=1)
                module.register_buffer("_fixed_bases", bases, persistent=False)

                def _deterministic_build_bases(
                    B, S, D, R, device=None, _mod=module
                ):
                    return _mod._fixed_bases.expand(B * S, D, R).to(device)

                module._build_bases = _deterministic_build_bases
                logging.info(f"Registered deterministic NMF2D bases for {name}")
            elif module.__class__.__name__ == "LightHamHead":

                def _dynamic_light_ham_forward(features, _mod=module):
                    inputs = [features["hl"][i] for i in _mod.in_index]
                    inputs = [
                        inputs[0]
                        if i == 0
                        else F.interpolate(
                            level,
                            scale_factor=float(2**i),
                            mode="bilinear",
                            align_corners=_mod.align_corners,
                        )
                        for i, level in enumerate(inputs)
                    ]
                    inputs = torch.cat(inputs, dim=1)
                    x = _mod.squeeze(inputs)
                    x = _mod.hamburger(x)
                    feats = _mod.align(x)
                    feats = F.interpolate(
                        feats,
                        scale_factor=2.0,
                        mode="bilinear",
                        align_corners=False,
                    )
                    feats = _mod.out_conv(feats)
                    feats = F.interpolate(
                        feats,
                        scale_factor=2.0,
                        mode="bilinear",
                        align_corners=False,
                    )
                    feats = _mod.ll_fusion(feats, features["ll"].clone())
                    uncertainty = _mod.linear_pred_uncertainty(feats).squeeze(1)
                    return feats, uncertainty

                module.forward = _dynamic_light_ham_forward

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hl = self.backbone({"image": image})["features"]
        ll = self.ll_enc({"image": image})["features"]
        out = self.perspective_decoder({"features": {"hl": hl, "ll": ll}})
        return (
            out["up_field"],
            out["up_confidence"].unsqueeze(1),
            out["latitude_field"],
            out["latitude_confidence"].unsqueeze(1),
        )


def load_geocalib_model(
    weights: str, repo_dir: Path, weights_path: Path | None = None
) -> nn.Module:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from geocalib.geocalib import GeoCalib

    if weights_path is None:
        default_cache = (
            Path.home()
            / ".cache"
            / "torch"
            / "hub"
            / "geocalib"
            / f"{weights}.tar"
        )
        if default_cache.exists():
            weights_path = default_cache

    if weights_path is not None and weights_path.exists():
        logging.info(f"Loading GeoCalib checkpoint from {weights_path}")
        checkpoint = torch.load(
            weights_path, map_location="cpu", weights_only=False
        )
        model_conf = checkpoint.get("conf", {}).get("model", {})
        model = GeoCalib(**model_conf)
        state_dict = checkpoint["model"]
        model.load_state_dict(state_dict, strict=False)
    else:
        from geocalib.extractor import GeoCalib as GeoCalibExtractor

        logging.info(f"Downloading/loading GeoCalib ({weights})...")
        extractor = GeoCalibExtractor(weights=weights).to("cpu")
        model = extractor.model

    model.eval()
    return model


def export(
    weights: str,
    output_path: Path,
    repo_dir: Path,
    weights_path: Path | None = None,
    opset_version: int = 18,
) -> None:
    geocalib_model = load_geocalib_model(weights, repo_dir, weights_path)
    wrapper = GeoCalibPerspectiveWrapper(geocalib_model).eval()

    dummy_image = torch.rand(1, 3, 320, 480, dtype=torch.float32)

    logging.info(f"Exporting ONNX model to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_names = [
        "up_field",
        "up_confidence",
        "latitude_field",
        "latitude_confidence",
    ]
    torch.onnx.export(
        wrapper,
        (dummy_image,),
        str(output_path),
        input_names=["image"],
        output_names=output_names,
        dynamic_axes={
            "image": {2: "height", 3: "width"},
            "up_field": {2: "height", 3: "width"},
            "up_confidence": {2: "height", 3: "width"},
            "latitude_field": {2: "height", 3: "width"},
            "latitude_confidence": {2: "height", 3: "width"},
        },
        opset_version=opset_version,
        dynamo=False,
    )

    # Internalize any external data into a single self-contained ONNX file.
    onnx_model = onnx.load(str(output_path), load_external_data=True)
    data_file = Path(str(output_path) + ".data")
    if data_file.exists():
        onnx.save_model(
            onnx_model, str(output_path), save_as_external_data=False
        )
        data_file.unlink()
    onnx.checker.check_model(onnx_model)

    # Verify numerical parity against PyTorch across multiple shapes.
    sess = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    for h, w in [(320, 480), (320, 320), (416, 320), (224, 320)]:
        torch.manual_seed(42)
        test_img = torch.rand(1, 3, h, w, dtype=torch.float32)
        with torch.no_grad():
            pt_outs = wrapper(test_img)
        ort_outs = sess.run(None, {"image": test_img.numpy()})

        for name, pt_out, ort_out in zip(
            output_names, pt_outs, ort_outs, strict=True
        ):
            max_diff = float(np.max(np.abs(pt_out.numpy() - ort_out)))
            mean_diff = float(np.mean(np.abs(pt_out.numpy() - ort_out)))
            logging.info(
                f"Shape ({h}, {w}) [{name}]: max_diff={max_diff:.2e}, "
                f"mean_diff={mean_diff:.2e}"
            )
            assert max_diff < 1e-4, (
                f"ONNX output mismatch for {name} at ({h}, {w}): "
                f"max_diff={max_diff}"
            )

    sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logging.info(
        f"Successfully exported {output_path} ({size_mb:.1f} MB, "
        f"SHA256: {sha256})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export GeoCalib perspective field network to ONNX."
    )
    parser.add_argument(
        "--weights",
        default="distorted",
        choices=["pinhole", "distorted"],
        help="GeoCalib pretrained weight variant.",
    )
    parser.add_argument(
        "--weights_path",
        type=Path,
        default=None,
        help="Optional explicit path to the .tar checkpoint.",
    )
    parser.add_argument(
        "--repo_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "GeoCalib",
        help="Path to the cloned GeoCalib repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .onnx file path.",
    )
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    output_path = args.output or Path(f"geocalib_{args.weights}.onnx")
    export(
        args.weights,
        output_path,
        args.repo_dir,
        args.weights_path,
        args.opset,
    )


if __name__ == "__main__":
    main()
