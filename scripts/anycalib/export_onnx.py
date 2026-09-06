#!/usr/bin/env python3
"""Export the AnyCalib feed-forward network to ONNX.

AnyCalib (https://github.com/javrtg/AnyCalib, Apache-2.0) calibrates a camera
from a single image in two stages:

  1. A feed-forward network (DINOv2 ViT-L/14 backbone + light DPT decoder +
     convex tangent head) regresses dense per-pixel rays and FoV (tangent)
     coordinates.
  2. Closed-form fitting + Gauss-Newton refinement fits intrinsics for a
     chosen camera model to those rays.

This script exports stage 1 to ONNX. Stage 2 is ported to C++ in
``src/colmap/calibration`` and consumes the exported model's outputs.

The exported model maps::

    image : (1, 3, 322, 322), RGB in [0, 1]
      -> rays           : (1, 322*322, 3) unit camera-ray vectors (x right,
                           y down, z forward), matching ``AnyCalib.forward``.
      -> tangent_coords : (1, 322*322, 2) FoV field in the tangent plane at
                           (0, 0, 1).

ImageNet normalization is part of the exported graph (DINOv2 wrapper buffers),
so callers must feed plain RGB in [0, 1], exactly like the PyTorch model.

The input size is fixed to 322x322 (the square training resolution
``round(sqrt(102400) / 14) * 14``). Upstream AnyCalib supports aspect ratios in
[0.5, 2] via dynamic resampling of the DINOv2 positional embeddings, but that
resampling bakes the input size into the graph under both the legacy tracer
and torch.export (the ``float(w0 + 0.1)`` scale factors specialize to
constants; forcing the explicit-size branch deviates from upstream by ~8e-3).
A fixed square input keeps the graph static -- maximizing execution-provider
compatibility -- at the cost of center-cropping non-square images to square in
preprocessing, mirroring ``AnyCalib.set_im_size`` with target AR 1. Square
inputs are in-distribution for the network. If full aspect-ratio support is
needed later, the fallback is a two-input export taking a precomputed,
bicubic-resampled positional embedding alongside the image.

Requirements: torch (CPU is fine, *without* xformers so attention traces to
plain matmul+softmax), onnx, onnxruntime, pillow, numpy. Example::

    uv venv && uv pip install torch onnx onnxruntime pillow numpy
    python scripts/anycalib/export_onnx.py --model_id anycalib_gen \
        --anycalib_dir /path/to/AnyCalib --output anycalib_gen.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.hub


class AnyCalibFeedForward(torch.nn.Module):
    """Backbone + decoder + head of AnyCalib, without the calibrator.

    Replicates ``AnyCalib.forward`` up to (but excluding) the
    ``self.calibrator(out, data)`` call, with ``data["image"]`` as the single
    input. Output layout matches the tensors consumed by
    ``Calibrator.__call__``.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.decoder = model.decoder
        self.head = model.head

    def forward(self, image: torch.Tensor):
        out = self.backbone(image)
        out = self.head(self.decoder(out))
        b, _, h, w = image.shape
        rays = out["rays"].permute(0, 2, 3, 1).reshape(b, h * w, 3)
        tangent_coords = (
            out["tangent_coords"].permute(0, 2, 3, 1).reshape(b, h * w, 2)
        )
        return rays, tangent_coords


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_id",
        default="anycalib_gen",
        choices=[
            "anycalib_pinhole",
            "anycalib_gen",
            "anycalib_dist",
            "anycalib_edit",
        ],
        help="AnyCalib weights to export.",
    )
    parser.add_argument(
        "--anycalib_dir",
        type=Path,
        required=True,
        help="Path to a checkout of https://github.com/javrtg/AnyCalib.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional local .pt weights file (else downloaded from GitHub).",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output .onnx path."
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version."
    )
    parser.add_argument(
        "--test_image",
        type=Path,
        default=None,
        help="Optional image for the ONNX-vs-torch parity check.",
    )
    parser.add_argument(
        "--parity_tol",
        type=float,
        default=1e-4,
        help="Max abs diff allowed between torch and ONNX outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.anycalib_dir))
    from anycalib.model.anycalib_pretrained import AnyCalib  # noqa: E402

    try:
        import xformers  # noqa: F401
    except ImportError:
        pass
    else:
        raise RuntimeError(
            "xformers is installed: AnyCalib would trace memory-efficient "
            "attention, which is not ONNX-exportable. Uninstall xformers so "
            "the model falls back to the standard attention path."
        )

    print(f"Loading {args.model_id} ...")
    model = AnyCalib(model_id=None)
    if args.weights is not None:
        state_dict = torch.load(args.weights, map_location="cpu")
    else:
        url = (
            "https://github.com/javrtg/AnyCalib/releases/download/"
            f"v1.0.0/{args.model_id}.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            url,
            f"{torch.hub.get_dir()}/anycalib",
            map_location="cpu",
            file_name=f"{args.model_id}.pt",
        )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    module = AnyCalibFeedForward(model).eval()

    # 322 = round(sqrt(102400) / 14) * 14: square training resolution.
    dummy = torch.rand(1, 3, 322, 322)
    with torch.inference_mode():
        ref_rays, ref_tangent = module(dummy)

    _ = ref_rays, ref_tangent  # recomputed in the parity check below

    print(f"Exporting to {args.output} (opset {args.opset}) ...")
    torch.onnx.export(
        module,
        (dummy,),
        args.output,
        input_names=["image"],
        output_names=["rays", "tangent_coords"],
        opset_version=args.opset,
        dynamo=True,
    )

    import onnx
    from onnx import TensorProto

    # dynamo export stores weights in a ``.onnx.data`` sidecar; merge them into
    # a single self-contained file, the format COLMAP loads.
    model = onnx.load(args.output)
    for init in model.graph.initializer:
        if (
            init.HasField("data_location")
            and init.data_location == TensorProto.EXTERNAL
        ):
            init.ClearField("external_data")
            init.data_location = TensorProto.DEFAULT
    onnx.save_model(model, args.output)
    # onnx appends ".data" to the full file name (model.onnx.data).
    sidecar = args.output.with_name(args.output.name + ".data")
    if sidecar.exists():
        sidecar.unlink()
    onnx.checker.check_model(onnx.load(args.output, load_external_data=False))
    print("ONNX checker passed (single file).")

    # Parity check: ONNX Runtime vs torch on dummy input and test image sizes.
    import onnxruntime as ort
    from PIL import Image

    session = ort.InferenceSession(
        args.output, providers=["CPUExecutionProvider"]
    )
    assert [i.name for i in session.get_inputs()] == ["image"]
    assert [o.name for o in session.get_outputs()] == [
        "rays",
        "tangent_coords",
    ]

    def prepare(im: Image.Image, size: tuple[int, int]) -> np.ndarray:
        arr = np.array(im.convert("RGB")).astype(np.float32) / 255.0
        if arr.shape[:2] != size:
            tmp = Image.fromarray((arr * 255).astype(np.uint8))
            arr = (
                np.array(tmp.resize((size[1], size[0]), Image.BICUBIC)).astype(
                    np.float32
                )
                / 255.0
            )
        return arr.transpose(2, 0, 1)[None].astype(np.float32)

    test_inputs = {"dummy-322x322": dummy.numpy()}
    if args.test_image is not None:
        # Center-crop to square, mirroring the COLMAP preprocessing, so the
        # parity check covers a realistic input.
        im = Image.open(args.test_image).convert("RGB")
        side = min(im.size)
        left = (im.size[0] - side) // 2
        top = (im.size[1] - side) // 2
        test_inputs["img-322x322"] = prepare(
            im.crop((left, top, left + side, top + side)), (322, 322)
        )

    with torch.inference_mode():
        for name, inp in test_inputs.items():
            torch_rays, torch_tangent = module(torch.from_numpy(inp))
            ort_rays, ort_tangent = session.run(None, {"image": inp})
            rays_diff = float(np.abs(torch_rays.numpy() - ort_rays).max())
            tangent_diff = float(
                np.abs(torch_tangent.numpy() - ort_tangent).max()
            )
            print(
                f"{name}: max|torch - onnx| rays={rays_diff:.3e} "
                f"tangent={tangent_diff:.3e}"
            )
            if max(rays_diff, tangent_diff) > args.parity_tol:
                raise RuntimeError(f"Parity check failed for {name}")

    print("Parity check passed.")
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
