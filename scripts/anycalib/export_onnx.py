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

The script exports two fixed-shape models::

    landscape image : (1, 3, 266, 392), RGB in [0, 1]
    portrait image  : (1, 3, 392, 266), RGB in [0, 1]
      -> rays           : (1, H*W, 3) unit camera-ray vectors (x right,
                           y down, z forward), matching ``AnyCalib.forward``.
      -> tangent_coords : (1, H*W, 2) FoV field in the tangent plane at
                           (0, 0, 1).

ImageNet normalization is part of the exported graph (DINOv2 wrapper buffers),
so callers must feed plain RGB in [0, 1], exactly like the PyTorch model.

Both input shapes are static for execution-provider compatibility. Their
dimensions are divisible by the DINOv2 patch size (14) and are close to the
102400-pixel AnyCalib training resolution. COLMAP rotates an image upright
from its gravity prior, then chooses the closer aspect ratio and center-crops
to that model's aspect ratio.

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
        "--output",
        type=Path,
        required=True,
        help=(
            "Output .onnx base path. The script appends _landscape and "
            "_portrait to its stem."
        ),
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

    import onnx
    import onnxruntime as ort
    from onnx import TensorProto
    from PIL import Image

    def prepare(im: Image.Image, height: int, width: int) -> np.ndarray:
        im = im.convert("RGB")
        if im.height < height or im.width < width:
            scale = max(height / im.height, width / im.width)
            im = im.resize(
                (int(im.width * scale), int(im.height * scale)),
                Image.Resampling.BICUBIC,
            )
        target_aspect = width / height
        image_aspect = im.width / im.height
        if image_aspect > target_aspect:
            crop_width = int(im.height * target_aspect + 0.5)
            left = (im.width - crop_width) // 2
            im = im.crop((left, 0, left + crop_width, im.height))
        else:
            crop_height = int(im.width / target_aspect + 0.5)
            top = (im.height - crop_height) // 2
            im = im.crop((0, top, im.width, top + crop_height))
        im = im.resize((width, height), Image.Resampling.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)[None]

    sizes = {"landscape": (266, 392), "portrait": (392, 266)}
    for orientation, (height, width) in sizes.items():
        output = args.output.with_name(
            f"{args.output.stem}_{orientation}{args.output.suffix}"
        )
        dummy = torch.rand(1, 3, height, width)
        print(f"Exporting to {output} (opset {args.opset}) ...")
        torch.onnx.export(
            module,
            (dummy,),
            output,
            input_names=["image"],
            output_names=["rays", "tangent_coords"],
            opset_version=args.opset,
            dynamo=True,
        )

        # dynamo export stores weights in a ``.onnx.data`` sidecar; merge them
        # into a single self-contained file, the format COLMAP loads.
        onnx_model = onnx.load(output)
        for init in onnx_model.graph.initializer:
            if (
                init.HasField("data_location")
                and init.data_location == TensorProto.EXTERNAL
            ):
                init.ClearField("external_data")
                init.data_location = TensorProto.DEFAULT
        onnx.save_model(onnx_model, output)
        sidecar = output.with_name(output.name + ".data")
        if sidecar.exists():
            sidecar.unlink()
        onnx.checker.check_model(onnx.load(output, load_external_data=False))

        session = ort.InferenceSession(
            output, providers=["CPUExecutionProvider"]
        )
        assert [i.name for i in session.get_inputs()] == ["image"]
        assert [o.name for o in session.get_outputs()] == [
            "rays",
            "tangent_coords",
        ]
        test_inputs = {f"dummy-{width}x{height}": dummy.numpy()}
        if args.test_image is not None:
            test_inputs[f"image-{width}x{height}"] = prepare(
                Image.open(args.test_image), height, width
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
        print(f"Checked: {output}")

    print("Parity checks passed.")


if __name__ == "__main__":
    main()
