"""Run Structure-from-Motion on 360-degree panorama images."""

import argparse
from pathlib import Path

from pycolmap.panorama import (
    Mapper,
    Matcher,
    PanoramaReconstructionOptions,
    PanoRenderType,
    reconstruct,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_image_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--matcher",
        type=Matcher,
        default=Matcher.SEQUENTIAL,
        choices=list(Matcher),
    )
    parser.add_argument(
        "--mapper",
        type=Mapper,
        default=Mapper.INCREMENTAL,
        choices=list(Mapper),
    )
    parser.add_argument(
        "--pano_render_type",
        type=PanoRenderType,
        default=PanoRenderType.PERSPECTIVE_OVERLAPPING,
        choices=list(PanoRenderType),
    )
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--gpu_index", default="-1")
    parser.add_argument("--use_gpu", default=True, action="store_true")
    parser.add_argument("--use_cpu", dest="use_gpu", action="store_false")
    args = parser.parse_args()

    reconstruct(
        args.input_image_path,
        args.output_path,
        PanoramaReconstructionOptions(
            matcher=args.matcher,
            mapper=args.mapper,
            render_type=args.pano_render_type,
            random_seed=args.random_seed,
            num_threads=args.num_threads,
            gpu_index=args.gpu_index,
            use_gpu=args.use_gpu,
        ),
    )


if __name__ == "__main__":
    main()
