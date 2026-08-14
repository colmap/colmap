import re
from pathlib import Path

import pycolmap

CAMERA_MODELS = Path(__file__).parents[1] / "viewer_src" / "camera_models.ts"


def parse_array(source: str, name: str) -> list[str]:
    match = re.search(
        rf"export const {name} = \[(.*?)\] as const;", source, re.DOTALL
    )
    if match is None:
        raise AssertionError(f"Could not find {name} in {CAMERA_MODELS}")
    return [
        value.strip().strip('"')
        for value in match.group(1).split(",")
        if value.strip()
    ]


def main() -> None:
    source = CAMERA_MODELS.read_text(encoding="utf-8")
    names = parse_array(source, "CAMERA_MODEL_NAMES")
    param_counts = [
        int(value) for value in parse_array(source, "CAMERA_MODEL_PARAM_COUNTS")
    ]
    expected = list(zip(names, param_counts, strict=True))

    actual: list[tuple[str, int] | None] = [None] * len(expected)
    for name, model in pycolmap.CameraModelId.__members__.items():
        if name == "INVALID":
            continue
        camera = pycolmap.Camera.create_from_model_id(1, model, 1.0, 100, 100)
        model_id = int(model.value)
        if model_id >= len(actual):
            raise AssertionError(
                f"Viewer is missing pycolmap camera model {name} ({model_id})"
            )
        actual[model_id] = (name, len(camera.params))

    message = (
        "Viewer camera model registry differs from pycolmap:\n"
        f"viewer={expected}\n"
        f"pycolmap={actual}"
    )
    assert actual == expected, message


if __name__ == "__main__":
    main()
