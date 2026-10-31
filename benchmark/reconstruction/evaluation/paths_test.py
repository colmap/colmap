# SPDX-License-Identifier: BSD-3-Clause

import sys
from pathlib import Path

import pytest

from .utils import colmap_reconstruction, parse_args


@pytest.mark.skipif(
    sys.platform == "win32", reason="uses a POSIX executable script"
)
@pytest.mark.parametrize("relative_data", [False, True])
@pytest.mark.parametrize("relative_run", [False, True])
def test_benchmark_paths_remain_valid_in_reconstruction_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_data, relative_run
):
    monkeypatch.chdir(tmp_path)
    image_dir = tmp_path / "data" / "scene" / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "image.png").touch()
    # This executable validates the same filesystem arguments as the COLMAP
    # CLI. Running it as a real child catches changes to the working directory.
    executable = tmp_path / "check_colmap_paths"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "images = Path(args[args.index('--image_path') + 1])\n"
        "workspace = Path(args[args.index('--workspace_path') + 1])\n"
        "assert (images / 'image.png').is_file(), images\n"
        "assert workspace.resolve() == Path.cwd(), workspace\n"
        "print('paths resolve in child')\n"
    )
    executable.chmod(0o755)
    data_path = Path("data") if relative_data else tmp_path / "data"
    run_path = Path("runs") if relative_run else tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--colmap_path",
            str(executable),
            "--data_path",
            str(data_path),
            "--run_path",
            str(run_path),
            "--use_cpu",
        ],
    )
    args = parse_args()
    workspace = args.run_path / "scene"
    colmap_reconstruction(args, workspace, args.data_path / "scene" / "images")
    assert args.data_path == tmp_path / "data"
    assert args.run_path == tmp_path / "runs"
    for phase in ["extraction", "matching", "reconstruction"]:
        assert (
            workspace / f"{phase}.log"
        ).read_text() == "paths resolve in child\n"
