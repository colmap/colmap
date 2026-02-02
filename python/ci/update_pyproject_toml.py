import argparse
from pathlib import Path

import tomlkit

# Set up command-line argument parser
parser = argparse.ArgumentParser(
    description="Modify pyproject.toml for a custom build."
)
parser.add_argument(
    "--name", required=True, help="The new package name for the wheel."
)
parser.add_argument(
    "--add-deps",
    nargs="+",
    default=[],
    help="Space-separated list of Python dependencies to add.",
)
args = parser.parse_args()

# Modify the pyproject.toml file
pyproject_path = Path("pyproject.toml")
if not pyproject_path.exists():
    raise FileNotFoundError(pyproject_path)

with open(pyproject_path, encoding="utf-8") as f:
    config = tomlkit.load(f)

config["project"]["name"] = args.name

if args.add_deps:
    if "dependencies" not in config["project"]:
        config["project"]["dependencies"] = []

    existing_deps = config["project"]["dependencies"]
    for req in args.add_deps:
        if req not in existing_deps:
            existing_deps.append(req)

with open(pyproject_path, "w", encoding="utf-8") as f:
    tomlkit.dump(config, f)
