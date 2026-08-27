# COLMAP Fork — Agent Instructions

This is the Medida COLMAP fork. Production and `medida-3d` consume both the **`colmap` CLI** and **`pycolmap`** from this tree.

For rebuild commands, one-time GPU setup, validation, and rollout, read:

**`~/medida-3d/.agents/skills/medida-colmap-fork/SKILL.md`**

Quick rebuild reference:

| Change type | Rebuild |
|-------------|---------|
| C++ in fork | `ninja && ninja install` in `~/colmap/build` |
| pycolmap bindings | `uv pip install --reinstall ~/colmap/pycolmap` from `~/medida-3d` |
| Both | CLI first, then pycolmap |

Branch **`release/v1.1.1`** matches production (`pipeline_runner_base.Dockerfile`).

Do not use `install_dev.sh` on GPU hosts (CPU-only). Do not run `uv sync` inside `pycolmap/`.
