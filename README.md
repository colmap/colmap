PYCOLMAP Wheels Built with CUDA (Linux only)
======

About
-----
This is a fork of COLMAP for building CUDA-enabled Python pycolmap wheels(currently only for Linux).

**[See the original repository for more details about Colmap!](https://github.com/colmap/colmap)**

This fork is currently experimental and unrelated from the original repository.
See this issue in the original COLMAP repository about plans for CUDA support in the official PYCOLMAP PyPi wheel.


## Building Python Wheels Locally
### Prerequisites
- a working docker installation since the build uses cibuildwheel which runs inside a container 
- you need to have uv installed ([see uv installation guide](https://docs.astral.sh/uv/getting-started/installation/))

### Building Wheels
- just execute `./local_cibuildwheel.sh`
- this sets up the environment and then calls cibuildwheel using which then builds the wheels inside a CUDA-enabled manylinux container (container image taken from [this repository](https://github.com/ameli/manylinux-cuda))
- the build might take several minutes up to about an hour
- **Once done, there should be a `wheelhouse` directory inside the repository's root folder which contains the built wheels**