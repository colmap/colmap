# How to build COLMAP using Docker

## Requirements

- Host machine with at least one NVIDIA GPU/CUDA support and installed drivers
  (to support dense reconstruction).
- Docker (for CUDA support 19.03+).

## Quick Start

1. Check that Docker >=19.03 installed on your host machine:

    ```bash
    docker --version
    ```

2. Setup the NVIDIA driver and nvidia-toolkit on your host machine:

    For Ubuntu host machines: `./setup-ubuntu.sh`
    For CentOS host machines: `./setup-centos.sh`

3. Run the *run* script, using the *full local path* to your preferred local
   working directory (a folder with your input files/images, etc.):

    ```bash
    ./run.sh /path/where/your/working/folder/is
    ```

    This will put you in a directory (inside the Docker container) mounted to
    the local path you specified. Now you can run COLMAP binaries on your own
    inputs like this:

    ```bash
    colmap automatic_reconstructor --image_path ./images --workspace_path .
    ```

    Alternatively, you can run the *run-gui* script, which will start the graphical user interface of COLMAP:

    ```bash
    ./run-gui.sh /path/where/your/working/folder/is
    ```

## Build from Scratch

After completing steps 1-2, you can build the Docker image from scratch using the **Dockerfile**.
First, update the CUDA and Ubuntu versions in Dockerfile lines 1-2 to match your system, then:

```bash
./build.sh
./run.sh /path/where/your/working/folder/is
```
