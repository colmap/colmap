# How to build COLMAP using Docker

## Requirements

- Host machine with at least one NVIDIA GPU/CUDA support and installed drivers
  (to support dense reconstruction).
- Docker (for CUDA support 19.03+).

## Quick Start

1. Check that Docker >=19.03 installed on your host machine:

    ```
    docker --version
    ```

2. Check that you have an NVIDIA driver installed on your host machine:

    ```
    nvidia-smi
    ```

3. Setup the nvidia-toolkit on your host machine:

    For Ubuntu host machines: `./setup-ubuntu.sh`
    For CentOS host machines: `./setup-centos.sh`

4. Run the *run* script, using the *full local path* to your preferred local
   working directory (a folder with your input files/images, etc.):

    ```
    ./run.sh /path/where/your/working/folder/is
    ```

    This will put you in a directory (inside the Docker container) mounted to
    the local path you specified. Now you can run COLMAP binaries on your own
    inputs like this:

    ```
    colmap automatic_reconstructor --image_path ./images --workspace_path .
    ```

## Build from Scratch

After completing steps 1-3, you can alternatively build the docker image from
scratch based on the **Dockerfile** (e.g., with your own modifications) using:

```
./build.sh
./run.sh /path/where/your/working/folder/is
```

## Troubleshooting

Install an NVIDIA driver and NVIDIA container runtime:

```
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
```

If you failed to install the above, check the appropriate NVIDIA driver by yourself and install it:

```
ubuntu-drivers devices
e.g.
sudo apt install nvidia-driver-455
```
