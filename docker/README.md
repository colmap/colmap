
# Building & running commandline COLMAP using Docker

These instructions are for compiling the libraries and binaries only; *not* the interactive GUI version. This assumes you have access to an ubuntu or centos machine with NVIDIA GPU hardware. 

## Quick start:

1. Make sure Docker 19.03[^1] or higher is installed on the host machine (perhaps your physical laptop or an AWS instance). Check by running:
	`docker --version`

2. Make sure you have an NVIDIA driver installed on your host machine. You can check by running:
	`nvidia-smi`

3. Setup the nvidia-toolkit on your host[^2]:

	For ubuntu host machines: `./setup-ubuntu.sh`
	For centos host machines: `./setup-centos.sh`

4. Run the 'QUICK_START' script, using the *full local path* to your prefered local working directory (perhaps a folder with your input files/images, etc):

	`./QUICK_START.sh /path/where/your/working/folder/is`

This will put you in a directory (inside the Docker container) mounted to the local path you specified so that you can run COLMAP binaries[^3] on your own inputs. Enjoy!

## Build from scratch:

Afer completing steps 1-3, you can alternatively build the docker image from scratch based on the **Dockerfile** (perhaps with your own changes / modifications) using:

	`./buildFromScratch.sh /path/where/your/working/folder/is`

## NOTES

[^1]: This is because Docker 19.03+ natively supports NVIDIA GPUs.

[^2]: You should get a similar output to what you get when you ran step 2 on your host, since the docker container is detecting the same GPU(s). If you have trouble, you may want to read the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) webpage, as the scripts `./setup-ubuntu.sh` and `./setup-centos.sh` are based on instructions posted there.

[^3]: Right now this workflow is pinned to build from [COLMAP 3.6-dev.3](https://github.com/tjdahlke/colmap/releases/tag/3.6-dev.3.). To build from a different release, or build from the latest commit on master, open up the Dockerfile and comment/modify as needed.

+ COLMAP needs NVIDA GPU compute hardware for dense reconstruction (as of 12/10/2019), and is optional for feature extraction and matching.

+ Running COLMAP binaries can use a lot of memory (depending on the size of your data set / imagery). Docker has a relatively small default memory setting (2Gb on Mac). You will probably want to increase this before you run any larger workflows. From Docker desktop on Mac for example, just open the Docker GUI, go to the *Advanced* tab and increase via the slider:

![alt text][dockerParam]

[dockerParam]: https://i.stack.imgur.com/6iWiW.png "Recommend increasing memory to >4Gb"


<!-- nvcc --version -->
<!-- lspci -->
