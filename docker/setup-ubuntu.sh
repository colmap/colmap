# Add the package repositories
# ref: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install nvidia-container-toolkit
sudo apt-get update 
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# Configure the Docker daemon to use nvidia runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify NVIDIA drivers are accessible
echo "Checking NVIDIA drivers..."
if nvidia-smi; then
    echo "✅ NVIDIA drivers working"
else
    echo "❌ NVIDIA drivers not found - install NVIDIA drivers first"
    exit 1
fi

# Configure Docker daemon to use nvidia as default runtime
echo "Configuring Docker daemon for NVIDIA runtime..."
sudo mkdir -p /etc/docker
echo '{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}' | sudo tee /etc/docker/daemon.json

sudo systemctl restart docker

# Check that it worked!
echo "Testing GPU support in Docker container..."
if docker run --rm --runtime=nvidia nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04 nvidia-smi; then
    echo "✅ GPU support successfully configured with --runtime=nvidia!"
elif docker run --rm nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi; then
    echo "✅ GPU support working with default nvidia runtime!"
elif docker run --rm --runtime=nvidia nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi; then
    echo "✅ GPU support working with CUDA 11.8!"
else
    echo "❌ GPU support test failed"
    echo "Debug information:"
    echo "Docker daemon config:"
    cat /etc/docker/daemon.json
    echo ""
    echo "Docker runtime info:"
    docker info | grep -A 5 -i runtime
    echo ""
    echo "Try manual test with: docker run --rm --runtime=nvidia nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi"
fi