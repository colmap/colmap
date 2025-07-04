#!/bin/bash

echo "🚀 Starting intelligent NVIDIA Docker setup..."

echo "📦 Updating NVIDIA driver to latest..."
sudo apt update
sudo ubuntu-drivers autoinstall
echo "🔄 Rebooting required after driver update. Run this script again after reboot."
# Check if reboot is needed
if [ -f /var/run/reboot-required ]; then
    echo "⚠️  System reboot required. Please reboot and run this script again."
    echo "After reboot, run: sudo reboot && ./setup-ubuntu.sh"
    exit 0
fi

echo "🔍 Detecting NVIDIA driver version..."
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
echo "Found NVIDIA driver: $DRIVER_VERSION"

echo "🔍 Determine compatible CUDA version based on driver..." 
get_compatible_cuda_version() {
    local driver_ver=$1
    
    # Extract major version (e.g., 560 from 560.35.03)
    local major_ver=$(echo $driver_ver | cut -d'.' -f1)
    
    # NVIDIA Driver-CUDA Compatibility Matrix
    if [ $major_ver -ge 565 ]; then
        echo "12.9"  # Driver 565+ supports CUDA 12.9
    elif [ $major_ver -ge 560 ]; then
        echo "12.6"  # Driver 560+ supports CUDA 12.6 (your current case)
    elif [ $major_ver -ge 555 ]; then
        echo "12.5"  # Driver 555+ supports CUDA 12.5
    elif [ $major_ver -ge 550 ]; then
        echo "12.4"  # Driver 550+ supports CUDA 12.4
    elif [ $major_ver -ge 535 ]; then
        echo "12.2"  # Driver 535+ supports CUDA 12.2
    elif [ $major_ver -ge 525 ]; then
        echo "12.0"  # Driver 525+ supports CUDA 12.0
    else
        echo "11.8"  # Fallback to CUDA 11.8
    fi
}
COMPATIBLE_CUDA=$(get_compatible_cuda_version $DRIVER_VERSION)
echo "✅ Compatible CUDA version: $COMPATIBLE_CUDA"

echo "📦 Installing latest nvidia-container-toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

echo "📦 Configure Docker"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "🔍 Finding latest patch version for CUDA $COMPATIBLE_CUDA..."
AVAILABLE_VERSIONS=$(curl -s "https://registry.hub.docker.com/v2/repositories/nvidia/cuda/tags/?page_size=100" | jq -r '.results[].name' | grep -E "^${COMPATIBLE_CUDA}\.[0-9]+-base-ubuntu24\.04$" | head -1)
if [ -n "$AVAILABLE_VERSIONS" ]; then
    # Extract full version (e.g., "12.9.1" from "12.9.1-base-ubuntu24.04")
    FULL_CUDA_VERSION=$(echo "$AVAILABLE_VERSIONS" | cut -d'-' -f1)
    echo "✅ Found CUDA version: $FULL_CUDA_VERSION"
else
    echo "❌ No CUDA $COMPATIBLE_CUDA images found for Ubuntu 24.04, trying Ubuntu 22.04..."
    AVAILABLE_VERSIONS=$(curl -s "https://registry.hub.docker.com/v2/repositories/nvidia/cuda/tags/?page_size=100" | jq -r '.results[].name' | grep -E "^${COMPATIBLE_CUDA}\.[0-9]+-base-ubuntu22\.04$" | head -1)
    
    if [ -n "$AVAILABLE_VERSIONS" ]; then
        FULL_CUDA_VERSION=$(echo "$AVAILABLE_VERSIONS" | cut -d'-' -f1)
        UBUNTU_VERSION="22.04"
        echo "✅ Found CUDA version: $FULL_CUDA_VERSION for Ubuntu 22.04"
    else
        echo "❌ No compatible CUDA images found"
        exit 1
    fi
fi

echo "🧪 Testing with automatically detected compatible CUDA version: $COMPATIBLE_CUDA..."
if docker run --rm --runtime=nvidia nvidia/cuda:${FULL_CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION:-24.04} nvidia-smi; then
    echo "✅ GPU support working with CUDA $FULL_CUDA_VERSION!"
    # Update Dockerfile with compatible version
    if [ -f "Dockerfile" ]; then
        sed -i "s/ARG NVIDIA_CUDA_VERSION=.*/ARG NVIDIA_CUDA_VERSION=${FULL_CUDA_VERSION}/" Dockerfile
        if [ "${UBUNTU_VERSION:-24.04}" = "22.04" ]; then
            sed -i "s/ARG UBUNTU_VERSION=.*/ARG UBUNTU_VERSION=22.04/" Dockerfile
        fi
        echo "✅ Updated Dockerfile to use CUDA $FULL_CUDA_VERSION"
    fi
else
    echo "❌ GPU test failed with CUDA $FULL_CUDA_VERSION"
fi