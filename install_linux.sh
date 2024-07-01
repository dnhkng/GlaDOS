#!/bin/bash

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Installing espeak-ng..."
if command_exists apt; then
    sudo add-apt-repository ppa:deadsnakes/ppa || { echo "Failed to add apt repository"; exit 1; }
    sudo apt install python3.12 python3.12-venv || { echo "Failed to install python3.12 with apt"; exit 1; }
    sudo apt-get update || { echo "Failed to update package list"; exit 1; }
    sudo apt-get install -y espeak-ng || { echo "Failed to install espeak-ng with apt"; exit 1; }
elif command_exists pacman; then
    sudo pacman -Syu --noconfirm || { echo "Failed to update package list"; exit 1; }
    sudo pacman -S --noconfirm espeak-ng || { echo "Failed to install espeak-ng with pacman"; exit 1; }
elif command_exists dnf; then
    sudo dnf check-update || { echo "Failed to check for updates"; exit 1; }
    sudo dnf install -y espeak-ng || { echo "Failed to install espeak-ng with dnf"; exit 1; }
else
    echo "No compatible package manager found (apt-get, pacman, dnf)"
    exit 1
fi

python3.12 -m venv venv || { echo "Failed to create virtual environment"; exit 1; }
source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
pip install -r requirements_cuda.txt || { echo "Failed to install Python dependencies"; exit 1; }

echo "Downloading Llama..."
curl -L "https://github.com/ggerganov/llama.cpp/releases/download/b3266/cudart-llama-bin-linux-cu12.2.0-x64.tar.gz" --output "cudart-llama-bin-linux-cu12.2.0-x64.tar.gz" || { echo "Failed to download cudart-llama"; exit 1; }
curl -L "https://github.com/ggerganov/llama.cpp/releases/download/b3266/llama-b3266-bin-linux-cuda-cu12.2.0-x64.tar.gz" --output "llama-bin-linux-cuda-cu12.2.0-x64.tar.gz" || { echo "Failed to download llama"; exit 1; }
echo "Unzipping Llama..."
tar -xzf cudart-llama-bin-linux-cu12.2.0-x64.tar.gz -C submodules/llama.cpp || { echo "Failed to extract cudart-llama"; exit 1; }
tar -xzf llama-bin-linux-cuda-cu12.2.0-x64.tar.gz -C submodules/llama.cpp || { echo "Failed to extract llama"; exit 1; }

echo "Downloading Whisper..."
curl -L "https://github.com/ggerganov/whisper.cpp/releases/download/v1.6.0/whisper-cublas-12.2.0-bin-x64.tar.gz" --output "whisper-cublas-12.2.0-bin-x64.tar.gz" || { echo "Failed to download whisper"; exit 1; }
echo "Unzipping Whisper..."
tar -xzf whisper-cublas-12.2.0-bin-x64.tar.gz -C submodules/whisper.cpp || { echo "Failed to extract whisper"; exit 1; }

echo "Cleaning up..."
rm whisper-cublas-12.2.0-bin-x64.tar.gz || { echo "Failed to remove whisper tarball"; exit 1; }
rm cudart-llama-bin-linux-cu12.2.0-x64.tar.gz || { echo "Failed to remove cudart-llama tarball"; exit 1; }
rm llama-bin-linux-cuda-cu12.2.0-x64.tar.gz || { echo "Failed to remove llama tarball"; exit 1; }

echo "Downloading Models..."
mkdir -p models || { echo "Failed to create models directory"; exit 1; }
curl -L "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin" --output "models/ggml-medium-32-2.en.bin" || { echo "Failed to download ggml-medium-32-2.en.bin"; exit 1; }
curl -L "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf?download=true" --output "models/Meta-Llama-3-8B-Instruct-Q6_K.gguf" || { echo "Failed to download Meta-Llama-3-8B-Instruct-Q6_K.gguf"; exit 1; }

echo "Done!"
