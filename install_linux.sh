#!/bin/bash
echo "Install espeak-ng..."
curl -L "https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-1.51.tar.gz" --output "espeak-ng-1.51.tar.gz"
tar -xzf espeak-ng-1.51.tar.gz
cd espeak-ng-1.51 || exit
./configure
make
sudo make install
cd ..
rm -rf espeak-ng-1.51
rm espeak-ng-1.51.tar.gz

python3.12 -m venv venv || { echo "Failed to create virtual environment"; exit 1; }
source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
pip install -r requirements_cuda.txt || { echo "Failed to install Python dependencies"; exit 1; }

echo "Downloading Llama..."
curl -L "https://github.com/ggerganov/llama.cpp/releases/download/b3266/cudart-llama-bin-linux-cu12.2.0-x64.tar.gz" --output "cudart-llama-bin-linux-cu12.2.0-x64.tar.gz"
curl -L "https://github.com/ggerganov/llama.cpp/releases/download/b3266/llama-b3266-bin-linux-cuda-cu12.2.0-x64.tar.gz" --output "llama-bin-linux-cuda-cu12.2.0-x64.tar.gz"
echo "Unzipping Llama..."
tar -xzf cudart-llama-bin-linux-cu12.2.0-x64.tar.gz -C submodules/llama.cpp
tar -xzf llama-bin-linux-cuda-cu12.2.0-x64.tar.gz -C submodules/llama.cpp

echo "Downloading Whisper..."
curl -L "https://github.com/ggerganov/whisper.cpp/releases/download/v1.6.0/whisper-cublas-12.2.0-bin-x64.tar.gz" --output "whisper-cublas-12.2.0-bin-x64.tar.gz"
echo "Unzipping Whisper..."
tar -xzf whisper-cublas-12.2.0-bin-x64.tar.gz -C submodules/whisper.cpp

echo "Cleaning up..."
rm whisper-cublas-12.2.0-bin-x64.tar.gz
rm cudart-llama-bin-linux-cu12.2.0-x64.tar.gz
rm llama-bin-linux-cuda-cu12.2.0-x64.tar.gz

echo "Downloading Models..."
mkdir -p models
curl -L "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin" --output  "models/ggml-medium-32-2.en.bin"
curl -L "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf?download=true" --output "models/Meta-Llama-3-8B-Instruct-Q6_K.gguf"

echo "Done!"
