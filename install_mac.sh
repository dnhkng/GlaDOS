#!/bin/sh
#
# Simple install script built for macOS to install the required components for the GLaDOS peoject
# https://github.com/dnhkng/GlaDOS

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Installing espeak and Homebrew if neccessary
echo "Installing espeak and Homebrew if necessary"
if [[ $(command -v brew) == "" ]] ; then
    # Install Homebrew
    echo "You do not have Homebrew installed, installing now"
    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
else
    echo You have Homebrew installed, updating now
    brew update
fi

brew install espeak-ng

python3.12 -m venv venv > /dev/null
source venv/bin/activate > /dev/null
python3.12 -m pip install -r requirements.txt > /dev/null

# Installing Whisper and llama
echo "Installing Whisper and llama"
git submodule update --init --recursive > /dev/null

# Compiling Whisper
echo "Compiling Whisper"
cd submodules/whisper.cpp
make libwhisper.so -j > /dev/null
cd ..
cd ..

# Compiling llama
echo "Compiling llama"
cd submodules/llama.cpp
make server > /dev/null
cd ..
cd ..

# Downloading ASR and LLM models
echo "Downloading Models"
curl -L "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin" --output  "models/ggml-medium-32-2.en.bin"
curl -L "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf?download=true" --output "models/Meta-Llama-3-8B-Instruct-Q6_K.gguf"

# Fixes ggml-metal.metal
echo Fixing Whisper.cpp
sed -i "1,6s|ggml-common.h|$SCRIPT_DIR/submodules/whisper.cpp/ggml-common.h|" submodules/whisper.cpp/ggml-metal.metal
