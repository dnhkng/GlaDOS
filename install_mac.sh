#!/bin/sh
#
# Simple install script built for macOS to install the required components for the GLaDOS peoject
# https://github.com/dnhkng/GlaDOS

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $SCRIPT_DIR"

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

brew install espeak

# Making espeak-ng become espeak in the tts.py and disables CUDA and also the ggml-metal.metal
echo "Applying fixes for the macOS usage"
git clone https://github.com/Ghostboo-124/espeak-fix.git
rm glados/tts.py > /dev/null 2>&1
cp espeak-fix/tts.py glados/tts.py > /dev/null 2>&1

python3.12 -m venv venv > /dev/null 2>&1
source venv/bin/activate > /dev/null 2>&1
python3.12 -m pip install -r requirements.txt > /dev/null 2>&1

# Installing Whisper and llama
echo "Installing Whisper and llama"
git submodule update --init --recursive > /dev/null 2>&1

# Compiling Whisper
echo "Compiling Whisper"
cd submodules/whisper.cpp
make libwhisper.so -j > /dev/null 2>&1
cd ..
cd ..

# Applying Whisper Fix
echo "Fixing Whisper for macOS"
if [[ $SCRIPT_DIR =~ "Downloads" ]]
then
    rm submodules/whisper.cpp/ggml-metal.metal
    cp espeak-fix/ggml-metal-zip_file.metal submodules/whisper.cpp/ggml-metal.metal
else
    rm submodules/whisper.cpp/ggml-metal.metal
    cp espeak-fix/ggml-metal-git_clone.metal submodules/whisper.cpp/ggml-metal.metal
fi

# Compiling llama
echo "Compiling llama"
cd submodules/llama.cpp
make server > /dev/null 2>&1
cd ..
cd ..

# Downloading ASR and LLM models
echo "Downloading Models"
curl -L "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin" --output  "models/ggml-medium-32-2.en.bin"
curl -L "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf?download=true" --output "models/Meta-Llama-3-8B-Instruct-Q6_K.gguf"

# Removing leftover files
echo Cleaning up
rm -rf espeak-fix