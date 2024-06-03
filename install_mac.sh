#!/bin/sh
#
# Simple install script built for macOS to install the required components for the GLaDOS peoject
# https://github.com/dnhkng/GlaDOS

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

# Making espeak-ng become espeak in the tts.py and disables CUDA
echo "Swaping the tts.py out for the one that uses espeak"
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

# Compiling llama
echo "Compiling llama"
cd submodules/llama.cpp
make server > /dev/null 2>&1
cd ..
cd ..

# Downloading ASR and LLM models
echo "Downloading Models"
curl -L "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin" --output  "models\ggml-medium-32-2.en.bin"
curl -L "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf?download=true" --output "models\Meta-Llama-3-8B-Instruct-Q6_K.gguf"

# Removing leftover files
echo Cleaning up
rm -rf espeak-fix