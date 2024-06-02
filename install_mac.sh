#!/bin/sh
#
# Simple install script built for macOS to install the required components for the GLaDOS peoject
# https://github.com/dnhkng/GlaDOS

# Installing espeak and Homebrew if neccessary
echo Install espeak
which -s brew
if [[ $? != 0 ]] ; then
    # Install Homebrew
    echo You do not have Homebrew installed, installing now
    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
else
    brew update
fi

brew install espeak

# Making espeak-ng become espeak in the tts.py
git clone https://github.com/Ghostboo-124/espeak-fix.git
rm glados/tts.py
cp espeak-fix/tts.py glados/tts.py

python3.12 -m venv venv
source venv/bin/activate
python3.12 -m pip install -r requirements.txt

# Downloading llama
echo Downloading llama
curl -L "https://github.com/ggerganov/llama.cpp/releases/download/b2839/llama-b2839-bin-macos-arm64.zip" --ouput "llama-b2839-bin-macos-arm64.zip"

# Unzipping llama
echo Unzipping llama
unzip llama-b2839-bin-macos-arm64.zip -d submodules/llama.cpp