#!/bin/bash

# First, change to the script's directory
cd "$(dirname "$0")"

# Check if UV is installed
if command -v uv &> /dev/null; then
    echo "UV is already installed."
else
    echo "UV is not installed. Installing UV..."
    # Run the installation script
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Verify if installation was successful
    if command -v uv &> /dev/null; then
        echo "UV installed successfully."
    else
        echo "Failed to install UV. Please check the installation script or your internet connection."
        exit 1
    fi
fi

echo "Creating Virtual Environment..."
uv venv --python 3.12.8
source .venv/bin/activate
echo "Installing Dependencies..."

if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
else
    echo "Error: requirements.txt not found in $(pwd)"
    exit 1
fi

echo "Downloading Models..."

# Use simple arrays instead of associative arrays for better compatibility
urls=(
    "https://github.com/dnhkng/GlaDOS/releases/download/0.1/glados.onnx"
    "https://github.com/dnhkng/GlaDOS/releases/download/0.1/nemo-parakeet_tdt_ctc_110m.onnx"
    "https://github.com/dnhkng/GlaDOS/releases/download/0.1/phomenizer_en.onnx"
    "https://github.com/dnhkng/GlaDOS/releases/download/0.1/silero_vad.onnx"
)
files=(
    "models/glados.onnx"
    "models/nemo-parakeet_tdt_ctc_110m.onnx"
    "models/phomenizer_en.onnx"
    "models/silero_vad.onnx"
)

# Check if curl is installed
if ! command -v curl &> /dev/null; then
    echo "curl is not installed. Installing curl..."
    sudo apt-get update
    sudo apt-get install -y curl
fi

# Loop through arrays by index
for i in "${!urls[@]}"; do
    echo "Checking file: ${files[$i]}"
    if [ -f "${files[$i]}" ]; then
        echo "File ${files[$i]} already exists."
    else
        echo "File ${files[$i]} does not exist. Downloading..."
        mkdir -p "$(dirname "${files[$i]}")" # Create the directory if it doesn't exist
        curl -L "${urls[$i]}" --output "${files[$i]}"
        if [ -f "${files[$i]}" ]; then
            echo "Download successful."
        else
            echo "Download failed."
        fi
    fi
done

echo "Installation Complete!"

# Keep the terminal window open to see any errors
echo "Press any key to close..."
read -n 1