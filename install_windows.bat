@echo off
REM Download and install the required dependencies for the project on Windows

echo Install espeak-ng...
curl -L "https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi" --output "espeak-ng-X64.msi"
espeak-ng-X64.msi
del espeak-ng-X64.msi

python3.12 -m venv venv
call .\venv\Scripts\activate
pip install -r requirements_cuda.txt

echo Downloading Llama...
curl -L "https://github.com/ggerganov/llama.cpp/releases/download/b3266/cudart-llama-bin-win-cu12.2.0-x64.zip" --output "cudart-llama-bin-win-cu12.2.0-x64.zip"
curl -L "https://github.com/ggerganov/llama.cpp/releases/download/b3266/llama-b3266-bin-win-cuda-cu12.2.0-x64.zip" --output "llama-bin-win-cuda-cu12.2.0-x64.zip"
echo Unzipping Llama...
tar -xf cudart-llama-bin-win-cu12.2.0-x64.zip -C submodules\llama.cpp
tar -xf llama-bin-win-cuda-cu12.2.0-x64.zip -C submodules\llama.cpp

echo Downloading Whisper...
curl -L "https://github.com/ggerganov/whisper.cpp/releases/download/v1.6.0/whisper-cublas-12.2.0-bin-x64.zip" --output "whisper-cublas-12.2.0-bin-x64.zip"
echo Unzipping Whisper...
tar -xf whisper-cublas-12.2.0-bin-x64.zip -C submodules\whisper.cpp

echo Cleaning up...
del whisper-cublas-12.2.0-bin-x64.zip
del cudart-llama-bin-win-cu12.2.0-x64.zip
del llama-bin-win-cuda-cu12.2.0-x64.zip

REM Download ASR and LLM Models
echo Downloading Models...
curl -L "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin" --output  "models\ggml-medium-32-2.en.bin"
curl -L "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf?download=true" --output "models\Meta-Llama-3-8B-Instruct-Q6_K.gguf"

echo Done!
