FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 as base-w
RUN mkdir /app
COPY submodules/whisper.cpp /app/
RUN cd /app && make libwhisper.so WHISPER_CUBLAS=1 CUDA_DOCKER_ARCH=all

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 as base-l
RUN mkdir /app
COPY submodules/llama.cpp /app/
RUN cd /app && make LLAMA_CUDA=1 LLAMA_CUBLAS=1 CUDA_DOCKER_ARCH=all

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update -y && apt install -qq  -y curl espeak-ng libportaudio2 libportaudiocpp0 portaudio19-dev python3.11 python3-pip python3.11-dev python3.11-venv pulseaudio libpulse-dev
COPY requirements.txt /app/requirements.txt
RUN python3.11 -m pip install -r /app/requirements.txt
COPY . /app
WORKDIR /app
COPY --from=base-l /app/server /app/submodules/llama.cpp/server
COPY --from=base-w /app/libwhisper.so /app/libwhisper.so
CMD ["python3.11", "glados.py"]
