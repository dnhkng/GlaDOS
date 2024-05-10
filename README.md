# GLaDOS Personality Core

This is a project dedicated to building a real-life version of GLaDOS.

[![localGLaDOS](https://img.youtube.com/vi/KbUfWpykBGg/0.jpg)](https://www.youtube.com/watch?v=KbUfWpykBGg)


*This is a hardware and software project that will create an aware, interactive, and embodied GLaDOS.*

This will entail:
- [x] Train GLaDOS voice generator
- [x] Generate a prompt that leads to a realistic "Personality Core"
- [ ] Generate a [MemGPT](https://memgpt.readthedocs.io/en/latest/) medium- and long-term memory for GLaDOS
- [ ] Give GLaDOS vision via [LLaVA](https://llava-vl.github.io/)
- [ ] Create 3D-printable parts
- [ ] Design the animatronics system
  


## Software Architecture
The initial goals are to develop a low-latency platform, where GLaDOS can respond to voice interactions within 600ms.

To do this, the system constantly records data to a circular buffer, waiting for [voice to be detected](https://github.com/snakers4/silero-vad). When it's determined that the voice has stopped (including detection of normal pauses), it will be [transcribed quickly](https://github.com/huggingface/distil-whisper). This is then passed to streaming [local Large Language Model](https://github.com/ggerganov/llama.cpp), where the streamed text is broken by sentence, and passed to a [text-to-speech system](https://github.com/rhasspy/piper). This means further sentences can be generated while the current is playing, reducing latency substantially.

### Subgoals
 - The other aim of the project is to minimize dependencies, so this can run on constrained hardware. That means no PyTorch or other large packages.  
 - As I want to fully understand the system, I have removed a large amount of redirection: which means extracting and rewriting code. i.e. as GLaDOS only speaks English, I have rewritten the wrapper around [espeak](https://espeak.sourceforge.net/) and the entire Text-to-Speech subsystem is about 500 LOC and has only 3 dependencies: numpy, onnxruntime, and sounddevice. 

## Hardware System
This will be based on servo- and stepper-motors. 3D printable STL will be provided to create GlaDOS's body, and she will be given a set of animations to express herself. The vision system will allow her to track and turn toward people and things of interest.

## Installation Instruction

### *New Simplified  Windows Installation Process*
Don't want to compile anything?  Try this process, but be aware its experimental!

1. Download and install eSpeak-ng: https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi
2. Download and unzip this repository somewhere
3. Open a command prompt in Windows (type `cmd` in the search bar)
4. in CMD, type `python`. If you dont see a python prompt, you will be redirected to install Python. Make sure you install version 3.12
5. In the command prompt navigate to where you unzipped this project, and run the command `install_windows.bat`  This *should*:
   1. Create a virtual environment and install the python requirements
   2. download the whisper and llama binaries, and unzip them to the correct locations
   3. download the whisper and llama-3 models
6. Once this is all done, you can start GLaDOS with `python glados.py`

If you have done all this before, just be sure to activate the virtual environment with `.\venv\Scripts\activate` before starting GLaDOS.

## Regular installation

If you want to install the TTS Engine on your machine, please follow the steps
below.  This has only been tested on Linux, but I think it will work on Windows with small tweaks.
If you are on windows, I would recommend WSL with an Ubuntu image.  Proper Windows and Mac support is in development.

1. Install the [`espeak`](https://github.com/espeak-ng/espeak-ng) synthesizer
   according to the [installation
   instructions](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md)
   for your operating system.
2. Install the required Python packages, e.g., by running `pip install -r
   requirements.txt`
3.  Download the models:
    1.  [voice recognition model](https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin?download=true)
    2.  [Llama-3 8B](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-IQ3_XS.gguf?download=true) or
    3.  [Llama-3 70B](https://huggingface.co/MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct.IQ4_XS.gguf?download=true)
    and put them in the ".models" directory.
4. For voice recognition, we use [Whisper.cpp](https://github.com/ggerganov/whisper.cpp)
   1. You can either download the compiled [whisper.cpp DLLs](https://github.com/ggerganov/whisper.cpp/releases) (recommended for windows), and copy the dll to the ./submodules/whisper.cpp directory
   2. Or compile them yourself. 
      1. To pull the code, from the GLaDOS directory use: `git submodule update --init --recursive`
      2. Move the the right subdirectory: `cd submodules/whisper.cpp`
      3. Compile for your system [(see the Documentation)](https://github.com/ggerganov/whisper.cpp), e.g.
         1. Linux with [CUDA](https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file#nvidia-gpu-support): `WHISPER_CUDA=1 make libwhisper.so -j`
         2. Mac with [CoreML](https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file#core-ml-support): `WHISPER_COREML=1 make -j`
5. For the LLM, you have two option:
   1. Compile llama.cpp:
      1. Use: `git submodule update --init --recursive` to pull the llama.cpp repo
      2. Move the the right subdirectory: `cd submodules/llama.cpp`
      3. Compile llama.cpp, [(see the Documentation)](https://github.com/ggerganov/whisper.cpp)
         1. Linux with [CUDA](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#cuda) `make server LLAMA_CUDA=1`
         2. MacOS with [Metal](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#metal-build) `make`
   2. Use a commercial API or install an inference backend yourself, such as Ollama or Llamafile:
      1. Find and install a backend with an OpenAI compatible API (most of them)
      2. Edit the glados_config.yaml
         1. update `completion_url` to the URL of your local server
         2. for commercial APIs, add the `api_key`
         3. remove the LlamaServer configurations (make them null)




## Running GLaDOS

To start GLaDOS, use:
`python glados.py`

You can stop with "Ctrl-c".


## Testing
You can test the systems by exploring the 'demo.ipynb'.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dnhkng/GlaDOS&type=Date)](https://star-history.com/#dnhkng/GlaDOS&Date)

<a href="https://trendshift.io/repositories/9828" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9828" alt="dnhkng%2FGlaDOS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
