# GLaDOS Personality Core

This is a project dedicated to building a real-life version of GLaDOS.

*This is a hardware and software project that will create an aware, interactive, and embodied GLaDOS.*

This will entail:
- [x] Train GLaDOS voice generator
- [x] Generate a prompt that leads to a realistic "Personality Core"
- [ ] Generate a [MemGPT](https://memgpt.readthedocs.io/en/latest/) medium- and long-term memory for GLaDOS
- [ ] Give GLaDOS vision via [LLaVA](https://llava-vl.github.io/)
- [ ] Create 3D-printable parts
- [ ] Design the animatronics system
  


## Sofware Architecture
The initial goals are to develop a low-latency platform, where GLaDOS can respond to voice interactions within 600ms.

To do this, the system constantly records data to a circular buffer, waiting for [voice to be detected](https://github.com/snakers4/silero-vad). When it's determined that the voice has stopped (including detection of normal pauses), it will be [transcribed quickly](https://github.com/huggingface/distil-whisper). This is then passed to streaming [local Large Language Model](https://github.com/ggerganov/llama.cpp), where the streamed text is broken by sentence, and passed to a [text-to-speech system](https://github.com/rhasspy/piper). This means further sentences can be generated while the current is playing, reducing latency substantially.

### Subgoals
 - The other aim of the project is to minimize dependencies, so this can run on constrained hardware. That means no PyTorch or other large packages.  
 - As I want to fully understand the system, I have removed a large amount of redirection: which means extracting and rewriting code. i.e. as GLaDOS only speaks English, I have rewritten the wrapper around [espeak](https://espeak.sourceforge.net/) and the entire Text-to-Speech subsystem is about 500 LOC and has only 3 dependencies: numpy, onnxruntime, and sounddevice. 

## Hardware System
This will be based on servo- and stepper-motors. 3D printable STL will be provided to create GlaDOS's body, and she will be given a set of animations to express herself. The vision system will allow her to track and turn toward people and things of interest.

## Installation Instruction
If you want to install the TTS Engine on your machine, please follow the steps
below.  This has only been tested on Linux, but I think it will work on Windows with small tweaks.

1. Install the [`espeak`](https://github.com/espeak-ng/espeak-ng) synthesizer
   according to the [installation
   instructions](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md)
   for your operating system.
2. Install the required Python packages, e.g., by running `pip install -r
   requirements.txt`
3. For the LLM, install [Llama.cpp](https://github.com/ggerganov/llama.cpp), by cloning it into `../llama.cpp` (relative to this folder), and compile it for your CPU or GPU. Edit the LLAMA_SERVER_PATH parameter in glados.py to match your installation path, if not cloning to `../llama.cpp`.
4. For voice recognition, install [Whisper.cpp](https://github.com/ggerganov/whisper.cpp), and after compiling, run ```make libwhisper.so``` and then move the "libwhisper.so" file to the "glados" folder or add it to your path.  For Windows, check out the discussion in my [whisper pull request](https://github.com/ggerganov/whisper.cpp/pull/1524).
5.  Download the models:
    1.  [voice recognition model](https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin?download=true)
    2.  [Llama-3 8B](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-IQ3_XS.gguf?download=true) or
    3.  [Llama-3 70B](https://huggingface.co/MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct.IQ4_XS.gguf?download=true)
   
    and put them in the "models" directory.

## Testing
You can test the systems by exploring the 'demo.ipynb'.
