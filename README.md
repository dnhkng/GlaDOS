# GLaDOS Personality Core

This is a project dedicated to building a real-life version of GLaDOS!

NEW: If you want to chat or join the community, [Join our discord!](https://discord.com/invite/ERTDKwpjNB)

[![localGLaDOS](https://img.youtube.com/vi/KbUfWpykBGg/0.jpg)](https://www.youtube.com/watch?v=KbUfWpykBGg)


*This is a hardware and software project that will create an aware, interactive, and embodied GLaDOS.*

This will entail:
- [x] Train GLaDOS voice generator
- [x] Generate a prompt that leads to a realistic "Personality Core"
- [ ] Generate a [MemGPT](https://github.com/cpacker/MemGPT) medium- and long-term memory for GLaDOS
- [ ] Give GLaDOS vision via [LLaVA](https://llava-vl.github.io/)
- [ ] Create 3D-printable parts
- [ ] Design the animatronics system



## Software Architecture
The initial goals are to develop a low-latency platform, where GLaDOS can respond to voice interactions within 600ms.

To do this, the system constantly records data to a circular buffer, waiting for [voice to be detected](https://github.com/snakers4/silero-vad). When it's determined that the voice has stopped (including detection of normal pauses), it will be [transcribed quickly](https://github.com/huggingface/distil-whisper). This is then passed to streaming [local Large Language Model](https://github.com/ggerganov/llama.cpp), where the streamed text is broken by sentence, and passed to a [text-to-speech system](https://github.com/rhasspy/piper). This means further sentences can be generated while the current is playing, reducing latency substantially.

### Subgoals
 - The other aim of the project is to minimize dependencies, so this can run on constrained hardware. That means no PyTorch or other large packages.
 - As I want to fully understand the system, I have removed a large amount of redirection: which means extracting and rewriting code.

## Hardware System
This will be based on servo- and stepper-motors. 3D printable STL will be provided to create GlaDOS's body, and she will be given a set of animations to express herself. The vision system will allow her to track and turn toward people and things of interest.

## Installation Instruction
Try this simplified process, but be aware it's still in the experimental stage!  For all operating systems, you'll first need to install Ollama to run the LLM.
### Set up a local LLM server:
1. Download and install [Ollama](https://github.com/ollama/ollama) for your operating system.
2. Once installed, download a small 2B model for testing, at a terminal or command prompt use: `ollama pull llama3.2`

### Windows Installation Process
1. Open the Microsoft Store, search for `python` and install Python 3.12
2. Downlod this repository, either:
   1. Download and unzip this repository somewhere in your home folder, or
   2. If you have Git set up, `git clone` this repository using `git clone github.com/dnhkng/glados.git`
3. In the repository folder, run the `install_windows.bat`, and wait until the installation in complete.
4. Double click `start_windows.bat` to start GLaDOS!

## macOS Installation Process
This is still experimental. Any issues can be addressed in the Discord server. If you create an issue related to this, you will be referred to the Discord server.


1. Downlod this repository, either:
   1. Download and unzip this repository somewhere in your home folder, or
   2. In a terminal, `git clone` this repository using `git clone github.com/dnhkng/glados.git`
2. In a terminal, go to the repository folder and run these commands:

         chmod +x install_mac.command
         chmod +x start_mac.command

3. In the Finder, double click `install_mac.command`, and wait until the installation in complete.
4. Double click `start_mac.sh` to start GLaDOS!

## Linux Installation Process
This is still experimental. Any issues can be addressed in the Discord server. If you create an issue related to this, you will be referred to the Discord server.  This has been tested on Ubuntu 24.04.1 LTS


1. Downlod this repository, either:
   1. Download and unzip this repository somewhere in your home folder, or
   2. In a terminal, `git clone` this repository using `git clone github.com/dnhkng/glados.git`
2. In a terminal, go to the repository folder and run these commands:
   
         chmod +x install_ubuntu.sh
         chmod +x start_ubuntu.sh

3. In the a terminal in the GLaODS folder, run `./install_ubuntu.sh`, and wait until the installation in complete.
4. Run  `./start_ubuntu.sh` to start GLaDOS!

## Changing the LLM Model

To use other models, use the conmmand:
```ollama pull {modelname}```
and then add {modelname} to glados_config.yaml as the model.

## Common Issues
1. If you find you are getting stuck in loops, as GLaDOS is hearing herself speak, you have two options:
   1. Solve this by upgrading your hardware. You need to you either headphone, so she can't physically hear herself speak, or a conference-style room microphone/speaker. These have hardware sound cancellation, and prevent these loops.
   2. Disable voice interruption. This means neither you nor GLaDOS can interrupt when GLaDOS is speaking. To accomplish this, edit the `glados_config.yaml`, and change `interruptible:` to  `false`.




## Testing the submodules
You can test the systems by exploring the 'demo.ipynb'.


## Star History
<a href="https://trendshift.io/repositories/9828" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9828" alt="dnhkng%2FGlaDOS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

[![Star History Chart](https://api.star-history.com/svg?repos=dnhkng/GlaDOS&type=Date)](https://star-history.com/#dnhkng/GlaDOS&Date)
