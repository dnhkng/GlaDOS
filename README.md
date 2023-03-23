# GLaDOS Personality Core

Automatic Speech Recognition from OpenAI Whisper
Text-to-Speech Engine based on Tacotron 2 and Wavenet vocoder.
"Brain" uses Davinci Text 03 from OpenAI.

This works as stand-alone.
```console
python3 glados.py
```

The wake-word is 'Glados', but this can be changed in the class variables.
Note: 'Glados' seems a hard word to accurately detect!


## Installation Instruction
If you want to install the TTS Engine on your machine, please follow the steps
below.

1. Install the [`espeak`](https://github.com/espeak-ng/espeak-ng) synthesizer
   according to the [installation
   instructions](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md)
   for your operating system.
2. Install the required Python packages, e.g., by running `pip install -r
   requirements.txt`
