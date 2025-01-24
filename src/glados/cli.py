import argparse
import hashlib
from pathlib import Path

import requests
import sounddevice as sd  # type: ignore

from .core import tts
from .engine import Glados, GladosConfig
from .utils import spoken_text_converter as stc

MODEL_CHECKSUMS = {
    "models/ASR/nemo-parakeet_tdt_ctc_110m.onnx": "313705ff6f897696ddbe0d92b5ffadad7429a47d2ddeef370e6f59248b1e8fb5",
    "models/ASR/silero_vad.onnx": "a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28",
    "models/TTS/glados.onnx": "17ea16dd18e1bac343090b8589042b4052f1e5456d42cad8842a4f110de25095",
    "models/phomenizer_en.onnx": "b64dbbeca8b350927a0b6ca5c4642e0230173034abd0b5bb72c07680d700c5a0",
}

MODEL_URLS = {
    "models/ASR/nemo-parakeet_tdt_ctc_110m.onnx": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/nemo-parakeet_tdt_ctc_110m.onnx",
    "models/ASR/silero_vad.onnx": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/silero_vad.onnx",
    "models/TTS/glados.onnx": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/glados.onnx",
    "models/phomenizer_en.onnx": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/phomenizer_en.onnx",
}


def verify_checksums() -> dict[str, bool]:
    """
    Verify the integrity of model files by comparing their SHA-256 checksums against expected values.
    
    This function checks each model file specified in MODEL_CHECKSUMS to ensure it exists 
    and has the correct checksum. Files that are missing or have incorrect checksums are 
    marked as invalid.
    
    Parameters:
        None
    
    Returns:
        dict[str, bool]: A dictionary where keys are model file paths and values are 
                         boolean indicators of checksum validity (True if valid, False if 
                         missing or checksum mismatch)
    
    Example:
        >>> verify_checksums()
        {
            'models/tts_model.pth': True, 
            'models/asr_model.bin': False
        }
    """
    results = {}
    for path, expected in MODEL_CHECKSUMS.items():
        model_path = Path(path)
        if not model_path.exists():
            results[path] = False
            continue

        sha = hashlib.sha256()
        sha.update(model_path.read_bytes())
        results[path] = sha.hexdigest() == expected

    return results


def download_models() -> None:
    """
    Download and verify model files for the GLaDOS voice assistant.
    
    This function checks the integrity of model files using their checksums and downloads
    any missing or invalid models from predefined URLs. It creates the necessary directory
    structure for the model files and streams the downloaded content to disk.
    
    Behavior:
        - Verifies existing model files using their SHA-256 checksums
        - Downloads models that are missing or have invalid checksums
        - Creates parent directories for model files if they do not exist
        - Prints a download message for each model being downloaded
    
    Side Effects:
        - Writes model files to the local filesystem
        - Prints download progress messages to the console
    
    Raises:
        requests.exceptions.RequestException: If there are network issues during download
    """
    results = verify_checksums()
    for path, is_valid in results.items():
        if not is_valid and path in MODEL_URLS:
            print(f"Downloading {path}...")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(MODEL_URLS[path], stream=True)
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)


def say(text: str, config_path: str | Path = "glados_config.yml") -> None:
    """
    Converts text to speech using the GLaDOS text-to-speech system and plays the generated audio.
    
    Parameters:
        text (str): The text to be spoken by the GLaDOS voice assistant.
        config_path (str | Path, optional): Path to the configuration YAML file. 
            Defaults to "glados_config.yml".
    
    Notes:
        - Uses a text-to-speech synthesizer to generate audio
        - Converts input text to a spoken format before synthesis
        - Plays the generated audio using the system's default sound device
        - Blocks execution until audio playback is complete
    
    Example:
        say("Hello, world!")  # Speaks the text using GLaDOS voice
    """
    glados_tts = tts.Synthesizer()
    converter = stc.SpokenTextConverter()
    converted_text = converter.text_to_spoken(text)
    # Generate the audio to from the text
    audio = glados_tts.generate_speech_audio(converted_text)

    # Play the audio
    sd.play(audio, glados_tts.rate)
    sd.wait()


def start(config_path: str | Path = "glados_config.yml") -> None:
    """
    Start the GLaDOS voice assistant and initialize its listening event loop.
    
    This function loads the GLaDOS configuration from a YAML file, creates a GLaDOS instance,
    and begins the continuous listening process for voice interactions.
    
    Parameters:
        config_path (str | Path, optional): Path to the configuration YAML file.
            Defaults to "glados_config.yml" in the current directory.
    
    Raises:
        FileNotFoundError: If the specified configuration file cannot be found.
        ValueError: If the configuration file is invalid or cannot be parsed.
    
    Example:
        start()  # Uses default configuration file
        start("/path/to/custom/config.yml")  # Uses a custom configuration file
    """
    glados_config = GladosConfig.from_yaml(str(config_path))
    glados = Glados.from_config(glados_config)
    glados.start_listen_event_loop()


def models_valid() -> bool:
    """
    Check the validity of all model files for the GLaDOS voice assistant.
    
    Verifies the integrity of model files by computing their checksums and comparing them against expected values.
    
    Returns:
        bool: True if all model files are valid and present, False otherwise. When False, prints a message 
              instructing the user to download the required model files.
    """
    results = verify_checksums()
    if not all(results.values()):
        print("Some model files are missing or invalid. Please run 'uv glados download'")
        return False
    return True


def main() -> None:
    """
    Command-line interface (CLI) entry point for the GLaDOS voice assistant.
    
    Provides three primary commands:
    - 'download': Download required model files
    - 'start': Launch the GLaDOS voice assistant
    - 'say': Generate speech from input text
    
    The function sets up argument parsing with optional configuration file paths and handles
    command execution based on user input. If no command is specified, it defaults to starting
    the assistant.
    
    Optional Arguments:
        --config (str): Path to configuration file, defaults to 'glados_config.yml'
    
    Raises:
        SystemExit: If invalid arguments are provided
    """
    parser = argparse.ArgumentParser(description="GLaDOS Voice Assistant")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Download command
    subparsers.add_parser("download", help="Download model files")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start GLaDOS voice assistant")
    start_parser.add_argument(
        "--config",
        type=str,
        default="glados_config.yml",
        help="Path to configuration file (default: glados_config.yml)",
    )

    # Say command
    say_parser = subparsers.add_parser("say", help="Make GLaDOS speak text")
    say_parser.add_argument("text", type=str, help="Text for GLaDOS to speak")
    say_parser.add_argument(
        "--config",
        type=str,
        default="glados_config.yml",
        help="Path to configuration file (default: glados_config.yml)",
    )

    args = parser.parse_args()

    if args.command == "download":
        download_models()
    else:
        if models_valid() is False:
            return
        if args.command == "say":
            say(args.text, args.config)
        elif args.command == "start":
            start(args.config)
        else:
            # Default to start if no command specified
            start("glados_config.yml")


if __name__ == "__main__":
    main()
