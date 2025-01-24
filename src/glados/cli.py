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
    """Verify checksums of model files."""
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
    """Download and verify model files."""
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
    """Use GLaDOS TTS to say the given text.

    Args:
        text: Text to speak
        config_path: Path to the configuration YAML file
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
    """Set up the LLM server and start GlaDOS.

    Args:
        config_path: Path to the configuration YAML file
    """
    glados_config = GladosConfig.from_yaml(str(config_path))
    glados = Glados.from_config(glados_config)
    glados.start_listen_event_loop()


def models_valid() -> bool:
    results = verify_checksums()
    if not all(results.values()):
        print("Some model files are missing or invalid. Please run 'uv glados download'")
        return False
    return True


def main() -> None:
    """CLI entry point for GLaDOS."""
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
