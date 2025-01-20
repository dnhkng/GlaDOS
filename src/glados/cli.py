import argparse
from pathlib import Path

import sounddevice as sd  # type: ignore

from .core import tts
from .engine import Glados, GladosConfig
from .utils import spoken_text_converter as stc


def say(text: str, config_path: str | Path = "glados_config.yml") -> None:
    """Use GLaDOS TTS to say the given text.

    Args:
        text: Text to speak
        config_path: Path to the configuration YAML file
    """
    glados_tts = tts.Synthesizer()
    converter = stc.SpokenTextConverter()

    print(text)
    converted_text = converter.text_to_spoken(text)
    print(converted_text)
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


def main() -> None:
    """CLI entry point for GLaDOS."""
    parser = argparse.ArgumentParser(description="GLaDOS Voice Assistant")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

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

    if args.command == "start":
        start(args.config)
    elif args.command == "say":
        say(args.text, args.config)
    else:
        # Default to start if no command specified
        start("glados_config.yml")


if __name__ == "__main__":
    main()
