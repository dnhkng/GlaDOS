# src/glados/cli.py
import argparse
from pathlib import Path

from .engine import Glados, GladosConfig


def start(config_path: str | Path = "glados_config.yml") -> None:
    """Set up the LLM server and start GlaDOS.

    Args:
        config_path: Path to the configuration YAML file
    """
    glados_config = GladosConfig.from_yaml(config_path)
    glados = Glados.from_config(glados_config)
    glados.start_listen_event_loop()


def main() -> None:
    """CLI entry point for GLaDOS."""
    parser = argparse.ArgumentParser(description="GLaDOS Voice Assistant")
    parser.add_argument(
        "--config",
        type=str,
        default="glados_config.yml",
        help="Path to configuration file (default: glados_config.yml)",
    )
    args = parser.parse_args()
    start(args.config)


if __name__ == "__main__":
    main()
