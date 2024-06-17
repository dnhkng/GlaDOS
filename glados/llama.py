import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self, Sequence

import requests
import yaml
from loguru import logger


class ServerStartupError(RuntimeError):
    pass


@dataclass
class LlamaServerConfig:
    llama_cpp_repo_path: str
    model_path: str
    context_length: int = 512  # Default value from the standard llama.cpp[server] repo
    port: int = 8080
    use_gpu: bool = True
    enable_split_mode: bool = False
    enable_flash_attn: bool = False

    @classmethod
    def from_yaml(
        cls, path: str, key_to_config: Sequence[str] | None = ("LlamaServer",)
    ) -> Self | None:
        key_to_config = key_to_config or []

        with open(path, "r") as file:
            data = yaml.safe_load(file)

        config = data
        for nested_key in key_to_config:
            config = config.get(nested_key, {})
        if not config:
            return None
        return cls(**config)


# TODO: extract abstract LLMServer class
class LlamaServer:
    def __init__(
        self,
        llama_cpp_repo_path: Path,
        model_path: Path,
        context_length: int = 512,
        port: int = 8080,
        use_gpu: bool = True,
        enable_split_mode: bool = True,
        enable_flash_attn: bool = True,
    ):
        self.llama_cpp_repo_path = llama_cpp_repo_path
        self.model_path = model_path
        self.context_length = context_length

        self.port = port
        self.process: subprocess.Popen | None = None
        self.use_gpu = use_gpu
        self.enable_split_mode = enable_split_mode
        self.enable_flash_attn = enable_flash_attn

        self.command = (
            [self.llama_cpp_repo_path, "--model"]
            + [self.model_path]
            + ["--ctx-size", str(context_length)]
            + ["--port", str(port)]
        )
        if self.use_gpu:
            self.command += [
                "--n-gpu-layers",
                "1000",
            ]  # More than we would ever need, just to be sure.
        if self.enable_split_mode:
            self.command += [
                "--split-mode",
                "row",
            ] # Split Mode Row significantly improves performance on multi-GPU setups
        if self.enable_flash_attn:
            self.command += [
                "--flash-attn",
            ]# Significantly improves performance on GPUs which support SM row
        logger.success(f"Command to start the server: {self.command}")

    @classmethod
    def from_config(cls, config: LlamaServerConfig):
        llama_cpp_repo_path = Path(config.llama_cpp_repo_path) / "llama-server"
        llama_cpp_repo_path = llama_cpp_repo_path.resolve()
        model_path = Path(config.model_path).resolve()

        return cls(
            llama_cpp_repo_path=llama_cpp_repo_path,
            model_path=model_path,
            context_length=config.context_length,
            port=config.port,
            use_gpu=config.use_gpu,
            enable_split_mode=config.enable_split_mode,
            enable_flash_attn=config.enable_flash_attn,
        )

    @property
    def base_url(self):
        return f"http://localhost:{self.port}"

    @property
    def completion_url(self):
        return f"{self.base_url}/completion"

    @property
    def health_check_url(self):
        return f"{self.base_url}/health"

    def start(self):
        logger.info(f"Starting the server by executing command {self.command=}")
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not self.is_running():
            self.stop()
            raise ServerStartupError("Failed to startup! Check the error log messages")

    def is_running(
        self,
        max_connection_attempts: int = 10,
        sleep_time_between_attempts: float = 0.01,
        max_wait_time_for_model_loading: float = 60.0,
    ) -> bool:
        if self.process is None:
            return False

        cur_attempt = 0
        model_loading_time = 0
        model_loading_log_time = 1
        while True:
            try:
                response = requests.get(self.health_check_url)

                if response.status_code == 503:
                    if model_loading_time > max_wait_time_for_model_loading:
                        logger.error(
                            f"Model failed to load in {max_wait_time_for_model_loading}. "
                            f"Consider increasing the waiting time for model loading."
                        )
                        return False

                    logger.info(
                        f"model is still being loaded, or at full capacity. "
                        f"Will wait for {max_wait_time_for_model_loading - model_loading_time} "
                        f"more seconds: {response=}"
                    )
                    time.sleep(model_loading_log_time)
                    model_loading_time += model_loading_log_time
                    continue
                if response.status_code == 200:
                    logger.debug(f"Server started successfully, {response=}")
                    return True
                logger.error(
                    f"Server is not responding properly, maybe model failed to load: {response=}"
                )
                return False

            except requests.exceptions.ConnectionError:
                logger.debug(
                    f"Couldn't establish connection, retrying with attempt: {cur_attempt}/{max_connection_attempts}"
                )
                cur_attempt += 1
                if cur_attempt > max_connection_attempts:
                    logger.error(
                        f"Couldn't establish connection after {max_connection_attempts=}"
                    )
                    return False
            time.sleep(sleep_time_between_attempts)

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def __del__(self):
        self.stop()
        del self
