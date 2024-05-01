import os
import subprocess
import time
import logging
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests


logger = logging.getLogger(__name__)


class BaseLlamaServer:
    def __init__(self, server_base_url: str, request_headers: Dict[str, str]):
        self._server_base_url = server_base_url
        self._request_headers = request_headers

    def start(self):
        pass

    def stop(self):
        pass

    def is_running(self) -> bool:
        try:
            health_url = urljoin(self._server_base_url, "./health")
            response = requests.get(health_url)
            return response.status_code == 200

        except requests.exceptions.ConnectionError:
            return False

    def await_running(self, timeout_secs=5.0, log_message: Optional[str] = None) -> bool:
        current_time = time.monotonic()
        max_time = current_time + timeout_secs

        if not log_message:
            log_message = "Awaiting LLM service availability"

        while current_time < max_time:
            running = self.is_running()

            if running:
                break

            logger.info(
                "%s: current time: %f; retrying for up to %f seconds (until %f)",
                log_message,
                current_time,
                timeout_secs,
                max_time
            )

            time.sleep(0.1)
            current_time = time.monotonic()

        return running

    def request(self, json: Dict[str, Any], stream=False) -> Dict[str, Any]:
        generate_url = urljoin(self._server_base_url, "./completion")

        response = requests.post(
            generate_url,
            headers=self._request_headers,
            json=json,
            stream=True,
        )

        if not response.ok:
            if response.status_code == 404:
                logger.error(f"Config.LLAMA_SERVER_BASE_URL {self._server_base_url!r} seems to be invalid")
            else:
                logger.error(f"Unrecognised error in LLM response. status_code: {response.status_code}, reason: {response.reason!r}")

        return response


class ExternalLlamaServer(BaseLlamaServer):
    pass

class ChildLlamaServer(BaseLlamaServer):
    def __init__(self, server_base_url: str, request_headers: Dict[str, str], llama_server_path, port=8080, model=None, external=False, use_gpu=False):
        super().__init__(server_base_url=server_base_url, request_headers=request_headers)

        self.model = model
        if not self.model.exists():
            raise FileNotFoundError(f"File {model!r} not found")

        self.port = port
        self.process = None

        # Specify the directory where the server executable is located if it's not the current directory
        self.llama_server_path = llama_server_path

        self._external = external

        if not self._external:
            # Define the fixed command and arguments
            self.command = ["./server", "-m", "chat-template", "llama3"]

        self._use_gpu = use_gpu

    def start(self):
        if not self._external:
            assert self.model is not None, "No Llama model provided"

            command = [os.path.join(self.llama_server_path, "server"), "-m"] + [self.model]

            if self._use_gpu:
                command += ["-ngl", "1000"]

            logger.warning("Running %r", command)

        self.process = subprocess.Popen(
            command,
            cwd=self.llama_server_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return self.is_running()

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
        self.process = None
