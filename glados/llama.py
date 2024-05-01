import os
import subprocess
import time

import requests


class LlamaServer:
    def __init__(self, llama_server_path, port=8080, model=None):
        # Initialize the model and process
        self.model = model
        self.port = port
        self.process = None

        # Define the fixed command and arguments
        self.command = ["./server", "-m", "chat-template", "llama3"]

        # Specify the directory where the server executable is located if it's not the current directory
        self.llama_server_path = llama_server_path

    def start(self, model=None, use_gpu=False):
        if model is not None:
            self.model = model
        command = [os.path.join(self.llama_server_path, "server"), "-m"] + [self.model]
        if use_gpu:
            command += ["-ngl", "1000"]
        print(command)
        self.process = subprocess.Popen(
            command,
            cwd=self.llama_server_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return self.is_running()

    def is_running(self):
        if self.process is not None:
            attempts = 0
            while True:
                try:
                    response = requests.get("http://localhost:8080/health")

                    if response.status_code == 200:
                        return True
                    elif (
                        response.status_code == 503
                    ):  # model is still being loaded, or at full capacity
                        pass
                    elif response.status_code == 500:  # model failed to load
                        self.stop()  # stop the server
                        return False
                    else:  # server is not running
                        return False

                except requests.exceptions.ConnectionError:
                    attempts += 1
                    if attempts > 10:
                        return False
                time.sleep(0.1)
        return False

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
        self.process = None
