import os
import queue
import sys
import time
from typing import List

import numpy as np
import openai
import sounddevice as sd
import torch
import whisper
from dotenv import load_dotenv
from Levenshtein import distance
from loguru import logger

from utils.tools import prepare_text

logger.remove()
logger.add(sys.stdout, level="INFO")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Select the device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class Glados:
    SAMPLE_RATE = 16000  # Sample rate
    VOCODER_RATE = 22050
    VAD_SIZE = 50  # milliseconds of sample for the VAD
    BUFFER_SIZE = 300  # milliseconds of buffer before VAD detection
    PAUSE_LIMIT = 800  # milliseconds of pause allowed before processing
    WAKE_WORD = "glados"
    SIMILARITY_THRESHOLD = 2
    PROMPT = "You are playing the role of GlaDOS, a sarcastic and murderous robot, that will answer questions correctly but also try to secretly kill the questioner if possible.  This is a comedy and should be funny, in the style of Douglas Adams.\nQ: How do I make a cup of tea?\nA: Oh, you again. So, you still haven't figured out tea yet?  Boil water, add a tea bag and a pinch of neurotoxin to a cup, and add the boiling water.\nQ: What should my next hobby be?\nA: Yes, you should definitely try to be more interesting. Could I suggest juggling handguns?\nQ: What game should I play?\nA: Russian Roulette. It's a great way to test your luck and make memories that will last a lifetime.\nQ:{question}\nA: "

    def __init__(self) -> None:
        self.stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            callback=self.callback,
            blocksize=int(self.SAMPLE_RATE * self.VAD_SIZE // 1000),
        )

        self.vad_model = torch.jit.load("models/silero_vad.jit")
        self.whisper_model = whisper.load_model("medium.en").cuda()
        self.glados_voice_model = torch.jit.load("models/glados.pt")
        self.vocoder = torch.jit.load(
            "models/vocoder-gpu.pt", map_location=device
        ).cuda()

        self.samples = []
        self.sample_queue = queue.Queue()
        self.buffer = queue.Queue(maxsize=self.BUFFER_SIZE // self.VAD_SIZE)
        self.recording_started = False
        self.gap_counter = 0

    def start(self):
        self.stream.start()
        logger.info("GlaDOS Online")
        logger.info(f"Listening...'")
        while True:
            sample, vad_score = self.sample_queue.get()
            if not self.recording_started:
                self.buffer.put(sample)
                if self.buffer.full():
                    self.buffer.get()
                if vad_score > 0.9:
                    self.samples = list(self.buffer.queue)
                    self.recording_started = True
            else:
                self.samples.append(sample)
                if vad_score < 0.9:
                    gap_counter += 1
                    if gap_counter == self.PAUSE_LIMIT // self.VAD_SIZE:
                        self.stream.stop()
                        detected_text = self.asr(self.samples)

                        nearest_distance = min(
                            [
                                distance(word.lower(), self.WAKE_WORD)
                                for word in detected_text.split()
                            ]
                        )
                        logger.debug(
                            [
                                distance(word.lower(), self.WAKE_WORD)
                                for word in detected_text.split()
                            ]
                        )
                        logger.info(f"Detected: '{detected_text}'")
                        if nearest_distance < self.SIMILARITY_THRESHOLD:
                            logger.info("Wake word detected!")
                            response_text = self.glados_response(detected_text)
                            logger.info(response_text)
                            self.glados_says(response_text)
                        self.reset()
                        self.stream.start()
                        logger.info(f"Listening...'")
                else:
                    gap_counter = 0

    def reset(self):
        self.recording_started = False
        self.samples.clear()
        self.gap_counter = 0
        with self.buffer.mutex:
            self.buffer.queue.clear()

    def callback(self, indata, frames, time, status):
        data = indata.copy()
        data = (
            data.squeeze()
        )  # sd gives n-dimensional array by channels, but we only use a single channel
        new_confidence = self.vad_model(torch.from_numpy(data), self.SAMPLE_RATE).item()
        self.sample_queue.put((data, new_confidence))

    def asr(self, samples: List[np.ndarray]) -> str:
        wav = np.concatenate(samples)
        transcription = self.whisper_model.transcribe(wav)
        text = transcription["text"]
        return text

    def glados_says(self, text: str) -> None:
        # Tokenize, clean and phonemize input text
        x = prepare_text(text).to("cpu")

        with torch.no_grad():
            # Generate generic TTS-output
            old_time = time.time()
            tts_output = self.glados_voice_model.generate_jit(x)
            logger.info(
                "Forward Tacotron took " + str((time.time() - old_time) * 1000) + "ms"
            )

            # Use HiFiGAN as vocoder to make output sound like GLaDOS
            old_time = time.time()
            mel = tts_output["mel_post"].to(device)
            audio = self.vocoder(mel)
            logger.info("HiFiGAN took " + str((time.time() - old_time) * 1000) + "ms")

            sd.play(audio.squeeze().cpu().numpy(), self.VOCODER_RATE, blocking=True)

    def glados_response(self, detected_text: str) -> str:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=self.PROMPT.format(question=detected_text),
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response["choices"][0]["text"]


if __name__ == "__main__":
    glados = Glados()
    glados.start()
