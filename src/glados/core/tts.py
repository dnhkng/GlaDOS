from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from pickle import load
from typing import Any

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore

from .phonemizer import Phonemizer

# Default OnnxRuntime is way to verbose
ort.set_default_logger_severity(4)

# Constants
MAX_WAV_VALUE = 32767.0

# Settings
MODEL_PATH = "./models/TTS/glados.onnx"
PHONEME_TO_ID_PATH = Path("./models/TTS/phoneme_to_id.pkl")
USE_CUDA = True

# Conversions
PAD = "_"  # padding (0)
BOS = "^"  # beginning of sentence
EOS = "$"  # end of sentence


@dataclass
class PiperConfig:
    """Piper configuration"""

    num_symbols: int
    """Number of phonemes"""

    num_speakers: int
    """Number of speakers"""

    sample_rate: int
    """Sample rate of output audio"""

    espeak_voice: str
    """Name of espeak-ng voice or alphabet"""

    length_scale: float
    noise_scale: float
    noise_w: float

    phoneme_id_map: Mapping[str, Sequence[int]]
    """Phoneme -> [id,]"""

    speaker_id_map: dict[str, int] | None = None

    @staticmethod
    def from_dict(config: dict[str, Any]) -> "PiperConfig":
        inference = config.get("inference", {})

        return PiperConfig(
            num_symbols=config["num_symbols"],
            num_speakers=config["num_speakers"],
            sample_rate=config["audio"]["sample_rate"],
            noise_scale=inference.get("noise_scale", 0.667),
            length_scale=inference.get("length_scale", 1.0),
            noise_w=inference.get("noise_w", 0.8),
            espeak_voice=config["espeak"]["voice"],
            phoneme_id_map=config["phoneme_id_map"],
            speaker_id_map=config.get("speaker_id_map", {}),
        )


class Synthesizer:
    """Synthesizer, based on the VITS model.

    Trained using the Piper project (https://github.com/rhasspy/piper)

    Attributes:
    -----------
    session: onnxruntime.InferenceSession
        The loaded VITS model.
    id_map: dict
        A dictionary mapping phonemes to ids.

    Methods:
    --------
    __init__(self, model_path, use_cuda):
        Initializes the Synthesizer class, loading the VITS model.

    generate_speech_audio(self, text):
        Generates speech audio from the given text.

    _phonemizer(self, input_text):
        Converts text to phonemes using espeak-ng.

    _phonemes_to_ids(self, phonemes):
        Converts the given phonemes to ids.

    _synthesize_ids_to_raw(self, phoneme_ids, speaker_id, length_scale, noise_scale, noise_w):
        Synthesizes raw audio from phoneme ids.

    say_phonemes(self, phonemes):
        Converts the given phonemes to audio.
    """

    def __init__(self, model_path: str = MODEL_PATH, speaker_id: int | None = None) -> None:
        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")

        self.session = ort.InferenceSession(
            model_path,
            sess_options=ort.SessionOptions(),
            providers=providers,
        )
        self.phonemizer = Phonemizer()
        # self.id_map = PHONEME_ID_MAP

        self.id_map = self._load_pickle(PHONEME_TO_ID_PATH)

        try:
            # Load the configuration file
            config_file_path = model_path + ".json"
            with open(config_file_path, encoding="utf-8") as config_file:
                config_dict = json.load(config_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at path: {config_file_path}") from None
        except json.JSONDecodeError as e:
            raise ValueError(f"Configuration file at path: {config_file_path} is not a valid JSON. Error: {e}") from e
        except Exception as e:
            raise RuntimeError(
                "An unexpected error occurred while reading the configuration file "
                f"at path: {config_file_path}. Error: {e}"
            ) from e
        self.config = PiperConfig.from_dict(config_dict)
        self.rate = self.config.sample_rate
        self.speaker_id = (
            self.config.speaker_id_map.get(str(speaker_id), 0)
            if self.config.num_speakers > 1 and self.config.speaker_id_map is not None
            else None
        )

    @staticmethod
    def _load_pickle(path: Path) -> dict[str, Any]:
        """Load a pickled dictionary from path."""
        with path.open("rb") as f:
            return dict(load(f))

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        phonemes = self._phonemizer(text)
        audio = self.say_phonemes(phonemes)
        return np.array(audio, dtype=np.float32)

    def say_phonemes(self, phonemes: list[str]) -> NDArray[np.float32]:
        audio_list = []
        for sentence in phonemes:
            audio_chunk = self._say_phonemes(sentence)
            audio_list.append(audio_chunk)
        if audio_list:
            audio: NDArray[np.float32] = np.concatenate(audio_list, axis=1).T
            return audio
        return np.array([], dtype=np.float32)

    def _phonemizer(self, input_text: str) -> list[str]:
        """Converts text to phonemes using espeak-ng."""
        phonemes = self.phonemizer.convert_to_phonemes([input_text], "en_us")

        return phonemes

    def _phonemes_to_ids(self, phonemes: str) -> list[int]:
        """Phonemes to ids."""

        ids: list[int] = list(self.id_map[BOS])

        for phoneme in phonemes:
            if phoneme not in self.id_map:
                continue

            ids.extend(self.id_map[phoneme])
            ids.extend(self.id_map[PAD])
        ids.extend(self.id_map[EOS])

        return ids

    def _synthesize_ids_to_raw(
        self,
        phoneme_ids: list[int],
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w: float | None = None,
    ) -> NDArray[np.float32]:
        """Synthesize raw audio from phoneme ids."""
        if length_scale is None:
            length_scale = self.config.length_scale

        if noise_scale is None:
            noise_scale = self.config.noise_scale

        if noise_w is None:
            noise_w = self.config.noise_w

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)

        scales = np.array(
            [noise_scale, length_scale, noise_w],
            dtype=np.float32,
        )

        sid = None

        if self.speaker_id is not None:
            sid = np.array([self.speaker_id], dtype=np.int64)

        # Synthesize through Onnx
        audio: NDArray[np.float32] = self.session.run(
            None,
            {
                "input": phoneme_ids_array,
                "input_lengths": phoneme_ids_lengths,
                "scales": scales,
                "sid": sid,
            },
        )[0].squeeze((0, 1))

        return audio

    def _say_phonemes(self, phonemes: str) -> NDArray[np.float32]:
        """Say phonemes."""

        phoneme_ids = self._phonemes_to_ids(phonemes)
        audio: NDArray[np.float32] = self._synthesize_ids_to_raw(phoneme_ids)

        return audio
