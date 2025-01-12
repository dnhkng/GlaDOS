import json
from dataclasses import dataclass
from pathlib import Path
from pickle import load
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import onnxruntime as ort

from . import phonemizer

# Default OnnxRuntime is way to verbose
ort.set_default_logger_severity(4)

# Constants
MAX_WAV_VALUE = 32767.0

# Settings
MODEL_PATH = "./models/glados.onnx"
PHONEME_TO_ID_PATH = Path("./models/phoneme_to_id.pkl")
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

    speaker_id_map: Optional[Dict[str, int]] = None

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> "PiperConfig":
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

    def __init__(self, model_path: str, speaker_id: Optional[int] = None):
        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")

        self.session = ort.InferenceSession(
            model_path,
            sess_options=ort.SessionOptions(),
            providers=providers,
        )
        self.phonemizer = phonemizer.Phonemizer()
        # self.id_map = PHONEME_ID_MAP

        self.id_map = self._load_pickle(PHONEME_TO_ID_PATH)

        try:
            # Load the configuration file
            config_file_path = model_path + ".json"
            with open(config_file_path, "r", encoding="utf-8") as config_file:
                config_dict = json.load(config_file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found at path: {config_file_path}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Configuration file at path: {config_file_path} is not a valid JSON. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while reading the configuration file at path: {config_file_path}. Error: {e}"
            )
        self.config = PiperConfig.from_dict(config_dict)
        self.rate = self.config.sample_rate
        self.speaker_id = (
            self.config.speaker_id_map.get(str(speaker_id), 0)
            if self.config.num_speakers > 1
            else None
        )

    @staticmethod
    def _load_pickle(path: Path) -> dict:
        """Load a pickled dictionary from path."""
        with path.open("rb") as f:
            return load(f)

    def generate_speech_audio(self, text: str) -> np.ndarray:
        phonemes = self._phonemizer(text)
        audio = self.say_phonemes(phonemes)
        return audio

    def say_phonemes(self, phonemes: List[str]) -> np.ndarray:
        audio = []
        for sentence in phonemes:
            audio_chunk = self._say_phonemes(sentence)
            audio.append(audio_chunk)
        if audio:
            return np.concatenate(audio, axis=1).T
        return np.array([])

    def _phonemizer(self, input_text: str) -> List[str]:
        """Converts text to phonemes using espeak-ng."""
        phonemes = self.phonemizer.convert_to_phonemes([input_text], "en_us")

        return phonemes

    def _phonemes_to_ids(self, phonemes: str) -> List[int]:
        """Phonemes to ids."""

        ids: List[int] = list(self.id_map[BOS])

        for phoneme in phonemes:
            if phoneme not in self.id_map:
                continue

            ids.extend(self.id_map[phoneme])
            ids.extend(self.id_map[PAD])
        ids.extend(self.id_map[EOS])

        return ids

    def _synthesize_ids_to_raw(
        self,
        phoneme_ids: List[int],
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
    ) -> bytes:
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
        audio = self.session.run(
            None,
            {
                "input": phoneme_ids_array,
                "input_lengths": phoneme_ids_lengths,
                "scales": scales,
                "sid": sid,
            },
        )[0].squeeze((0, 1))

        return audio

    def _say_phonemes(self, phonemes: str) -> bytes:
        """Say phonemes."""

        phoneme_ids = self._phonemes_to_ids(phonemes)
        audio = self._synthesize_ids_to_raw(phoneme_ids)

        return audio
