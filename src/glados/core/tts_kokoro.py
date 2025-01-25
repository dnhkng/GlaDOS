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
VOICES_PATH = "./models/TTS/voices.bin"
MODEL_PATH = "./models/TTS/kokoro-v0_19.onnx"
PHONEME_TO_ID_PATH = Path("./models/TTS/phoneme_to_id.pkl")
USE_CUDA = True


MAX_PHONEME_LENGTH = 510
SAMPLE_RATE = 24000

# Conversions
PAD = "_"  # padding (0)
BOS = "^"  # beginning of sentence
EOS = "$"  # end of sentence



def get_vocab() -> dict[str, int]:
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»“” '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad, *_punctuation, *_letters, *_letters_ipa]
    dicts = {}
    for i in range(len(symbols)):
        dicts[symbols[i]] = i
    return dicts
VOCAB = get_vocab()


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
        """
        Create a PiperConfig instance from a configuration dictionary.

        This class method parses a configuration dictionary and constructs a PiperConfig object with specified
        parameters. It allows flexible configuration by providing default values for optional inference settings.

        Parameters:
            config (dict[str, Any]): A dictionary containing configuration parameters for the text-to-speech model.
                Required keys:
                    - "num_symbols": Number of unique phoneme symbols
                    - "num_speakers": Total number of available speakers
                    - "audio": Dictionary containing "sample_rate"
                    - "espeak": Dictionary containing "voice"
                    - "phoneme_id_map": Mapping of phonemes to their corresponding IDs

                Optional keys:
                    - "inference": Dictionary with optional scaling parameters
                        - "noise_scale": Controls audio noise (default: 0.667)
                        - "length_scale": Controls speech duration (default: 1.0)
                        - "noise_w": Additional noise parameter (default: 0.8)
                    - "speaker_id_map": Mapping of speaker names to IDs (default: empty dictionary)

        Returns:
            PiperConfig: A configured PiperConfig instance with the specified parameters.

        Example:
            config = {
                "num_symbols": 100,
                "num_speakers": 5,
                "audio": {"sample_rate": 22050},
                "espeak": {"voice": "en-us"},
                "phoneme_id_map": {...},
                "inference": {
                    "noise_scale": 0.5,
                    "length_scale": 1.2
                }
            }
            piper_config = PiperConfig.from_dict(config)
        """
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
        """
        Initialize the text-to-speech synthesizer with a specified model and optional speaker configuration.

        Parameters:
            model_path (str, optional): Path to the ONNX model file. Defaults to the predefined MODEL_PATH.
            speaker_id (int | None, optional): Identifier for the desired speaker voice. Defaults to None.

        Raises:
            FileNotFoundError: If the configuration file cannot be found.
            ValueError: If the configuration file contains invalid JSON.
            RuntimeError: If an unexpected error occurs while reading the configuration file.

        Notes:
            - Removes TensorRT execution provider from available providers to prevent potential compatibility issues.
            - Loads the ONNX model using the specified providers.
            - Initializes a phonemizer and loads phoneme-to-ID mapping.
            - Configures speaker settings based on the model's configuration.
        """

        self.voices: NDArray[np.float32] = np.load(VOICES_PATH)

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
        """
        Load a pickled dictionary from the specified file path.

        Args:
            path (Path): The file path to the pickle file containing the dictionary.

        Returns:
            dict[str, Any]: A dictionary loaded from the pickle file, ensuring type consistency.

        Raises:
            FileNotFoundError: If the specified pickle file does not exist.
            PickleError: If there are issues during pickle deserialization.
        """
        with path.open("rb") as f:
            return dict(load(f))

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        """
        Convert input text to synthesized speech audio.

        Converts the input text to phonemes using the internal phonemizer, then generates audio from those phonemes.
        The result is returned as a NumPy array of 32-bit floating point audio samples.

        Parameters:
            text (str): The text to be converted to speech

        Returns:
            NDArray[np.float32]: An array of audio samples representing the synthesized speech
        """
        phonemes = self._phonemizer(text)
        audio = self._say_phonemes(phonemes)
        return np.array(audio, dtype=np.float32)

    def _say_phonemes(self, phonemes: list[str]) -> NDArray[np.float32]:
        """
        Convert a list of phoneme sentences to synthesized audio.

        Generates audio for each phoneme sentence and concatenates the results into a single audio array.

        Parameters:
            phonemes (list[str]): A list of phoneme sentences to convert to speech.

        Returns:
            NDArray[np.float32]: A numpy array containing the synthesized audio, with each sentence concatenated.
            Returns an empty float32 numpy array if no audio could be generated.

        Notes:
            - Processes each phoneme sentence individually using _say_phonemes()
            - Concatenates audio chunks along the time axis
            - Transposes the final audio array to ensure correct audio format
        """
        audio_list = []
        for sentence in phonemes:
            audio_chunk = self._say_phonemes(sentence)
            audio_list.append(audio_chunk)
        if audio_list:
            audio: NDArray[np.float32] = np.concatenate(audio_list, axis=1).T
            return audio
        return np.array([], dtype=np.float32)

    def _phonemizer(self, input_text: str) -> list[str]:
        """
        Convert input text to phonemes using espeak-ng phonemization.

        This method transforms plain text into a sequence of phonetic representations
        using the English (US) phoneme set. It leverages the pre-configured phonemizer
        to break down text into its constituent phonetic components.

        Parameters:
            input_text (str): The text to be converted into phonemes.

        Returns:
            list[str]: A list of phoneme strings representing the input text's pronunciation.

        Example:
            phonemes = synthesizer._phonemizer("Hello world")
            # Might return something like ['hh', 'AH0', 'l', 'oW1', 'r', 'AO1', 'l', 'd']
        """
        phonemes = self.phonemizer.convert_to_phonemes([input_text], "en_us")

        return phonemes

    def _phonemes_to_ids(self, phonemes: str) -> list[int]:
        """
        Convert a sequence of phonemes to their corresponding integer IDs.

        This method transforms phonemes into a list of integer identifiers used by the text-to-speech model. It handles the conversion by:
        - Starting with the beginning-of-sentence (BOS) marker
        - Mapping each valid phoneme to its corresponding ID
        - Adding padding between phonemes
        - Appending the end-of-sentence (EOS) marker

        Parameters:
            phonemes (str): A string of phonemes to convert

        Returns:
            list[int]: A list of integer IDs representing the input phonemes, including sentence boundary markers and padding

        Notes:
            - Skips phonemes not found in the ID mapping
            - Adds padding between each valid phoneme
            - Ensures consistent input format for the speech synthesis model
        """

        ids: list[int] = list(self.id_map[BOS])

        for phoneme in phonemes:
            if phoneme not in self.id_map:
                continue

            ids.extend(self.id_map[phoneme])
            ids.extend(self.id_map[PAD])
        ids.extend(self.id_map[EOS])

        return ids


    def tokenize(self, phonemes: str) -> list[int]:
        if len(phonemes) > MAX_PHONEME_LENGTH:
            raise ValueError(
                f"text is too long, must be less than {MAX_PHONEME_LENGTH} phonemes"
            )
        return [i for i in map(VOCAB.get, phonemes) if i is not None]


    def _synthesize_ids_to_raw(
            self, phonemes: list[int], voice: NDArray[np.float32], speed: float
        ) -> NDArray[np.float32]:
        # if len(phonemes) > MAX_PHONEME_LENGTH:
        #     logger.warning(
        #         f"Phonemes are too long, truncating to {MAX_PHONEME_LENGTH} phonemes"
        #     )
        # phonemes = phonemes[:MAX_PHONEME_LENGTH]
        # start_t = time.time()
        # tokens = self.tokenizer.tokenize(phonemes)
        # assert (
        #     len(tokens) <= MAX_PHONEME_LENGTH
        # ), f"Context length is {MAX_PHONEME_LENGTH}, but leave room for the pad token 0 at the start & end"

        voice = voice[len(phonemes)]
        tokens = [[0, *phonemes, 0]]

        audio = self.session.run(
            None,
            {
                "tokens": tokens, "style": voice, "speed": np.ones(1, dtype=np.float32) * speed
            },
        )[0]
        # audio_duration = len(audio) / SAMPLE_RATE
        # create_duration = time.time() - start_t
        # speedup_factor = audio_duration / create_duration
        # log.debug(
        #     f"Created audio in length of {audio_duration:.2f}s for {len(phonemes)} phonemes in {create_duration:.2f}s (More than {speedup_factor:.2f}x real-time)"
        # )
        return audio #, SAMPLE_RATE



    def _say_phonemes(self, phonemes: str) -> NDArray[np.float32]:
        """
        Convert a string of phonemes to synthesized audio.

        This method transforms phonemes into their corresponding numeric IDs and then generates raw audio using the VITS model.

        Parameters:
            phonemes (str): A string containing phonemes to be synthesized into speech.

        Returns:
            NDArray[np.float32]: A NumPy array representing the synthesized audio waveform.

        Example:
            synthesizer = Synthesizer()
            audio = synthesizer._say_phonemes("h ɛ l oʊ")  # Generates audio for "hello"
        """

        phoneme_ids = self._phonemes_to_ids(phonemes)
        audio: NDArray[np.float32] = self._synthesize_ids_to_raw(phoneme_ids)

        return audio
