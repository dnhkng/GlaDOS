import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore

from .phonemizer import Phonemizer

# Default OnnxRuntime is way to verbose
ort.set_default_logger_severity(3)

# Settings
MODEL_PATH = "./models/TTS/kokoro-v1.0.fp16.onnx"
VOICES_PATH = "./models/TTS/kokoro-voices-v1.0.bin"
DEFAULT_VOICE = "af_alloy"
MAX_PHONEME_LENGTH = 510
SAMPLE_RATE = 24000


def get_voices(path: str = VOICES_PATH) -> list[str]:
    voices = np.load(path)
    return list(voices.keys())


class Synthesizer:
    def __init__(self, model_path: str = MODEL_PATH, voice: str = DEFAULT_VOICE) -> None:
        self.rate = SAMPLE_RATE
        self.voices: dict[str, NDArray[np.float32]] = np.load(VOICES_PATH)
        self.vocab = self._get_vocab()
        self.voice = voice
        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")

        self.session = ort.InferenceSession(
            model_path,
            sess_options=ort.SessionOptions(),
            providers=providers,
        )
        self.phonemizer = Phonemizer()

    def generate_speech_audio(self, text: str, voice: str | None = None) -> NDArray[np.float32]:
        """
        Convert input text to synthesized speech audio.

        Converts the input text to phonemes using the internal phonemizer, then generates audio from those phonemes.
        The result is returned as a NumPy array of 32-bit floating point audio samples.

        Parameters:
            text (str): The text to be converted to speech

        Returns:
            NDArray[np.float32]: An array of audio samples representing the synthesized speech
        """
        if voice is None:
            voice = self.voice
        else:
            assert voice in self.voices.keys(), f"voice '{voice}' not found"
        phonemes = self.phonemizer.convert_to_phonemes([text], "en_us")
        ids = self._phonemes_to_ids(phonemes[0])
        audio = self._synthesize_ids_to_audio(ids, voice)
        return np.array(audio, dtype=np.float32)

    @staticmethod
    def _get_vocab() -> dict[str, int]:
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»“” '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        symbols = [_pad, *_punctuation, *_letters, *_letters_ipa]
        dicts = {}
        for i in range(len(symbols)):
            dicts[symbols[i]] = i
        return dicts

    def _phonemes_to_ids(self, phonemes: str) -> list[int]:
        if len(phonemes) > MAX_PHONEME_LENGTH:
            raise ValueError(f"text is too long, must be less than {MAX_PHONEME_LENGTH} phonemes")
        return [i for i in map(self.vocab.get, phonemes) if i is not None]

    def _synthesize_ids_to_audio(self, ids: list[int], voice: str | None = None) -> NDArray[np.float32]:
        if voice is None:
            voice = self.voice
        else:
            assert voice in self.voices, f"voice '{voice}' not found"
        voice_vector = self.voices[voice]
        voice_array = voice_vector[len(ids)]

        tokens = [[0, *ids, 0]]
        speed = 1.0
        audio = self.session.run(
            None,
            {
                "tokens": tokens,
                "style": voice_array,
                "speed": np.ones(1, dtype=np.float32) * speed,
            },
        )[0]
        return np.array(audio[:-8000], dtype=np.float32)  # Remove the last 1/3 of a second, as kokoro adds a lot of silence at the end
