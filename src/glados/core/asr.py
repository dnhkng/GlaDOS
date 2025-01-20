import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore
import soundfile as sf  # type: ignore

from .mel_spectrogram import MelSpectrogramCalculator

# Default OnnxRuntime is way to verbose
ort.set_default_logger_severity(4)

# Settings
MODEL_PATH = "./models/ASR/nemo-parakeet_tdt_ctc_110m.onnx"
TOKEN_PATH = "./models/ASR/nemo-parakeet_tdt_ctc_110m_tokens.txt"


class AudioTranscriber:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        tokens_file: str = TOKEN_PATH,
    ) -> None:
        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")

        self.session = ort.InferenceSession(
            model_path,
            sess_options=ort.SessionOptions(),
            providers=providers,
        )
        self.vocab = self._load_vocabulary(tokens_file)

        # Standard mel spectrogram parameters
        self.melspectrogram = MelSpectrogramCalculator()

    def _load_vocabulary(self, tokens_file: str) -> dict[int, str]:
        vocab = {}
        with open(tokens_file, encoding="utf-8") as f:
            for line in f:
                token, index = line.strip().split()
                vocab[int(index)] = token
        return vocab

    def process_audio(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Load and process audio file into mel spectrogram with improved normalization.
        """


        mel_spec = self.melspectrogram.compute(audio)

        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)

        # Add batch dimension and ensure correct shape
        mel_spec = np.expand_dims(mel_spec, axis=0)  # [1, n_mels, time]

        return mel_spec

    def decode_output(self, output_logits: NDArray[np.float32]) -> list[str]:
        """Decode model output logits into text with improved token handling."""
        predictions = np.argmax(output_logits, axis=-1)

        decoded_texts = []
        for batch_idx in range(predictions.shape[0]):
            tokens = []
            prev_token = None

            for idx in predictions[batch_idx]:
                if idx in self.vocab:
                    token = self.vocab[idx]
                    # Skip <blk> tokens and repeated tokens
                    if token != "<blk>" and token != prev_token:
                        tokens.append(token)
                        prev_token = token

            # Combine tokens with improved handling
            text = ""
            for token in tokens:
                if token.startswith("▁"):
                    text += " " + token[1:]
                else:
                    text += token

            # Clean up the text
            text = text.strip()
            text = " ".join(text.split())  # Remove multiple spaces

            decoded_texts.append(text)

        return decoded_texts

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        """
        Transcribe an audio file to text.
        """

        # Process audio
        mel_spec = self.process_audio(audio)

        # Prepare length input
        length = np.array([mel_spec.shape[2]], dtype=np.int64)

        # Create input dictionary
        input_dict = {"audio_signal": mel_spec, "length": length}

        # Run inference
        outputs = self.session.run(None, input_dict)

        # Decode output
        transcription = self.decode_output(outputs[0])

        return transcription[0]

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.
        """

        # Load audio
        audio, sr = sf.read(audio_path, dtype='float32')

        return self.transcribe(audio)
