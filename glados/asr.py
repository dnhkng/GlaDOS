from typing import Dict, List

import librosa
import numpy as np
import onnxruntime as ort

# Default OnnxRuntime is way to verbose
ort.set_default_logger_severity(4)

# Settings
MODEL_PATH = "./models/nemo-parakeet_tdt_ctc_110m.onnx"
TOKEN_PATH = "./models/nemo-parakeet_tdt_ctc_110m_tokens.txt"

# Constants
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400


class AudioTranscriber:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        tokens_file: str = TOKEN_PATH,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.sample_rate = sample_rate
        
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
        self.n_mels = N_MELS
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH
        self.win_length = WIN_LENGTH

    def _load_vocabulary(self, tokens_file: str) -> Dict[int, str]:
        vocab = {}
        with open(tokens_file, "r", encoding="utf-8") as f:
            for line in f:
                token, index = line.strip().split()
                vocab[int(index)] = token
        return vocab

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Load and process audio file into mel spectrogram with improved normalization.
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            power=2.0,
        )

        # Convert to log scale with improved scaling
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)

        # Add batch dimension and ensure correct shape
        mel_spec = np.expand_dims(mel_spec, axis=0)  # [1, n_mels, time]

        return mel_spec

    def decode_output(self, output_logits: np.ndarray) -> List[str]:
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

    def transcribe(self, audio: np.ndarray) -> str:
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
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        return self.transcribe(audio)
