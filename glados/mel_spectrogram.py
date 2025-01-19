import numpy as np
from numba import jit  # type: ignore


# Constants
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400


@jit(nopython=True)
def _extract_windows(audio_padded, window, n_fft, hop_length, n_frames):
    """Extract and window frames efficiently"""
    # Pre-allocate output matrix
    frames = np.zeros((n_frames, n_fft), dtype=np.float32)

    for t in range(n_frames):
        start = t * hop_length
        # Extract and window the frame
        frames[t] = audio_padded[start : start + n_fft] * window

    return frames


class MelSpectrogramCalculator:
    def __init__(
        self,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=0.0,
        fmax=None,
        top_db=80.0,
    ):
        if fmax is None:
            fmax = float(sr) / 2

        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.top_db = top_db

        # Pre-compute constants
        self.mel_filterbank = self._create_mel_filterbank(fmin, fmax)
        self.window = np.hanning(n_fft).astype(np.float32)

    def _create_mel_filterbank(self, fmin, fmax):
        """Create mel filterbank matrix"""
        # Center frequencies of each FFT bin
        n_freqs = int(1 + self.n_fft // 2)
        fftfreqs = np.linspace(0, self.sr / 2, n_freqs)

        # Mel points
        mel_f = 2595.0 * np.log10(1.0 + np.array([fmin, fmax]) / 700.0)
        mel_points = np.linspace(mel_f[0], mel_f[1], self.n_mels + 2)
        freq_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        # Initialize weights matrix
        weights = np.zeros((self.n_mels, n_freqs), dtype=np.float32)

        # Vectorized filter creation
        for i in range(self.n_mels):
            f_left = freq_points[i]
            f_center = freq_points[i + 1]
            f_right = freq_points[i + 2]

            # Left side of triangle
            indices = (fftfreqs >= f_left) & (fftfreqs <= f_center)
            weights[i, indices] = (fftfreqs[indices] - f_left) / (f_center - f_left)

            # Right side of triangle
            indices = (fftfreqs >= f_center) & (fftfreqs <= f_right)
            weights[i, indices] = (f_right - fftfreqs[indices]) / (f_right - f_center)

        # Normalize
        enorm = 2.0 / (freq_points[2:] - freq_points[:-2])
        weights *= enorm[:, np.newaxis]

        return weights.astype(np.float32)

    def compute(self, audio):
        """Compute mel spectrogram efficiently"""
        # Ensure audio is float32
        audio = np.asarray(audio, dtype=np.float32)

        # Pad audio
        padding = int(self.n_fft // 2)
        audio_padded = np.pad(audio, (padding, padding), mode="reflect")

        # Calculate frames
        n_frames = 1 + (len(audio_padded) - self.n_fft) // self.hop_length

        # Extract and window frames using Numba
        frames = _extract_windows(
            audio_padded, self.window, self.n_fft, self.hop_length, n_frames
        )

        # Compute FFT (no Numba)
        stft = np.fft.rfft(frames, axis=1).T

        # Power spectrum
        power_spec = np.abs(stft, dtype=np.float32) ** 2

        # Apply mel filterbank
        mel_spec = self.mel_filterbank @ power_spec

        # Convert to dB scale
        ref = np.maximum(1e-10, mel_spec.max())
        mel_spec = 10.0 * np.log10(np.maximum(1e-10, mel_spec / ref))

        return np.maximum(mel_spec, -self.top_db)
