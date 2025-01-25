from numba import jit  # type: ignore
import numpy as np

# Constants
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400


@jit(nopython=True)
def _extract_windows(
    audio_padded: np.ndarray,
    window: np.ndarray,
    n_fft: int,
    hop_length: int,
    n_frames: int,
) -> np.ndarray:
    """
    Extract overlapping, windowed frames from a padded audio signal.

    Efficiently pre-allocates a matrix and applies a windowing function to each frame of the audio signal.

    Parameters:
        audio_padded (np.ndarray): Padded input audio signal
        window (np.ndarray): Window function to apply to each frame
        n_fft (int): Number of FFT points defining the frame size
        hop_length (int): Number of samples between successive frames
        n_frames (int): Total number of frames to extract

    Returns:
        np.ndarray: 2D array of shape (n_frames, n_fft) containing windowed audio frames
    """
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
        sr: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        fmin: float = 0.0,
        fmax: float | None = None,
        top_db: float = 80.0,
    ) -> None:
        """
        Initialize a Mel spectrogram calculator with configurable audio processing parameters.

        Parameters:
            sr (int, optional): Sampling rate of the audio signal. Defaults to SAMPLE_RATE.
            n_mels (int, optional): Number of Mel frequency bands. Defaults to N_MELS.
            n_fft (int, optional): Number of FFT points for spectral analysis. Defaults to N_FFT.
            hop_length (int, optional): Number of samples between successive frames. Defaults to HOP_LENGTH.
            win_length (int, optional): Length of the window function. Defaults to WIN_LENGTH.
            fmin (float, optional): Minimum frequency for Mel filterbank. Defaults to 0.0 Hz.
            fmax (float | None, optional): Maximum frequency for Mel filterbank. Defaults to half the sampling rate.
            top_db (float, optional): Decibel threshold for limiting spectrogram dynamic range. Defaults to 80.0 dB.

        Attributes:
            sr (int): Sampling rate of the audio signal.
            n_mels (int): Number of Mel frequency bands.
            n_fft (int): Number of FFT points.
            hop_length (int): Number of samples between frames.
            win_length (int): Length of the window function.
            top_db (float): Decibel threshold for spectrogram.
            mel_filterbank (np.ndarray): Pre-computed Mel filterbank matrix.
            window (np.ndarray): Hanning window function for spectral analysis.
        """
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

    def _create_mel_filterbank(self, fmin: float, fmax: float) -> np.ndarray:
        """
        Create a Mel filterbank matrix for converting linear frequency spectra to Mel scale.

        This method generates a matrix of triangular filters that map linear frequency bins to Mel-scale bands.
        Each filter is a triangular window centered at a specific Mel frequency point, with weights that
        smoothly transition from zero to peak and back to zero.

        Parameters:
            fmin (float): Minimum frequency in Hz for the Mel filterbank
            fmax (float): Maximum frequency in Hz for the Mel filterbank

        Returns:
            np.ndarray: A 2D matrix of shape (n_mels, n_fft//2 + 1) representing Mel filterbank weights,
                        where each row corresponds to a Mel band and each column represents a linear frequency bin.

        Notes:
            - Converts linear frequencies to Mel scale using the formula: mel = 2595 * log10(1 + f/700)
            - Creates triangular filters with normalized weights
            - Ensures efficient computation using vectorized NumPy operations
        """
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

    def compute(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute the Mel spectrogram from an input audio signal.

        This method efficiently transforms the input audio into a Mel spectrogram by performing
        the following steps:
        - Convert input to float32
        - Pad the audio signal
        - Extract and window audio frames
        - Compute Short-Time Fourier Transform (STFT)
        - Calculate power spectrum
        - Apply Mel filterbank
        - Convert to decibel scale

        Parameters:
            audio (np.ndarray): Input audio signal as a NumPy array

        Returns:
            np.ndarray: Mel spectrogram with values in decibel scale,
                        limited to a maximum of `top_db` below the peak

        Notes:
            - Uses Numba-optimized window extraction
            - Applies reflection padding
            - Normalizes spectrogram relative to its maximum value
        """
        # Ensure audio is float32
        audio = np.asarray(audio, dtype=np.float32)

        # Pad audio
        padding = int(self.n_fft // 2)
        audio_padded = np.pad(audio, (padding, padding), mode="reflect")

        # Calculate frames
        n_frames = 1 + (len(audio_padded) - self.n_fft) // self.hop_length

        # Extract and window frames using Numba
        frames = _extract_windows(audio_padded, self.window, self.n_fft, self.hop_length, n_frames)

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
