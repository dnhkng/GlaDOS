"""Core speech and audio processing components."""

from .asr import AudioTranscriber
from .mel_spectrogram import MelSpectrogramCalculator
from .tts_glados import Synthesizer
from .vad import VAD

__all__ = ["VAD", "AudioTranscriber", "MelSpectrogramCalculator", "Synthesizer"]
