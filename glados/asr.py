import ctypes

import numpy as np
from loguru import logger

from . import whisper_cpp_wrapper

LANG = "en"
WORD_LEVEL_TIMINGS = False
BEAM_SEARCH = True

# Translate ggml log levels to loguru levels
# From ggml.h:
# enum ggml_log_level {
#   GGML_LOG_LEVEL_ERROR = 2,
#   GGML_LOG_LEVEL_WARN  = 3,
#   GGML_LOG_LEVEL_INFO  = 4,
#   GGML_LOG_LEVEL_DEBUG = 5
#};
_log_at_level = {2: logger.error, 3: logger.warning, 4: logger.info, 5: logger.debug}

def _unlog(level: ctypes.c_int, text: ctypes.c_char_p, user_data: ctypes.c_void_p) -> None:
        """Callback function to log output from whisper.cpp to the loguru log.
        """
        _log_at_level[level](text.rstrip())

_unlog_func = whisper_cpp_wrapper.ggml_log_callback(_unlog)


class ASR:
    """Wrapper around whisper.cpp, which is a C++ implementation of the Whisper
    speech recognition model.

    This class is not thread-safe, so you should only use it from one thread.

    Args:
        model: The path to the model file to use.
    """

    def __init__(self, model: str) -> None:
        # set whisper's logging to use the _unlog callback function, so that messages
        # are logged to the loguru logger instead of stdout/stderr
        whisper_cpp_wrapper.whisper_log_set(_unlog_func, ctypes.c_void_p(0))
        self.ctx = whisper_cpp_wrapper.whisper_init_from_file(model.encode("utf-8"))
        self.params = self._whisper_cpp_params(
            language=LANG,
            word_level_timings=WORD_LEVEL_TIMINGS,
            beam_search=BEAM_SEARCH,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using the given parameters.

        Any is whisper_cpp.WhisperParams, but we can't import that here
        because it's a C++ class.
        """

        # Run the model
        whisper_cpp_audio = audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result = whisper_cpp_wrapper.whisper_full(
            self.ctx, self.params, whisper_cpp_audio, len(audio)
        )
        if result != 0:
            raise Exception(f"Error from whisper.cpp: {result}")

        # Get the text
        n_segments = whisper_cpp_wrapper.whisper_full_n_segments((self.ctx))
        text = [
            whisper_cpp_wrapper.whisper_full_get_segment_text((self.ctx), i)
            for i in range(n_segments)
        ]

        if not text:
            return ""
        else:
            return text[0].decode("utf-8")[1:]

    def __del__(self):
        """
        Free the C++ object when this Python object is garbage collected.
        """
        whisper_cpp_wrapper.whisper_free(self.ctx)

    def _whisper_cpp_params(
        self,
        language: str,
        word_level_timings: bool,
        beam_search: bool = True,
        print_realtime=False,
        print_progress=False,
    ):
        if beam_search:
            params = whisper_cpp_wrapper.whisper_full_default_params(
                whisper_cpp_wrapper.WHISPER_SAMPLING_BEAM_SEARCH
            )
        else:
            params = whisper_cpp_wrapper.whisper_full_default_params(
                whisper_cpp_wrapper.WHISPER_SAMPLING_GREEDY
            )

        params.print_realtime = print_realtime
        params.print_progress = print_progress
        params.language = whisper_cpp_wrapper.String(language.encode("utf-8"))
        params.max_len = ctypes.c_int(100)
        params.max_len = 1 if word_level_timings else 0
        params.token_timestamps = word_level_timings
        params.no_timestamps = True
        return params
