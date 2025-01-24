import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore

# Default OnnxRuntime is way to verbose
ort.set_default_logger_severity(4)

VAD_MODEL = "./models/ASR/silero_vad.onnx"
SAMPLE_RATE = 16000


class VAD:
    _initial_h = np.zeros((2, 1, 64)).astype("float32")
    _initial_c = np.zeros((2, 1, 64)).astype("float32")

    def __init__(self, model_path: str = VAD_MODEL, window_size_samples: int = int(SAMPLE_RATE / 10)) -> None:
        """
        Initialize a Voice Activity Detection (VAD) model with an ONNX runtime inference session.
        
        Parameters:
            model_path (str, optional): Path to the ONNX VAD model. Defaults to VAD_MODEL.
            window_size_samples (int, optional): Size of audio chunks to process. 
                Defaults to 1/10th of the sample rate (1600 samples).
        
        Notes:
            - Configures ONNX runtime providers, excluding TensorrtExecutionProvider
            - Sets up inference session with the specified model
            - Initializes internal state variables for processing audio chunks
        """
        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")

        self.ort_sess = ort.InferenceSession(
            model_path,
            sess_options=ort.SessionOptions(),
            providers=providers,
        )
        self.window_size_samples = window_size_samples
        self.sr = SAMPLE_RATE
        self._h = self._initial_h
        self._c = self._initial_c

    def reset(self) -> None:
        """
        Reset the hidden and cell states of the Voice Activity Detection (VAD) model to their initial values.
        
        This method restores the internal LSTM states to their original configuration, effectively clearing any previous 
        processing context and preparing the model for a new audio sequence.
        """
        self._h = self._initial_h
        self._c = self._initial_c

    def process_chunk(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Process an audio chunk using the Voice Activity Detection (VAD) ONNX model.
        
        Prepares the input audio chunk for inference by expanding its dimensions and running the ONNX model with the current hidden and cell states. Updates the internal hidden and cell states after processing.
        
        Parameters:
            chunk (NDArray[np.float32]): A single audio chunk of float32 values to be processed by the VAD model.
        
        Returns:
            NDArray[np.float32]: Processed output from the VAD model after removing singleton dimensions.
        
        Notes:
            - The method modifies the internal hidden state (`_h`) and cell state (`_c`) of the VAD model.
            - Assumes the input chunk is a single-dimensional NumPy array of float32 values.
            - The sample rate is passed as an integer to the ONNX model.
        """
        ort_inputs = {
            "input": np.expand_dims(chunk, 0),
            "h": self._h,
            "c": self._c,
            "sr": np.array(self.sr, dtype="int64"),
        }
        out: NDArray[np.float32]
        out, self._h, self._c = self.ort_sess.run(None, ort_inputs)
        return np.squeeze(out)

    def process_file(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Process an entire audio file for Voice Activity Detection (VAD) using sliding window inference.
        
        This method processes the input audio in fixed-size chunks, running the ONNX model on each chunk
        and tracking the hidden and cell states across the entire sequence. It breaks processing if a 
        final incomplete chunk is encountered.
        
        Parameters:
            audio (NDArray[np.float32]): Input audio time series data as a 1D NumPy float32 array.
        
        Returns:
            NDArray[np.float32]: Processed VAD results for each audio chunk, stacked as a 2D array.
        
        Notes:
            - Resets internal model states before processing
            - Uses a sliding window of size `window_size_samples`
            - Stops processing if the final chunk is smaller than the window size
            - Maintains recurrent model state across chunk processing
        """
        self.reset()
        results_list = []
        for i in range(0, len(audio), self.window_size_samples):
            chunk = audio[i : i + self.window_size_samples]
            if len(chunk) < self.window_size_samples:
                break
            ort_inputs = {
                "input": np.expand_dims(chunk, 0),
                "h": self._h,
                "c": self._c,
                "sr": np.array(self.sr, dtype="int64"),
            }
            out, self._h, self._c = self.ort_sess.run(None, ort_inputs)
            results_list.append(np.squeeze(out))
        results: NDArray[np.float32] = np.stack(results_list, axis=0)
        return results
