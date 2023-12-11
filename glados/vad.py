import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000


class VAD:
    def __init__(self, model_path, window_size_samples=SAMPLE_RATE / 10):
        self.ort_sess = ort.InferenceSession(model_path)
        self.window_size_samples = window_size_samples
        self.sr = SAMPLE_RATE
        self.reset()

    def reset(self):
        self._h = np.zeros((2, 1, 64)).astype("float32")
        self._c = np.zeros((2, 1, 64)).astype("float32")

    def process_chunk(self, chunk):
        ort_inputs = {
            "input": np.expand_dims(chunk, 0),
            "h": self._h,
            "c": self._c,
            "sr": np.array(self.sr, dtype="int64"),
        }
        out, self._h, self._c = self.ort_sess.run(None, ort_inputs)
        return np.squeeze(out)

    def process_file(self, audio):
        self.reset()
        results = []
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
            results.append(np.squeeze(out))
        results = np.stack(results, axis=0)
        return results
