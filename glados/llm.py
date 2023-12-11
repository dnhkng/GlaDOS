from llama_cpp import Llama

N_GPU_LAYERS = 1000
N_CTX = 2048
TENSOR_SPLIT = [1, 0]
MAX_TOKENS = 512
STOP_SYMBOLS = ["}"]  # ["USER:", "\n"]
STREAM = True
MODEL_PATH = "../models/mistral-7b-instruct-v0.1.Q8_0.gguf"


class LLM:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        n_gpu_layers: int = N_GPU_LAYERS,
        n_ctx: int = N_CTX,
        tensor_split: list = TENSOR_SPLIT,
        max_tokens: int = MAX_TOKENS,
        stop_symbols: list = STOP_SYMBOLS,
        stream: bool = STREAM,
    ):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.tensor_split = tensor_split
        self.max_tokens = max_tokens
        self.stop_symbols = stop_symbols
        self.stream = stream
        self.llama = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            tensor_split=self.tensor_split,
        )

    def is_pause(self, token: str) -> bool:
        """
        Checks if a token marks the end of a sentence.
        """
        return token in [".", "?", "!", ":"]

    # def generate_response(self, prompt: str) -> str:
    #     stream = self.llama(
    #         prompt,
    #         max_tokens=self.max_tokens,
    #         stop=self.stop_symbols,
    #         stream=self.stream,
    #     )

    #     response_text = ""
    #     sentence_buffer = ""
    #     for output in stream:
    #         token = output["choices"][0]["text"]
    #         if self.is_pause(token):
    #             sentence_buffer += token
    #             response_text += sentence_buffer
    #             sentence_buffer = ""
    #         else:
    #             sentence_buffer += token

    #     return response_text

    def generate_response(self, prompt: str) -> str:
        stream = self.llama(
            prompt,
            max_tokens=self.max_tokens,
            stop=self.stop_symbols,
            stream=self.stream,
        )

        response_text = ""
        for output in stream:
            token = output["choices"][0]["text"]

            response_text += token

        return response_text

    def response_sentences(self, prompt: str) -> str:
        stream = self.llama(
            prompt,
            max_tokens=self.max_tokens,
            stop=self.stop_symbols,
            stream=self.stream,
        )

        response_text = ""
        sentence_buffer = ""
        for output in stream:
            token = output["choices"][0]["text"]
            if self.is_pause(token):
                sentence_buffer += token
                yield sentence_buffer
                response_text += sentence_buffer
                sentence_buffer = ""
            else:
                sentence_buffer += token
        return sentence_buffer
