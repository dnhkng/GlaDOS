# ruff: noqa: RUF001
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import cache
from itertools import zip_longest
from pathlib import Path
from pickle import load
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore

# Default OnnxRuntime is way to verbose
ort.set_default_logger_severity(4)


@dataclass
class ModelConfig:
    MODEL_NAME: str = "models/TTS/phomenizer_en.onnx"
    PHONEME_DICT_PATH: Path = Path("./models/TTS/lang_phoneme_dict.pkl")
    TOKEN_TO_IDX_PATH: Path = Path("./models/TTS/token_to_idx.pkl")
    IDX_TO_TOKEN_PATH: Path = Path("./models/TTS/idx_to_token.pkl")
    CHAR_REPEATS: int = 3
    MODEL_INPUT_LENGTH: int = 64
    EXPAND_ACRONYMS: bool = True
    USE_CUDA: bool = True


class SpecialTokens(Enum):
    PAD = "_"
    START = "<start>"
    END = "<end>"
    EN_US = "<en_us>"


class Punctuation(Enum):
    PUNCTUATION = "().,:?!/–"
    HYPHEN = "-"
    SPACE = " "

    @classmethod
    @cache
    def get_punc_set(cls) -> set[str]:
        return set(cls.PUNCTUATION.value + cls.HYPHEN.value + cls.SPACE.value)

    @classmethod
    @cache
    def get_punc_pattern(cls) -> re.Pattern[str]:
        return re.compile(f"([{cls.PUNCTUATION.value + cls.SPACE.value}])")


class Phonemizer:
    """Phonemizer class for converting text to phonemes.

    This class uses an ONNX model to predict phonemes for a given text.

    Attributes:
        phoneme_dict: Dictionary of phonemes for each word.
        token_to_idx: Mapping of tokens to indices.
        idx_to_token: Mapping of indices to tokens.
        ort_session: ONNX runtime session for the model.
        special_tokens: Set of special tokens.

    Methods:
        _load_pickle(path: Path) -> dict:
            Load a pickled dictionary from path.

        _unique_consecutive(arr: np.ndarray) -> List[np.ndarray]:
            Equivalent to torch.unique_consecutive for numpy arrays.

        _remove_padding(arr: np.ndarray, padding_value: int = 0) -> List[np.ndarray]:
            Remove padding from an array.

        _trim_to_stop(arr: np.ndarray, end_index: int = 2) -> List[np.ndarray]:
            Trim an array to the stop index.

        _process_model_output(arr: np.ndarray) -> List[np.ndarray]:
            Process the output of the model to get the phoneme indices.

        _expand_acronym(word: str) -> str:
            Expand an acronym into its subwords.

        encode(sentence: Iterable[str]) -> List[int]:
            Map a sequence of symbols to a sequence of indices.

        decode(sequence: np.ndarray) -> str:
            Map a sequence of indices to a sequence of symbols.

        _pad_sequence_fixed(v: List[np.ndarray], target_length: int = ModelConfig.MODEL_INPUT_LENGTH) -> np.ndarray:
            Pad or truncate a list of arrays to a fixed length.

        _get_dict_entry(word: str, lang: str, punc_set: set[str]) -> str | None:
            Get the phoneme entry for a word in the dictionary.

        _get_phonemes(word: str, word_phonemes: Dict[str, Union[str, None]], word_splits: Dict[str, List[str]]) -> str:
            Get the phonemes for a word.

        _clean_and_split_texts(
            texts: List[str],
            punc_set: set[str],
            punc_pattern: re.Pattern
        ) -> tuple[List[List[str]], set[str]]:
            Clean and split texts.

        convert_to_phonemes(texts: List[str], lang: str) -> List[str]:
            Convert a list of texts to phonemes using a phonemizer.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        if config is None:
            config = ModelConfig()
        self.config = config
        self.phoneme_dict: dict[str, str] = self._load_pickle(self.config.PHONEME_DICT_PATH)

        self.phoneme_dict['glados'] = 'ɡlˈɑːdɑːs' # Add GLaDOS to the phoneme dictionary!

        self.token_to_idx = self._load_pickle(self.config.TOKEN_TO_IDX_PATH)
        self.idx_to_token = self._load_pickle(self.config.IDX_TO_TOKEN_PATH)

        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")

        self.ort_session = ort.InferenceSession(
            self.config.MODEL_NAME,
            sess_options=ort.SessionOptions(),
            providers=providers,
        )

        self.special_tokens: set[str] = {
            SpecialTokens.PAD.value,
            SpecialTokens.END.value,
            SpecialTokens.EN_US.value,
        }

    @staticmethod
    def _load_pickle(path: Path) -> dict[str, Any]:
        """Load a p_ickled dictionary from path."""
        with path.open("rb") as f:
            return load(f)  # type: ignore

    @staticmethod
    def _unique_consecutive(arr: list[NDArray[np.int64]]) -> list[NDArray[np.int64]]:
        """
        Equivalent to torch.unique_consecutive for numpy arrays.

        :param arr: Array to process.
        :return: Array with consecutive duplicates removed.
        """
        
        result = []
        for row in arr:
            if len(row) == 0:
                result.append(row)
            else:
                mask = np.concatenate(([True], row[1:] != row[:-1]))
                result.append(row[mask])

        return result

    @staticmethod
    def _remove_padding(arr: list[NDArray[np.int64]], padding_value: int = 0) -> list[NDArray[np.int64]]:
        return [row[row != padding_value] for row in arr]

    @staticmethod
    def _trim_to_stop(arr: list[NDArray[np.int64]], end_index: int = 2) -> list[NDArray[np.int64]]:
        """
        Trims the array to the stop index.

        :param arr: Array to trim.
        :param end_index: Index to stop at.
        """
        result = []
        for row in arr:
            stop_index = np.where(row == end_index)[0]
            if len(stop_index) > 0:
                result.append(row[: stop_index[0] + 1])
            else:
                result.append(row)
        return result

    def _process_model_output(self, arr: list[NDArray[np.int64]]) -> list[NDArray[np.int64]]:
        """
        Processes the output of the model to get the phoneme indices.

        :param arr: Model output.
        :return: List of phoneme indices.
        """

        arr_processed: list[NDArray[np.int64]] = np.argmax(arr[0], axis=2)
        arr_processed = self._unique_consecutive(arr_processed)
        arr_processed = self._remove_padding(arr_processed)
        arr_processed = self._trim_to_stop(arr_processed) 

        return arr_processed

    @staticmethod
    def _expand_acronym(word: str) -> str:
        """
        Expands an acronym into its subwords.
        Now only handles true acronyms since mixed case is handled by SpokenTextConverter.
        """
        # Only split on hyphens if they exist
        if Punctuation.HYPHEN.value in word:
            return word
        
        # For acronyms, just return as is - they're already preprocessed
        return word

    def encode(self, sentence: Iterable[str]) -> list[int]:
        """
        Maps a sequence of symbols for a language to a sequence of indices.

        :param sentence: Sentence (or word) as a sequence of symbols.
        :return: Sequence of token indices.
        """
        sentence = [item for item in sentence for _ in range(self.config.CHAR_REPEATS)]
        sentence = [s.lower() for s in sentence]
        sequence = [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]
        return [self.token_to_idx[SpecialTokens.START.value], *sequence, self.token_to_idx[SpecialTokens.END.value]]

    def decode(self, sequence: NDArray[np.int64]) -> str:
        """
        Maps a sequence of indices to an array of symbols.

        :param sequence: Encoded sequence to be decoded.
        :return: Decoded sequence of symbols.
        """
        decoded = []

        for t in sequence:
            idx = t.item()
            token = self.idx_to_token[idx]
            decoded.append(token)

        result = "".join(d for d in decoded if d not in self.special_tokens)
        return result


    @staticmethod
    def pad_sequence_fixed(v: list[list[int]], target_length: int) -> NDArray[np.int64]:
        """
        Pad or truncate a list of arrays to a fixed length.

        :param v: List of arrays.
        :param target_length: Target length to pad or truncate to.
        :return: Padded array.
        """

        result: NDArray[np.int64] = np.zeros((len(v), target_length), dtype=np.int64)

        for i, seq in enumerate(v):
            length = min(len(seq), target_length)  # Handle both shorter and longer sequences
            result[i, :length] = seq[:length]  # Copy either the full sequence or its truncated version

        return result

    def _get_dict_entry(self, word: str, punc_set: set[str]) -> str | None:
        """
        Gets the phoneme entry for a word in the dictionary.

        :param word: Word to get phoneme entry for.
        :param lang: Language of the word.
        :param punc_set: Set of punctuation characters.
        :return: Phoneme entry for the word.
        """
        if word in punc_set or len(word) == 0:
            return word
        if word in self.phoneme_dict:
            return self.phoneme_dict[word]

        elif word.lower() in self.phoneme_dict:
            return self.phoneme_dict[word.lower()]

        elif word.title() in self.phoneme_dict:
            return self.phoneme_dict[word.title()]
        else:
            return None

    @staticmethod
    def _get_phonemes(
        word: str,
        word_phonemes: dict[str, str | None],
        word_splits: dict[str, list[str]],
    ) -> str:
        """
        Gets the phonemes for a word. If the word is not in the phoneme dictionary, it is split into subwords.

        :param word: Word to get phonemes for.
        :param word_phonemes: Dictionary of word phonemes.
        """
        phons = word_phonemes[word]
        if phons is None:
            subwords = word_splits[word]
            subphons_converted = [word_phonemes[w] for w in subwords]
            phons = "".join([subphon for subphon in subphons_converted if subphon is not None])
        return phons

    def _clean_and_split_texts(
        self, texts: list[str], punc_set: set[str], punc_pattern: re.Pattern[str]
    ) -> tuple[list[list[str]], set[str]]:
        split_text, cleaned_words = [], set[str]()
        for text in texts:
            cleaned_text = "".join(t for t in text if t.isalnum() or t in punc_set)
            split = [s for s in re.split(punc_pattern, cleaned_text) if len(s) > 0]
            split_text.append(split)
            cleaned_words.update(split)
        return split_text, cleaned_words

    def convert_to_phonemes(self, texts: list[str], lang: str = "en_us") -> list[str]:
        """
        Converts a list of texts to phonemes using a phonemizer.

        :param texts: List of texts to convert.
        :param lang: Language of the texts.
        :return: List of phonemes.
        """
        split_text: list[list[str]] = []
        cleaned_words = set[str]()


        punc_set = Punctuation.get_punc_set()
        punc_pattern = Punctuation.get_punc_pattern()

        # Step 1: Preprocess texts
        split_text, cleaned_words = self._clean_and_split_texts(texts, punc_set, punc_pattern)

        # Step 2: Collect dictionary phonemes for words and hyphenated words
        for punct in punc_set:
            self.phoneme_dict[punct] = punct
        word_phonemes = {word: self.phoneme_dict.get(word.lower()) for word in cleaned_words}

        # Step 3: If word is not in dictionary, split it into subwords
        words_to_split = [w for w in cleaned_words if word_phonemes[w] is None]

        word_splits = {
            key: re.split(
                r"([-])",
                self._expand_acronym(word) if self.config.EXPAND_ACRONYMS else word,
            )
            for key, word in zip(words_to_split, words_to_split, strict=False)
        }

        subwords = {w for values in word_splits.values() for w in values if w not in word_phonemes}

        for subword in subwords:
            word_phonemes[subword] = self._get_dict_entry(word=subword, punc_set=punc_set)

        # Step 4: Predict all subwords that are missing in the phoneme dict
        words_to_predict = [
            word for word, phons in word_phonemes.items() if phons is None and len(word_splits.get(word, [])) <= 1
        ]

        if words_to_predict:
            input_batch = [self.encode(word) for word in words_to_predict]
            input_batch_padded: NDArray[np.int64] = self.pad_sequence_fixed(input_batch, self.config.MODEL_INPUT_LENGTH)

            ort_inputs = {self.ort_session.get_inputs()[0].name: input_batch_padded}
            ort_outs = self.ort_session.run(None, ort_inputs)

            ids = self._process_model_output(ort_outs)

            # Step 5: Add predictions to the dictionary
            for id, word in zip(ids, words_to_predict, strict=False):
                word_phonemes[word] = self.decode(id)

        # Step 6: Get phonemes for each word in the text
        phoneme_lists = []
        for text in split_text:
            text_phons = [
                self._get_phonemes(word=word, word_phonemes=word_phonemes, word_splits=word_splits) for word in text
            ]
            phoneme_lists.append(text_phons)

        return ["".join(phoneme_list) for phoneme_list in phoneme_lists]
