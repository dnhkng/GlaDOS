from collections.abc import Sequence
import json
from pathlib import Path

from jinja2 import Template
import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer

from glados.Vision.image_preprocessor import ImagePreprocessor


class PromptProcessor:
    """
    A class for processing and tokenizing multimodal (text and image) inputs.

    This class handles the preprocessing of images and text for vision-language models,
    including chat template application and token management.

    Attributes:
        image_seq_len (int): Length of the image token sequence
        image_processor (ImagePreprocessor): Processor for handling image inputs
        tokenizer (Tokenizer): Tokenizer instance for text processing
        chat_template (Template): Jinja2 template for chat formatting
    """

    # Default paths
    IMAGE_CONFIG_PATH = Path("./models/Vision/preprocessor_config.json")
    SPECIAL_TOKENS_PATH = Path("./models/Vision/special_tokens_map.json")
    TOKENIZER_CONFIG_PATH = Path("./models/Vision/tokenizer.json")
    TEMPLATE_PATH = Path("./models/Vision/chat_template.json")

    def __init__(
        self,
        image_seq_len: int = 64,
        image_config_path: Path = IMAGE_CONFIG_PATH,
        special_tokens_path: Path = SPECIAL_TOKENS_PATH,
        tokenizer_config_path: Path = TOKENIZER_CONFIG_PATH,
        template_path: Path = TEMPLATE_PATH,
    ) -> None:
        """
        Initialize the SmolVLM instance.

        Args:
            image_seq_len: Length of image token sequence
            image_config_path: Path to image preprocessor configuration
            special_tokens_path: Path to special tokens mapping
            tokenizer_config_path: Path to tokenizer configuration
            template_path: Path to chat template
        """
        self.image_seq_len = image_seq_len
        self._initialize_special_tokens(special_tokens_path)
        self._initialize_tokenizer_and_processor(tokenizer_config_path, image_config_path, template_path)

    def _initialize_special_tokens(self, special_tokens_path: Path) -> None:
        """Initialize special tokens from configuration file."""
        with open(special_tokens_path) as f:
            special_tokens = json.load(f)
            self.fake_image_token = special_tokens["additional_special_tokens"][0]
            self.image_token = special_tokens["additional_special_tokens"][1]
            self.end_of_utterance = special_tokens["additional_special_tokens"][2]
            self.bos_token = special_tokens["bos_token"]
            self.eos_token = special_tokens["eos_token"]
            self.pad_token = special_tokens["pad_token"]
            self.unknown_token = special_tokens["unk_token"]
        self.global_image_tag = "<global-img>"

    def _initialize_tokenizer_and_processor(
        self, tokenizer_config_path: Path, image_config_path: Path, template_path: Path
    ) -> None:
        """Initialize tokenizer, chat template, and image processor."""
        self.tokenizer = Tokenizer.from_file(str(tokenizer_config_path))

        with open(template_path) as f:
            chat_template_data = json.load(f)
            self.chat_template = Template(chat_template_data["chat_template"])

        with open(image_config_path) as f:
            image_config = json.load(f)
            self.image_processor = ImagePreprocessor(image_config)

    @staticmethod
    def _create_single_image_prompt(
        image_seq_len: int, fake_token_around_image: str, image_token: str, global_img_token: str
    ) -> str:
        """
        Create a prompt string for a single image.

        Args:
            image_seq_len: Length of the image token sequence
            fake_token_around_image: Token to wrap around image sequence
            image_token: Token representing an image
            global_img_token: Global image token

        Returns:
            Formatted prompt string for a single image
        """
        return f"{fake_token_around_image}{global_img_token}{image_token * image_seq_len}{fake_token_around_image}"

    @staticmethod
    def _create_split_image_prompt(
        image_seq_len: int,
        image_rows: int,
        image_cols: int,
        fake_token_around_image: str,
        image_token: str,
        global_img_token: str,
    ) -> str:
        """
        Create a prompt string for an image split into patches.

        Args:
            image_seq_len: Length of the image token sequence
            image_rows: Number of rows in the image grid
            image_cols: Number of columns in the image grid
            fake_token_around_image: Token to wrap around image sequence
            image_token: Token representing an image
            global_img_token: Global image token

        Returns:
            Formatted prompt string for a split image
        """
        text_split_images = ""
        for n_h in range(image_rows):
            for n_w in range(image_cols):
                text_split_images += (
                    f"{fake_token_around_image}<row_{n_h + 1}_col_{n_w + 1}>{image_token * image_seq_len}"
                )
            text_split_images += "\n"

        text_split_images += (
            f"\n{fake_token_around_image}{global_img_token}{image_token * image_seq_len}{fake_token_around_image}"
        )
        return text_split_images

    def get_image_prompt_string(
        self,
        image_rows: int,
        image_cols: int,
        image_seq_len: int,
        fake_token_around_image: str,
        image_token: str,
        global_img_token: str,
    ) -> str:
        """
        Get the appropriate prompt string based on image configuration.

        Args:
            image_rows: Number of rows in the image grid
            image_cols: Number of columns in the image grid
            image_seq_len: Length of the image token sequence
            fake_token_around_image: Token to wrap around image sequence
            image_token: Token representing an image
            global_img_token: Global image token

        Returns:
            Formatted prompt string for the image
        """
        if image_rows == 0 and image_cols == 0:
            return self._create_single_image_prompt(
                image_seq_len,
                fake_token_around_image=fake_token_around_image,
                image_token=image_token,
                global_img_token=global_img_token,
            )
        return self._create_split_image_prompt(
            image_seq_len,
            image_rows,
            image_cols,
            fake_token_around_image,
            image_token,
            global_img_token,
        )

    def apply_chat_template(
        self, messages: list[dict[str, Sequence[object]]], add_generation_prompt: bool = True
    ) -> str:
        """
        Apply the chat template to the messages.

        Args:
            messages: List of message dictionaries containing role and content
            add_generation_prompt: Whether to add a generation prompt

        Returns:
            Formatted chat string
        """
        prompt = self.chat_template.render(messages=messages, add_generation_prompt=add_generation_prompt)
        return prompt

    def _create_prompt_strings(
        self,
        prompt: str,
        image_rows: list[list[int]],
        image_cols: list[list[int]],
        image_seq_len: int,
        fake_token_around_image: str,
        image_token: str,
        global_img_token: str,
    ) -> list[str]:
        """
        Create prompt strings with image tokens properly placed.

        Args:
            prompt: Base prompt text
            image_rows: Number of rows for each image
            image_cols: Number of columns for each image
            image_seq_len: Length of image token sequence
            fake_token_around_image: Token to wrap around image sequence
            image_token: Token representing an image
            global_img_token: Global image token

        Returns:
            List of formatted prompt strings

        Raises:
            ValueError: If image token is missing from text
        """
        prompt_strings = []

        for sample, sample_rows, sample_cols in zip([prompt], image_rows, image_cols, strict=False):
            image_prompt_strings = []

            for n_rows, n_cols in zip(sample_rows, sample_cols, strict=False):
                image_prompt_string = self.get_image_prompt_string(
                    n_rows,
                    n_cols,
                    image_seq_len,
                    image_token=image_token,
                    fake_token_around_image=fake_token_around_image,
                    global_img_token=global_img_token,
                )
                image_prompt_strings.append(image_prompt_string)

            split_sample = sample.split(image_token)
            if len(split_sample) == 0:
                raise ValueError("The image token should be present in the text.")

            final_sample = split_sample[0]
            for i, image_prompt_string in enumerate(image_prompt_strings):
                final_sample += image_prompt_string + split_sample[i + 1]

            prompt_strings.append(final_sample)

        return prompt_strings

    def preprocess(self, text: str | list[str], images: list[NDArray[np.uint8]]) -> dict[str, NDArray[np.uint8]]:
        """
        Preprocess text and images for model input.

        Args:
            text: Input text or list of texts
            images: List of images as numpy arrays

        Returns:
            Dictionary containing preprocessed inputs

        Raises:
            ValueError: If text format is invalid or image count mismatch
        """

        if type(images) is list:
            images = images[0]

        inputs = self.image_processor.preprocess_image(images)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text}
                ]
            },
        ]
        prompt = self.apply_chat_template(messages, add_generation_prompt=True)

        if not isinstance(text, str | list):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        text_list = [text] if isinstance(text, str) else text
        if not all(isinstance(t, str) for t in text_list):
            raise ValueError("All elements in text list must be strings")

        image_rows = inputs.pop("rows", [[0] * len(text_list)])
        image_cols = inputs.pop("cols", [[0] * len(text_list)])

        prompt_strings = self._create_prompt_strings(
            prompt,
            image_rows,
            image_cols,
            self.image_seq_len,
            self.fake_image_token["content"],
            self.image_token["content"],
            self.global_image_tag,
        )

        input_ids = self.tokenizer.encode(prompt_strings[0])

        inputs["input_ids"] = np.asarray([input_ids.ids])
        inputs["attention_mask"] = np.ones_like(inputs["input_ids"])

        return inputs
