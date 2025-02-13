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
    A class for processing and tokenizing multimodal (text and multiple images) inputs.
    """

    # Default paths remain the same
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
        """Initialize as before"""
        self.image_seq_len = image_seq_len
        self._initialize_special_tokens(special_tokens_path)
        self._initialize_tokenizer_and_processor(tokenizer_config_path, image_config_path, template_path)

    def _initialize_special_tokens(self, special_tokens_path: Path) -> None:
        """Initialize special tokens from configuration file."""
        # Same as before
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
        # Same as before
        self.tokenizer = Tokenizer.from_file(str(tokenizer_config_path))

        with open(template_path) as f:
            chat_template_data = json.load(f)
            self.chat_template = Template(chat_template_data["chat_template"])

        with open(image_config_path) as f:
            image_config = json.load(f)
            self.image_processor = ImagePreprocessor(image_config)

    def _create_single_image_prompt(
        self, image_seq_len: int, fake_token_around_image: str, image_token: str, global_img_token: str
    ) -> str:
        """Create a prompt string for a single image."""
        # Same as before
        return f"{fake_token_around_image}{global_img_token}{image_token * image_seq_len}{fake_token_around_image}"

    def _create_split_image_prompt(
        self,
        image_seq_len: int,
        image_rows: int,
        image_cols: int,
        fake_token_around_image: str,
        image_token: str,
        global_img_token: str,
    ) -> str:
        """Create a prompt string for an image split into patches."""
        # Same as before
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
        """Get the appropriate prompt string based on image configuration."""
        # Same as before
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
        """Apply the chat template to the messages."""
        # Same as before
        prompt = self.chat_template.render(messages=messages, add_generation_prompt=add_generation_prompt)
        return prompt

    def _create_prompt_strings(
        self,
        prompt: str,
        image_configs: list[tuple[list[int], list[int]]],  # List of (rows, cols) tuples for each image
        image_seq_len: int,
        fake_token_around_image: str,
        image_token: str,
        global_img_token: str,
    ) -> list[str]:
        """
        Create prompt strings with image tokens properly placed for multiple images.

        Args:
            prompt: Base prompt text
            image_configs: List of (rows, cols) tuples for each image
            image_seq_len: Length of image token sequence
            fake_token_around_image: Token to wrap around image sequence
            image_token: Token representing an image
            global_img_token: Global image token

        Returns:
            List of formatted prompt strings
        """
        prompt_strings = []
        split_sample = prompt.split(image_token)
        
        if len(split_sample) - 1 != len(image_configs):
            raise ValueError(f"Number of image tokens ({len(split_sample) - 1}) does not match number of images ({len(image_configs)})")

        final_sample = split_sample[0]
        for i, (rows, cols) in enumerate(image_configs):
            # Extract integer values from the rows and cols lists
            current_rows = rows[0][0] if rows and rows[0] else 0
            current_cols = cols[0][0] if cols and cols[0] else 0
            image_prompt_string = self.get_image_prompt_string(
                current_rows,
                current_cols,
                image_seq_len,
                fake_token_around_image=fake_token_around_image,
                image_token=image_token,
                global_img_token=global_img_token,
            )
            final_sample += image_prompt_string + split_sample[i + 1]

        prompt_strings.append(final_sample)
        return prompt_strings

    def preprocess(self, text: str | list[str], images: list[NDArray[np.uint8]]) -> dict[str, NDArray[np.uint8]]:
        """
        Preprocess text and multiple images for model input.

        Args:
            text: Input text or list of texts
            images: List of images as numpy arrays

        Returns:
            Dictionary containing preprocessed inputs
        """
        if not isinstance(images, list):
            images = [images]

        # Process all images
        all_inputs = [self.image_processor.preprocess_image(img) for img in images]
        


        # Create message content with multiple images
        content = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]
        prompt = self.apply_chat_template(messages, add_generation_prompt=True)

        if not isinstance(text, str | list):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        text_list = [text] if isinstance(text, str) else text
        if not all(isinstance(t, str) for t in text_list):
            raise ValueError("All elements in text list must be strings")

        image_configs = [(inp['rows'], inp['cols']) for inp in all_inputs]
        prompt_strings = self._create_prompt_strings(
            prompt,
            image_configs,
            self.image_seq_len,
            self.fake_image_token["content"],
            self.image_token["content"],
            self.global_image_tag,
        )

        input_ids = self.tokenizer.encode(prompt_strings[0])

        input_ids = np.asarray([input_ids.ids])
        attention_mask = np.ones_like(input_ids)

        return all_inputs, input_ids, attention_mask