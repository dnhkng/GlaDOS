from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .model_loader import ModelLoader
from .prompt_processor import PromptProcessor


class TextGenerator:
    def __init__(self, model_loader: ModelLoader, prompt_processor: PromptProcessor):
        """Initialize text generator with model loader and prompt processor"""
        self.model_loader = model_loader
        self.prompt_processor = prompt_processor

    def _compute_vision_features(self, inputs: dict[str, NDArray]) -> NDArray[np.float32]:
        """Compute vision features from image inputs"""
        return self.model_loader.sessions["vision"].run(
            ["image_features"],
            {
                "pixel_values": inputs["pixel_values"],
                "pixel_attention_mask": inputs["pixel_attention_mask"].astype(np.bool_),
            },
        )[0]

    def _update_embeddings(
        self, inputs_embeds: NDArray[np.float32], input_ids: NDArray[np.int64], image_features: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Update input embeddings with vision features"""
        inputs_embeds[input_ids == self.model_loader.model_config.image_token_id] = image_features.reshape(
            -1, image_features.shape[-1]
        )
        return inputs_embeds

    def generate(
        self, prompt: str, image: Image.Image, max_new_tokens: int = 1024, callback: Callable[[str], None] | None = None
    ) -> str:
        inputs = self.prompt_processor.preprocess(prompt, [image])

        batch_size = inputs["input_ids"].shape[0]
        past_key_values = self.model_loader.create_past_key_values(batch_size)
        image_features = None
        input_ids = inputs["input_ids"]

        # import matplotlib.pyplot as plt

        # for i in range(inputs['pixel_values'].shape[1]):
        #     plt.subplot(4, 4, i+1)
        #     plt.imshow(inputs['pixel_values'][0, i].transpose(1, 2, 0))
        #     plt.axis('off')
        # plt.show()

        attention_mask = inputs["attention_mask"]
        position_ids = np.cumsum(inputs["attention_mask"], axis=-1)
        generated_tokens = np.array([[]], dtype=np.int64)

        for _ in range(max_new_tokens):
            inputs_embeds = self.model_loader.sessions["embed"].run(None, {"input_ids": input_ids})[0]

            if image_features is None:
                image_features = self._compute_vision_features(inputs)
                inputs_embeds[inputs["input_ids"] == self.model_loader.model_config.image_token_id] = (
                    image_features.reshape(-1, image_features.shape[-1])
                )

            logits, *present_key_values = self.model_loader.sessions["decoder"].run(
                None,
                {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    **past_key_values,
                },
            )

            input_ids = logits[:, -1].argmax(-1, keepdims=True)
            attention_mask = np.ones_like(input_ids)
            position_ids = position_ids[:, -1:] + 1
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]

            generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)

            if callback:
                callback(self.prompt_processor.tokenizer.decode(input_ids[0]))

            if (input_ids == self.model_loader.model_config.eos_token_id).all():
                break

        return self.prompt_processor.tokenizer.decode(generated_tokens[0])
