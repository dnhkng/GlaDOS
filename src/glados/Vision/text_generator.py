from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from .model_loader import ModelLoader
from .prompt_processor import PromptProcessor


class TextGenerator:
    MODEL_PATH = Path("./models/Vision")

    def __init__(self) -> None:
        """Initialize text generator with model loader and prompt processor"""
        self.model_loader = ModelLoader(model_dir=self.MODEL_PATH)
        self.prompt_processor = PromptProcessor()

    def generate(
        self,
        prompt: str,
        images: list[Image.Image],
        max_new_tokens: int = 1024,
        callback: Callable[[str], None] | None = None,
        batch_size: int = 1,
    ) -> str:
        images_patches, input_ids, attention_mask = self.prompt_processor.preprocess(
            prompt, [np.asarray(image) for image in images]
        )

        past_key_values = self.model_loader.create_past_key_values(batch_size)
        image_features = None

        position_ids = np.cumsum(attention_mask, axis=-1)
        generated_tokens = np.array([[]], dtype=np.int64)

        for _ in range(max_new_tokens):
            inputs_embeds = self.model_loader.sessions["embed"].run(None, {"input_ids": input_ids})[0]

            if image_features is None:
                ## Only compute vision features if not already computed
                images_features = []
                for image_patches in images_patches:
                    features = self.model_loader.sessions["vision"].run(
                        ["image_features"],
                        {
                            "pixel_values": image_patches["pixel_values"],
                            "pixel_attention_mask": image_patches["pixel_attention_mask"].astype(np.bool_),
                        },
                    )[0]

                    images_features.append(features)

                image_features = np.concatenate(images_features, axis=0)

                inputs_embeds[input_ids == self.model_loader.model_config.image_token_id] = image_features.reshape(
                    -1, image_features.shape[-1]
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

        output: str = self.prompt_processor.tokenizer.decode(generated_tokens[0])

        return output
