from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort


@dataclass
class ModelConfig:
    num_key_value_heads: int
    head_dim: int 
    num_hidden_layers: int
    eos_token_id: int
    image_token_id: int

class ModelLoader:
    CONFIG_PATH:Path = Path("models/Vision/config.json")
    
    def __init__(self,model_dir: Path, config_path: Path = CONFIG_PATH) -> None:
        """Initialize model loader with config path and model directory"""
        self.model_config = self._load_config(config_path)
        self.sessions = self._initialize_sessions(model_dir)


    def _load_config(self, config_path: Path) -> ModelConfig:
        """Load model configuration from JSON"""
        with open(config_path) as f:
            config = json.load(f)
            return ModelConfig(
                num_key_value_heads=config["text_config"]["num_key_value_heads"],
                head_dim=config["text_config"]["head_dim"],
                num_hidden_layers=config["text_config"]["num_hidden_layers"], 
                eos_token_id=config["text_config"]["eos_token_id"],
                image_token_id=config["image_token_id"]
            )
        

    def _initialize_sessions(self, model_dir: Path) -> dict[str, ort.InferenceSession]:
        """Initialize ONNX Runtime sessions"""
        providers = [p for p in ort.get_available_providers() 
                    if p not in ("TensorrtExecutionProvider", "CoreMLExecutionProvider")]

        return {
            'vision': ort.InferenceSession(str(model_dir / "vision_encoder.onnx"), providers=providers),
            'embed': ort.InferenceSession(str(model_dir / "embed_tokens.onnx"), providers=providers),
            'decoder': ort.InferenceSession(str(model_dir / "decoder_model_merged.onnx"), providers=providers)
        }

    def create_past_key_values(self, batch_size: int) -> dict[str, NDArray[np.float32]]:
        """Create initial past key values for decoder"""
        return {
            f'past_key_values.{layer}.{kv}': np.zeros(
                [batch_size, self.model_config.num_key_value_heads, 0, self.model_config.head_dim], 
                dtype=np.float32
            )
            for layer in range(self.model_config.num_hidden_layers)
            for kv in ('key', 'value')
        }

    def __del__(self) -> None:
        """Clean up ONNX sessions"""
        if hasattr(self, "sessions"):
            del self.sessions