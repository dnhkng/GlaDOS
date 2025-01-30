"""
This module provides functionality to convert phonemizer models to ONNX format.

It was used to convert the model from:
https://github.com/NeuralVox/OpenPhonemizer

Classes:
    ModelType (Enum): Enum class representing different types of models.
    Model (ABC): Abstract base class for phonemizer models.
    ForwardTransformer (Model): Implementation of a forward transformer model for phonemization.

Functions:
    ModelType.is_autoregressive: Checks if the model type is autoregressive.
    Model.generate: Abstract method to generate phonemes for a text batch.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from dp.model.utils import PositionalEncoding, _generate_square_subsequent_mask, _make_len_mask, get_dedup_tokens
from dp.preprocessing.text import Preprocessor
import torch
import torch.nn as nn
from torch.nn import LayerNorm, TransformerEncoder, TransformerEncoderLayer


class ModelType(Enum):
    TRANSFORMER = "transformer"
    AUTOREG_TRANSFORMER = "autoreg_transformer"

    def is_autoregressive(self) -> bool:
        """
        Returns: bool: Whether the model is autoregressive.
        """
        return self in {ModelType.AUTOREG_TRANSFORMER}


class Model(torch.nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def generate(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates phonemes for a text batch

        Args:
          batch (Dict[str, torch.Tensor]): Dictionary containing 'text' (tokenized text tensor),
                       'text_len' (text length tensor),
                       'start_index' (phoneme start indices for AutoregressiveTransformer)

        Returns:
          Tuple[torch.Tensor, torch.Tensor]: The predictions. The first element is a tensor (phoneme tokens)
          and the second element  is a tensor (phoneme token probabilities)
        """
        pass


class ForwardTransformer(Model):
    def __init__(
        self,
        encoder_vocab_size: int,
        decoder_vocab_size: int,
        d_model: int = 512,
        d_fft: int = 1024,
        layers: int = 4,
        dropout: float = 0.1,
        heads: int = 1,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(encoder_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_fft, dropout=dropout, activation="relu"
        )
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=layers, norm=encoder_norm)

        self.fc_out = nn.Linear(d_model, decoder_vocab_size)

    def forward(self, input_tensor: torch.tensor) -> torch.Tensor:  # shape: [N, T]
        """
        Forward pass of the model on a data batch.

        Args:
         batch (Dict[str, torch.Tensor]): Input batch entry 'text' (text tensor).

        Returns:
          Tensor: Predictions.
        """

        x = input_tensor.transpose(0, 1)  # shape: [T, N]

        # x = x.transpose(0, 1)        # shape: [T, N]
        src_pad_mask = _make_len_mask(x).to(x.device)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_pad_mask)
        x = self.fc_out(x)
        x = x.transpose(0, 1)
        return x

    @torch.jit.export
    def generate(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference pass on a batch of tokenized texts.

        Args:
          batch (Dict[str, torch.Tensor]): Input batch with entry 'text' (text tensor).

        Returns:
          Tuple: The first element is a Tensor (phoneme tokens) and the second element
                 is a tensor (phoneme token probabilities).
        """

        with torch.no_grad():
            x = self.forward(batch)
        tokens, logits = get_dedup_tokens(x)
        return tokens, logits

    @classmethod
    def from_config(cls, config: dict) -> "ForwardTransformer":
        preprocessor = Preprocessor.from_config(config)
        return ForwardTransformer(
            encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
            decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
            d_model=config["model"]["d_model"],
            d_fft=config["model"]["d_fft"],
            layers=config["model"]["layers"],
            dropout=config["model"]["dropout"],
            heads=config["model"]["heads"],
        )


class AutoregressiveTransformer(Model):
    def __init__(
        self,
        encoder_vocab_size: int,
        decoder_vocab_size: int,
        end_index: int,
        d_model: int = 512,
        d_fft: int = 1024,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        dropout: float = 0.1,
        heads: int = 1,
    ) -> None:
        super().__init__()

        self.end_index = end_index
        self.d_model = d_model
        self.encoder = nn.Embedding(encoder_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.decoder = nn.Embedding(decoder_vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=d_fft,
            dropout=dropout,
            activation="relu",
        )
        self.fc_out = nn.Linear(d_model, decoder_vocab_size)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:  # shape: [N, T]
        """
        Foward pass of the model on a data batch.

        Args:
          batch (Dict[str, torch.Tensor]): Input batch with entries 'text' (text tensor) and 'phonemes'
                                           (phoneme tensor for teacher forcing).

        Returns:
          Tensor: Predictions.
        """

        src = batch["text"]
        trg = batch["phonemes"][:, :-1]

        src = src.transpose(0, 1)  # shape: [T, N]
        trg = trg.transpose(0, 1)

        trg_mask = _generate_square_subsequent_mask(len(trg)).to(trg.device)

        src_pad_mask = _make_len_mask(src).to(trg.device)
        trg_pad_mask = _make_len_mask(trg).to(trg.device)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(
            src,
            trg,
            src_mask=None,
            tgt_mask=trg_mask,
            memory_mask=None,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=trg_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        output = self.fc_out(output)
        output = output.transpose(0, 1)
        return output

    @torch.jit.export
    def generate(self, batch: dict[str, torch.Tensor], max_len: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference pass on a batch of tokenized texts.

        Args:
          batch (Dict[str, torch.Tensor]): Dictionary containing the input to the model with entries 'text'
                                           and 'start_index'
          max_len (int): Max steps of the autoregressive inference loop.

        Returns:
          Tuple: Predictions. The first element is a Tensor of phoneme tokens and the second element
                 is a Tensor of phoneme token probabilities.
        """

        input = batch["text"]
        start_index = batch["start_index"]

        batch_size = input.size(0)
        input = input.transpose(0, 1)  # shape: [T, N]
        src_pad_mask = _make_len_mask(input).to(input.device)
        with torch.no_grad():
            input = self.encoder(input)
            input = self.pos_encoder(input)
            input = self.transformer.encoder(input, src_key_padding_mask=src_pad_mask)
            out_indices = start_index.unsqueeze(0)
            out_logits = []
            for i in range(max_len):
                tgt_mask = _generate_square_subsequent_mask(i + 1).to(input.device)
                output = self.decoder(out_indices)
                output = self.pos_decoder(output)
                output = self.transformer.decoder(
                    output, input, memory_key_padding_mask=src_pad_mask, tgt_mask=tgt_mask
                )
                output = self.fc_out(output)  # shape: [T, N, V]
                out_tokens = output.argmax(2)[-1:, :]
                out_logits.append(output[-1:, :, :])

                out_indices = torch.cat([out_indices, out_tokens], dim=0)
                stop_rows, _ = torch.max(out_indices == self.end_index, dim=0)
                if torch.sum(stop_rows) == batch_size:
                    break

        out_indices = out_indices.transpose(0, 1)  # out shape [N, T]
        out_logits = torch.cat(out_logits, dim=0).transpose(0, 1)  # out shape [N, T, V]
        out_logits = out_logits.softmax(-1)
        out_probs = torch.ones((out_indices.size(0), out_indices.size(1)))
        for i in range(out_indices.size(0)):
            for j in range(0, out_indices.size(1) - 1):
                out_probs[i, j + 1] = out_logits[i, j].max()
        return out_indices, out_probs

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AutoregressiveTransformer":
        """
        Initializes an autoregressive Transformer model from a config.
        Args:
          config (dict): Configuration containing the hyperparams.

        Returns:
          AutoregressiveTransformer: Model object.
        """

        preprocessor = Preprocessor.from_config(config)
        return AutoregressiveTransformer(
            encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
            decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
            end_index=preprocessor.phoneme_tokenizer.end_index,
            d_model=config["model"]["d_model"],
            d_fft=config["model"]["d_fft"],
            encoder_layers=config["model"]["layers"],
            decoder_layers=config["model"]["layers"],
            dropout=config["model"]["dropout"],
            heads=config["model"]["heads"],
        )


def create_model(model_type: ModelType, config: dict[str, Any]) -> Model:
    """
    Initializes a model from a config for a given model type.

    Args:
        model_type (ModelType): Type of model to be initialized.
        config (dict): Configuration containing hyperparams.

    Returns: Model: Model object.
    """

    if model_type is ModelType.TRANSFORMER:
        model = ForwardTransformer.from_config(config)
    elif model_type is ModelType.AUTOREG_TRANSFORMER:
        model = AutoregressiveTransformer.from_config(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {[t.value for t in ModelType]}")
    return model


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> tuple[Model, dict[str, Any]]:
    """
    Initializes a model from a checkpoint (.pt file).

    Args:
        checkpoint_path (str): Path to checkpoint file (.pt).
        device (str): Device to put the model to ('cpu' or 'cuda').

    Returns: Tuple: The first element is a Model (the loaded model)
             and the second element is a dictionary (config).
    """

    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint["config"]["model"]["type"]
    model_type = ModelType(model_type)
    model = create_model(model_type, config=checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


if __name__ == "__main__":
    model_name = "best_model"
    model, checkpoint = load_checkpoint(model_name + ".pt", device="cpu")
    model.eval()

    input_dummy = torch.tensor(
        [
            [
                1,
                18,
                18,
                18,
                10,
                10,
                10,
                17,
                17,
                17,
                16,
                16,
                16,
                7,
                7,
                7,
                15,
                15,
                15,
                11,
                11,
                11,
                28,
                28,
                28,
                7,
                7,
                7,
                20,
                20,
                20,
                2,
            ]
        ]
    )
    model.forward(input_dummy)
    torch.onnx.export(
        model,
        input_dummy,
        model_name + ".onnx",
        export_params=True,
        do_constant_folding=True,
        input_names=["modelInput"],
        output_names=["modelOutput"],
        dynamic_axes={"modelInput": {0: "batch_size"}, "modelOutput": {0: "batch_size"}},
    )

    # Inference example
    import onnxruntime

    onnx_runtime_input = input_dummy.detach().numpy()
    ort_session = onnxruntime.InferenceSession(model_name + ".onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: onnx_runtime_input}
    ort_outs = ort_session.run(None, ort_inputs)[0]
