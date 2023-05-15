"""Models used in mCLIP"""
import logging
import os

import torch
import torch.nn.functional as F
from multilingual_clip import Config_MCLIP
from torch import nn
from transformers import AutoModel, PreTrainedModel
from typing import Union
from open_clip.model import VisualTransformer
from peft import (
    LoraConfig,
    get_peft_model
)

logger = logging.getLogger()


class LinearProjection(PreTrainedModel):  # pylint: disable=abstract-method
    """Linear projection"""

    config_class = Config_MCLIP.MCLIPConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.LinearTransformation = torch.nn.Linear(
            in_features=config.transformerDimensions, out_features=config.numDims
        )

    def forward(self, embs):
        """forward pass"""
        return self.LinearTransformation(embs)


class XMLRoBERTaLargeTextEncoder(torch.nn.Module):
    """Multilingual text encoder"""

    @classmethod
    def from_pretrained(cls, model_dir: str):
        return cls(os.path.join(model_dir, "transformer"), os.path.join(model_dir, "linear_projection"))

    def __init__(self, transformer_path: str, linear_projection_path: str):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_path)
        self.linear_transformation = LinearProjection.from_pretrained(linear_projection_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass"""
        embs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
        embs = (embs * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(dim=1)[:, None]
        text_features = self.linear_transformation(embs)
        return text_features


class mDeBERTaTextEncoder(torch.nn.Module):
    """Multilingual text encoder"""

    @classmethod
    def from_pretrained(cls, model_dir: str):
        return cls(model_dir)

    def __init__(self, transformer_path: str):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_path)#, torch_dtype=torch.float16)
        self.linear_transformation = torch.nn.Linear(in_features=768, out_features=640)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass"""
        embs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
        embs = (embs * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(dim=1)[:, None]
        text_features = self.linear_transformation(embs)
        return text_features


class mCLIP(nn.Module):
    """Multilingual CLIP"""

    def __init__(
        self,
        text_transformer: Union[XMLRoBERTaLargeTextEncoder, mDeBERTaTextEncoder],
        visual_transformer: VisualTransformer,
        logit_scale: nn.Parameter,
        use_lora: bool = False
    ) -> None:
        super().__init__()
        self.text_transformer = text_transformer
        if use_lora:
            logger.info("LORA enabled")
            self.text_transformer.transformer = self.apply_lora(self.text_transformer.transformer)
            self.text_transformer.transformer.print_trainable_parameters()
        else:
            logger.info("LORA disabled")
        self.visual_transformer = visual_transformer
        self.logit_scale = logit_scale

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        text_features = self.text_transformer(input_ids, attention_mask)
        text_features = F.normalize(text_features, dim=-1)
        image_features = self.visual_transformer(image)
        image_features = F.normalize(image_features, dim=-1)
        return image_features, text_features, self.logit_scale.exp()

    def apply_lora(
        self,
        text_encoder,
        lora_r = 8,
        lora_alpha= 16,
        lora_dropout = 0.05
    ):
        lora_target_modules = [
            "query_proj", "value_proj",
        ]
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
        )
        return get_peft_model(text_encoder, config)
