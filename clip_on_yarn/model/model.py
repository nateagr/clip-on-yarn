"""Models used in mCLIP"""
import logging
import os

import torch
import torch.nn.functional as F
from multilingual_clip import Config_MCLIP
from torch import nn
from transformers import AutoModel, PreTrainedModel

logger = logging.getLogger()


class mCLIP(nn.Module):
    """Multilingual CLIP"""

    def __init__(self, text_transformer: nn.Module, visual_transformer: nn.Module, logit_scale: nn.Parameter) -> None:
        super().__init__()
        self.text_transformer = text_transformer
        self.visual_transformer = visual_transformer
        self.logit_scale = logit_scale

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        text_features = self.text_transformer(input_ids, attention_mask)
        text_features = F.normalize(text_features, dim=-1)
        image_features = self.visual_transformer(image)
        image_features = F.normalize(image_features, dim=-1)
        return image_features, text_features, self.logit_scale.exp()


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
        self.transformer = AutoModel.from_pretrained(transformer_path, torch_dtype=torch.float16)
        self.linear_transformation = torch.nn.Linear(in_features=768, out_features=640)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass"""
        embs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
        embs = (embs * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(dim=1)[:, None]
        text_features = self.linear_transformation(embs)
        return text_features
