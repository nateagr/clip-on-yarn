"""Text transformer utils"""
import os

import torch
from multilingual_clip import Config_MCLIP
from transformers import AutoModel, PreTrainedModel


class LinearProjection(PreTrainedModel):
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


class MultilingualTextEncoder(torch.nn.Module):
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
