"""DDP Clip loss"""
from typing import Tuple

import torch
from torch import nn
from torch.distributed.nn.functional import all_gather
from torch.nn import functional as F


def gather_features(image_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    all_image_features = torch.cat(all_gather(image_features), dim=0)
    all_text_features = torch.cat(all_gather(text_features), dim=0)
    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    """Implement Clip loss for DDP training"""

    def __init__(self, local_loss: bool = False, rank: int = 0, world_size: int = 1):
        super().__init__()
        self.local_loss = local_loss
        self.rank = rank
        self.world_size = world_size

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: float):
        """Compute the loss"""
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(image_features, text_features)
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        if self.world_size > 1 and self.local_loss:
            labels = labels + num_logits * self.rank
        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss
