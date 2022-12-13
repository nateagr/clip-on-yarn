# type: ignore
"""Torch utils"""
import logging

import torch

logger = logging.getLogger()


def _unfreeze_last_n_layers_of_vit(model: torch.nn.Module, n: int) -> torch.nn.Module:
    unfreeze_last_n_layers(model.visual_transformer, n, num_layers=11)
    # Unfreeze layers after transformer
    unfreeze_layer(model.visual_transformer, "ln_post")
    return model


def _unfreeze_last_n_layers_of_xlm_roberta_large(model: torch.nn.Module, n: int) -> torch.nn.Module:
    unfreeze_last_n_layers(model.text_transformer.transformer, n, num_layers=23)
    # Unfreeze layers after transformer
    unfreeze_layer(model.text_transformer.transformer, "pooler")
    unfreeze_layer(model.text_transformer, "linear_transformation")
    return model


def _unfreeze_last_n_layers_of_mdeberta(model: torch.nn.Module, n: int) -> torch.nn.Module:
    unfreeze_last_n_layers(model.text_transformer.transformer, n, num_layers=11)
    # Unfreeze layers after transformer
    unfreeze_layer(model.text_transformer.transformer, "rel_embeddings")
    unfreeze_layer(model.text_transformer.transformer, "LayerNorm")
    unfreeze_layer(model.text_transformer, "linear_transformation")
    return model


def unfreeze_last_n_layers_of_transformers(model: torch.nn.Module, n: int) -> torch.nn.Module:
    """Unfreeze last n layers of CLIP transformers"""
    _unfreeze_last_n_layers_of_vit(model, n)
    _unfreeze_last_n_layers_of_mdeberta(model, n)
    model.logit_scale.requires_grad = True
    return model


def log_parameters(model: torch.nn.Module) -> None:
    """Log model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params/1e6:.2f}M")
    logger.info(f"Trainable params: {trainable_params/1e6:.2f}M")


def apply_freezing_strategy(model: torch.nn.Module, epoch: int) -> torch.nn.Module:
    """Define and apply the freezing strategy to the model"""
    # freeze visual transformer for n epoch such that it will act as a teacher for the text transformer
    freeze_model(model)
    N_VISUAL_FREEZING_EPOCH = 1
    if epoch < N_VISUAL_FREEZING_EPOCH:
        _unfreeze_last_n_layers_of_mdeberta(model, 5)
    else:
        unfreeze_last_n_layers_of_transformers(model, 2)
    log_parameters(model)
    return model


def freeze_model(model: torch.nn.Module) -> None:
    """Freeze model"""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: torch.nn.Module) -> None:
    """Unfreeze model"""
    for param in model.parameters():
        param.requires_grad = True


def freeze_layer(model: torch.nn.Module, name: str) -> None:
    """Freeze layer"""
    for n, p in list(model.named_parameters()):
        if name in n:
            p.requires_grad = False


def unfreeze_layer(model: torch.nn.Module, name: str) -> None:
    """Unfreeze layer"""
    for n, p in list(model.named_parameters()):
        if name in n:
            p.requires_grad = True


def unfreeze_last_n_layers(model: torch.nn.Module, n: int, num_layers: int) -> None:
    """Unfreeze the last n layers on the model"""
    for name, _ in model.named_parameters():
        for i in range(max(num_layers - n, 0), num_layers):
            if (f"resblocks.{i}" in name) or (f"layer.{i}" in name):
                unfreeze_layer(model, name)
