# type: ignore
"""Torch utils"""
import logging

import torch

logger = logging.getLogger()


def _unfreeze_last_n_layers_of_vit(model: torch.nn.Module, n: int) -> torch.nn.Module:
    unfreeze_last_n_layers(model.visual_transformer, n, num_layers=12)
    # Unfreeze layers after transformer
    unfreeze_layer(model.visual_transformer, "ln_post")


def _unfreeze_last_n_layers_of_mdeberta(model: torch.nn.Module, n: int) -> torch.nn.Module:
    unfreeze_last_n_layers(model.text_transformer.transformer, n, num_layers=12)
    # Unfreeze layers after transformer
    unfreeze_layer(model.text_transformer.transformer, "rel_embeddings")
    unfreeze_layer(model.text_transformer.transformer, "encoder.LayerNorm")
    unfreeze_layer(model.text_transformer, "linear_transformation")


def unfreeze_last_n_layers_of_transformers(model: torch.nn.Module, n: int) -> torch.nn.Module:
    """Unfreeze last n layers of CLIP transformers"""
    _unfreeze_last_n_layers_of_vit(model, n)
    _unfreeze_last_n_layers_of_mdeberta(model, n)
    model.logit_scale.requires_grad = True


def log_parameters(model: torch.nn.Module) -> None:
    """Log model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params/1e6:.2f}M")
    logger.info(f"Trainable params: {trainable_params/1e6:.2f}M")


def apply_freezing_strategy(model: torch.nn.Module, step: int, optimizer) -> torch.nn.Module:
    """Define and apply the freezing strategy to the model"""
    # freeze visual transformer for n epoch such that it will act as a teacher for the text transformer
    N_VISUAL_FREEZING_STEPS = 300_000
    if step == 0:
        freeze_model(model)
        unfreeze_model(model.text_transformer)
        log_parameters(model)
    elif step == N_VISUAL_FREEZING_STEPS:
        # Make sure no grad is kept in the model
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        freeze_model(model)
        unfreeze_last_n_layers_of_transformers(model, 6)
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
