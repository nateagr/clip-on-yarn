# type: ignore
"""Torch utils"""
import torch


def apply_freezing_strategy(model: torch.nn.Module) -> torch.nn.Module:
    """Freeze model & unfreeze layers of interest"""
    unfreeze_n_last_layers = 2
    freeze_model(model)
    # Unfreeze last n layers of the transformers
    unfreeze_last_n_layers(model.visual_transformer, unfreeze_n_last_layers, num_layers=11)
    unfreeze_last_n_layers(model.text_transformer.transformer, unfreeze_n_last_layers, num_layers=23)
    # Unfreeze layers after transformers
    unfreeze_layer(model.visual_transformer, "ln_post")
    unfreeze_layer(model.text_transformer.transformer, "pooler")
    unfreeze_layer(model.text_transformer, "linear_transformation")
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
            if f"resblocks.{i}" in name:
                unfreeze_layer(model, name)
