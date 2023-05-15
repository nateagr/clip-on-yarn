"""Torch utils"""
import logging
from typing import Union

from clip_on_yarn.model.model import XMLRoBERTaLargeTextEncoder, mCLIP, mDeBERTaTextEncoder
from torch import nn

logger = logging.getLogger()


def _unfreeze_last_n_layers_of_vit(model: mCLIP, n: int) -> None:
    unfreeze_last_n_layers(model.visual_transformer, n, num_layers=12)
    # Unfreeze layers after transformer
    unfreeze_layer(model.visual_transformer, "ln_post")


def _unfreeze_last_n_layers_of_xml_roberta_large(model: mCLIP, n: int) -> None:
    unfreeze_last_n_layers(model.text_transformer.transformer.encoder, n, num_layers=24)
    # Unfreeze layers after transformer
    unfreeze_model(model.text_transformer.transformer.pooler)
    unfreeze_model(model.text_transformer.linear_transformation)


def _unfreeze_last_n_layers_of_mdeberta(model: mCLIP, n: int) -> None:
    unfreeze_last_n_layers(model.text_transformer.transformer, n, num_layers=12)
    # Unfreeze layers after transformer
    unfreeze_layer(model.text_transformer.transformer, "rel_embeddings")
    unfreeze_layer(model.text_transformer.transformer, "encoder.LayerNorm")
    unfreeze_layer(model.text_transformer, "linear_transformation")


def unfreeze_last_n_layers_of_transformers(model: mCLIP, n: int) -> None:
    """Unfreeze last n layers of CLIP transformers"""
    _unfreeze_last_n_layers_of_vit(model, n)
    _unfreeze_last_n_layers_of_xml_roberta_large(model, n)
    model.logit_scale.requires_grad = True


def log_parameters(model: mCLIP) -> None:
    """Log model parameters"""
    print(f"model.text_transformer: {model.text_transformer}")
    print(f"model.text_transformer.transformer: {model.text_transformer.transformer}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params/1e6:.2f}M")
    logger.info(f"Trainable params: {trainable_params/1e6:.2f}M")
    trainable_params = sum(p.numel() for p in model.text_transformer.parameters() if p.requires_grad)
    logger.info(f"Trainable params text encoder: {trainable_params/1e6:.2f}M")
    trainable_params = sum(p.numel() for p in model.text_transformer.transformer.parameters() if p.requires_grad)
    logger.info(f"Trainable params text transformer: {trainable_params/1e6:.2f}M")


def apply_freezing_strategy(model: mCLIP, epoch: int) -> mCLIP:
    """Define and apply the freezing strategy to the model"""
    # freeze visual transformer for n epoch such that it will act as a teacher for the text transformer
    N_VISUAL_FREEZING_EPOCHS = 0
    if epoch == 0:
        freeze_model(model.visual_transformer)
        #unfreeze_model(model.text_transformer)
        model.logit_scale.requires_grad = True
        log_parameters(model)
    if epoch >= N_VISUAL_FREEZING_EPOCHS:
        freeze_model(model.visual_transformer)
        #unfreeze_model(model.text_transformer)
        model.logit_scale.requires_grad = True
        log_parameters(model)
    return model


def freeze_model(model: mCLIP) -> None:
    """Freeze model"""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: Union[XMLRoBERTaLargeTextEncoder, mDeBERTaTextEncoder]) -> None:
    """Unfreeze model"""
    for param in model.parameters():
        param.requires_grad = True


def freeze_layer(model: mCLIP, name: str) -> None:
    """Freeze layer"""
    for n, p in list(model.named_parameters()):
        if name in n:
            p.requires_grad = False


def unfreeze_layer(model: nn.Module, name: str) -> None:
    """Unfreeze layer"""
    for n, p in list(model.named_parameters()):
        if name in n:
            p.requires_grad = True


def unfreeze_last_n_layers(model: nn.Module, n: int, num_layers: int) -> None:
    """Unfreeze the last n layers on the model"""
    for name, _ in model.named_parameters():
        for i in range(max(num_layers - n, 0), num_layers):
            if (f"resblocks.{i}" in name) or (f"layer.{i}" in name):
                unfreeze_layer(model, name)
