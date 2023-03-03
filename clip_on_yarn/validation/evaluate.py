"""Evaluation scripts"""
import gc
import logging
import os
import pickle
from contextlib import suppress
from typing import Any, Dict, List

import fsspec
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import webdataset as wds
from clip_on_yarn.config import Config
from clip_on_yarn.data.dataset import create_webdataset
from clip_on_yarn.model.model import mCLIP
from clip_on_yarn.utils.uc import CAT_LANGUAGES_OF_INTEREST
from tf_yarn.pytorch.model_ckpt import _unwrap_model
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger()
CONFIG = Config()


def create_validation_dataloader_per_lang(
    webdataset_paths_per_lang: Dict[str, List[str]],
    samples_per_lang: Dict[str, int],
    batch_size: int,
    num_workers: int,
    image_transform_val: Compose,
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, DataLoader]:
    """Create a validation dataloader per lang"""
    dataloader_per_lang = {}
    for lang in CAT_LANGUAGES_OF_INTEREST:
        dataloader_per_lang[lang] = create_validation_dataloader(
            webdataset_paths_per_lang[lang],
            samples_per_lang[lang],
            batch_size,
            num_workers,
            image_transform_val,
            tokenizer,
        )
    return dataloader_per_lang


def create_validation_dataloader(
    webdataset_paths: List[str],
    num_samples: int,
    batch_size: int,
    num_workers: int,
    image_transform_val: Compose,
    tokenizer: PreTrainedTokenizer,
) -> DataLoader:
    """Create validation dataloader"""
    url_paths = ["pipe:hdfs dfs -cat viewfs://root" + path for path in webdataset_paths]
    validation_dataset = create_webdataset(
        url_paths,
        image_transform_val,
        tokenizer,
        is_train=False,
        enable_metadata=True,
        num_samples=num_samples,
        batch_size=batch_size,
    )
    return wds.WebLoader(
        validation_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )


def zero_shot_classifier(
    model: mCLIP,
    tokenizer: PreTrainedTokenizer,
    templates_per_uc_id: Dict[int, np.ndarray],
    device: str,
) -> torch.Tensor:
    """Create class embeddings"""
    unwrapped_model = _unwrap_model(model)
    with torch.inference_mode():
        zeroshot_classifier = []
        for templates in templates_per_uc_id.values():
            tokenized_text = tokenizer(
                templates.tolist(), padding="max_length", truncation=True, max_length=77, return_tensors="pt"
            )
            input_ids = tokenized_text["input_ids"].to(device)
            attention_mask = tokenized_text["attention_mask"].to(device)
            class_embeddings = unwrapped_model.text_transformer(input_ids, attention_mask)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_classifier.append(class_embedding)
    # stack along second dimension to avoid transpose when matrix multiplication
    return torch.stack(zeroshot_classifier, dim=1).to(device).type(torch.float32)


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> List[float]:
    """Compute top k accuracies"""
    # Indices of K largest logits (shape: (K, batch size))
    top_classes = logits.topk(max(topk), 1, True, True)[1].t()
    # True if the indice matches the target else False (shape: (K, batch size))
    correct = top_classes.eq(target.view(1, -1).expand_as(top_classes))
    # Flatten, convert to 1.0 or 0.0 and sum values
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def evaluate(
    model,
    tokenizer,
    validation_dataloader_per_lang,
    device,
    precision,
    epoch,
    enable_wandb,
):
    """Zero shot evaluation"""
    rank = dist.get_rank()
    if rank == 0 and CONFIG.valid_cfg:
        model.eval()
        logger.info("Starting zero shot evaluation")
        # Recompute the new class embeddings
        metrics = compute_metrics(
            model,
            tokenizer,
            validation_dataloader_per_lang,
            device,
            precision,
        )
        logger.info("Finished zero_shot_eval")
        logger.info(f"[{os.getpid()}]" f"zero shot evaluation metrics: {metrics}")
        if enable_wandb:
            for name, val in metrics.items():
                name = "eval/" + name
                wandb.log({name: val, "epoch": epoch})


def zero_shot_metrics(
    model: mCLIP,
    classifier: torch.Tensor,
    dataloader: DataLoader,
    device: str,
    precision: str,
) -> dict:
    """Evaluate the accuracies of the model"""
    unwrapped_model = _unwrap_model(model)
    autocast = torch.cuda.amp.autocast if precision == "amp" else suppress
    logger.info(f"Precision: {precision}")
    with torch.inference_mode():
        top1, top5, top10, n = 0.0, 0.0, 0.0, 0.0
        for images, target in dataloader:
            images = images.to(device)
            target = target.to(device)

            with autocast():
                image_features = unwrapped_model.visual_transformer(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100.0 * image_features @ classifier

            acc1, acc5, acc10 = accuracy(logits, target, topk=(1, 5, 10))
            top1 += acc1
            top5 += acc5
            top10 += acc10
            n += images.size(0)
    top1 = top1 / n
    top5 = top5 / n
    top10 = top10 / n
    return {"zeroshot-val-top1": top1, "zeroshot-val-top5": top5, "zeroshot-val-top10": top10}


def compute_metrics(
    model: mCLIP,
    tokenizer: PreTrainedTokenizer,
    dataloader_per_lang: Dict[str, DataLoader],
    device: str,
    precision: str,
) -> Dict[str, Any]:
    """Compute validation metrics"""
    metrics = {}
    templates_per_uc_id_x_lang = pickle.load(
        fsspec.filesystem("hdfs").open(CONFIG.templates_per_lang_x_uc_id_path, "rb")
    )
    for lang in CAT_LANGUAGES_OF_INTEREST:
        logger.info(f"Computing {lang} evaluation metrics")
        classifier = zero_shot_classifier(model, tokenizer, templates_per_uc_id_x_lang[lang], device)
        lang_metrics = zero_shot_metrics(
            model,
            classifier=classifier,
            dataloader=dataloader_per_lang[lang],
            device=device,
            precision=precision,
        )
        metrics.update({f"{lang}_{k}": v for k, v in lang_metrics.items()})
        # Clear memory
        del classifier
        torch.cuda.empty_cache()
        gc.collect()
    del templates_per_uc_id_x_lang
    gc.collect()
    return metrics
