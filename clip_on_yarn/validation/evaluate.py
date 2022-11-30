"""Evaluation scripts"""
import logging
import pickle
from contextlib import suppress
from functools import partial
from typing import Any, Dict, List

import fsspec
import torch
import torch.nn.functional as F
from clip_on_yarn.config import CONFIG
from clip_on_yarn.dataset.dataset import create_webdataset
from clip_on_yarn.model.mclip import mCLIP
from clip_on_yarn.utils.uc import CAT_LANGUAGES_OF_INTEREST
from tf_yarn.pytorch.model_ckpt import _unwrap_model
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger()


def extract_image_label(row, uc_id_to_idx_mapping):
    """Extract image and target from the wds sample"""
    label = uc_id_to_idx_mapping[row["metadata"]["uc_id"]]
    return row["image_tensor"], label


def filter_unsupported_uc_id(row, uc_ids):
    return row["metadata"]["uc_id"] in uc_ids


def create_validation_dataloader_per_lang(
    webdataset_dir_per_lang: Dict[str, str],
    batch_size: int,
    num_workers: int,
    uc_id_to_idx_mapping: Dict[int, int],
    image_transform_val: Compose,
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, DataLoader]:
    """Create a validation dataloader per lang"""
    dataloader_per_lang = {}
    for lang in CAT_LANGUAGES_OF_INTEREST:
        dataloader_per_lang[lang] = create_validation_dataloader(
            webdataset_dir_per_lang[lang], batch_size, num_workers, uc_id_to_idx_mapping, image_transform_val, tokenizer
        )
    return dataloader_per_lang


def create_validation_dataloader(
    validation_webdataset_dir: str,
    batch_size: int,
    num_workers: int,
    uc_id_to_idx_mapping: Dict[int, int],
    image_transform_val: Compose,
    tokenizer: PreTrainedTokenizer,
) -> DataLoader:
    """Create validation dataloader"""
    fs, path = fsspec.core.url_to_fs(validation_webdataset_dir)
    url_paths = [
        "pipe:hdfs dfs -cat viewfs://root" + path for path in fs.ls(path, detail=False) if path.endswith(".tar")
    ]
    extract_fb = partial(extract_image_label, uc_id_to_idx_mapping=uc_id_to_idx_mapping)
    validation_dataset = create_webdataset(url_paths, image_transform_val, tokenizer, enable_metadata=True).map(
        extract_fb
    )
    return DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def zero_shot_classifier(
    model: mCLIP,
    tokenizer: PreTrainedTokenizer,
    templates_per_uc_id: Dict[int, List[str]],
    device: str,
) -> torch.Tensor:
    """Create class embeddings"""
    unwrapped_model = _unwrap_model(model)
    unwrapped_model.eval()  # Make sure the model is in eval mode
    with torch.no_grad():
        zeroshot_classifier = []
        for templates in tqdm(templates_per_uc_id.values()):
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


def zero_shot_classifier_per_lang(model: mCLIP, tokenizer: PreTrainedTokenizer, device: str) -> Dict[str, torch.Tensor]:
    """Create class embeddings per lang"""
    classifier_per_lang = {}
    templates_per_uc_id_x_lang = pickle.load(
        fsspec.filesystem("hdfs").open(CONFIG.templates_per_lang_x_uc_id_path, "rb")
    )
    for lang in CAT_LANGUAGES_OF_INTEREST:
        logger.info(f"Creating {lang} classifier")
        classifier_per_lang[lang] = zero_shot_classifier(model, tokenizer, templates_per_uc_id_x_lang[lang], device)
    return classifier_per_lang


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> List[float]:
    """Compute top k accuracies"""
    # Indices of K largest logits (shape: (K, batch size))
    top_classes = logits.topk(max(topk), 1, True, True)[1].t()
    # True if the indice matches the target else False (shape: (K, batch size))
    correct = top_classes.eq(target.view(1, -1).expand_as(top_classes))
    # Flatten, convert to 1.0 or 0.0 and sum values
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def evaluate(
    model: mCLIP,
    classifier: torch.Tensor,
    dataloader: DataLoader,
    device: str,
    precision: str,
    n_steps: int = None,
) -> dict:
    """Evaluate the accuracies of the model"""
    unwrapped_model = _unwrap_model(model)
    unwrapped_model.eval()  # Make sure the model is in eval mode
    autocast = torch.cuda.amp.autocast if precision == "amp" else suppress
    with torch.no_grad():
        top1, top5, top10, n = 0.0, 0.0, 0.0, 0.0
        for i, (images, target) in enumerate(dataloader):
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
            if n_steps and i >= n_steps:
                break

    top1 = top1 / n
    top5 = top5 / n
    top10 = top10 / n
    return {"zeroshot-val-top1": top1, "zeroshot-val-top5": top5, "zeroshot-val-top10": top10}


def compute_metrics(
    model: mCLIP,
    classifier_per_lang: Dict[str, torch.Tensor],
    dataloader_per_lang: Dict[str, DataLoader],
    device: str,
    precision: str,
    steps_per_lang: Dict[str, int],
) -> Dict[str, Any]:
    """Compute validation metrics"""
    metrics = {}
    for lang in CAT_LANGUAGES_OF_INTEREST:
        logger.info(f"Computing {lang} evaluation metrics")
        lang_metrics = evaluate(
            model,
            classifier=classifier_per_lang[lang],
            dataloader=dataloader_per_lang[lang],
            device=device,
            precision=precision,
            n_steps=steps_per_lang[lang],
        )
        metrics.update({f"{lang}_{k}": v for k, v in lang_metrics.items()})
    return metrics
