import logging
from contextlib import suppress
import json
from typing import NamedTuple, List, Callable

import fsspec
from tqdm import tqdm
import torch
import torch.nn.functional as F
from clip import tokenize
from tf_yarn.pytorch.model_ckpt import _unwrap_model

from clip_on_yarn.validation.imagenet_zeroshot_data import (
    imagenet_classnames, openai_imagenet_template
)
from clip_on_yarn.dataset.dataset import create_webdataset
from clip_on_yarn.model import transform


logger = logging.getLogger()


class ValidationConfig(NamedTuple):
    validation_webdataset_dir: str # Path to validation webdataset
    batch_size: int
    num_workers: int
    classnames: List[str] # List of classes
    templates: List[Callable[[str], str]] # List of functions classname -> text
    period_in_steps: int # validation period in steps
    n_batches: int # number of batches to process during validation


def extract_image_label(row):
    metadata = json.loads(row["metadata"])
    return row["image_tensor"], metadata["training_category_id"]


def create_validation_dataloader(validation_webdataset_dir: str, batch_size: int, num_workers: int):
    # dataset
    fs, path = fsspec.core.url_to_fs(validation_webdataset_dir)
    url_paths = ["pipe:hdfs dfs -cat viewfs://root"+ path for path in fs.ls(path, detail=False)]
    validation_dataset = create_webdataset(url_paths, transform(224, False), enable_metadata=True) \
        .map(extract_image_label)
    return torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, num_workers=num_workers
    )


def zero_shot_classifier(model, classes, templates, device):
    with torch.no_grad():
        zeroshot_classifier = []
        for classname in tqdm(classes, total=len(classes)):
            texts = [template(classname) for template in templates] 
            texts = tokenize(texts).to(device) 
            class_embeddings = _unwrap_model(model).encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_classifier.append(class_embedding)
     # stack along second dimension to avoid transpose when matrix multiplication
    return torch.stack(zeroshot_classifier, dim=1).to(device)


def accuracy(logits, target, topk=(1,)):
    # Indices of K largest logits (shape: (K, batch size))
    top_classes = logits.topk(max(topk), 1, True, True)[1].t()
    # True if the indice matches the target else False (shape: (K, batch size))
    correct = top_classes.eq(target.view(1, -1).expand_as(top_classes))
    # Flatten, convert to 1.0 or 0.0 and sum values
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def evaluate(model, classifier, dataloader, device, precision, n_steps=None):
    autocast = torch.cuda.amp.autocast if precision == 'amp' else suppress
    with torch.no_grad():
        top1, top5, top10, n = 0., 0., 0., 0.
        for i, (images, target) in tqdm(enumerate(dataloader)):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                image_features = _unwrap_model(model).encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            acc1, acc5, acc10 = accuracy(logits, target, topk=(1, 5, 10))
            top1 += acc1
            top5 += acc5
            top10 += acc10
            n += images.size(0)
            if n_steps and i >= n_steps:
                break

    top1 = (top1 / n)
    top5 = (top5 / n)
    top10 = (top10 / n)
    return {
        'zeroshot-val-top1': top1,
        'zeroshot-val-top5': top5,
        'zeroshot-val-top10': top10
    }


def zero_shot_eval(
    model, dataloader, device, precision,
    classes=imagenet_classnames, templates=openai_imagenet_template, n_steps=None
):
    logger.info('Building zero shot classifier')
    classifier = zero_shot_classifier(model, classes, templates, device)
    return evaluate(model, classifier, dataloader, device, precision, n_steps)
