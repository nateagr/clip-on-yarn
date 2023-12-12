"""Utils to load mCLIP model"""
import os
import shutil
import uuid
from typing import Optional

import fsspec
import torch
from open_clip.factory import load_checkpoint
from open_clip.model import CLIP
from open_clip.transform import image_transform
from transformers import AutoTokenizer

from clip_on_yarn.config import Config
from clip_on_yarn.model.model import mCLIP

CONFIG = Config()
CLIP_MODEL_CONFIG = {
    "embed_dim": 640,
    "vision_cfg": {
        "image_size": 240,
        "layers": 12,
        "width": 896,
        "patch_size": 16,
        # "patch_dropout": CONFIG.train_cfg.patch_dropout,
    },
    "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 640, "heads": 10, "layers": 12},
}


def download_artifact(artifact_hdfs_path: str, download_dir: str) -> str:
    artifact_name = os.path.basename(artifact_hdfs_path)
    artifact_local_path = os.path.join(download_dir, artifact_name)
    if os.path.exists(artifact_local_path):
        return artifact_local_path
    fs = fsspec.filesystem("hdfs")
    fs.get(artifact_hdfs_path, artifact_local_path, recursive=True)
    return artifact_local_path


def _load_visual_transformer_and_transformers(visual_transformer_hdfs_path: str, download_root: str):
    """Load visual transformer and transformers from hdfs"""
    local_path = download_artifact(visual_transformer_hdfs_path, download_root)
    model = CLIP(**CLIP_MODEL_CONFIG)
    load_checkpoint(model, local_path)
    image_preprocessing_train = image_transform(model.visual.image_size, is_train=True, mean=None, std=None)
    image_preprocessing_val = image_transform(model.visual.image_size, is_train=False, mean=None, std=None)
    return model.visual, model.logit_scale, image_preprocessing_train, image_preprocessing_val


def _load_text_transformer_and_tokenizer(text_transformer_hdfs_path: str, tokenizer_hdfs_path: str, download_root: str):
    """Load text transformer and tokenizer from hdfs"""
    # Load text transformer
    local_path = download_artifact(text_transformer_hdfs_path, download_root)
    text_transfomer = CONFIG.text_model.from_pretrained(local_path)
    # Load tokenizer
    local_path = download_artifact(tokenizer_hdfs_path, download_root)
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    return text_transfomer, tokenizer


def load_model_tokenizer_and_transforms(
    text_transformer_hdfs_path: str, visual_transformer_hdfs_path: str, tokenizer_hdfs_path: str, download_root: str
):
    """Load mCLIP model, tokenizer and image transforms"""
    if not os.path.exists(download_root):
        os.mkdir(download_root)
    text_transformer, tokenizer = _load_text_transformer_and_tokenizer(
        text_transformer_hdfs_path, tokenizer_hdfs_path, download_root
    )
    (
        visual_transformer,
        logit_scale,
        image_preprocessing_train,
        image_preprocessing_val,
    ) = _load_visual_transformer_and_transformers(visual_transformer_hdfs_path, download_root)
    model = mCLIP(text_transformer, visual_transformer, logit_scale)
    return model, tokenizer, image_preprocessing_train, image_preprocessing_val


def load_artifacts(
    text_transformer_hdfs_path: str,
    visual_transformer_hdfs_path: str,
    tokenizer_hdfs_path: str,
    download_root: str,
    ckpt_path: Optional[str] = None,
):
    """Load mCLIP model, tokenizer and image transforms"""
    os.makedirs(download_root, exist_ok=True)
    text_transformer, tokenizer = _load_text_transformer_and_tokenizer(
        text_transformer_hdfs_path, tokenizer_hdfs_path, download_root
    )
    (
        visual_transformer,
        logit_scale,
        image_preprocessing_train,
        image_preprocessing_val,
    ) = _load_visual_transformer_and_transformers(visual_transformer_hdfs_path, download_root)
    model = mCLIP(text_transformer, visual_transformer, logit_scale)
    if ckpt_path:
        local_path = download_artifact(ckpt_path, download_root)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(local_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model"])
    model.eval()
    model = model.float()
    return model, tokenizer, image_preprocessing_train, image_preprocessing_val
