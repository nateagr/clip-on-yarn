"""Utils to load mCLIP model"""
import os

import fsspec
from clip_on_yarn.config import Config
from clip_on_yarn.model.model import mCLIP
from open_clip.factory import load_checkpoint
from open_clip.model import CLIP
from open_clip.transform import image_transform
from transformers import AutoTokenizer

CONFIG = Config()
CLIP_MODEL_CONFIG = {
    "embed_dim": 640,
    "vision_cfg": {"image_size": 240, "layers": 12, "width": 896, "patch_size": 16},
    "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 640, "heads": 10, "layers": 12},
}


def _load_visual_transformer_and_transformers(visual_transformer_hdfs_path: str, download_root: str):
    """Load visual transformer and transformers from hdfs"""
    model = CLIP(**CLIP_MODEL_CONFIG)
    load_checkpoint(
        model,
        _download_artifact(visual_transformer_hdfs_path, download_root)
    )
    image_preprocessing_train = image_transform(model.visual.image_size, is_train=True, mean=None, std=None)
    image_preprocessing_val = image_transform(model.visual.image_size, is_train=False, mean=None, std=None)
    return model.visual, model.logit_scale, image_preprocessing_train, image_preprocessing_val


def _load_text_transformer_and_tokenizer(text_transformer_hdfs_path: str, tokenizer_hdfs_path: str, download_root: str):
    """Load text transformer and tokenizer from hdfs"""
    text_transfomer = CONFIG.text_model.from_pretrained(
        _download_artifact(text_transformer_hdfs_path, download_root)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        _download_artifact(tokenizer_hdfs_path, download_root)
    )
    return text_transfomer, tokenizer


def _download_artifact(artifact_src_path: str, download_root: str) -> str:
    artifact_dst_path = os.path.join(download_root, os.path.basename(artifact_src_path))
    if not os.path.exists(artifact_dst_path):
        fs, _ = fsspec.core.url_to_fs(artifact_src_path, use_listings_cache=False)
        fs.get(artifact_src_path, artifact_dst_path, recursive=True)
    return artifact_dst_path


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
