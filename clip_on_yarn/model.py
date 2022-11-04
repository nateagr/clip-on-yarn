import io
import os

import torch
from clip.clip import build_model, tokenize
from clip.model import CLIP, convert_weights
from cluster_pack import filesystem
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomResizedCrop, Resize, ToTensor

vit_b_32_config = {
    "embed_dim": 512,
    "image_resolution": 224,
    "vision_layers": 12,
    "vision_width": 768,
    "vision_patch_size": 32,
    "context_length": 77,
    "vocab_size": 49408,
    "transformer_width": 512,
    "transformer_heads": 8,
    "transformer_layers": 12,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def transform(n_px: int, is_train: bool):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return Compose(
            [
                RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
                _convert_image_to_rgb,
                ToTensor(),
                normalize,
            ]
        )
    else:
        return Compose(
            [
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                _convert_image_to_rgb,
                ToTensor(),
                normalize,
            ]
        )


def _convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def load_model(precision, model_config):
    model = CLIP(**model_config)
    if precision == "amp" or precision == "fp32":
        _convert_models_to_fp32(model)
    elif precision == "fp16":
        convert_weights(model)
    return model


def load_pretrained_model(model_hdfs_path: str, download_root: str, use_gpu: bool) -> torch.nn.Module:
    """_summary_

    Args:
        model_hdfs_path (str): hdfs model path
        download_root (str): local directory to save the model
        use_gpu (bool): True if model should be mapped on GPU

    Returns:
        torch.nn.Module: clip model
    """
    if not os.path.exists(download_root):
        os.mkdir(download_root)
    model_name = os.path.basename(model_hdfs_path)
    local_model_path = os.path.join(download_root, model_name)
    fs, _ = filesystem.resolve_filesystem_and_path(model_hdfs_path)
    fs.get(model_hdfs_path, local_model_path)
    model = torch.jit.load(local_model_path, map_location="cpu").eval()
    model = build_model(model.state_dict()).to("cpu")
    if not use_gpu:
        model.float()
    return model


def preprocessing(n_px: int, is_train: bool):
    preprocess_img = transform(n_px, is_train)

    def _preprocess_fn(pyarrow_batch):
        images, texts = pyarrow_batch.columns
        image_tensor = torch.stack([preprocess_img(Image.open(io.BytesIO(img.as_py()))) for img in images])
        text_tensor = torch.stack([tokenize([text.as_py()], truncate=True)[0] for text in texts])
        return image_tensor, text_tensor

    return _preprocess_fn
