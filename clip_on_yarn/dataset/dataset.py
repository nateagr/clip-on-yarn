import io
from functools import partial
from typing import List, Union

import clip
import torch
import webdataset as wds

# from imagenetv2_pytorch import ImageNetV2Dataset
from PIL import Image
from torchvision.transforms import Compose


def preprocess_dataset(
    row: dict,
    caption_key: Union[str, List[str]],
    image_key: str,
    metadata_key: str,
    image_transform: Compose,
    enable_text: bool,
    enable_image: bool,
    enable_metadata: bool,
) -> dict:
    """Preprocessing of the wds sample"""
    output = {}
    if enable_image:
        image_data = row[image_key]
        image = Image.open(io.BytesIO(image_data))
        image_tensor = image_transform(image)
        output["image_filename"] = row["__key__"]
        output["image_tensor"] = image_tensor

    if enable_text:
        if isinstance(caption_key, str):
            text = row[caption_key]
            caption = text.decode("utf-8")
            tokenized_text = clip.tokenize([caption], truncate=True)[0]
            output["text_tokens"] = tokenized_text
            output["text"] = caption
        elif isinstance(caption_key, list):
            text = row[caption_key]
            caption = "\n".join([row[key].decode("utf-8") for key in caption_key])
            tokenized_text = clip.tokenize([caption], truncate=True)[0]
            output["text_tokens"] = tokenized_text
            output["text"] = caption
    if enable_metadata:
        metadata_file = row[metadata_key]
        metadata = metadata_file.decode("utf-8")
        output["metadata"] = metadata
    return output


def filter_row(
    row: dict,
    caption_key: Union[str, List[str]],
    image_key: str,
    metadata_key: str,
    enable_text: bool,
    enable_image: bool,
    enable_metadata: bool,
) -> bool:
    """Filter wds sample"""
    if isinstance(caption_key, str):
        return (
            (not enable_text or caption_key in row)
            and (not enable_image or image_key in row)
            and (not enable_metadata or metadata_key in row)
        )
    elif isinstance(caption_key, list):
        return (
            (not enable_text or all(key in row for key in caption_key))
            and (not enable_image or image_key in row)
            and (not enable_metadata or metadata_key in row)
        )


def create_webdataset(
    urls: List[str],
    image_transform: Compose,
    enable_text: bool = True,
    enable_image: bool = True,
    enable_metadata: bool = False,
    image_key: str = "jpg",
    caption_key: Union[str, List[str]] = "txt",
    metadata_key: str = "json",
    cache_path: str = None,
) -> wds.WebDataset:
    """Create the WebDataset pipeline"""
    dataset = wds.WebDataset(urls, cache_dir=cache_path, cache_size=10**10, handler=wds.handlers.warn_and_continue)

    filter_fn = partial(
        filter_row,
        caption_key=caption_key,
        image_key=image_key,
        metadata_key=metadata_key,
        enable_text=enable_text,
        enable_image=enable_image,
        enable_metadata=enable_metadata,
    )
    filtered_dataset = dataset.select(filter_fn)

    transform_fn = partial(
        preprocess_dataset,
        caption_key=caption_key,
        image_key=image_key,
        metadata_key=metadata_key,
        image_transform=image_transform,
        enable_image=enable_image,
        enable_text=enable_text,
        enable_metadata=enable_metadata,
    )
    transformed_dataset = filtered_dataset.map(transform_fn, handler=wds.handlers.warn_and_continue)
    return transformed_dataset


# def create_imagenetv2_dataset(preprocess_val, batch_size=64, num_worker=4):
#     dataset = ImageNetV2Dataset(location=".", transform=preprocess_val)
#     return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_worker)
