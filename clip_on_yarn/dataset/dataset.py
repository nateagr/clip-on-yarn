"""Webdataset functions"""
import io
import json
import os
from functools import partial
from typing import List, Union

import fsspec
import webdataset as wds
from PIL import Image
from torchvision.transforms import Compose
from transformers.tokenization_utils import PreTrainedTokenizer


def preprocess_dataset(
    row: dict,
    caption_key: Union[str, List[str]],
    image_key: str,
    metadata_key: str,
    image_transform: Compose,
    tokenizer: PreTrainedTokenizer,
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
        output["image_tensor"] = image_tensor

    if enable_text:
        if isinstance(caption_key, str):
            text = row[caption_key]
            caption = text.decode("utf-8")
            tokenized_text = tokenizer(
                caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
            )
            output["input_ids"] = tokenized_text["input_ids"]
            output["attention_mask"] = tokenized_text["attention_mask"]
            output["text"] = caption
        elif isinstance(caption_key, list):
            caption = "\n".join([row[key].decode("utf-8") for key in caption_key])
            tokenized_text = tokenizer(
                caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
            )
            output["input_ids"] = tokenized_text["input_ids"]
            output["attention_mask"] = tokenized_text["attention_mask"]
            output["text"] = caption
    if enable_metadata:
        metadata_file = row[metadata_key]
        metadata = metadata_file.decode("utf-8")
        output["metadata"] = json.loads(metadata)
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

    if isinstance(caption_key, list):
        return (
            (not enable_text or all(key in row for key in caption_key))
            and (not enable_image or image_key in row)
            and (not enable_metadata or metadata_key in row)
        )
    else:
        return (
            (not enable_text or caption_key in row)
            and (not enable_image or image_key in row)
            and (not enable_metadata or metadata_key in row)
        )


def create_webdataset(
    urls: List[str],
    image_transform: Compose,
    tokenizer: PreTrainedTokenizer,
    enable_text: bool = True,
    enable_image: bool = True,
    enable_metadata: bool = False,
    image_key: str = "image.jpg",
    caption_key: Union[str, List[str]] = ["title.txt", "description.txt"],
    metadata_key: str = "metadata.json",
    cache_path: str = None,
) -> wds.WebDataset:
    """Create the WebDataset pipeline"""
    dataset = wds.WebDataset(urls, cache_dir=cache_path, cache_size=10**10, handler=wds.handlers.warn_and_stop)

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
        tokenizer=tokenizer,
        enable_image=enable_image,
        enable_text=enable_text,
        enable_metadata=enable_metadata,
    )
    transformed_dataset = filtered_dataset.map(transform_fn, handler=wds.handlers.warn_and_stop)
    return transformed_dataset


def get_number_of_samples(dataset_path: str) -> int:
    fs = fsspec.filesystem("hdfs")
    n_samples = 0
    with fs.open(os.path.join(dataset_path, "metadata")) as f:
        metadata = json.load(f)
    for item in metadata.values():
        n_samples += item["nb_of_samples"]
    return n_samples
