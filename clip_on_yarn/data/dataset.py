"""Webdataset functions"""
import io
import json
import math
import os
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import fsspec
import webdataset as wds
from clip_on_yarn.utils.uc import CAT_LANGUAGES_OF_INTEREST
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
    caption_key: Union[str, List[str]] = "title.txt",
    metadata_key: str = "metadata.json",
    cache_path: Optional[str] = None,
) -> wds.WebDataset:
    """Create the WebDataset pipeline"""
    dataset = wds.PytorchShardList(urls, shuffle=False, split_by_node=False)
    dataset = dataset.then(wds.tariterators.url_opener, handler=wds.handlers.warn_and_continue)
    if cache_path:
        dataset = dataset.then(
            wds.shardcache.cache_shards,
            cache_dir=cache_path,
        )
    dataset = dataset.then(wds.tariterators.tar_file_expander, handler=wds.handlers.warn_and_continue)
    dataset = dataset.then(wds.tariterators.group_by_keys)
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
    transformed_dataset = filtered_dataset.map(transform_fn, handler=wds.handlers.warn_and_continue)
    return transformed_dataset


def get_paths_and_nb_of_samples(dataset_path: str) -> Tuple[List[str], List[int]]:
    """Return shard paths and the number of samples"""
    fs = fsspec.filesystem("hdfs")
    shard_ids, n_samples = [], []
    with fs.open(os.path.join(dataset_path, "metadata")) as f:
        metadata = json.load(f)
    for item in metadata.values():
        n_samples.append(item["nb_of_samples"])
        shard_ids.extend(item["shard_ids"])
    key_format = math.ceil(math.log10(max(shard_ids)))

    paths = [os.path.join(dataset_path, f"{shard_id:0{key_format}d}.tar") for shard_id in shard_ids]
    return paths, n_samples


def get_number_of_samples(dataset_path: str) -> int:
    """Compute the number of samples include"""
    _, nb_of_samples = get_paths_and_nb_of_samples(dataset_path)
    return sum(nb_of_samples)


def generate_wds_paths_and_samples_per_lang(
    base_path: str, max_samples=500_000
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """Return paths and the number of samples corresponding, capped by the max_samples parameter"""
    webdataset_paths_per_lang: Dict[str, List[str]] = {}
    samples_per_lang: Dict[str, int] = {}
    for lang in CAT_LANGUAGES_OF_INTEREST:
        paths, n_samples = get_paths_and_nb_of_samples(os.path.join(base_path, f"language={lang}"))
        webdataset_paths_per_lang[lang] = []
        samples_per_lang[lang] = 0
        for p, n in zip(paths, n_samples):
            webdataset_paths_per_lang[lang].append(p)
            samples_per_lang[lang] += n
            if samples_per_lang[lang] >= max_samples:
                break
    return webdataset_paths_per_lang, samples_per_lang
