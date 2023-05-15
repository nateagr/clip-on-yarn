"""Webdataset functions"""
import io
import json
import logging
import math
import os
import pickle
import random
import sys
from functools import partial
from typing import Dict, List, Tuple, Union

import fsspec
import torch.distributed as dist
import webdataset as wds
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms import Compose
from transformers.tokenization_utils import PreTrainedTokenizer
from webdataset.filters import _shuffle

from clip_on_yarn.config import Config
from clip_on_yarn.data.utils import SharedEpoch
from clip_on_yarn.utils.uc import CAT_LANGUAGES_OF_INTEREST

CONFIG = Config()
_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

logger = logging.getLogger()


class Shuffle(wds.PipelineStage):  # pylint: disable=abstract-method
    """Shuffle stage"""

    def __init__(
        self,
        epoch: SharedEpoch,
        bufsize: int = 1000,
        initial: int = 100,
        seed: int = 0,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        """Stage run"""
        epoch = self.epoch.get_value()
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


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
        elif isinstance(caption_key, list):
            caption = " ".join([row[key].decode("utf-8") for key in caption_key])
        tokenized_text = tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        output["input_ids"] = tokenized_text["input_ids"]
        output["attention_mask"] = tokenized_text["attention_mask"]

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


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


class ResampledShards(IterableDataset):  # pylint: disable=abstract-method
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        epoch: SharedEpoch,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        epoch = self.epoch.get_value()
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


def extract_image_label(row, uc_id_to_label_mapping):
    """Extract image and target from the wds sample"""
    label = uc_id_to_label_mapping[row["metadata"]["uc_id"]]
    return {"image_tensor": row["image_tensor"], "label": label}


def create_webdataset(
    urls: List[str],
    image_transform: Compose,
    tokenizer: PreTrainedTokenizer,
    is_train: bool,
    num_samples: int,
    batch_size: int,
    epoch: int = 0,
    enable_text: bool = True,
    enable_image: bool = True,
    enable_metadata: bool = False,
    image_key: str = "image.jpg",
    caption_key: Union[str, List[str]] = ["title.txt", "description.txt"],
    metadata_key: str = "metadata.json",
) -> wds.WebDataset:
    """Create the WebDataset pipeline"""
    resampled = CONFIG.train_cfg.dataset_resampled and is_train
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [ResampledShards(urls, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(urls)]
    if is_train:
        if not resampled:
            pipeline.extend(
                [
                    Shuffle(
                        bufsize=_SHARD_SHUFFLE_SIZE,
                        initial=_SHARD_SHUFFLE_INITIAL,
                        seed=CONFIG.seed,
                        epoch=shared_epoch,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )
        pipeline.extend(
            [
                # at this point, we have an iterator over the shards assigned to each worker at each node
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                ),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
            ]
        )
    filter_fn = partial(
        filter_row,
        caption_key=caption_key,
        image_key=image_key,
        metadata_key=metadata_key,
        enable_text=enable_text,
        enable_image=enable_image,
        enable_metadata=enable_metadata,
    )

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
    if is_train:
        pipeline.extend(
            [
                wds.select(filter_fn),
                wds.map(transform_fn),
                wds.to_tuple("image_tensor", "input_ids", "attention_mask"),
                wds.batched(batch_size, partial=not is_train),
            ]
        )
    else:
        uc_id_to_label_mapping = pickle.load(fsspec.filesystem("hdfs").open(CONFIG.uc_id_to_idx_mapping_path, "rb"))
        extract_fb = partial(extract_image_label, uc_id_to_label_mapping=uc_id_to_label_mapping)
        pipeline.extend(
            [
                wds.select(filter_fn),
                wds.map(transform_fn),
                wds.map(extract_fb),
                wds.to_tuple("image_tensor", "label"),
                wds.batched(batch_size, partial=not is_train),
            ]
        )
    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        global_batch_size = batch_size * world_size
        num_batches = math.ceil(num_samples / global_batch_size)
        num_workers = max(1, CONFIG.train_cfg.num_workers)
        num_worker_batches = math.ceil(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
        CONFIG.train_cfg.num_samples = num_samples
        CONFIG.train_cfg.num_batches = num_batches
        CONFIG.train_cfg.shared_epoch = shared_epoch
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / batch_size)
    return dataset


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
