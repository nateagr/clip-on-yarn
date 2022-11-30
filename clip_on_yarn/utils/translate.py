"""Tanslation model utils"""
import os
import shutil
import uuid
import zipfile
from pathlib import Path

import fsspec
import nltk
from easynmt import EasyNMT, models

PUNKT_HDFS_PATH = "hdfs://root/user/g.racic/punkt.zip"
MODEL_HDFS_PATH = "hdfs://root/user/g.racic/m2m100_1.2B"


def _create_dir(directory: str) -> None:
    """Create a directory"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def _download_punkt(destination_dir: str) -> None:
    """Download nltk punkt"""
    tokentizer_dir = os.path.join(destination_dir, "tokenizers")
    tokentizer_path = Path(tokentizer_dir)
    tokentizer_path.mkdir(parents=True, exist_ok=True)

    filename = os.path.basename(PUNKT_HDFS_PATH)
    destination_file_path = os.path.join(tokentizer_dir, filename)
    if not os.path.exists(destination_file_path):
        fs, _ = fsspec.core.url_to_fs(PUNKT_HDFS_PATH, use_listings_cache=False)
        fs.get(PUNKT_HDFS_PATH, destination_file_path)
        with zipfile.ZipFile(destination_file_path, "r") as zip_ref:
            zip_ref.extractall(tokentizer_dir)


def load_m2m100_12B(download_root: str, on_yarn: bool = True):
    """Load translation model"""
    if on_yarn:
        download_root = os.path.join(download_root, str(uuid.uuid4()))
        _create_dir(download_root)
    model_name = os.path.basename(MODEL_HDFS_PATH)
    model_local_path = os.path.join(download_root, model_name)
    if not os.path.exists(model_local_path):
        fs, _ = fsspec.core.url_to_fs(MODEL_HDFS_PATH, use_listings_cache=False)
        fs.get(MODEL_HDFS_PATH, model_local_path, recursive=True)

    _download_punkt(download_root)
    nltk.data.path.append(download_root)

    return EasyNMT(translator=models.AutoModel(model_local_path))
