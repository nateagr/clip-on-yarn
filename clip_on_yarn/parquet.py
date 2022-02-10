import logging

from torch.utils.data import IterableDataset
import torch.distributed as dist
from cluster_pack import filesystem
import pyarrow.parquet as pq


logger = logging.getLogger()


class ParquetDataset(IterableDataset):
    def __init__(self, dataset_path, num_samples, batch_size):
        self.num_samples = num_samples
        self.fs, _ = filesystem.resolve_filesystem_and_path(dataset_path)
        self.dataset_file_paths = [
            f for f in self.fs.base_fs.ls(dataset_path) if f.endswith(".parquet")
        ]
        self.batch_size = batch_size
        self.worker_id = dist.get_rank() if dist.is_initialized() else 0
        self.num_workers = dist.get_world_size() if dist.is_initialized() else 1
        logger.info(f"worker_id: {self.worker_id}; num_workers: {self.num_workers}")

    def __iter__(self):
        for dataset_file_path in self.dataset_file_paths:
            with self.fs.base_fs.open(dataset_file_path) as f:
                parquet_file = pq.ParquetFile(f)
                # Drop last batch because all-reduce ops require batches to have the same
                # sizes
                batches = list(parquet_file.iter_batches(batch_size=self.batch_size))[:-1]
                n_batches = len(batches)
                # FIXME: what if batch size is smaller than number of workers ?
                n_batches_per_worker = n_batches // self.num_workers
                start = self.worker_id * n_batches_per_worker
                end = start + n_batches_per_worker
                logger.info(f"worker_id: {self.worker_id}; n_batches: {n_batches}; start: {start}; end: {end}")
                for b in batches[start:end]:
                    yield (b.column("image"), b.column("description"))

    def __len__(self):
        return self.num_samples // self.batch_size // self.num_workers
