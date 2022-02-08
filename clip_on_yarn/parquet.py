from torch.utils.data import IterableDataset
import torch
from cluster_pack import filesystem
import pyarrow.parquet as pq


class ParquetDataset(IterableDataset):
    def __init__(self, dataset_path, length, batch_size):
        self.length = length
        self.fs, _ = filesystem.resolve_filesystem_and_path(dataset_path)
        self.dataset_file_paths = [
            f for f in self.fs.base_fs.ls(dataset_path) if f.endswith(".parquet")
        ]
        self.batch_size = batch_size
        worker = torch.utils.data.get_worker_info()
        self.worker_id = int(worker.id) if worker else 0
        self.num_workers = worker.num_workers if worker else 1

    def __iter__(self):
        for dataset_file_path in self.dataset_file_paths:
            with self.fs.base_fs.open(dataset_file_path) as f:
                parquet_file = pq.ParquetFile(f)
                batch = list(parquet_file.iter_batches(batch_size=self.batch_size))
                batch_size = len(batch)
                # FIXME: what if batch size is smaller than number of workers ?
                subbatch_size = batch_size // self.num_workers
                start = self.worker_id * subbatch_size
                end = start + subbatch_size if self.worker_id < (self.num_workers - 1) else batch_size
                for b in batch[start:end]:
                    yield (b.column("image"), b.column("description"))

    def __len__(self):
        return self.length
