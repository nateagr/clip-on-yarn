import fsspec
from torch.utils.data import IterableDataset
import torch
from fastparquet import ParquetFile
import math


class ParquetDataset(IterableDataset):
    def __init__(self, dataset_path, length):
        self.length = length
        self.fs, path_in_fs = fsspec.core.url_to_fs(dataset_path)
        self.dataset_file_paths = self.fs.glob(path_in_fs + '/*.parquet')
        worker = torch.utils.data.get_worker_info()
        self.worker_id = int(worker.id) if worker else 0
        self.num_workers = worker.num_workers if worker else 1

    def __iter__(self):
        for dataset_file_path in self.dataset_file_paths:
            parquet_file = ParquetFile(dataset_file_path, fs=self.fs)
            for row_group in parquet_file.iter_row_groups():
                group_size = len(row_group)
                subgroup_size = math.ceil(group_size / self.num_workers)
                start = self.worker_id * subgroup_size
                end = start + subgroup_size
                subgroup = row_group[start:end]
                for row in zip(subgroup["image"], subgroup["description"]):
                    yield row

    def __len__(self):
        return self.length
