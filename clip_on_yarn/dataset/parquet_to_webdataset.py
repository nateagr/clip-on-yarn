import tempfile
from typing import List
import json
import os

import numpy as np
import pyarrow.parquet as pq
import webdataset as wds
import fsspec


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def create_webdataset_from_parquet(
    spark_session, parquet_files: List[str], webdataset_hdfs_dir: str, filter_fn = None
) -> None:
    indexed_parquet_file = list(enumerate(sorted(parquet_files)))
    rdd = spark_session.sparkContext.parallelize(indexed_parquet_file, len(parquet_files))
    rdd.foreach(
        lambda indexed_parquet_file: _from_parquet_to_webdataset(
            indexed_parquet_file, webdataset_hdfs_dir, filter_fn
        )
    )


class WebDatasetSampleWriter:
    def __init__(self, shard_id, webdataset_hdfs_dir, tmp_dir):
        shard_name = f"{shard_id:05d}"
        self.local_path = os.path.join(tmp_dir, shard_name+".tar") 
        self.hdfs_path = os.path.join(webdataset_hdfs_dir, shard_name+".tar") 
        self.tarwriter = wds.TarWriter(self.local_path)

    def write(self, sample):
        self.tarwriter.write(sample)

    def close(self):
        self.tarwriter.close()
        fs, _ = fsspec.core.url_to_fs(self.hdfs_path)
        fs.put(self.local_path, self.hdfs_path)


def _from_parquet_to_webdataset(indexed_parquet_file, webdataset_hdfs_dir, filter_fn):
    index_file, parquet_file = indexed_parquet_file
    with tempfile.TemporaryDirectory() as tmp:
        try:
            parquet_file = parquet_file.replace("viewfs", "hdfs")
            wdsw = WebDatasetSampleWriter(
                shard_id=index_file, tmp_dir=tmp, webdataset_hdfs_dir=webdataset_hdfs_dir
            )
            pdf = pq.read_table(parquet_file).to_pandas()
            for index, row in pdf.iterrows():
                if not filter_fn(row):
                    continue
                print(f"Processing row {index}")
                title = "" if row["title"] is None else row["title"]
                description = "" if row["description"] is None else row["description"]
                text = ''.join((title, "\n", description))
                meta = {col: row[col] for col in pdf.columns if col != "image"}
                sample = {
                    "__key__": "%010d" % (index_file*100000+index),
                    "txt": text,
                    "json": json.dumps(meta, indent=4, cls=NpEncoder)
                }
                if row["image"]:
                    sample["jpg"] = row["image"]
                wdsw.write(sample)
            wdsw.close()
        except Exception as e: 
            raise e
