"""HDFS utils"""
import os

from cluster_pack import filesystem


def upload_dir(local_dir: str, hdfs_dir: str) -> None:
    resolved_fs, _ = filesystem.resolve_filesystem_and_path(hdfs_dir)
    if not resolved_fs.exists(hdfs_dir):
        resolved_fs.mkdir(hdfs_dir)
    for f in os.listdir(local_dir):
        hdfs_file_path = os.path.join(hdfs_dir, f)
        local_file_path = os.path.join(local_dir, f)
        resolved_fs.put(local_file_path, hdfs_file_path)
