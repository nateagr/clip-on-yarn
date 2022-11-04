"""Pytorch profiler utils"""

import torch


def create_profiler(rank: int, local_dir: str):
    """Return torch profiler"""
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=20, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(local_dir, worker_name=f"worker{rank}"),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    )
