import os
import logging
import uuid
import fsspec
from functools import partial
from typing import List, NamedTuple, Callable

import wandb
import torch
from torch.cuda.amp import GradScaler
from tf_yarn.pytorch import (
    run_on_yarn, TaskSpec, NodeLabel, PytorchExperiment,
    DataLoaderArgs
)
from tf_yarn.pytorch import model_ckpt
import torch.distributed as dist
from webdataset.extradatasets import FakeLength


from clip_on_yarn.dataset.dataset import create_webdataset
from clip_on_yarn.optimizer import get_adamw_optimize, cosine_lr
from clip_on_yarn.train import train_and_evaluate
from clip_on_yarn.model import load_pretrained_model, transform
from clip_on_yarn.hdfs import upload_dir
from clip_on_yarn.validation.evaluate import zero_shot_classifier


class ValidationConfig(NamedTuple):
    dataloader: torch.utils.data.dataloader.DataLoader
    classnames: List[str]
    templates: List[Callable[[str], str]]
    period_in_steps: int # validation period in steps
    n_batches: int # number of batches to process during validation


logger = logging.getLogger()
default_config = {
    "n_epochs": 32,
    "precision": "fp32",
    "learning_rate": 5.0e-4,
    "beta1": 0.9,
    "beta2": 0.98,
    "eps": 1.0e-6,
    "weight_decay": 0.2,
    "warmup": 10000, # number of steps to warm up
    "aggregate": True, # whether to gather all image and text embeddings
    "wandb_config": {
        "api_key": None,
        "entity": None,
        "project": None
    },
    "model_dir": None, # Directory where model is checkpointed
    "profiling_hdfs_dir": None, # Directory where profiling results will be written
    "validation_config_fn": None # function that generate and return an instance of ValidationConfig
}


def create_profiler(rank: int, local_dir: str):
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=20,
            repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(local_dir, worker_name=f'worker{rank}'),
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    )


def fill_config(args):
    if args is None:
        return default_config
    for name, value in default_config.items():
        if name not in args:
            args[name] = value
    return args


def training_loop(
    model: torch.nn.Module,
    trainloader: torch.utils.data.dataloader.DataLoader,
    device: str,
    rank: int,
    tb_writer,
    args
):
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    cudnn.deterministic = False

    torch.manual_seed(rank)

    config = fill_config(args)
    logger.info(f"config: {config}")

    n_epochs = config["n_epochs"]
    precision = config["precision"]
    learning_rate = config["learning_rate"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    eps = config["eps"]
    weight_decay = config["weight_decay"]
    warmup = config["warmup"]
    wandb_config = config["wandb_config"]
    model_dir = config["model_dir"]
    profiling_hdfs_dir = config["profiling_hdfs_dir"]

    if rank == 0 and wandb_config and wandb_config["api_key"] and wandb_config["entity"] \
        and wandb_config["project"]:
        enable_wandb = True
        os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
        os.environ["WANDB_ENTITY"] = wandb_config["entity"]
        os.environ["WANDB_PROJECT"] = wandb_config["project"]
        os.environ["WANDB_CONFIG_DIR"] = "."
        wandb.init(config=config, dir=".")
    else:
        enable_wandb = False

    validation_config = config["validation_config_fn"]() \
        if rank == 0 and config["validation_config_fn"] else None
    validation_classifier = zero_shot_classifier(
        model, validation_config.classnames, validation_config.templates, device
    ) if validation_config else None
    
    train_steps_per_epoch = len(trainloader)
    total_steps = train_steps_per_epoch * n_epochs
    logger.info(
        f"n_epochs: {n_epochs}; train_steps_per_epoch: {train_steps_per_epoch}; "
        f"total_steps: {total_steps}"
    )
    optimizer = get_adamw_optimize(model.module, weight_decay, learning_rate, beta1, beta2, eps)
    scaler = GradScaler() if precision == "amp" else None
    scheduler = cosine_lr(optimizer, learning_rate, warmup, total_steps)

    if precision == "amp" or precision == "fp32":
        model.module.float()

    start_epoch = 0
    if model_dir:
        ckpt = model_ckpt.load_latest_ckpt(model_dir, model, optimizer, device)
        if ckpt:
            start_epoch = ckpt["epoch"]
            if scaler is not None and 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])

    profiler = None
    if profiling_hdfs_dir:
        profiling_local_dir = f"./{str(uuid.uuid4())}"
        profiler = create_profiler(rank, profiling_local_dir)
    
    for epoch in range(start_epoch, n_epochs):
        train_and_evaluate(
            model, trainloader, epoch, optimizer, scaler, scheduler, device,
            precision, model_dir, tb_writer, enable_wandb, profiler, validation_config, validation_classifier
        )
    if enable_wandb:
        wandb.finish()

    if profiling_hdfs_dir and profiling_local_dir:
        logger.info("Uploading profiling data to HDFS")
        upload_dir(profiling_local_dir, profiling_hdfs_dir)


def get_experiment_fn(model_hdfs_path, trainset_path, batch_size, args=None):
    def _experiment_fn():
        model = load_pretrained_model(model_hdfs_path, "./" + str(uuid.uuid4()), True)
        
        webdataset = create_webdataset(trainset_path, transform(224, True)) \
            .shuffle(1000)
        num_workers = dist.get_world_size() if dist.is_initialized() else 1
        wds = FakeLength(webdataset, int(140000 * len(trainset_path) / num_workers))

        return PytorchExperiment(
            model=model,
            main_fn=partial(training_loop, args=args),
            train_dataset=wds,
            dataloader_args=DataLoaderArgs(batch_size=batch_size, num_workers=8, pin_memory=False),
            n_workers_per_executor=2
        )
    return _experiment_fn


if __name__ == "__main__":
    model_hdfs_path = "viewfs://root/user/g.racic/ViT-B-32.pt"
    trainset_path = "hdfs://root/user/u.tanielian/EU_img_titles/"
    fs, path = fsspec.core.url_to_fs(trainset_path)
    url_paths = fs.ls(path, detail=False)
    url_paths = ["pipe:hdfs dfs -cat viewfs://root"+ path for path in url_paths]
    batch_size = 32
    run_on_yarn(
        experiment_fn=get_experiment_fn(model_hdfs_path, url_paths, batch_size),
        task_specs={
            "worker": TaskSpec(memory=72*2**10, vcores=80, instances=2, label=NodeLabel.GPU)
        },
        queue="ml-gpu",
        pyenv_zip_path="viewfs://root/user/g.racic/envs/pytorch_distributed_env.pex"
    )
