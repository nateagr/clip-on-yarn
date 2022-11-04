"""Main script"""
import logging
import os
import uuid
from functools import partial
from typing import List

import fsspec
import torch
import torch.distributed as dist
import wandb
from tf_yarn.pytorch import (DataLoaderArgs, NodeLabel, PytorchExperiment,
                             TaskSpec, model_ckpt, run_on_yarn)
from torch.cuda.amp import GradScaler
from webdataset.extradatasets import FakeLength

from clip_on_yarn.config import Config
from clip_on_yarn.dataset.dataset import create_webdataset
from clip_on_yarn.model import load_pretrained_model, transform
from clip_on_yarn.optimizer import cosine_lr, get_adamw_optimize
from clip_on_yarn.train import train_and_evaluate
from clip_on_yarn.utils.hdfs import upload_dir
from clip_on_yarn.utils.profiler import create_profiler
from clip_on_yarn.utils.seed import seed_everything
from clip_on_yarn.validation.evaluate import (create_validation_dataloader,
                                              zero_shot_classifier)

logger = logging.getLogger()
CONFIG = Config()


def training_loop(
    model: torch.nn.Module,
    trainloader: torch.utils.data.dataloader.DataLoader,
    device: str,
    rank: int,
    tb_writer,
    **kwds,
):
    """Training loop"""
    seed_everything(rank)
    config = CONFIG.update(kwds)
    logger.info(f"config: {config}")

    n_epochs = config.n_epochs
    precision = config.precision
    learning_rate = config.learning_rate
    beta1 = config.beta1
    beta2 = config.beta2
    eps = config.eps
    weight_decay = config.weight_decay
    warmup = config.warmup
    wandb_config = config.wandb_config
    model_dir = config.model_dir
    profiling_hdfs_dir = config.profiling_hdfs_dir
    validation_config = config.validation_config

    if rank == 0 and wandb_config and wandb_config["api_key"] and wandb_config["entity"] and wandb_config["project"]:
        enable_wandb = True
        os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
        os.environ["WANDB_ENTITY"] = wandb_config["entity"]
        os.environ["WANDB_PROJECT"] = wandb_config["project"]
        os.environ["WANDB_CONFIG_DIR"] = "."
        wandb.init(config=config, dir=".")
    else:
        enable_wandb = False

    validation_dataloader = None
    validation_classifier = None
    if validation_config and rank == 0:
        validation_classifier = zero_shot_classifier(
            model, validation_config.classnames, validation_config.templates, device
        )
        validation_classifier = validation_classifier.type(torch.float32)
        validation_dataloader = create_validation_dataloader(
            validation_config.validation_webdataset_dir, validation_config.batch_size, validation_config.num_workers
        )

    train_steps_per_epoch = len(trainloader)
    total_steps = train_steps_per_epoch * n_epochs
    logger.info(f"n_epochs: {n_epochs}; train_steps_per_epoch: {train_steps_per_epoch}; " f"total_steps: {total_steps}")
    optimizer = get_adamw_optimize(model.module, weight_decay, learning_rate, beta1, beta2, eps)
    scaler = GradScaler() if precision == "amp" else None
    scheduler = cosine_lr(optimizer, learning_rate, warmup, total_steps)

    if precision in ("amp", "fp32"):
        model.module.float()

    start_epoch = 0
    if model_dir:
        ckpt = model_ckpt.load_latest_ckpt(model_dir, model, optimizer, device)
        if ckpt:
            start_epoch = ckpt["epoch"]
            logger.info(f"Successfully loaded latest checkpoint from {model_dir}")
            logger.info(f"Resuming training at epoch {start_epoch}")
            if scaler is not None and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])

    profiler = None
    if profiling_hdfs_dir:
        profiling_local_dir = f"./{str(uuid.uuid4())}"
        profiler = create_profiler(rank, profiling_local_dir)

    for epoch in range(start_epoch, n_epochs):
        train_and_evaluate(
            model,
            trainloader,
            epoch,
            optimizer,
            scaler,
            scheduler,
            device,
            precision,
            model_dir,
            tb_writer,
            enable_wandb,
            profiler,
            validation_config,
            validation_classifier,
            validation_dataloader,
        )
    if enable_wandb:
        wandb.finish()

    if profiling_hdfs_dir and profiling_local_dir:
        logger.info("Uploading profiling data to HDFS")
        upload_dir(profiling_local_dir, profiling_hdfs_dir)


def get_experiment_fn(model_hdfs_path: str, trainset_paths: List[str], batch_size: int, **kwds) -> PytorchExperiment:
    """Generate tf_yarn PytorchExperiment"""

    def _experiment_fn():
        model = load_pretrained_model(model_hdfs_path, "./" + str(uuid.uuid4()), True)
        webdataset = create_webdataset(trainset_paths, transform(224, True)).shuffle(1000)
        num_workers = dist.get_world_size() if dist.is_initialized() else 1
        wds = FakeLength(
            webdataset, int(140000 * len(trainset_paths) / num_workers)
        )  # 14_000 should be the number of samples per shard?

        return PytorchExperiment(
            model=model,
            main_fn=partial(training_loop, **kwds),
            train_dataset=wds,
            dataloader_args=DataLoaderArgs(batch_size=batch_size, num_workers=8, pin_memory=False),
            n_workers_per_executor=2,
        )

    return _experiment_fn


if __name__ == "__main__":
    
    fs, path = fsspec.core.url_to_fs(CONFIG.trainset_folder)
    url_trainset_paths = fs.ls(path, detail=False)
    url_trainset_paths = ["pipe:hdfs dfs -cat viewfs://root" + path for path in url_trainset_paths]
    run_on_yarn(
        experiment_fn=get_experiment_fn(CONFIG.model_hdfs_path, url_trainset_paths, CONFIG.batch_size),
        task_specs={"worker": TaskSpec(memory=72 * 2**10, vcores=80, instances=2, label=NodeLabel.GPU)},
        queue="ml-gpu",
        # pyenv_zip_path=CONFIG.pyenv_zip_path,
    )
