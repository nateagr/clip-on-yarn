"""Main script"""
import gc
import logging
import os
import uuid
from functools import partial

import fsspec
import numpy as np
import torch
import wandb
from tf_yarn.pytorch import DataLoaderArgs, PytorchExperiment, model_ckpt, run_on_yarn
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from transformers.tokenization_utils import PreTrainedTokenizer

from clip_on_yarn.config import Config
from clip_on_yarn.data.dataset import create_webdataset, generate_wds_paths_and_samples_per_lang, get_number_of_samples
from clip_on_yarn.model.load import load_model_tokenizer_and_transforms
from clip_on_yarn.model.model import mCLIP
from clip_on_yarn.optimizer import cosine_lr, get_adamw_optimize
from clip_on_yarn.train import get_start_epoch, train_one_epoch
from clip_on_yarn.utils.hdfs import upload_dir
from clip_on_yarn.utils.profiler import create_profiler
from clip_on_yarn.utils.seed import seed_everything
from clip_on_yarn.validation.evaluate import create_validation_dataloader_per_lang, evaluate
from clip_on_yarn.validation.templates import create_templates_per_lang_x_uc_id, create_uc_id_to_idx_mapping

logger = logging.getLogger()


def training_loop(
    model: mCLIP,
    trainloader: DataLoader,
    device: str,
    rank: int,
    tb_writer,  # pylint: disable=unused-argument
    image_transform_val: Compose,
    tokenizer: PreTrainedTokenizer,
    **kwds,
):
    """Training loop"""
    seed_everything(rank)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Remove tokenizer error
    config = Config()
    config.update(kwds)
    logger.info(f"config: {config}")
    n_epochs = config.train_cfg.n_epochs
    accumulate_grad_batches = config.train_cfg.accumulate_grad_batches
    num_batches = config.train_cfg.num_batches
    precision = config.train_cfg.precision
    learning_rate = config.train_cfg.learning_rate
    beta1 = config.train_cfg.beta1
    beta2 = config.train_cfg.beta2
    eps = config.train_cfg.eps
    weight_decay = config.train_cfg.weight_decay
    warmup = config.train_cfg.warmup
    wandb_config = config.wandb_config
    ckpt_dir = config.ckpt_dir
    profiling_hdfs_dir = config.profiling_hdfs_dir
    validation_config = config.valid_cfg

    if rank == 0 and wandb_config and wandb_config["api_key"] and wandb_config["entity"] and wandb_config["project"]:
        enable_wandb = True
        os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
        os.environ["WANDB_ENTITY"] = wandb_config["entity"]
        os.environ["WANDB_PROJECT"] = wandb_config["project"]
        os.environ["WANDB_CONFIG_DIR"] = "."
        wandb.init(config=config.__dict__, dir=".")
    else:
        enable_wandb = False

    validation_dataloader_per_lang = None
    if validation_config and rank == 0:
        webdataset_paths_per_lang, samples_per_lang = generate_wds_paths_and_samples_per_lang(
            base_path=validation_config.webdataset_dir,
            max_samples=validation_config.max_samples,
        )
        validation_dataloader_per_lang = create_validation_dataloader_per_lang(
            webdataset_paths_per_lang,
            samples_per_lang,
            validation_config.batch_size,
            validation_config.num_workers,
            image_transform_val,
            tokenizer,
        )

    total_steps = (num_batches // accumulate_grad_batches) * n_epochs

    logger.info(
        f"n_epochs: {n_epochs}; train_steps_per_epoch: {num_batches //accumulate_grad_batches}; "
        f"total_steps: {total_steps}"
    )
    optimizer = get_adamw_optimize(model.module, weight_decay, learning_rate, beta1, beta2, eps)  # type: ignore[arg-type]
    scaler = GradScaler() if precision == "amp" else None

    scheduler = cosine_lr(optimizer, learning_rate, warmup, total_steps)

    if precision in ("amp", "fp32"):
        model.module.float()
    start_epoch = get_start_epoch()
    if ckpt_dir:
        ckpt = model_ckpt.load_latest_ckpt(ckpt_dir, model, optimizer, device)
        if ckpt:
            logger.info(f"Successfully loaded checkpoint from {ckpt_dir}")
            logger.info(f"Resuming training at epoch {start_epoch}")
            if scaler is not None and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            del ckpt
    profiler = None
    if profiling_hdfs_dir:
        profiling_local_dir = f"./{str(uuid.uuid4())}"
        profiler = create_profiler(rank, profiling_local_dir)
    for epoch in range(start_epoch, n_epochs):
        train_one_epoch(
            model,
            trainloader,
            epoch,
            optimizer,
            scaler,
            scheduler,
            device,
            precision,
            enable_wandb,
            profiler,
        )
        evaluate(
            model,
            tokenizer,
            validation_dataloader_per_lang,
            device,
            precision,
            epoch,
            enable_wandb,
        )
        # Saving the model
        if ckpt_dir:
            others = {}
            if scaler:
                others["scaler"] = scaler.state_dict()
            model_ckpt.save_ckpt(ckpt_dir, model, optimizer, epoch, **others)
    if enable_wandb:
        wandb.finish()

    if profiling_hdfs_dir and profiling_local_dir:
        logger.info("Uploading profiling data to HDFS")
        upload_dir(profiling_local_dir, profiling_hdfs_dir)


def get_experiment_fn(
    text_transformer_hdfs_path: str,
    visual_transformer_hdfs_path: str,
    tokenizer_hdfs_path: str,
    trainset_base_path: str,
    **kwds,
) -> PytorchExperiment:
    """Generate tf_yarn PytorchExperiment"""
    config = Config()

    def _experiment_fn():
        model, tokenizer, image_preprocessing_train, image_preprocessing_val = load_model_tokenizer_and_transforms(
            text_transformer_hdfs_path, visual_transformer_hdfs_path, tokenizer_hdfs_path, "./" + str(uuid.uuid4())
        )

        fs = fsspec.filesystem("hdfs")
        url_trainset_paths = fs.ls(trainset_base_path, detail=False)
        url_trainset_paths = [
            "pipe:hdfs dfs -cat viewfs://root" + path for path in url_trainset_paths if path.endswith(".tar")
        ]
        num_samples_per_epoch = (
            config.train_cfg.num_samples if config.train_cfg.num_samples else get_number_of_samples(trainset_base_path)
        )
        start_epoch = get_start_epoch()
        webdataset = create_webdataset(
            url_trainset_paths,
            image_preprocessing_train,
            tokenizer,
            is_train=True,
            num_samples=num_samples_per_epoch,
            batch_size=config.train_cfg.batch_size,
            epoch=start_epoch,
        )

        return PytorchExperiment(
            model=model,
            main_fn=partial(
                training_loop,
                image_transform_val=image_preprocessing_val,
                tokenizer=tokenizer,
                **kwds,
            ),
            train_dataset=webdataset,
            dataloader_args=DataLoaderArgs(
                batch_size=None, num_workers=config.train_cfg.num_workers, persistent_workers=False, drop_last=False
            ),
            n_workers_per_executor=config.train_cfg.n_workers_per_executor,
        )

    return _experiment_fn


if __name__ == "__main__":
    # Create artifacts
    config = Config()
    create_templates_per_lang_x_uc_id()
    create_uc_id_to_idx_mapping()

    # Launch training
    run_on_yarn(
        experiment_fn=get_experiment_fn(
            config.text_transformer_hdfs_path,
            config.visual_transformer_hdfs_path,
            config.tokenizer_hdfs_path,
            config.train_cfg.webdataset_dir,
        ),
        task_specs={"worker": config.yarn_worker_spec},
        queue="ml-gpu",
        pyenv_zip_path="viewfs://root/user/r.fabre/envs/.venv.pex.zip",
    )
