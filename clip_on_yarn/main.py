"""Main script"""
import logging
import os
import pickle
import uuid
from functools import partial

import fsspec
import torch
import torch.distributed as dist
from tf_yarn.pytorch import DataLoaderArgs, PytorchExperiment, model_ckpt, run_on_yarn
from torch.cuda.amp import GradScaler
from torchvision.transforms import Compose
from transformers.tokenization_utils import PreTrainedTokenizer
from webdataset.extradatasets import FakeLength

import wandb
from clip_on_yarn.config import CONFIG
from clip_on_yarn.dataset.dataset import create_webdataset
from clip_on_yarn.model.load import load_model_tokenizer_and_transforms
from clip_on_yarn.model.model import mCLIP
from clip_on_yarn.optimizer import cosine_lr, get_adamw_optimize
from clip_on_yarn.train import train_and_evaluate
from clip_on_yarn.utils.hdfs import upload_dir
from clip_on_yarn.utils.profiler import create_profiler
from clip_on_yarn.utils.seed import seed_everything
from clip_on_yarn.validation.evaluate import create_validation_dataloader_per_lang
from clip_on_yarn.validation.templates import create_templates_per_lang_x_uc_id, create_uc_id_to_idx_mapping

logger = logging.getLogger()


def training_loop(
    model: mCLIP,
    trainloader: torch.utils.data.dataloader.DataLoader,
    device: str,
    rank: int,
    tb_writer,
    image_transform_val: Compose,
    tokenizer: PreTrainedTokenizer,
    **kwds,
):
    """Training loop"""
    seed_everything(rank)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Remove tokenizer error
    CONFIG.update(kwds)
    logger.info(f"config: {CONFIG}")
    n_epochs = CONFIG.train_cfg.n_epochs
    precision = CONFIG.train_cfg.precision
    learning_rate = CONFIG.train_cfg.learning_rate
    beta1 = CONFIG.train_cfg.beta1
    beta2 = CONFIG.train_cfg.beta2
    eps = CONFIG.train_cfg.eps
    weight_decay = CONFIG.train_cfg.weight_decay
    warmup = CONFIG.train_cfg.warmup
    wandb_config = CONFIG.wandb_config
    model_dir = CONFIG.model_dir
    profiling_hdfs_dir = CONFIG.profiling_hdfs_dir
    validation_config = CONFIG.valid_cfg
    if rank == 0 and wandb_config and wandb_config["api_key"] and wandb_config["entity"] and wandb_config["project"]:
        enable_wandb = True
        os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
        os.environ["WANDB_ENTITY"] = wandb_config["entity"]
        os.environ["WANDB_PROJECT"] = wandb_config["project"]
        os.environ["WANDB_CONFIG_DIR"] = "."
        wandb.init(config=CONFIG.__dict__, dir=".")
    else:
        enable_wandb = False

    validation_dataloader_per_lang = None
    if validation_config and rank == 0:
        uc_id_to_idx_mapping = pickle.load(fsspec.filesystem("hdfs").open(CONFIG.uc_id_to_idx_mapping_path, "rb"))
        validation_dataloader_per_lang = create_validation_dataloader_per_lang(
            validation_config.webdataset_paths_per_lang,
            validation_config.batch_size,
            validation_config.num_workers,
            uc_id_to_idx_mapping,
            image_transform_val,
            tokenizer,
        )
        logger.info("Validation dataloader per lang created")
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
            validation_dataloader_per_lang,
            tokenizer,
        )
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

    def _experiment_fn():
        model, tokenizer, image_preprocessing_train, image_preprocessing_val = load_model_tokenizer_and_transforms(
            text_transformer_hdfs_path, visual_transformer_hdfs_path, tokenizer_hdfs_path, "./" + str(uuid.uuid4())
        )
        fs = fsspec.filesystem("hdfs")
        url_trainset_paths = fs.ls(trainset_base_path, detail=False)
        url_trainset_paths = [
            "pipe:hdfs dfs -cat viewfs://root" + path for path in url_trainset_paths if path.endswith(".tar")
        ]
        webdataset = create_webdataset(url_trainset_paths, image_preprocessing_train, tokenizer)
        num_workers = dist.get_world_size() if dist.is_initialized() else 1
        wds = FakeLength(webdataset, int(CONFIG.train_cfg.nb_of_samples / num_workers))
        return PytorchExperiment(
            model=model,
            main_fn=partial(training_loop, image_transform_val=image_preprocessing_val, tokenizer=tokenizer, **kwds),
            train_dataset=wds,
            dataloader_args=DataLoaderArgs(
                batch_size=CONFIG.train_cfg.batch_size, num_workers=CONFIG.train_cfg.num_workers, pin_memory=False
            ),
            n_workers_per_executor=CONFIG.train_cfg.n_workers_per_executor,
        )

    return _experiment_fn


if __name__ == "__main__":
    # Create artifacts
    create_templates_per_lang_x_uc_id()
    create_uc_id_to_idx_mapping()

    # Launch training
    run_on_yarn(
        experiment_fn=get_experiment_fn(
            CONFIG.text_transformer_hdfs_path,
            CONFIG.visual_transformer_hdfs_path,
            CONFIG.tokenizer_hdfs_path,
            CONFIG.train_cfg.webdataset_dir,
        ),
        task_specs={"worker": CONFIG.yarn_worker_spec},
        queue="ml-gpu",
    )
