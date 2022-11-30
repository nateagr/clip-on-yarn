"""Training scrips"""
import logging
import os
import time
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.nn
from tf_yarn.pytorch import model_ckpt
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import wandb
from clip_on_yarn.config import ValidationConfig
from clip_on_yarn.loss import ClipLoss
from clip_on_yarn.model.mclip import mCLIP
from clip_on_yarn.validation.evaluate import compute_metrics

logger = logging.getLogger()


def model_inference(
    model: mCLIP, images: torch.Tensor, texts: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Inference"""
    image_features = model.encode_image(images)
    text_features = model.encode_text(texts)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return image_features, text_features, model.logit_scale.exp()


def train_and_evaluate(
    model: mCLIP,
    trainloader: DataLoader,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    scheduler: Callable,
    device: str,
    precision: str,
    model_dir: Optional[str],
    tb_writer,  # type: ignore
    enable_wandb: bool,
    profiler: Optional[torch.profiler.profile],
    validation_config: ValidationConfig,
    validation_classifier_per_lang: Optional[Dict[str, torch.Tensor]],
    validation_dataloader_per_lang: Optional[Dict[str, DataLoader]],
    local_loss: bool = True,
) -> None:
    """Train and evaluate for one epoch"""

    def _get_progress() -> str:
        num_samples = i * batch_size * world_size
        percent_complete = 100.0 * i / n_batches_per_epoch
        return f"Train Epoch: {epoch} [{num_samples}/{n_samples_per_epoch  * world_size} ({percent_complete:.0f}%)]\t"

    def _log_metrics(metrics: dict, taining: bool) -> None:
        for name, val in metrics.items():
            name = f"{'train/' if taining else 'eval/'}" + name
            if tb_writer:
                tb_writer.add_scalar(name, val, current_step)
            if enable_wandb:
                wandb.log({name: val, "step": current_step})

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    loss = ClipLoss(local_loss=local_loss, rank=rank, world_size=world_size)

    n_batches_per_epoch = len(trainloader)
    n_samples_per_epoch = len(trainloader.dataset)
    n_done_steps = n_batches_per_epoch * epoch

    if profiler:
        profiler.start()

    logging_n_steps = 100
    batch_time_acc = 0.0
    end = time.perf_counter()
    model.train()  # Make sure the model is in train mode
    for i, batch in enumerate(trainloader):
        current_step = n_done_steps + i
        scheduler(current_step)
        optimizer.zero_grad()
        images = batch["image_tensor"]
        text_input_ids = batch["input_ids"]
        text_attention_mask = batch["attention_mask"]
        images = images.to(device, non_blocking=True)
        text_input_ids = text_input_ids.squeeze().to(device, non_blocking=True)
        text_attention_mask = text_attention_mask.squeeze().to(device, non_blocking=True)
        batch_size = images.shape[0]
        data_time = time.time() - end

        # with automatic mixed precision.
        if precision == "amp" and scaler:
            with autocast():
                image_features, text_features, logit_scale = model.module(images, text_input_ids, text_attention_mask)
                total_loss = loss(image_features, text_features, logit_scale)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            image_features, text_features, logit_scale = model.module(images, text_input_ids, text_attention_mask)
            total_loss = loss(image_features, text_features, logit_scale)
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            model.module.logit_scale.clamp_(0, 4.6052)

        batch_time = time.perf_counter() - end
        batch_time_acc += batch_time
        end = time.perf_counter()

        if (
            rank == 0
            and validation_config
            and (i % validation_config.period_in_steps) == 0
            and validation_classifier_per_lang
            and validation_dataloader_per_lang
        ):
            logger.info("Starting zero shot evaluation")
            metrics = compute_metrics(
                model,
                validation_classifier_per_lang,
                validation_dataloader_per_lang,
                device,
                precision,
                validation_config.steps_per_lang,
            )
            model.train()
            logger.info("Finished zero_shot_eval")
            logger.info(f"[{os.getpid()}] {_get_progress()}" f"zero shot evaluation metrics: {metrics}")
            _log_metrics(metrics, False)

        if (i % logging_n_steps) == 0:
            logger.info(
                f"[{os.getpid()}] {_get_progress()}"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {model.module.logit_scale.data:.3f}"
            )

            if rank == 0:
                log_data = {
                    "loss": total_loss.item(),
                    "data_time": data_time,
                    "batch_time": batch_time,
                    "scale": model.module.logit_scale.data.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "samples_per_second": world_size * batch_size * logging_n_steps / batch_time_acc,
                }
                _log_metrics(log_data, True)

            batch_time_acc = 0

        if profiler:
            profiler.step()

    if profiler:
        profiler.stop()

    if model_dir:
        others = {}
        if scaler:
            others["scaler"] = scaler.state_dict()
        model_ckpt.save_ckpt(model_dir, model, optimizer, epoch, **others)
