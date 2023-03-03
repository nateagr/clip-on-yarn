"""Training scrips"""
import logging
import math
import os
import time
from contextlib import suppress
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn
import wandb
from cluster_pack import filesystem
from tf_yarn.pytorch.model_ckpt import find_latest_ckpt
from torch.utils.data import DataLoader

from clip_on_yarn.config import Config
from clip_on_yarn.loss import ClipLoss
from clip_on_yarn.model.freeze import apply_freezing_strategy
from clip_on_yarn.model.model import mCLIP

logger = logging.getLogger()
CONFIG = Config()


def get_start_epoch() -> int:
    start_epoch = 0
    if CONFIG.ckpt_dir:
        latest_ckpt = find_latest_ckpt(CONFIG.ckpt_dir)
        if not latest_ckpt:
            return start_epoch
        start_epoch = int(latest_ckpt.split("/")[-1].split(".")[0].split("_")[-1])
        start_epoch += 1
    return start_epoch


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def step(optimizer, scaler):
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def train_one_epoch(
    model: mCLIP,
    trainloader: DataLoader,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    scheduler: Callable,
    device: str,
    precision: str,
    enable_wandb: bool,
    profiler: Optional[torch.profiler.profile],
    local_loss: bool = True,
) -> None:
    """Train and evaluate for one epoch"""

    def _get_progress() -> str:
        num_samples = batch_count * batch_size * world_size * accumulate_grad_batches
        percent_complete = 100.0 * batch_count / num_batches_per_epoch
        return f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"

    autocast = torch.cuda.amp.autocast if precision == "amp" else suppress
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    loss = ClipLoss(local_loss=local_loss, rank=rank, world_size=world_size)
    CONFIG.train_cfg.shared_epoch.set_value(epoch)
    num_batches = CONFIG.train_cfg.num_batches
    samples_per_epoch = CONFIG.train_cfg.num_samples
    accumulate_grad_batches = CONFIG.train_cfg.accumulate_grad_batches
    num_batches_per_epoch = num_batches // accumulate_grad_batches
    sample_digits = math.ceil(math.log(samples_per_epoch + 1, 10))
    if CONFIG.apply_freezing_strategy:
        model.module = apply_freezing_strategy(model.module, epoch)  # type: ignore[arg-type]
    if profiler:
        profiler.start()
    logging_n_steps = 50
    batch_time_acc = 0.0
    end = time.perf_counter()
    model.train()  # Make sure the model is in train mode
    if accumulate_grad_batches > 1:
        accum_images, accum_text_input_ids, accum_text_attention_masks, accum_image_features, accum_text_features = (
            [],
            [],
            [],
            [],
            [],
        )
    for i, batch in enumerate(trainloader):
        i_accum = i // accumulate_grad_batches
        current_step = num_batches_per_epoch * epoch + i_accum
        scheduler(current_step)
        images, text_input_ids, text_attention_masks = batch
        images = images.to(device, non_blocking=True)
        text_input_ids = text_input_ids.squeeze().to(device, non_blocking=True)
        text_attention_masks = text_attention_masks.squeeze().to(device, non_blocking=True)
        batch_size = images.shape[0]
        data_time = time.time() - end
        optimizer.zero_grad(set_to_none=True)

        if accumulate_grad_batches == 1:
            with autocast():
                image_features, text_features, logit_scale = model.module(images, text_input_ids, text_attention_masks)
                total_loss = loss(image_features, text_features, logit_scale)
            backward(total_loss, scaler)
        else:
            with torch.inference_mode():
                with autocast():
                    chunk_image_features, chunk_text_features, _ = model.module(
                        images, text_input_ids, text_attention_masks
                    )
                accum_image_features.append(chunk_image_features)
                accum_text_features.append(chunk_text_features)

                accum_images.append(images)
                accum_text_input_ids.append(text_input_ids)
                accum_text_attention_masks.append(text_attention_masks)
            # If (i + 1) % accumulate_grad_batches is not zero, move on to the next batch.
            if ((i + 1) % accumulate_grad_batches) > 0:
                continue
            optimizer.zero_grad()
            for j in range(accumulate_grad_batches):
                images = accum_images[j]
                text_input_ids = accum_text_input_ids[j]
                text_attention_masks = accum_text_attention_masks[j]
                with autocast():
                    chunk_image_features, chunk_text_features, logit_scale = model.module(
                        images, text_input_ids, text_attention_masks
                    )
                    image_features = torch.cat(
                        accum_image_features[:j] + [chunk_image_features] + accum_image_features[j + 1 :]
                    )
                    text_features = torch.cat(
                        accum_text_features[:j] + [chunk_text_features] + accum_text_features[j + 1 :]
                    )
                    total_loss = loss(image_features, text_features, logit_scale)
                backward(total_loss, scaler)
        step(optimizer, scaler)
        if accumulate_grad_batches > 1:
            (
                accum_images,
                accum_text_input_ids,
                accum_text_attention_masks,
                accum_image_features,
                accum_text_features,
            ) = (
                [],
                [],
                [],
                [],
                [],
            )
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.inference_mode():
            model.module.logit_scale.clamp_(0, 4.6052)  # type: ignore[union-attr]

        batch_time = time.perf_counter() - end
        batch_time_acc += batch_time
        end = time.perf_counter()
        batch_count = i_accum + 1
        if (i_accum % logging_n_steps == 0) or (batch_count == num_batches_per_epoch):
            logger.info(
                f"[{os.getpid()}] {_get_progress()}"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {model.module.logit_scale.data:.3f}"  # type: ignore[union-attr]
            )

            if rank == 0:
                log_data = {
                    "loss": total_loss.item(),
                    "data_time": data_time,
                    "batch_time": batch_time,
                    "scale": model.module.logit_scale.data.item(),  # type: ignore[union-attr]
                    "lr": optimizer.param_groups[0]["lr"],
                    "samples_per_second": world_size
                    * batch_size
                    * logging_n_steps
                    * accumulate_grad_batches
                    / batch_time_acc,
                }
                if enable_wandb:
                    for name, val in log_data.items():
                        name = "train/" + name
                        wandb.log({name: val, "step": current_step})

            batch_time_acc = 0

        if profiler:
            profiler.step()

    if profiler:
        profiler.stop()
