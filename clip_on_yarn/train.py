"""Training scrips"""
import logging
import os
import time
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn
from cluster_pack import filesystem
from tf_yarn.pytorch.model_ckpt import _unwrap_model
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

import wandb
from clip_on_yarn.config import CONFIG, ValidationConfig
from clip_on_yarn.loss import ClipLoss
from clip_on_yarn.model.freeze import apply_freezing_strategy
from clip_on_yarn.model.model import mCLIP
from clip_on_yarn.validation.evaluate import compute_metrics, zero_shot_classifier_per_lang

logger = logging.getLogger()


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
    validation_config: Optional[ValidationConfig],
    validation_dataloader_per_lang: Optional[Dict[str, DataLoader]],
    tokenizer: PreTrainedTokenizer,
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
    if CONFIG.apply_freezing_strategy:
        model.module = apply_freezing_strategy(model.module, epoch)
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
        optimizer.zero_grad(set_to_none=True)
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
            if ((i + 1) % CONFIG.train_cfg.accumulate_grad_batches == 0) or (i + 1 == n_batches_per_epoch):
                with autocast():
                    image_features, text_features, logit_scale = model.module(
                        images, text_input_ids, text_attention_mask
                    )
                    total_loss = loss(image_features, text_features, logit_scale)
                    total_loss = total_loss / CONFIG.train_cfg.accumulate_grad_batches
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with model.no_sync():
                    with autocast():
                        image_features, text_features, logit_scale = model.module(
                            images, text_input_ids, text_attention_mask
                        )
                        total_loss = loss(image_features, text_features, logit_scale)
                        total_loss = total_loss / CONFIG.train_cfg.accumulate_grad_batches
                    scaler.scale(total_loss).backward()
        else:
            if ((i + 1) % CONFIG.train_cfg.accumulate_grad_batches == 0) or (i + 1 == n_batches_per_epoch):
                image_features, text_features, logit_scale = model.module(images, text_input_ids, text_attention_mask)
                total_loss = loss(image_features, text_features, logit_scale)
                total_loss = total_loss / CONFIG.train_cfg.accumulate_grad_batches
                total_loss.backward()
                optimizer.step()
            else:
                with model.no_sync():
                    image_features, text_features, logit_scale = model.module(
                        images, text_input_ids, text_attention_mask
                    )
                    total_loss = loss(image_features, text_features, logit_scale)
                    total_loss = total_loss / CONFIG.train_cfg.accumulate_grad_batches
                    total_loss.backward()

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
            and validation_dataloader_per_lang
        ):
            logger.info("Starting zero shot evaluation")
            # Recompute the new class embeddings
            validation_classifier_per_lang = zero_shot_classifier_per_lang(model, tokenizer, device)
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
        dist.barrier()
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

    dist.barrier()
    if model_dir and rank == 0:
        others = {}
        if scaler:
            others["scaler"] = scaler.state_dict()
        state = {
            "model": _unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            **others,
        }
        resolved_fs, path = filesystem.resolve_filesystem_and_path(model_dir)
        if not resolved_fs.exists(model_dir):
            resolved_fs.mkdir(model_dir)
        model_ckpt_path = os.path.join(path, f"model_{epoch}.pt")
        with TemporaryDirectory() as tmpdir:
            tmp_file = os.path.join(tmpdir, f"model_{epoch}.pt")
            torch.save(state, tmp_file)
            resolved_fs.put(tmp_file, model_ckpt_path)
