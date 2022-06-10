import os
import time
import logging

import wandb
import torch
from torch.cuda.amp import autocast
import torch.distributed as dist
import torch.distributed.nn
from tf_yarn.pytorch import model_ckpt

from clip_on_yarn.loss import ClipLoss
from clip_on_yarn.validation.evaluate import evaluate


logger = logging.getLogger()


def model_inference(model, images, texts):
    image_features = model.encode_image(images)
    text_features = model.encode_text(texts)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return image_features, text_features, model.logit_scale.exp()


def train_and_evaluate(
    model, trainloader, epoch, optimizer, scaler, scheduler, device,
    precision, model_dir, tb_writer, enable_wandb, profiler, validation_config, validation_classifier,
    local_loss=True
):
    def _get_progress():
        num_samples = i * batch_size * world_size
        percent_complete = 100.0 * i / n_batches_per_epoch
        return f"Train Epoch: {epoch} [{num_samples}/{n_samples_per_epoch  * world_size} ({percent_complete:.0f}%)]\t"

    def _log_metrics(metrics, taining: bool):
        for name, val in metrics.items():
            name = f"{'train/' if taining else 'eval/'}" + name
            if tb_writer:
                tb_writer.add_scalar(name, val, current_step)
            if enable_wandb:
                wandb.log({name: val, 'step': current_step})

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    loss = ClipLoss(
        local_loss=local_loss,
        rank=rank,
        world_size=world_size
    )
    
    n_batches_per_epoch = len(trainloader)
    n_samples_per_epoch = len(trainloader.dataset)
    n_done_steps = n_batches_per_epoch * epoch

    if profiler:
        profiler.start()

    logging_n_steps = 100
    batch_time_acc = 0
    end = time.perf_counter()
    for i, batch in enumerate(trainloader):
        current_step = n_done_steps +  i
        scheduler(current_step)

        optimizer.zero_grad()

        images = batch['image_tensor']
        texts = batch['text_tokens']
        images = images.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)

        batch_size = images.shape[0]
        data_time = time.time() - end

        # with automatic mixed precision.
        if precision == "amp":
            with autocast():
                image_features, text_features, logit_scale = model_inference(model.module, images, texts)
                total_loss = loss(image_features, text_features, logit_scale)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            image_features, text_features, logit_scale = model_inference(model.module, images, texts)
            total_loss = loss(image_features, text_features, logit_scale)
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            model.module.logit_scale.clamp_(0, 4.6052)

        batch_time = time.perf_counter() - end
        batch_time_acc += batch_time
        end = time.perf_counter()

        if rank == 0 and validation_config and (validation_config.period_in_steps % i) == 0:
            model.eval()
            logger.info("Beginning zero_shot_eval")
            metrics= evaluate(
                model, validation_classifier, validation_config.dataloader,
                device, precision, validation_config.n_batches
            )
            logger.info("Finished zero_shot_eval")
            model.train()
            logger.info(
                f"[{os.getpid()}] {_get_progress()}"
                f"zero shot evaluation metrics: {metrics}"
            )
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
                    "scale":  model.module.logit_scale.data.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "samples_per_second": world_size * batch_size * logging_n_steps / batch_time_acc
                }
                _log_metrics(log_data, True)

            batch_time_acc = 0

        if profiler:
            profiler.step()
    
    if profiler:
        profiler.stop()

    if model_dir:
        others = dict()
        if scaler:
            others["scaler"] = scaler.state_dict()
        model_ckpt.save_ckpt(
            model_dir, model, optimizer, epoch, **others
        )
