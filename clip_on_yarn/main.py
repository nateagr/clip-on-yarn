import os
import logging
import uuid
import fsspec

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


from clip_on_yarn.dataset import create_webdataset
from clip_on_yarn.optimizer import get_adamw_optimize, cosine_lr
from clip_on_yarn.train import train
from clip_on_yarn.model import load_pretrained_model, transform
from clip_on_yarn.hdfs import upload_dir


logger = logging.getLogger()


def training_loop(
    model: torch.nn.Module,
    trainloader: torch.utils.data.dataloader.DataLoader,
    device: str,
    rank: int,
    tb_writer
):
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    cudnn.deterministic = False
    
    # Inputs
    n_epochs = 10
    precision = "fp32"
    learning_rate = 5.0e-4
    beta1 = 0.9
    beta2 = 0.98
    eps = 1.0e-6
    weight_decay = 0.2
    warmup = 10000 # number of steps to warm up
    aggregate = True # whether to gather all image and text embeddings
    enable_wandb = False
    model_save_ckpt_dir = None # Directory where to save model checkpoints
    n_steps_ckpt = 2000 # Model will be checkpointed every n_steps_ckpt steps
    model_load_ckpt_path = None # Path of a checkpoint to reload
    profiling_local_dir = None # f"profiling_result_{str(uuid.uuid4())}"
    profiling_hdfs_dir = os.path.join("viewfs://prod-am6/user/g.racic", profiling_local_dir) \
        if profiling_local_dir else None

    if rank == 0 and enable_wandb:
        os.environ["WANDB_API_KEY"] = None # Replace by your API key
        os.environ["WANDB_ENTITY"] = None # Replace by your entity name
        os.environ["WANDB_PROJECT"] = "clip-fine-tuning"
        os.environ["WANDB_CONFIG_DIR"] = "."
        config = {
            "n_epochs": n_epochs,
            "precision": precision,
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "weight_decay": weight_decay,
            "warmup": warmup,
            "aggregate": aggregate,
            "model_save_ckpt_dir": model_save_ckpt_dir,
            "n_steps_ckpt": n_steps_ckpt,
            "model_load_ckpt_path": model_load_ckpt_path
        }
        wandb.init(config=config, dir=".")
    
    train_steps_per_epoch = len(trainloader)
    total_steps = train_steps_per_epoch * n_epochs
    logger.info(
        f"n_epochs: {n_epochs}; train_steps_per_epoch: {train_steps_per_epoch}; "
        f"total_steps: {total_steps}"
    )
    preprocess_train = transform(model.module.visual.input_resolution, True)
    preprocess_val = transform(model.module.visual.input_resolution, False)
    optimizer = get_adamw_optimize(model.module, weight_decay, learning_rate, beta1, beta2, eps)
    scaler = GradScaler() if precision == "amp" else None
    scheduler = cosine_lr(optimizer, learning_rate, warmup, total_steps)

    if model_load_ckpt_path:
        model_ckpt.load_ckpt(model_load_ckpt_path, model, optimizer, device)

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=20,
            repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiling_local_dir, worker_name=f'worker{rank}'),
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) if profiling_local_dir else None
    
    start_epoch = 0
    for epoch in range(start_epoch, n_epochs):
        train(
            model, trainloader, epoch, optimizer, scaler, scheduler, device,
            precision, aggregate, model_save_ckpt_dir, n_steps_ckpt, tb_writer, enable_wandb, profiler
        )
    if rank == 0 and enable_wandb:
        wandb.finish()

    if profiling_local_dir:
        logger.info("Uploading profiling data to HDFS")
        upload_dir(profiling_local_dir, profiling_hdfs_dir)


def get_experiment_fn(model_hdfs_path, trainset_path, batch_size):
    def _experiment_fn():
        model = load_pretrained_model(model_hdfs_path, "./" + str(uuid.uuid4()), True)
        
        worker_id = dist.get_rank() if dist.is_initialized() else 0
        num_workers = dist.get_world_size() if dist.is_initialized() else 1
        
        trainset_subset = [path for n, path in enumerate(trainset_path) if n % num_workers == worker_id]
        webdataset = create_webdataset(trainset_subset, transform(224, True))
        wds = FakeLength(webdataset, 419_000 * len(trainset_subset))
        return PytorchExperiment(
            model=model,
            train_fn=training_loop,
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
