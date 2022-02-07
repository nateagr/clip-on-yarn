from clip.clip import _transform
import torch
from torch.cuda.amp import GradScaler
from tf_yarn.pytorch import run_on_yarn, TaskSpec, NodeLabel, PytorchExperiment, DataLoaderArgs

from clip_on_yarn.optimizer import get_adamw_optimize, cosine_lr
from clip_on_yarn.train import train
from clip_on_yarn.model import load_model, preprocessing
from clip_on_yarn.parquet import ParquetDataset


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
        
    total_steps = len(trainloader) * n_epochs
    preprocess_train = _transform(model.module.visual.input_resolution)
    preprocess_val = _transform(model.module.visual.input_resolution)
    optimizer = get_adamw_optimize(model.module, weight_decay, learning_rate, beta1, beta2, eps)
    scaler = GradScaler() if precision == "amp" else None
    scheduler = cosine_lr(optimizer, learning_rate, warmup, total_steps)
    
    start_epoch = 0
    for epoch in range(start_epoch, n_epochs):
        trainloader.sampler.set_epoch(epoch)
        train(model, trainloader, epoch, optimizer, scaler, scheduler, device, precision, aggregate, tb_writer)


def experiment_fn():
    trainset_path = "hdfs://root/user/g.racic/filtered-image-text-pipeline/EU/resized-images/day=20220130000000"
    trainset = ParquetDataset(trainset_path).map(preprocessing)

    model = load_model("fp32")
    
    return PytorchExperiment(
        model=model,
        train_fn=training_loop,
        train_dataset=trainset,
        dataloader_args=DataLoaderArgs(batch_size=256, num_workers=0),
        n_workers_per_executor=2
    )


if __name__ == "__main__":
    run_on_yarn(
        experiment_fn=experiment_fn,
        task_specs={
            "worker": TaskSpec(memory=48*2**10, vcores=48, instances=2, label=NodeLabel.GPU)
        },
        queue="ml-gpu",
        pyenv_zip_path="viewfs://root/user/g.racic/envs/pytorch_distributed_env.pex"
    )
