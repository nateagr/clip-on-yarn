"""Configuration"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from tf_yarn.pytorch import NodeLabel, TaskSpec

from clip_on_yarn.data.utils import SharedEpoch
from clip_on_yarn.model.model import mDeBERTaTextEncoder


class SingletonMetaclass(type):
    """Singleton base class"""

    _cls_instances: Dict = {}

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        if cls not in cls._cls_instances:
            instance = super().__call__(*args, **kwds)
            cls._cls_instances[cls] = instance
        return cls._cls_instances[cls]


@dataclass
class TrainingConfig:
    """Training parameters"""

    webdataset_dir: str = "/user/cailimage/dev/users/r.fabre/catalog_filtered_0.2_wds/train"  # Path to webdataset
    batch_size: int = 128
    num_workers: int = 8
    n_workers_per_executor: int = 2
    n_epochs: int = 50
    precision: str = "amp"
    learning_rate: float = 5e-6
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1.0e-6
    weight_decay: float = 0.2
    warmup: int = 0
    accumulate_grad_batches: int = 6  # accumulate_grad_batches*batch_size samples per batch per GPU
    dataset_resampled: bool = True
    num_samples: int = 90_000_000  # 1/10 of all training samples
    num_batches: int = 0  # Placeholder, will be updated in accordance with the total training size
    shared_epoch: SharedEpoch = None
    patch_dropout: float = 0


@dataclass
class ValidationConfig:
    """Validation parameters"""

    webdataset_dir: str = "/user/cailimage/dev/users/r.fabre/catalog_filtered_0.2_wds/valid"
    max_samples: int = 50_000
    batch_size: int = 128
    num_workers: int = 8


class Config(metaclass=SingletonMetaclass):
    """Singleton containing all the configuration"""

    def __init__(self) -> None:
        self.seed = 42
        # Model training configuration
        self.train_cfg: TrainingConfig = TrainingConfig()

        # Model validation configuration
        self.valid_cfg: Optional[ValidationConfig] = ValidationConfig()

        # Wandb config
        self.wandb_config: Dict = {
            "api_key": None,
            "entity": None,
            "project": None,
        }

        # Directory where model is checkpointed
        self.ckpt_dir: Optional[str] = "viewfs://root/user/r.fabre/models/mdeberta_finetuned"

        # Directory where profiling results will be written
        self.profiling_hdfs_dir: Optional[str] = None

        # Model paths
        self.text_transformer_hdfs_path: str = "viewfs://root/user/r.fabre/models/mdeberta/model"
        self.visual_transformer_hdfs_path: str = "viewfs://root/user/r.fabre/models/mclip_xlm_roberta_large_vit_b_16_plus/visual_transformer/vit_b_16_plus_240_laion400m_e32.pt"  # pylint: disable=line-too-long
        self.tokenizer_hdfs_path: str = "viewfs://root/user/r.fabre/models/mdeberta/tokenizer"

        # Text model
        self.text_model = mDeBERTaTextEncoder

        # Yarn config
        self.yarn_worker_spec: TaskSpec = TaskSpec(memory=72 * 2**10, vcores=80, instances=14, label=NodeLabel.GPU)

        # Class template path
        self.templates_per_lang_x_uc_id_path: str = "/user/r.fabre/clip_on_yarn/templates_per_lang_x_uc_id.pkl"

        # Label encoder path
        self.uc_id_to_idx_mapping_path: str = "/user/r.fabre/clip_on_yarn/uc_id_to_idx_mapping.pkl"

        # Freezing strategy
        self.apply_freezing_strategy: bool = True

    def update(self, kwds: dict) -> None:
        """Update config"""
        self.__dict__.update(kwds)

    def __repr__(self) -> str:
        return str(self.__dict__)
