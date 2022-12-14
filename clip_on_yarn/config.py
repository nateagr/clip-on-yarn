"""Configuration"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tf_yarn.pytorch import NodeLabel, TaskSpec

from clip_on_yarn.data.dataset import (generate_wds_paths_and_samples_per_lang,
                                       get_number_of_samples)
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

    webdataset_dir: str = "/user/cailimage/dev/users/r.fabre/mclip_finetuning/train"  # Path to webdataset
    batch_size: int = 64
    num_workers: int = 16
    n_workers_per_executor: int = 2
    n_epochs: int = 10
    precision: str = "fp32"
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1.0e-6
    weight_decay: float = 0.2
    warmup: int = 200_000  # number of steps to warm up
    aggregate: bool = True  # whether to gather all image and text embeddings
    nb_of_samples: int = get_number_of_samples("/user/cailimage/dev/users/r.fabre/mclip_finetuning/train")
    accumulate_grad_batches: int = 16


@dataclass
class ValidationConfig:
    """Validation parameters"""
    webdataset_paths_per_lang: Dict[str, List[str]] = field(default_factory=dict)
    steps_per_lang: Dict[str, int] = field(default_factory=dict)
    max_samples: int = 100_000
    batch_size: int = 64
    num_workers: int = 0
    period_in_steps: int = int(1e10)  # validation period in steps, int() for linting purposes
    
    def __post_init__(self) -> None:
        webdataset_paths_per_lang, samples_per_lang = generate_wds_paths_and_samples_per_lang(
            base_path="/user/cailimage/dev/users/r.fabre/mclip_finetuning/valid", max_samples=100_000
        )
        self.webdataset_paths_per_lang = webdataset_paths_per_lang
        self.steps_per_lang = {k: v // 32 for k, v in samples_per_lang.items()}

class Config(metaclass=SingletonMetaclass):
    """Singleton containing all the configuration"""

    def __init__(self) -> None:
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
        self.model_dir: Optional[str] = "viewfs://root/user/r.fabre/models/mdeberta_finetuned"

        # Directory where profiling results will be written
        self.profiling_hdfs_dir: Optional[str] = None

        # Model paths
        self.text_transformer_hdfs_path: str = "viewfs://root/user/r.fabre/models/mdeberta/model"
        self.visual_transformer_hdfs_path: str = "viewfs://root/user/r.fabre/models/mclip_xlm_roberta_large_vit_b_16_plus/visual_transformer/vit_b_16_plus_240_laion400m_e32.pt"  # pylint: disable=line-too-long
        self.tokenizer_hdfs_path: str = "viewfs://root/user/r.fabre/models/mdeberta/tokenizer"

        # Text model
        self.text_model = mDeBERTaTextEncoder

        # Yarn config
        self.yarn_worker_spec: TaskSpec = TaskSpec(memory=72 * 2**10, vcores=80, instances=10, label=NodeLabel.GPU)

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


CONFIG = Config()
