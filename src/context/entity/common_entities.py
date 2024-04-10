"""Module containing common entities used by the pipelines."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    untrained_model_path: Path
    source_breed_model_url: str
    params_input_size: list
    params_num_classes: int
    params_weights: str
    params_learning_rate: float


@dataclass(frozen=True)
class TrainModelConfig:
    root_dir: Path
    trained_model_path: Path
    full_model_path: Path
    breed_model_path: Path
    training_data_path: Path
    params_input_size: list
    params_batch_size: int
    params_augmentation: bool
    params_epoch: int