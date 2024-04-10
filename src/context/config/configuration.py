"""Module with the utilities to handle configurations."""
import os
from pathlib import Path
from context import constants
from context.utils import common
from context.entity import common_entities

class ConfigurationManager:
    """Class to store and handle the configurations for the pipelines."""
    def __init__(self,
                 config_filepath: Path = constants.CONFIG_FILE_PATH,
                 params_filepath: Path = constants.PARAMS_FILE_PATH):
        self.config = common.read_yaml(config_filepath)
        self.params = common.read_yaml(params_filepath)

        common.create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> common_entities.DataIngestionConfig:
        """Extracts the data ingestion pipeline configuration."""
        config = self.config.data_ingestion
        common.create_directories([config.root_dir])

        return common_entities.DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_url=config.source_url,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    def get_model_config(self) -> common_entities.BaseModelConfig:
        """Extracts the base model creation pipeline configuration."""
        config = self.config.create_model
        common.create_directories([config.root_dir])

        return common_entities.BaseModelConfig(
            root_dir=Path(config.root_dir),
            untrained_model_path=Path(config.untrained_model_path),
            source_breed_model_url=config.source_breed_model_url,
            params_input_size=self.params.INPUT_SIZE,
            params_num_classes=self.params.NUM_CLASSES,
            params_weights=self.params.WEIGHTS,
            params_learning_rate=self.params.LEARNING_RATE,
        )

    def get_training_config(self) -> common_entities.TrainModelConfig:
        """Extracts the config to train the full model."""
        config = self.config.train_model
        common.create_directories([config.root_dir])

        return common_entities.TrainModelConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            full_model_path=Path(self.config.create_model.untrained_model_path),
            breed_model_path=Path(config.breed_model_path),
            training_data_path=Path(self.config.data_ingestion.unzip_dir),
            params_input_size=self.params.INPUT_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_augmentation=self.params.AUGMENTATION,
            params_epoch=self.params.EPOCHS,
        )