"""Module with the utilities to handle configurations."""
from pathlib import Path
from ml_project import constants
from ml_project.utils import common
from ml_project.entity import common_entities


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