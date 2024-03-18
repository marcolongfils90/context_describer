import gdown
import os
import zipfile
from ml_project import logger
from ml_project.entity.common_entities import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_file(self):
        """Load data based on the DataIngestionConfig."""
        try:
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading the data from {self.config.source_url}"
                        f" to file {self.config.local_data_file}")

            file_id = self.config.source_url.split("/")[-2]
            prefix_str = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix_str + file_id, str(self.config.local_data_file))

        except Exception as e:
            raise e


    def unzip_file(self):
        """Unzips a zipped file to extract its content."""
        os.makedirs(self.config.unzip_dir, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as f:
            f.extractall(self.config.unzip_dir)