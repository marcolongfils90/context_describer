from ml_project.config import configuration
from ml_project.components import data_ingestion
from ml_project import logger

STAGE_NAME = "Data Ingestion"


class DataIngestionPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            config = configuration.ConfigurationManager().get_data_ingestion_config()
            data_ingestion_config = data_ingestion.DataIngestion(config=config)
            data_ingestion_config.load_file()
            data_ingestion_config.unzip_file()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f"*** Start of stage: {STAGE_NAME}. ***")
        DataIngestionPipeline().run()
        logger.info(f"*** Successfully completed stage: {STAGE_NAME}. ***")
    except Exception as e:
        logger.exception(e)