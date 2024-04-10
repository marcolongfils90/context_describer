"""Simple binary to test execution."""
import base64
from src.context.pipeline import inference_pipeline
from src.context.pipeline import data_ingestion_pipeline
from src.context.pipeline import base_model_pipeline
from src.context.pipeline import train_model_pipeline

from src.context import logger


STAGE_NAME = "Data Ingestion"
try:
    logger.info(f"*** Start of stage: {STAGE_NAME}. ***")
    data_ingestion_pipeline.DataIngestionPipeline().run()
    logger.info(f"*** Successfully completed stage: {STAGE_NAME}. ***")
except Exception as e:
    logger.exception(e)


STAGE_NAME = "Base Model Creation"
try:
    logger.info(f"*** Start of stage: {STAGE_NAME}. ***")
    base_model_pipeline.BaseModelPipeline().run()
    logger.info(f"*** Successfully completed stage: {STAGE_NAME}. ***")
except Exception as e:
    logger.exception(e)


STAGE_NAME = "Model Training"
try:
    logger.info(f"*** Start of stage: {STAGE_NAME}. ***")
    train_model_pipeline.TrainModelPipeline().run()
    logger.info(f"*** Successfully completed stage: {STAGE_NAME}. ***")
except Exception as e:
    logger.exception(e)

# Function to encode the image
def encode_image(image_path):
    """Encodes an image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# FILENAME = "artifacts/data_ingestion/images/8f6446af-img10.jpg" # Both
FILENAME = "artifacts/data_ingestion/images/1cac97d1-img8.jpg"  # Flora alone
# FILENAME = "artifacts/data_ingestion/images/a863e3ae-img6.jpg"    # Psoti alone
#
# objects = {'Flora': 'left', 'Psoti': 'right'}
description = inference_pipeline.InferencePipeline(FILENAME).predict()
print(description)
