"""Pipeline to train the model and store it for inference."""
from context.config import configuration
from context.components import train_model
from context import logger


STAGE_NAME = "Model Training"


class TrainModelPipeline:
    """Class for model training pipeline."""
    def __init__(self):
        pass

    def run(self):
        """Run the model training pipeline."""
        try:
            config = configuration.ConfigurationManager().get_training_config()
            model_config = train_model.TrainModel(config=config)
            model_config.load_model()
            model_config.create_data_generator()
            model_config.train_model()
            model_config.save_trained_model()
        except Exception as exc:
            raise exc


if __name__ == "__main__":
    try:
        logger.info(f"*** Start of stage: {STAGE_NAME}. ***")
        TrainModelPipeline().run()
        logger.info(f"*** Successfully completed stage: {STAGE_NAME}. ***")
    except Exception as e:
        logger.exception(e)