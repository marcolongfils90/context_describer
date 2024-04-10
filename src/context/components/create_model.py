"""Module containing the basic functionalities to create a base model."""
import gdown
import torch
import torchvision
from context import logger
from context.entity.common_entities import BaseModelConfig


def get_model_instance_segmentation():
    # load a model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    return model


class BaseModel:
    """Base model class to load a pretrained model."""
    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    def load_model(self):
        """Load and store a pretrained model."""
        logger.info('Building model and restoring weights for fine-tuning')
        logger.info('Weights restored!')

        self.model = get_model_instance_segmentation()

    def save_untrained_model(self):
        """Store full model so that we can later train it."""
        torch.save(self.model.state_dict(), self.config.untrained_model_path)
        file_id = self.config.source_breed_model_url.split("/")[-2]
        prefix_str = "https://drive.google.com/uc?/export=download&id="
        gdown.download(prefix_str + file_id, str(self.config.root_dir) + "/breed.h5")
        logger.info(f"Full model correctly compiled and"
                    f" stored in {self.config.untrained_model_path}.")