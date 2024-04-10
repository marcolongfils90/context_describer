"""Module containing the basic functionalities to train the model."""

import os
import pybboxes as pbx
import torch
import torchvision
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
from context import logger
from context.entity.common_entities import TrainModelConfig


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.label_id_offset = 1
        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = [image for image in sorted(os.listdir(image_dir))
                     if image[-4:] == '.jpg']
        self.labels = [label for label in sorted(os.listdir(label_dir))
                       if label[-4:] == '.txt']

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.image_dir, img_name)

        img = torchvision.io.read_image(image_path)

        width = img.shape[2]
        height = img.shape[1]

        boxes = []
        labels = []

        # box coordinates for xml files are extracted and corrected for image size given
        with open(os.path.join(self.label_dir, self.labels[idx]), 'r') as file:
            lines = [line.rstrip() for line in file]
        bbox = [pd.to_numeric(line.split(" "))[1:] for line in lines]
        for box in bbox:
            boxes.append(pbx.convert_bbox(box, from_type="yolo", to_type="voc", image_size=(width,height)))

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        for line in lines:
          labels.append(int(line.split(" ")[0]) + self.label_id_offset)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["masks"] = tv_tensors.Mask(boxes)
        # image_id
        image_id = idx
        target["image_id"] = image_id


        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class TrainModel:
    """Class to train a model."""
    def __init__(self, config: TrainModelConfig):
        self.config = config
        self.model = None
        self.training_generator = None
        self.validation_generator = None

    def load_model(self):
        """Load and store a pretrained model."""
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        model.load_state_dict(torch.load(self.config.full_model_path))
        self.model = model
        logger.info("Untrained model correctly loaded.")

    def create_data_generator(self):
        """Split data into training and validation and augment them."""
        # use our dataset and defined transformations
        img_dir = os.path.join(self.config.training_data_path,
                               "images")
        label_dir = os.path.join(self.config.training_data_path,
                                 "labels")
        dataset = ObjectDetectionDataset(img_dir, label_dir,
                                         get_transform(train=True))
        dataset_test = ObjectDetectionDataset(img_dir, label_dir,
                                              get_transform(train=True))

        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-5])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-5:])

        # define training and validation data loaders
        self.training_generator = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )

        self.training_generator = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )

    def save_trained_model(self):
        """Stores model so that we can later use it for inference."""
        torch.save(self.model.state_dict(), self.config.trained_model_path)
        logger.info(f"Model correctly trained and"
                    f" stored in {self.config.trained_model_path}.")

    def train_model(self):
        """Trains the model using the data generators."""

        if not self.model:
            self.load_model()

        if not self.training_generator:
            self.create_data_generator()

        ###
        # ADD HERE CODE TO FINE TUNE
        ###

        self.save_trained_model()