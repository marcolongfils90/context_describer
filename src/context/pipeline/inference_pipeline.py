"""Module containing the inference pipeline."""
import google.generativeai as genai
import numpy as np
import cv2
import tensorflow as tf
import torch
import torchvision
import torchvision.transforms.functional as transform
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from PIL import Image
from context import constants
from context.components import train_model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = genai.GenerativeModel('gemini-pro-vision')

DESCRIBE_SYSTEM_PROMPT = '''
    You are a system generating descriptions for cats in a cute way and with lots of cat-related puns.
    Provided with an image, you will describe the cats that you see in the image, giving details but staying concise.
    You can describe unambiguously what the cat is doing, its color, and breed if clearly identifiable.
    '''

def describe_image(pil_image, name):
    """Calls the OpenAI API and return the results."""
    context = f"The name of the cat is {name}, and don't forget the puns!"
    print(context)
    full_prompt = DESCRIBE_SYSTEM_PROMPT + context
    response = model.generate_content([full_prompt, pil_image], stream=True)
    response.resolve()

    return response.text


def process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (350, 350))
    img = tf.keras.applications.mobilenet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def get_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class InferencePipeline:
    """Class for the inference pipeline."""
    def __init__(self, image):
        self.image = image
        self.objects = None

    def predict(self):
        """Predicts attribute for the input."""
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        model.load_state_dict(torch.load(constants.MODEL_PATH))
        breed_model = load_model("artifacts/model/breed.h5")

        eval_transform = train_model.get_transform(train=False)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.eval()
        pil_image = Image.open(self.image)
        image = transform.to_tensor(pil_image)
        with torch.no_grad():
            x = eval_transform(image)
            # convert RGBA -> RGB and move to device
            x = x.to(device)
            predictions = model([x, ])
            pred = predictions[0]

        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        image = image[:3, ...]
        indices = torch.BoolTensor([(int(label) == constants.CAT_CLASS_ID) and
                                    (score >= constants.MIN_SCORE) for label, score in
                                    zip(pred["labels"], pred["scores"])])
        pred_labels = [f"{int(label)}: {score:.3f}" for label, score in
                       zip(pred["labels"][indices], pred["scores"][indices])]
        pred_boxes = pred["boxes"].long()
        label_plot = []
        boxes = pred_boxes[indices]
        description = ""
        classes = np.array(constants.CLASSES)
        for box in boxes:
            crop_image = image.numpy()[0][box[1]:box[3], box[0]:box[2]]
            test_images_np = process(crop_image)
            y_pred = classes[np.argmax(breed_model.predict(
                test_images_np,
                verbose=0), axis=1)]

            cat_name = 'Flora'
            if y_pred[0] in ['Abyssinian', 'Egyptian', 'Maine', 'Persian', 'Russian']:
                cat_name = 'Psoti'
            np_box = box.numpy()
            crop_pil_image = pil_image.crop((np_box[0], np_box[1], np_box[2], np_box[3]))
            description += describe_image(crop_pil_image, cat_name)
            label_plot.append(cat_name)

        # colors = plt.get_cmap("gnuplot")(np.linspace(0.2, 0.7, len(pred_labels)))
        # output_image = draw_bounding_boxes(image, pred_boxes[indices], pred_labels)
        #
        # masks = (pred["masks"][indices] > min_score).squeeze(1)

        ###
        # The idea is to pass each bounding box to a cat breed classifier,
        # and then pass each single bounded image with its classification to the LLM
        # separately, and then merge the descriptions provided for each single sub image.
        ###

        return description


# def describe_image(image, objects):
#     """Calls the OpenAI API and return the results."""
#     context = ""
#     for name, position in objects.items():
#         context += f"The name of the cat on the {position} is {name}, and "
#
#     context += "don't forget the puns!"
#     print(context)
#     full_prompt = DESCRIBE_SYSTEM_PROMPT + context
#     pil_image = Image.open(image)
#     response = model.generate_content([full_prompt, pil_image], stream=True)
#     response.resolve()
#
#     return response.text
#
#
# class InferencePipeline:
#     """Class for the inference pipeline."""
#     def __init__(self, image, objects):
#         self.image = image
#         self.objects = objects
#
#     def predict(self):
#         """Predicts attribute for the input."""
#         return describe_image(self.image, self.objects)
