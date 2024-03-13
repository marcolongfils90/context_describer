"""Simple binary to test execution."""
import base64
from PIL import Image
from src.context.pipeline import inference_pipeline


# Function to encode the image
def encode_image(image_path):
    """Encodes an image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


FILENAME = "artifacts/images/IMG-20240121-WA0001.jpg"
description = inference_pipeline.InferencePipeline(Image.open(FILENAME)).predict()
print(description)
