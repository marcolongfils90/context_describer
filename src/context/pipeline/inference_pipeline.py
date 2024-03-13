"""Module containing the inference pipeline."""
import google.generativeai as genai
from PIL import Image

model = genai.GenerativeModel('gemini-pro-vision')

DESCRIBE_SYSTEM_PROMPT = '''
    You are a system generating descriptions for cats in a cute and with lots of cat-related puns.
    Provided with an image, you will describe the cats that you see in the image, giving details but staying concise.
    You can describe unambiguously what the cat is doing, its color, and breed if clearly identifiable.
    '''


def describe_image(image):
    """Calls the OpenAI API and return the results."""
    pil_image = Image.open(image)
    response = model.generate_content([DESCRIBE_SYSTEM_PROMPT, pil_image], stream=True)
    response.resolve()

    return response.text


class InferencePipeline:
    """Class for the inference pipeline."""
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        """Predicts attribute for the input."""
        return describe_image(self.filename)
