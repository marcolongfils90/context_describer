"""Module containing the inference pipeline."""
import google.generativeai as genai
from PIL import Image

model = genai.GenerativeModel('gemini-pro-vision')

DESCRIBE_SYSTEM_PROMPT = '''
    You are a system generating descriptions for cats in a cute way and with lots of cat-related puns.
    Provided with an image, you will describe the cats that you see in the image, giving details but staying concise.
    You can describe unambiguously what the cat is doing, its color, and breed if clearly identifiable.
    '''


def describe_image(image, objects):
    """Calls the OpenAI API and return the results."""
    context = ""
    for name, position in objects.items():
        context += f"The name of the cat on the {position} is {name}, and "

    context += "don't forget the puns!"
    print(context)
    full_prompt = DESCRIBE_SYSTEM_PROMPT + context
    pil_image = Image.open(image)
    response = model.generate_content([full_prompt, pil_image], stream=True)
    response.resolve()

    return response.text


class InferencePipeline:
    """Class for the inference pipeline."""
    def __init__(self, image, objects):
        self.image = image
        self.objects = objects

    def predict(self):
        """Predicts attribute for the input."""
        return describe_image(self.image, self.objects)
