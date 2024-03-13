"""Module with common functions."""
import base64


def decode_image(imgstring, filename):
    """Decodes image from a base 64 string."""
    imgdata = base64.b64decode(imgstring)
    with open(filename, 'wb') as f:
        f.write(imgdata)
        f.close()


def encode_image_to_base64(image_path):
    """Encodes an image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read())
