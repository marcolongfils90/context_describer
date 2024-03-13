"""Simple web App to run context_describer."""
import os
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from PIL import Image
from src.context.pipeline import inference_pipeline
# from vertexai.generative_models import GenerativeModel, Image

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    """Client App."""
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = inference_pipeline.InferencePipeline(self.filename)


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    """Initial page for the app."""
    return render_template("index.html")




@app.route("/train", methods=["GET","POST"])
@cross_origin()
def train_route():
    """Trains the model on the app."""
    os.system("dvc repro")
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_route():
    """Callbacks for the predict button."""
    image = request.json['image']
    # common.decode_image(Image.open(image), clApp.filename)
    clApp.filename = Image.open(image)
    result = clApp.classifier.predict()
    return [result]

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080)  # for AWS
