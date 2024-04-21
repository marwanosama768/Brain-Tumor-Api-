import cv2
import os
import numpy as np
from PIL import Image
from keras.models import load_model
from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route('/', methods=["post"])
def index():
    # load the model
    model = load_model('BTDCNN.h5')
    # loadind the photo
    image = request.files['image']
    image.save('img.jpg')

    image = cv2.imread('img.jpg')
    img = Image.fromarray(image)
    img = img.resize((64, 64))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)

    result = model.predict(input_img)

    results = ''
    if result == 0:
        results = {"diag" : "no tumor"}
    elif result == 1:
        results = {"diag": "tumor"}
    else:
        results = {"diag" : "none"}

    return jsonify(results)

if __name__ == "__main__":
    app.run('0.0.0.0',9090)