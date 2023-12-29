import os
import base64
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app, resources={r"/": {"origins": "*"}})

UPLOAD_FOLDER = os.path.basename('.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# URL of the model file to be downloaded
MODEL_DOWNLOAD_URL = "https://download.wetransfer.com/eugv/296b273b60f0b8e45f7e71d55254cf0320231229051202/37fe05436824a04fd6acdc21b5527a9da2130141/new_model_mobileNet.h5?cf=y&token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImRlZmF1bHQifQ.eyJleHAiOjE3MDM4Mjc0NjAsImlhdCI6MTcwMzgyNjg2MCwiZG93bmxvYWRfaWQiOiI0N2I2NzAxZS05YzgwLTQ5MzctOWU2Ni1jNGRlNDZlMTBhMWEiLCJzdG9yYWdlX3NlcnZpY2UiOiJzdG9ybSJ9.J7M4DGHAidQga3Z4ctvrkbnsc3umRg1wQ7WMXWvAErs"

def download_model():
    model_response = requests.get(MODEL_DOWNLOAD_URL)
    
    if model_response.status_code == 200:
        model_filename = os.path.join(app.config['UPLOAD_FOLDER'], "new_model_mobileNet.h5")
        with open(model_filename, 'wb') as model_file:
            model_file.write(model_response.content)
        return model_filename
    else:
        return None

# Initialize the model on startup
model_path = download_model()
if model_path is None:
    print("Unable to download the model. Exiting.")
    exit()

# Load the model once during startup
model = tf.keras.models.load_model(model_path)

def predict(img_path):
    labels = {0: 'Cardboard', 1: 'Glass', 2: 'Metal', 3: 'Paper', 4: 'Plastic', 5: 'Trash'}
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img, dtype=np.uint8)
    img = np.array(img) / 255.0
    predicted = model.predict(img[np.newaxis, ...])
    prob = np.max(predicted[0], axis=-1)
    prob = prob * 100
    prob = round(prob, 2)
    prob = str(prob) + '%'
    predicted_class = labels[np.argmax(predicted[0], axis=-1)]
    result = {"category": "", "predicted_class": "", "probability": ""}
    
    if predicted_class in ['Cardboard', 'Paper']:
        result["category"] = "Biodegradable"
        result["predicted_class"] = str(predicted_class)
        result["probability"] = str(prob)
    elif predicted_class in ['Metal', 'Glass', 'Plastic']:
        result["category"] = "Non-Biodegradable"
        result["predicted_class"] = str(predicted_class)
        result["probability"] = str(prob)
    else:
        result["category"] = "Categorizing Difficult"
        result["predicted_class"] = str(predicted_class)
        result["probability"] = str(prob)

    return result

@app.route("/", methods=['POST'])
def predict_trash_type():
    try:
        # Get base64-encoded image data from the JSON payload
        image_data = request.json.get("image", "")

        # Decode base64 data
        image_data_decoded = base64.b64decode(image_data)

        if not image_data_decoded:
            return jsonify({"error": "No image data provided"}), 400

        # Save the image to a temporary file
        temp_filename = os.path.join(app.config['UPLOAD_FOLDER'], "temp_image.png")
        with open(temp_filename, 'wb') as temp_file:
            temp_file.write(image_data_decoded)

        # Use the predict function to get the result
        result = predict(temp_filename)

        if not result:
            return jsonify({"error": "Sorry! Unable to predict"}), 500

        # Debugging statements
        print("Prediction Result:", result)

        return jsonify({"result": result}), 200

    except Exception as e:
        # Print the exception for debugging
        print("Error:", str(e))
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
