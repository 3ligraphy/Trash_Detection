import os
import base64
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import json
from flask import Flask, request, jsonify, abort
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = os.path.basename('.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict(img_path):
    labels = {0: 'Cardboard', 1: 'Glass', 2: 'Metal', 3: 'Paper', 4: 'Plastic', 5: 'Trash'}
    img = image.load_img(img_path, target_size=(300, 300))  # Update target_size to match the model's expected input shape
    img = image.img_to_array(img, dtype=np.uint8)
    img = np.array(img) / 255.0
    model = tf.keras.models.load_model("trained_model.h5")
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
