from flask import Flask, request, jsonify
from prediction import *
import os
import base64
from flask_cors import CORS  # Import the CORS module


app = Flask(__name__)
CORS(app, resources={r"/": {"origins": "*"}})

UPLOAD_FOLDER = os.path.basename('.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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
        result, prop = predict(temp_filename)



        if not result:
            return jsonify({"error": "Sorry! Unable to predict"}), 500

        # Debugging statements
        print("Prediction Result:", result)

        return jsonify({"result": result, "prop":prop}), 200

    except Exception as e:
        # Print the exception for debugging
        print("Error:", str(e))
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
