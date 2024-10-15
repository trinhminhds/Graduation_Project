from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import base64
import re
import io
import joblib
import os

# Load the trained SVM model (replace with the correct path to your model)
MODEL_PATH = 'D:/Graduation_Project/Model/svm_digit_classifier.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

svm_model = joblib.load(MODEL_PATH)

app = Flask(__name__)


# Helper function to process image
def process_image(img):
    # Resize the image to 28x28 like MNIST
    img = img.resize((28, 28)).convert('L')  # Convert to grayscale
    img_np = np.array(img)

    # Normalize the image
    img_np = img_np.astype('float32') / 255

    # Flatten the image to 1D (required by SVM input)
    img_np = img_np.reshape(1, -1)

    return img_np


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')


# Route to handle prediction from canvas drawing
@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'prediction': 'Error: No image data found.'})

    try:
        img_data = data['image']
        # Decode base64 image
        img_str = re.search(r'base64,(.*)', img_data).group(1)
        img_bytes = io.BytesIO(base64.b64decode(img_str))
        img = Image.open(img_bytes)

        # Process the image and make a prediction
        processed_image = process_image(img)
        prediction = svm_model.predict(processed_image)

        print("Prediction Result:", prediction.shape)  # Debugging line
        # Convert prediction to a list or int before returning
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'prediction': f'Error in processing image: {str(e)}'})


# Route to handle image upload
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'prediction': 'Error: No image file uploaded.'})

    try:
        file = request.files['image']
        img = Image.open(file.stream)

        # Process the image and make a prediction
        processed_image = process_image(img)
        print("Processed Image Shape:", processed_image.shape)

        prediction = svm_model.predict(processed_image)
        prediction_digit = int(prediction[0])
        print("Prediction Result:", prediction_digit)
        # Convert prediction to a list or int before returning
        return jsonify({'prediction': prediction_digit})
    except Exception as e:
        return jsonify({'prediction': f'Error in processing image: {str(e)}'})
    

# Inside your Flask app
@app.route('/upload_image2', methods=['POST'])
def upload_image2():
    if 'image' not in request.files:
        return jsonify({'prediction': 'Error: No image file uploaded.'})

    try:
        file = request.files['image']
        img = Image.open(file.stream)

        # Process the image and make a prediction
        processed_image = process_image(img)
        prediction = svm_model.predict(processed_image)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'prediction': f'Error in processing image: {str(e)}'})


# Route to handle basic calculations
@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        first_number = int(data['firstNumber'])
        operation = data['operation'].strip()  # Trim whitespace
        second_number = int(data['secondNumber'])

        # Evaluate the expression based on the operation
        if operation == '+':
            result = first_number + second_number
        elif operation == '-':
            result = first_number - second_number
        elif operation == '*':
            result = first_number * second_number
        elif operation == '/':
            result = first_number / second_number if second_number != 0 else 'Error: Division by zero'
        else:
            return jsonify({'result': 'Error: Invalid operation'})

        return jsonify({'result': result})
    except ValueError:
        return jsonify({'result': 'Error: Invalid number'})
    except Exception as e:
        return jsonify({'result': f'Error in calculation: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True)
