import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from keras.models import load_model, Model
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the MobileNetV2 model for feature extraction
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Load the face recognition model
model = load_model('models/face_recognition_model.keras')

# Load the class-to-label mapping
def load_class_mapping(file_path):
    class_mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            if "Class:" in line and "Label:" in line:
                parts = line.strip().split("->")
                class_name = parts[0].split(":")[1].strip()
                label = int(parts[1].split(":")[1].strip())
                class_mapping[label] = class_name
    return class_mapping

class_mapping = load_class_mapping('models/class_label_mapping.txt')
print(f"Class Mapping Loaded: {class_mapping}")

# Function to process the uploaded image
def process_image(file_path):
    try:
        print(f"Processing file: {file_path}")

        # Load and preprocess the image
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Failed to read uploaded image")

        # Resize the image for MobileNetV2
        img = cv2.resize(img, (224, 224))
        img = preprocess_input(img)  # MobileNetV2 preprocessing
        img = np.expand_dims(img, axis=0)

        # Extract embeddings using MobileNetV2
        features = feature_extractor.predict(img)
        print(f"Extracted features shape: {features.shape}")

        # Perform prediction using the face recognition model
        predictions = model.predict(features)
        print(f"Raw predictions: {predictions}")

        # Get the predicted class index
        predicted_class = np.argmax(predictions)
        print(f"Predicted class index: {predicted_class}")

        # Get the corresponding class name from the mapping
        predicted_label = class_mapping.get(predicted_class, "Unknown Class")
        print(f"Predicted label: {predicted_label}")

        return predicted_label, None

    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return None, {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            raise ValueError("No file part in the request")

        file = request.files['file']
        if file.filename == '':
            raise ValueError("No file selected for upload")

        # Save the uploaded file as PNG
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Convert to PNG using Pillow
        img = Image.open(file.stream)
        png_filename = os.path.splitext(filename)[0] + ".png"  # Change extension to .png
        png_file_path = os.path.join(app.config['UPLOAD_FOLDER'], png_filename)
        img = img.convert("RGB")  # Ensure compatibility for saving
        img.save(png_file_path, format="PNG")
        print(f"File converted to PNG and saved at: {png_file_path}")

        # Process the PNG file
        result, error = process_image(png_file_path)

        # Remove the file after processing
        try:
            os.remove(png_file_path)
            print(f"File deleted: {png_file_path}")
        except Exception as e:
            print(f"Error deleting file: {str(e)}")

        if error:
            print(f"Error returned by process_image: {error}")
            return jsonify(error), 500

        print(f"Result sent to client: {result}")
        return jsonify({"result": result}), 200

    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
