from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import zipfile

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = os.path.join("models","model1.keras")
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

# Function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 128  # Your model's expected input size
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode='grayscale')

    img_array = img_to_array(img) / 255.0  # Normalize pixel values (0-1)
    img_array = np.expand_dims(img_array, axis=-1)  # Add grayscale channel dimension
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension

    # Perform prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = float(np.max(predictions))  # Convert to float

    # Map the predicted index to the corresponding class label
    predicted_label = class_labels[predicted_class_index]

    if predicted_label == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {predicted_label}", confidence_score

# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files.get('file')
        if file and file.filename.lower().endswith(('jpg','jpeg','png')):
            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Predict the tumor
            result, confidence = predict_tumor(file_path)

            # Return result along with image path for display
            return render_template('index.html',
                                    result=result,
                                    confidence=f"{confidence*100:.2f}%",
                                    file_path=f'/uploads/{file.filename}')

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
