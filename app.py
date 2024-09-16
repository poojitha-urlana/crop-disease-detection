from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the trained model
model = load_model('model/train_model.keras')

# Define class names and solutions as per your model's output
class_name = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Define solutions for each disease
solutions = {
    'Apple___Apple_scab': 'Apply fungicides, remove infected leaves, and improve air circulation around the plant.',
    'Apple___Black_rot': 'Prune infected branches, use resistant varieties, and apply appropriate fungicides.',
    'Apple___Cedar_apple_rust': 'Remove cedar trees nearby, use fungicides, and prune affected parts.',
    'Cherry_(including_sour)___Powdery_mildew': 'Apply fungicides, ensure good air circulation, and remove infected parts.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Use fungicides and practice crop rotation.',
    'Corn_(maize)___Common_rust_': 'Use rust-resistant varieties and apply fungicides.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant varieties, apply fungicides, and practice crop rotation.',
    'Grape___Black_rot': 'Apply fungicides and remove infected parts.',
    'Grape___Esca_(Black_Measles)': 'Prune affected vines, ensure proper vine management, and use resistant varieties.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply fungicides and remove infected leaves.',
    'Orange___Haunglongbing_(Citrus_greening)': 'Use resistant varieties, apply appropriate insecticides, and maintain proper nutrition.',
    'Peach___Bacterial_spot': 'Apply copper-based fungicides and remove infected plant parts.',
    'Pepper,_bell___Bacterial_spot': 'Use copper-based fungicides, practice crop rotation, and ensure good plant spacing.',
    'Potato___Early_blight': 'Apply fungicides, practice crop rotation, and remove infected plant debris.',
    'Potato___Late_blight': 'Use resistant varieties, apply fungicides, and remove infected parts.',
    'Squash___Powdery_mildew': 'Apply fungicides, ensure proper spacing, and remove infected parts.',
    'Strawberry___Leaf_scorch': 'Improve soil drainage, use mulch, and avoid overhead irrigation.',
    'Tomato___Bacterial_spot': 'Use resistant varieties, apply copper-based fungicides, and remove infected leaves.',
    'Tomato___Early_blight': 'Apply fungicides, use resistant varieties, and practice crop rotation.',
    'Tomato___Late_blight': 'Use resistant varieties, apply fungicides, and remove infected plant parts.',
    'Tomato___Leaf_Mold': 'Ensure good air circulation, remove affected leaves, and apply fungicides.',
    'Tomato___Septoria_leaf_spot': 'Use resistant varieties, apply fungicides, and remove infected leaves.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Use miticides and improve plant humidity.',
    'Tomato___Target_Spot': 'Apply fungicides and practice good crop management.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Use resistant varieties and control vector insects.',
    'Tomato___Tomato_mosaic_virus': 'Use resistant varieties, control aphid populations, and remove infected plants.'
}

@app.route('/')
def home():
    # Render the HTML file located in the 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.content_type.startswith('image'):
        return jsonify({'error': 'File is not an image'}), 400

    try:
        # Read the image file as a stream
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((128, 128))  # Resize the image to the input size expected by the model
        input_arr = img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)

        # Predict the class
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        prediction_label = class_name[result_index]
        solution = solutions.get(prediction_label, "No solution available")

        return jsonify({"prediction": prediction_label, "solution": solution})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
