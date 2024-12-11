from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import tempfile
import requests
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

# Konfigurasi URL model (GCS bucket publik)
MODEL_URL = os.getenv("MODEL_URL", "https://storage.googleapis.com/bucketsformodel/Recommender/recommender_fix_savemodel.h5")
IMAGE_MODEL_URL = os.getenv("IMAGE_MODEL_URL", "https://storage.googleapis.com/bucketsformodel/imageclass_model_new1.h5")
TEMP_DIR = tempfile.gettempdir()

# Fungsi untuk mengunduh model dari GCS
def download_model(model_url, model_path):
    response = requests.get(model_url)
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded successfully from {model_url}")
    else:
        raise Exception(f"Failed to download model from {model_url}, status code: {response.status_code}")

# Tentukan jalur model lokal
model_path = os.path.join(TEMP_DIR, 'recommender_fix_savemodel.h5')
image_model_path = os.path.join(TEMP_DIR, 'imageclass_model_new1.h5')

# Download model dari GCS (hanya sekali saat aplikasi dimulai)
download_model(MODEL_URL, model_path)
download_model(IMAGE_MODEL_URL, image_model_path)

# Memuat model
model = tf.keras.models.load_model(model_path, custom_objects={'l2_normalize': tf.linalg.l2_normalize})
image_model = tf.keras.models.load_model(image_model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request data
        input_data = request.json

        # Validate input data
        if not input_data or 'user_id' not in input_data:
            return jsonify({'error': 'Invalid input data'}), 400

        # Create a DataFrame from the input data
        num_rows = len(input_data['user_id'])
        data = pd.DataFrame({
            'user_id': input_data['user_id'],
            'rating_count': np.random.randint(5, 15, size=num_rows),
            'Zoo': np.random.uniform(3.0, 5.0, size=num_rows),
            'Historical': np.random.uniform(3.0, 5.0, size=num_rows),
            'Nature & Adventure': np.random.uniform(3.0, 5.0, size=num_rows),
            'Waterpark': np.random.uniform(3.0, 5.0, size=num_rows),
            'Museum': np.random.uniform(3.0, 5.0, size=num_rows),
            'Food': np.random.uniform(3.0, 5.0, size=num_rows),
            'Park': np.random.uniform(3.0, 5.0, size=num_rows),
        })

        # Prepare input features for the model
        input_features = data.drop(columns=['user_id']).values

        # Make predictions
        predictions = model.predict(input_features)

        # Process predictions
        relevance_scores = np.max(predictions, axis=1)  # Take the highest probability
        predicted_classes = np.argmax(predictions, axis=1)  # Get the index of the highest probability

        # Add results to DataFrame
        data['relevance_score'] = relevance_scores
        data['predicted_class'] = predicted_classes

        # Convert to JSON for the response
        result = data.to_dict(orient='records')

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['POST'])
def image_predict():
    labels = ['historical', 'makanan', 'museum', 'nature_adventure', 'park', 'waterpark', 'zoo']
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    imagefile.save(image_path)

    # Load the image using Keras
    image = load_img(image_path, target_size=(256, 256))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Predict the class of the image
    yhat = image_model.predict(image)
    predicted_class = labels[yhat.argmax()]
    confidence = yhat.max()

    return jsonify({
        "prediction": predicted_class,
        "confidence": f"{confidence * 100:.2f}%"
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
