from flask_cors import CORS
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

# === App Setup ===
app = Flask(__name__)
CORS(app)

# === Load Trained Model ===
model = tf.keras.models.load_model('model.h5')

# === Your Class Labels ===
CLASS_LABELS = [
    'books_stationery',
    'clothing_fashion',
    'cosmetics',
    'electronics',
    'food_dry',
    'home_decor',
    'jewelry'
]

# === Packaging Suggestions ===
suggestions = {
    'books_stationery': {
        'internal': 'recycled kraft paper',
        'external': 'corrugated envelopes'
    },
    'clothing_fashion': {
        'internal': 'compostable polybags',
        'external': 'jute bags'
    },
    'cosmetics': {
        'internal': 'molded pulp',
        'external': 'cardboard boxes'
    },
    'electronics': {
        'internal': 'mushroom foam',
        'external': 'corrugated fiberboard'
    },
    'food_dry': {
        'internal': 'cornstarch plastic',
        'external': 'compostable wraps'
    },
    'home_decor': {
        'internal': 'shredded paper',
        'external': 'recycled carton'
    },
    'jewelry': {
        'internal': 'paper pulp trays',
        'external': 'reusable fabric boxes'
    }
}

# === Image Preprocessing ===
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert('RGB').resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# === Classification Endpoint ===
@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        img_file = request.files['image']
        image = Image.open(img_file)
        processed = preprocess_image(image)

        prediction = model.predict(processed)[0]
        predicted_class = CLASS_LABELS[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return jsonify({
            'product_type': predicted_class,
            'confidence': confidence,
            'packaging_suggestion': suggestions.get(predicted_class, {})
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
