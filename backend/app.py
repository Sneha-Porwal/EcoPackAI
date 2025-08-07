from flask_cors import CORS
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('model.h5')

# Update to match your dataset folders
CLASS_LABELS = [
    'books_stationery',
    'clothing_fashion',
    'cosmetic_bottle',
    'cosmetic_tube',
    'electronics',
    'food_dry',
    'home_decor',
    'jewelry'
]

suggestions = {
    'books_stationery': {
        'internal': 'recycled kraft paper',
        'external': 'corrugated envelopes'
    },
    'clothing_fashion': {
        'internal': 'compostable polybags',
        'external': 'jute bags'
    },
    'cosmetic_bottle': {
        'internal': 'paper mesh wrap',
        'external': 'compostable boxes'
    },
    'cosmetic_tube': {
        'internal': 'biodegradable bubble wrap',
        'external': 'eco-mailer'
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

def preprocess_image(image, target_size=(224, 224)):
    image = image.convert('RGB').resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        img_file = request.files['image']
        image = Image.open(img_file)
        processed = preprocess_image(image)

        prediction = model.predict(processed)[0]
        print("Raw prediction:", prediction)  # DEBUG

        predicted_class = CLASS_LABELS[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        if np.isnan(confidence):
            confidence_display = "Unknown"
        else:
            confidence_display = f"{round(confidence * 100, 2)}%"

        return jsonify({
            'product_type': predicted_class,
            'prediction_accuracy': confidence_display,
            'packaging_suggestion': suggestions.get(predicted_class, {})
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
