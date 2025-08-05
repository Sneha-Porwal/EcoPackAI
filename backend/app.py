from flask_cors import CORS
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model('model.h5')  # Placeholder model file

CLASS_LABELS = ['skincare', 'ceramic_mug', 'glass_bottle', 'electronics', 'food_item']

def preprocess_image(image, target_size=(224, 224)):
    image = image.convert('RGB').resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = Image.open(request.files['image'])
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    predicted_class = CLASS_LABELS[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    suggestions = {
        'skincare': {'internal': 'cornstarch wrap', 'external': 'recycled cardboard'},
        'ceramic_mug': {'internal': 'mushroom foam', 'external': 'kraft box'},
        'glass_bottle': {'internal': 'paper wrap', 'external': 'recycled cardboard'},
        'electronics': {'internal': 'shredded paper', 'external': 'tape-free cardboard'},
        'food_item': {'internal': 'cornstarch bubble', 'external': 'kraft bag'}
    }

    return jsonify({
        'product_type': predicted_class,
        'confidence': confidence,
        'packaging_suggestion': suggestions.get(predicted_class, {})
    })

if __name__ == '__main__':
    app.run(debug=True)
