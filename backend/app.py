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
        'internal': {
            'material': 'recycled kraft paper',
            'reason': 'Eco-friendly cushioning for books, pens, notebooks and other stationery; reduces waste and is recyclable.'
        },
        'external': {
            'material': 'corrugated envelopes',
            'reason': 'Protective and lightweight, suitable for most stationery items, and made from recycled paper.'
        }
    },
    'clothing_fashion': {
        'internal': {
            'material': 'compostable polybags',
            'reason': 'Protects garments from dust/moisture while decomposing naturally in compost.'
        },
        'external': {
            'material': 'jute bags',
            'reason': 'Reusable and biodegradable material, perfect for shipping apparel sustainably.'
        }
    },
    'cosmetic_bottle': {
        'internal': {
            'material': 'paper mesh wrap',
            'reason': 'Provides shock absorption for fragile cosmetic bottles while being biodegradable.'
        },
        'external': {
            'material': 'compostable boxes',
            'reason': 'Sturdy and sustainable boxes ideal for cosmetics packaging.'
        }
    },
    'cosmetic_tube': {
        'internal': {
            'material': 'biodegradable bubble wrap',
            'reason': 'Protects small cosmetic tubes during transport while reducing plastic use.'
        },
        'external': {
            'material': 'eco-mailer',
            'reason': 'Lightweight, recyclable and eco-friendly for shipping compact items.'
        }
    },
    'electronics': {
        'internal': {
            'material': 'mushroom foam',
            'reason': 'Natural material molded to fit electronics and absorb impact; compostable.'
        },
        'external': {
            'material': 'corrugated fiberboard',
            'reason': 'Durable and protective outer packaging made from recycled paper fibers.'
        }
    },
    'food_dry': {
        'internal': {
            'material': 'cornstarch plastic',
            'reason': 'Compostable material that protects dry food like chocolates, snacks, and grains.'
        },
        'external': {
            'material': 'compostable wraps',
            'reason': 'Eco-packaging suitable for various dry food items, reducing landfill waste.'
        }
    },
    'home_decor': {
        'internal': {
            'material': 'shredded paper',
            'reason': 'Reusable and cushioning for vases, lamps, and other decor items.'
        },
        'external': {
            'material': 'recycled carton',
            'reason': 'Sturdy and recyclable box ideal for decorative and fragile items.'
        }
    },
    'jewelry': {
        'internal': {
            'material': 'paper pulp trays',
            'reason': 'Secure and molded compartments for rings, earrings, and more — fully recyclable.'
        },
        'external': {
            'material': 'reusable fabric boxes',
            'reason': 'Elegant and reusable boxes that reduce single-use waste.'
        }
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
            'packaging_suggestion': {
            'internal': suggestions[predicted_class]['internal'],
            'external': suggestions[predicted_class]['external']
       }})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
