'''from flask_cors import CORS
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("best_model.keras")

# Load class labels from file (auto sync with dataset)
with open("class_labels.txt", "r") as f:
    CLASS_LABELS = [line.strip() for line in f.readlines()]

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
    },
    'Biscuits': {
        'internal': {
            'material': 'kraft paper or compostable film',
            'reason': 'Keeps biscuits fresh while using recyclable materials.'
        },
        'external': {
            'material': 'recyclable cardboard box',
            'reason': 'Eco-friendly box suitable for shipping biscuit packs.'
        }
    },
    'DryFruits': {
        'internal': {
            'material': 'paper pouch / compostable bag',
            'reason': 'Protects nuts and dry fruits and is biodegradable.'
        },
        'external': {
            'material': 'cardboard box / paper bag / glass jar',
            'reason': 'Eco-friendly options for storing and shipping dry fruits.'
        }
    },
    'chocolates': {
        'internal': {
            'material': 'compostable foil wrap or paper-based wrap',
            'reason': 'Keeps chocolates fresh and protected while being biodegradable and reducing plastic waste.'
        },
        'external': {
            'material': 'recyclable cardboard box',
            'reason': 'Strong outer packaging that protects chocolates during shipping and is eco-friendly.'
        }
    },
    'Snacks': {
        'internal': {
            'material': 'compostable film or kraft paper pouch',
            'reason': 'Provides freshness and protection for chips, namkeen, and other snacks while reducing single-use plastic.'
        },
        'external': {
            'material': 'paperboard carton or recycled cardboard box',
            'reason': 'Sturdy and stackable, suitable for bulk snack shipments while being recyclable.'
        }
    },
}

# ==============================
# Preprocess image for model input
# ==============================
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB").resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# ==============================
# API endpoint for classification
# ==============================
@app.route("/classify", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img_file = request.files["image"]
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
            "product_type": predicted_class,
            "prediction_accuracy": confidence_display,
            "packaging_suggestion": {
                "internal": suggestions[predicted_class]["internal"],
                "external": suggestions[predicted_class]["external"]
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
'''


from flask_cors import CORS
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ==============================
# Load model & labels
# ==============================
model = tf.keras.models.load_model("best_model.keras")

with open("class_labels.txt", "r") as f:
    CLASS_LABELS = [line.strip() for line in f.readlines()]

# ==============================
# Packaging Suggestions
# ==============================
suggestions = {
    # ------------------- Existing categories -------------------
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
    },
    'Biscuits': {
        'internal': {
            'material': 'kraft paper or compostable film',
            'reason': 'Keeps biscuits fresh while using recyclable materials.'
        },
        'external': {
            'material': 'recyclable cardboard box',
            'reason': 'Eco-friendly box suitable for shipping biscuit packs.'
        }
    },
    'DryFruits': {
        'internal': {
            'material': 'paper pouch / compostable bag',
            'reason': 'Protects nuts and dry fruits and is biodegradable.'
        },
        'external': {
            'material': 'cardboard box / paper bag / glass jar',
            'reason': 'Eco-friendly options for storing and shipping dry fruits.'
        }
    },
    'chocolates': {
        'internal': {
            'material': 'compostable foil wrap or paper-based wrap',
            'reason': 'Keeps chocolates fresh and protected while being biodegradable and reducing plastic waste.'
        },
        'external': {
            'material': 'recyclable cardboard box',
            'reason': 'Strong outer packaging that protects chocolates during shipping and is eco-friendly.'
        }
    },
    'Snacks': {
        'internal': {
            'material': 'compostable film or kraft paper pouch',
            'reason': 'Provides freshness and protection for chips, namkeen, and other snacks while reducing single-use plastic.'
        },
        'external': {
            'material': 'paperboard carton or recycled cardboard box',
            'reason': 'Sturdy and stackable, suitable for bulk snack shipments while being recyclable.'
        }
    },

    # ------------------- New categories -------------------
    'mobile_wearables': {
        'internal': {
            'material': 'molded paper pulp trays',
            'reason': 'Holds phones, smartwatches securely while avoiding plastic inserts.'
        },
        'external': {
            'material': 'corrugated recyclable boxes',
            'reason': 'Strong outer box for safe delivery of mobiles and wearables.'
        }
    },
    'computing_devices': {
        'internal': {
            'material': 'molded pulp & corrugated dividers',
            'reason': 'Protects laptops, desktops, and other electronics from impact.'
        },
        'external': {
            'material': 'reinforced corrugated cartons',
            'reason': 'Durable eco-friendly outer packaging for heavy devices.'
        }
    },
    'cameras_drones': {
        'internal': {
            'material': 'biodegradable molded inserts',
            'reason': 'Keeps drones/cameras stable and protected during shipping.'
        },
        'external': {
            'material': 'compostable corrugated boxes',
            'reason': 'Eco-friendly and durable for fragile tech.'
        }
    },
    'remotes_calculators': {
        'internal': {
            'material': 'compostable film wrap',
            'reason': 'Lightweight and eco alternative to plastic pouches.'
        },
        'external': {
            'material': 'paper padded mailers',
            'reason': 'Slim protective mailers reduce material use.'
        }
    },
    'input_devices': {
        'internal': {
            'material': 'corrugated inserts or paper wrap',
            'reason': 'Prevents scratches and movement of keyboards/mice.'
        },
        'external': {
            'material': 'recycled cardboard cartons',
            'reason': 'Protective and stackable for bulk logistics.'
        }
    },
    'power_speakers': {
        'internal': {
            'material': 'biodegradable pulp holders',
            'reason': 'Keeps power banks & speakers secure in place.'
        },
        'external': {
            'material': 'kraft paperboard boxes',
            'reason': 'Strong eco-friendly outer pack.'
        }
    }
}

# ==============================
# Preprocess image for model input
# ==============================
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB").resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# ==============================
# API endpoint for classification
# ==============================
@app.route("/classify", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img_file = request.files["image"]
        image = Image.open(img_file)
        processed = preprocess_image(image)

        prediction = model.predict(processed)[0]
        predicted_class = CLASS_LABELS[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        confidence_display = "Unknown" if np.isnan(confidence) else f"{round(confidence * 100, 2)}%"

        # Safely fetch packaging info
        packaging_info = suggestions.get(predicted_class, {
            "internal": {"material": "N/A", "reason": "No packaging suggestion available"},
            "external": {"material": "N/A", "reason": "No packaging suggestion available"}
        })

        return jsonify({
            "product_type": predicted_class,
            "prediction_accuracy": confidence_display,
            "packaging_suggestion": packaging_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# Run Flask
# ==============================
if __name__ == "__main__":
    app.run(debug=True)

