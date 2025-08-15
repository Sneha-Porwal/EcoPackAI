import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# === Configurable Parameters ===
DATASET_DIR = 'dataset'       # Path to dataset folder
IMAGE_SIZE = (224, 224)       # Resize all images to this size
BATCH_SIZE = 16               # Balanced for speed & memory
EPOCHS = 10                   # Number of training epochs

# === Image Data Generators ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2       # 80% training, 20% validation
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# === Model Architecture ===
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze base model weights

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),    # Extra dense layer for better learning
    layers.Dropout(0.3),                     # Prevent overfitting
    layers.Dense(train_data.num_classes, activation='softmax') 
])

# === Compile Model ===
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Train Model ===
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

model.save('model.h5')
print("Model saved as model.h5")
