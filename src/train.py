import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from model import build_pneumonia_cnn
import os
from collections import Counter

# ----------------- CONFIG -----------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "Data")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_model", "pneumonia_cnn.h5")

MODEL_SAVE_PATH = "saved_model/pneumonia_cnn.h5"

# ----------------- DATA GENERATORS -----------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ----------------- CLASS WEIGHTS -----------------
# Handles imbalance: Pneumonia > Normal


counter = Counter(train_generator.classes)
print("Class distribution:", counter)

total = sum(counter.values())

# Compute class weights
class_weight = {
    cls: total / (len(counter) * count)
    for cls, count in counter.items()
}

print("Class Weights:", class_weight)

# ----------------- MODEL -----------------
model = build_pneumonia_cnn(input_shape=(224, 224, 3))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# ----------------- CALLBACKS -----------------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_recall",
        mode="max",
        save_best_only=True
    )
]

# ----------------- TRAINING -----------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)

print("Training completed. Best model saved.")
