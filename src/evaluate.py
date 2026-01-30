import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = "Data"
MODEL_PATH = "saved_model/pneumonia_cnn.h5"

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# ---------------- DATA GENERATOR ----------------
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ---------------- PREDICTIONS ----------------
y_true = test_generator.classes
y_prob = model.predict(test_generator)
y_pred = (y_prob > 0.4).astype(int).ravel()

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Pneumonia"],
    yticklabels=["Normal", "Pneumonia"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ---------------- CLASSIFICATION REPORT ----------------
print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Normal", "Pneumonia"]
))

# ---------------- ROC CURVE ----------------
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
