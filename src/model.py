import tensorflow as tf
from tensorflow.keras import layers, models


def build_pneumonia_cnn(input_shape=(224, 224, 3)):
    """
    Custom CNN model for Pneumonia Detection from Chest X-rays
    Binary Classification: NORMAL vs PNEUMONIA
    """

    model = models.Sequential(name="Pneumonia_CNN_From_Scratch")

    # -------- Block 1 --------
    model.add(layers.Conv2D(32, (3, 3), padding="same",
                            activation="relu", input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    # -------- Block 2 --------
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    # -------- Block 3 --------
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu",name="last_conv_layer"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.30))

    # -------- Classification Head --------
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))  # Binary output

    return model
