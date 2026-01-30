import tensorflow as tf

model = tf.keras.models.load_model("saved_model/pneumonia_cnn.h5")
model.summary()
