# Pneumonia Detection System

This project is a deep learning-based medical imaging solution designed to detect pneumonia from chest X-ray images. It features a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras and an interactive web dashboard for clinical decision support.

## Project Overview

The system provides a complete pipeline from model training to explainable AI (XAI) deployment:

- **Model Architecture:** A custom CNN featuring multiple convolutional blocks, batch normalization, and dropout layers for robust feature extraction.
- **Explainability:** Integrated **Grad-CAM** (Gradient-weighted Class Activation Mapping) to highlight specific lung regions that influenced the model's decision.
- **Web Interface:** A professional, Streamlit-based dashboard for image uploading, real-time analysis, and clinical recommendations.

---

## File Structure

| File | Description |
| --- | --- |
| `model.py` | Contains the `build_pneumonia_cnn` architecture using TensorFlow/Keras. |
| `train.py` | Handles data augmentation, class imbalance via class weights, and model training. |
| `evaluate.py` | Generates performance metrics including Confusion Matrix, Classification Report, and ROC/AUC curves. |
| `app.py` | The Streamlit web application providing the user interface and results visualization. |
| `gradcam.py` | Implementation of the Grad-CAM algorithm for generating heatmaps on X-rays. |

---

## Key Features

### 1. Advanced CNN Architecture

The model uses a sequential design with specialized blocks:

- **Input Layer:** RGB images.
- **Convolutional Blocks:** Multiple layers of 2D convolutions with ReLU activation and MaxPooling.
- **Regularization:** Batch Normalization and Dropout layers to prevent overfitting.
- **Global Average Pooling:** Used before the final dense layers to reduce parameter count.

### 2. Robust Training Pipeline

- **Data Augmentation:** Utilizes `ImageDataGenerator` for rotation, zooming, and horizontal flips to improve generalization.
- **Class Imbalance Handling:** Automatically calculates class weights to ensure the model doesn't become biased toward the majority class.
- **Callback Strategy:** Implements `EarlyStopping` to prevent overfitting and `ReduceLROnPlateau` for fine-tuning the learning rate during training.

### 3. Clinical Web Dashboard

The Streamlit app includes:

- **Medical Disclaimer:** Clear warnings regarding the tool's use for educational purposes.
- **Visual Analysis:** Displays the original X-ray alongside the Grad-CAM activation map.
- **Confidence Scores:** Provides a percentage-based confidence level for every prediction.
- **Recommendations:** Generates specific clinical follow-up steps based on whether pneumonia was detected.

---

## Technical Requirements

To run this project, ensure you have the following dependencies installed:

- `tensorflow`
- `streamlit`
- `opencv-python`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `Pillow`

---

