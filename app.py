import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

from gradcam import make_gradcam_heatmap, overlay_heatmap


# ---------------- CONFIG ----------------
MODEL_PATH = "saved_model/pneumonia_cnn.h5"
IMG_SIZE = (224, 224)

# ---------------- CUSTOM CSS ----------------
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Professional Medical Fonts */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Main Title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        color: #4a5568;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Clinical Card Design */
    .clinical-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    
    .clinical-card-header {
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #718096;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Upload Section */
    .upload-section {
        background: white;
        border: 2px dashed #cbd5e0;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin: 2rem 0;
    }
    
    .upload-section:hover {
        border-color: #4299e1;
        background: #f7fafc;
    }
    
    /* Results Card */
    .result-card {
        background: white;
        border-left: 4px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    .result-card.normal {
        border-left-color: #48bb78;
        background: linear-gradient(90deg, #f0fff4 0%, #ffffff 100%);
    }
    
    .result-card.pneumonia {
        border-left-color: #f56565;
        background: linear-gradient(90deg, #fff5f5 0%, #ffffff 100%);
    }
    
    .result-label {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .result-label.normal {
        color: #2d7a4f;
    }
    
    .result-label.pneumonia {
        color: #c53030;
    }
    
    /* Confidence Metric */
    .confidence-container {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .confidence-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .confidence-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
        color: #2d3748;
    }
    
    /* Warning Banner */
    .medical-warning {
        background: linear-gradient(135deg, #fff5e6 0%, #ffe8cc 100%);
        border: 1px solid #f6ad55;
        border-left: 4px solid #ed8936;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1.5rem 0;
    }
    
    .warning-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #7c2d12;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .warning-text {
        font-size: 0.95rem;
        color: #92400e;
        line-height: 1.6;
    }
    
    /* Info Section */
    .info-section {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-left: 4px solid #0ea5e9;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1.5rem 0;
    }
    
    .info-text {
        font-size: 0.95rem;
        color: #0c4a6e;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Analysis Button */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.875rem 2rem;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        transition: all 0.3s ease;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
        transform: translateY(-1px);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a202c;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Image Display */
    .image-container {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    /* Grad-CAM Section */
    .gradcam-header {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #4299e1;
    }
    
    .gradcam-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #2d3748;
        margin: 0;
    }
    
    /* Spinner Override */
    .stSpinner > div {
        border-top-color: #2563eb !important;
    }
    
    /* File Uploader Styling */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #cbd5e0;
        border-radius: 12px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #4299e1;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: #718096 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        color: #2d3748 !important;
    }
                
                [data-testid="stExpander"] * {
    color: black !important;
}
    </style>
    """, unsafe_allow_html=True)

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Pneumonia Detection System | Medical AI",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
load_custom_css()

# ---------------- HEADER ----------------
st.markdown('<h1 class="main-title">🫁 Pneumonia Detection System</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Clinical decision support powered by deep learning for chest X-ray analysis</p>',
    unsafe_allow_html=True
)

# ---------------- MEDICAL DISCLAIMER ----------------
st.markdown("""
<div class="medical-warning">
    <div class="warning-title">⚕️ Medical Disclaimer</div>
    <div class="warning-text">
        This tool is designed for educational and screening purposes only. 
        It is NOT a substitute for professional medical diagnosis. 
        All results should be reviewed by a qualified healthcare professional.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

with st.spinner("Loading AI model..."):
    model = load_model()
    last_conv_layer_name = "last_conv_layer"

# ---------------- SYSTEM INFO ----------------
with st.expander("ℹ️ About This System", expanded=False):
    st.markdown("""
    ### Model Architecture
    - **Type:** Convolutional Neural Network (CNN)
    - **Input:** 224×224 RGB Chest X-ray Images
    - **Output:** Binary Classification (Normal / Pneumonia)
    - **Explainability:** Grad-CAM visualization
    
    ### Recommended Use
    - Upload high-quality frontal chest X-rays
    - Ensure proper image orientation
    - Review confidence scores carefully
    - Always consult with medical professionals
    """)

# ---------------- IMAGE UPLOAD ----------------
st.markdown('<div class="section-header">📤 Upload Chest X-ray</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Select a chest X-ray image (JPG, JPEG, or PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a frontal chest X-ray for pneumonia screening"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    st.markdown('<div class="section-header">🖼️ Input Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, IMG_SIZE)
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # ---------------- ANALYSIS BUTTON ----------------
    if st.button("🔬 Analyze X-ray", use_container_width=True):
        with st.spinner("Analyzing image with AI model..."):
            prediction = model.predict(img_input)[0][0]

        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        # ---------------- RESULTS DISPLAY ----------------
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)

        if prediction > 0.5:
            st.markdown("""
            <div class="result-card pneumonia">
                <div class="result-label pneumonia">🛑 Pneumonia Detected</div>
                <p style="color: #742a2a; margin: 0; font-size: 0.95rem;">
                    The model has identified radiological features consistent with pneumonia.
                    Immediate clinical review is recommended.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card normal">
                <div class="result-label normal">✅ Normal Appearance</div>
                <p style="color: #22543d; margin: 0; font-size: 0.95rem;">
                    The model has not detected significant pneumonia indicators.
                    Routine follow-up as clinically indicated.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence Score
        st.markdown(f"""
        <div class="confidence-container">
            <div class="confidence-label">Model Confidence</div>
            <div class="confidence-value">{confidence * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- GRAD-CAM VISUALIZATION ----------------
        st.markdown("""
        <div class="gradcam-header">
            <div class="gradcam-title">🔍 Model Explainability (Grad-CAM)</div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Generating explainability heatmap..."):
            heatmap = make_gradcam_heatmap(
                img_input,
                model,
                last_conv_layer_name
            )

            gradcam_img = overlay_heatmap(
                heatmap,
                img_resized,
            )

        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(
            gradcam_img,
            caption="Grad-CAM Activation Map - Regions of Interest",
            use_column_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-section">
            <p class="info-text">
                <strong>How to interpret:</strong> The heatmap highlights lung regions that most influenced 
                the model's prediction. Red/warm colors indicate areas of high activation. This visualization 
                helps clinicians understand the model's decision-making process and verify that predictions 
                are based on relevant anatomical features.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- RECOMMENDATIONS ----------------
        st.markdown('<div class="section-header">📋 Clinical Recommendations</div>', unsafe_allow_html=True)
        
        if prediction > 0.5:
            st.markdown("""
            <div class="clinical-card">
                <ul style="margin: 0; padding-left: 1.25rem; color: #2d3748; line-height: 1.8;">
                    <li>Immediate review by a radiologist or attending physician</li>
                    <li>Consider correlation with clinical symptoms and patient history</li>
                    <li>Evaluate need for additional imaging (lateral view, CT scan)</li>
                    <li>Initiate appropriate treatment protocols if clinically indicated</li>
                    <li>Follow institutional guidelines for pneumonia management</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="clinical-card">
                <ul style="margin: 0; padding-left: 1.25rem; color: #2d3748; line-height: 1.8;">
                    <li>No immediate radiological concern for pneumonia</li>
                    <li>Continue clinical correlation with patient symptoms</li>
                    <li>Consider repeat imaging if clinical suspicion remains high</li>
                    <li>Maintain standard follow-up protocols</li>
                    <li>Document AI-assisted screening in patient records</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div class="upload-section">
        <p style="font-size: 1.1rem; color: #4a5568; margin: 0;">
            👆 Upload a chest X-ray image to begin analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #718096; font-size: 0.875rem;">
    <p style="margin: 0;">
        Developed by <strong style="color:#2d3748;">Aaseem Mhaskar</strong>, M.Tech (AI)
    </p>
    <p style="margin: 0.4rem 0;">
        <a href="https://github.com/aaseem22" target="_blank" style="color:#4a90e2; text-decoration:none;">
            GitHub
        </a>
        &nbsp;|&nbsp;
        <a href="https://www.linkedin.com/in/aaseem-mhaskar-007279203" target="_blank" style="color:#0a66c2; text-decoration:none;">
            LinkedIn
        </a>
    </p>
    <p style="margin: 0.4rem 0 0 0;">
        For Research & Educational Use Only
    </p>
</div>

""", unsafe_allow_html=True)