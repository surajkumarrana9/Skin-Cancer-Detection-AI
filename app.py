import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from fpdf import FPDF
import datetime
from pathlib import Path

# 1. Page Settings
st.set_page_config(page_title="Skin Cancer Detector", layout="centered")
st.title("🛡️ AI Skin Cancer Diagnostic System")
st.write("---")

# Session State initialize karein (Data yaad rakhne ke liye)
if 'final_label' not in st.session_state:
    st.session_state.final_label = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None

# 2. Model Load Karo
@st.cache_resource
def load_my_model():
    model_path = Path(__file__).resolve().parent / "skin_cancer_model.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)

model = load_my_model()

# 3. Categories
class_names = [
    'Actinic_keratoses', 'Basal_cell_carcinoma', 'Benign_keratosis_like_lesions',
    'Dermatofibroma', 'Melanoma', 'Melanocytic_nevi', 'Vascular_lesions'
]

# PDF Function
def create_report(result, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="AI Skin Cancer Diagnostic Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date: {datetime.date.today()}", ln=True)
    pdf.cell(200, 10, txt=f"Patient Name: Suraj Kumar Rana (Demo)", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Diagnosis Results:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Detected Condition: {result}", ln=True)
    pdf.cell(200, 10, txt=f"AI Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(10)
    if result == 'Melanoma':
        pdf.set_text_color(255, 0, 0)
        pdf.multi_cell(0, 10, txt="URGENT: High probability of Melanoma. Consult a doctor immediately.")
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# 4. UI
uploaded_file = st.file_uploader("Image select karein...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Photo', use_container_width=True)
    
    if st.button('Analyze & Predict'):
        with st.spinner('AI Analyze kar raha hai...'):
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            
            # Session state mein save kar rahe hain
            st.session_state.final_label = class_names[np.argmax(predictions[0])]
            st.session_state.confidence = 100 * np.max(predictions[0])

    # Agar prediction ho chuki hai toh result aur PDF dikhao
    if st.session_state.final_label:
        st.write("---")
        st.subheader(f"Result: {st.session_state.final_label}")
        st.write(f"Confidence: {st.session_state.confidence:.2f}%")
        
        pdf_data = create_report(st.session_state.final_label, st.session_state.confidence)
        st.download_button(label="📥 Download PDF Report",
                           data=pdf_data,
                           file_name="Skin_Report.pdf",
                           mime="application/pdf")
        

st.write("---")
st.info("ℹ️ **Disclaimer:** Yeh AI model sirf educational purposes ke liye hai. Ise asali medical advice na maanein.")