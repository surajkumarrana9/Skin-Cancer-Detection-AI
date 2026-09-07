# 🛡️ AI-Powered Skin Cancer Diagnostic System

An end-to-end Deep Learning web application designed to assist in the early detection of skin cancer. Built with **Python**, **TensorFlow**, and **Streamlit**.

---

## 🚀 Overview

Skin cancer is one of the most common forms of cancer, but early detection significantly improves survival rates. This project leverages the **MobileNetV2** architecture (Transfer Learning) to classify dermatological lesions into 7 diagnostic categories from the ISIC Archive.

### Key Features:
* **Instant Prediction:** Upload a dermoscopic image and get classification results in seconds.
* **Robust Performance:** High multiclass diagnostic sensitivity evaluated against benchmark ISIC samples.
* **Automated PDF Reports:** Generates a structured clinical diagnostic summary PDF using FPDF.
* **Streamlit UI:** Clean, intuitive interface for seamless user interaction.

---

## 🛠️ Tech Stack

* **Deep Learning:** TensorFlow, Keras, MobileNetV2
* **Web Framework:** Streamlit
* **Data Processing & Vision:** NumPy, Pandas, Pillow (PIL)
* **Report Generation:** FPDF
* **Language:** Python 3.x

---

## ⚙️ Installation & Usage

1. Clone the repository:
   git clone https://github.com/surajkumarrana9/Skin-Cancer-Detection-AI.git
   cd Skin-Cancer-Detection-AI

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Streamlit Application:
   streamlit run app.py

---

## 📊 Model & Dataset Details

* Base Architecture: MobileNetV2 (Pre-trained on ImageNet, fine-tuned for lesion classification)
* Dataset: ISIC (International Skin Imaging Collaboration) Archive
* Optimization: Data Augmentation (rotation, zoom, flip) and Dropout layers to prevent overfitting.

---

## ⚠️ Disclaimer

This project is built strictly for educational and research purposes. It is not intended to replace certified medical diagnosis or clinical consultation.

---

**Developed by Suraj Kumar Rana**  
📍 Ranchi, Jharkhand  
🔗 [LinkedIn](https://www.linkedin.com/in/surajrana-ai/) | [GitHub](https://github.com/surajkumarrana9)
